"""Opt-in local USB Reachy Mini feedback; imports have no hardware side effects."""

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time


EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
EMOTION_NAMES = {"celebrate": "success1", "encourage": "helpful1",
                 "calm": "calming1", "thanks": "grateful1"}
EMOTION_TRANSITION_SECONDS = 0.8
EMOTION_SOUND_GAIN = 0.2

GAME_NAMES = {1: "Sound Sweeps", 2: "Target Tracker", 3: "Mixed Signals", 4: "Delayed Task Switching"}


@dataclass(frozen=True)
class FeedbackPlan:
    text: str
    gesture: str
    accuracy: float
    accuracy_high: bool
    fatigue: bool | None
    current_game: int
    current_level: int
    next_game: int
    next_level: int
    final_block: bool


def build_feedback(*, accuracy, accuracy_high, fatigue, current_game, current_level,
                   next_game, next_level, final_block=False):
    """Describe the final settings that will be sent, never recompute decisions."""
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be a finite fraction between 0 and 1")
    current_game, next_game = int(current_game), int(next_game)
    current_level, next_level = int(current_level), int(next_level)
    if current_game not in GAME_NAMES or next_game not in GAME_NAMES:
        raise ValueError("Unknown game identifier")
    if not 0 <= current_level <= 999 or not 0 <= next_level <= 999:
        raise ValueError("Expected a three-digit encoded game level")

    prefix = "Nice work!" if accuracy_high else "Thank you for your effort."
    parts = [f"{prefix} You got {accuracy:.0%} correct in that block."]
    gesture = "celebrate" if accuracy_high else "encourage"
    if fatigue is True:
        # Respond supportively without presenting a model flag as a diagnosis.
        parts.append("Take a moment to relax.")
        gesture = "calm"
    if final_block:
        gesture = "thanks"
        parts.append("That was the last block. Thank you for completing today's session.")
    elif next_game != current_game:
        # Levels belong to individual games: do not compare difficulty across games.
        parts.append(f"Next, we'll switch to {GAME_NAMES[next_game]}.")
        parts.append("Follow the instructions for the new game when it starts.")
    else:
        parts.append(f"Next, we'll continue with {GAME_NAMES[next_game]}.")
        old_difficulty, new_difficulty = (current_level // 10) % 10, (next_level // 10) % 10
        if new_difficulty > old_difficulty:
            parts.append("The difficulty will increase.")
        elif new_difficulty < old_difficulty:
            parts.append("The difficulty will decrease.")
        else:
            parts.append("The difficulty will stay the same.")
        if current_level // 100 != next_level // 100:
            parts.append("You'll also see a different task variation.")
        elif current_level % 10 != next_level % 10:
            parts.append("The task setup will also change slightly.")

    return FeedbackPlan(" ".join(parts), gesture, accuracy, bool(accuracy_high), fatigue,
                        current_game, current_level, next_game, next_level, bool(final_block))


FEEDBACK_TIMEOUT = 45


def _daemon_listening():
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
            return True
    except OSError:
        return False


def start_local_daemon(output_dir):
    """Reuse a local daemon or launch the SDK daemon once, before recording.

    A successfully started daemon stays available after the FACE session. Only a
    process started here that fails startup is terminated; existing services are
    never replaced or killed.
    """
    process = None
    try:
        for module in ("reachy_mini", "pyttsx3", "soundfile", "scipy"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError("Install feedback dependencies in the FACE environment: "
                                   "python -m pip install -r src/requirements-reachy.txt")
        if _daemon_listening():
            return True
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / "reachy_daemon.log"
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                   else {"start_new_session": True})
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "reachy_mini.daemon.app.main",
                 "--fastapi-host", "127.0.0.1", "--fastapi-port", "8000"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"USB daemon exited; see {log_path}")
            if _daemon_listening():
                print(f"[Reachy] Local USB daemon started; log: {log_path}")
                return True
            time.sleep(0.25)
        raise TimeoutError(f"USB daemon startup timed out; see {log_path}")
    except Exception as exc:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            except Exception as cleanup_error:
                print(f"[Reachy] Daemon cleanup failed: {cleanup_error}")
        print(f"[Reachy] Startup failed; continuing without robot feedback: {exc}")
        return False


def _synthesize_speech(text, path):
    """Use the OS English voice through one cross-platform TTS library."""
    import pyttsx3

    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices")
        english = []
        for voice in voices:
            for language in voice.languages:
                if isinstance(language, bytes):
                    language = language.decode("utf-8", errors="ignore")
                # Some drivers prefix the language tag with a priority byte.
                tag = str(language).lstrip("".join(chr(i) for i in range(33))).lower()
                if tag == "en" or tag.startswith(("en-", "en_")):
                    english.append(voice)
                    break
        if not english:
            raise RuntimeError("Install an English system text-to-speech voice before using Reachy feedback")
        default_voice = engine.getProperty("voice")
        voice = next((voice for voice in english if voice.id == default_voice), english[0])
        engine.setProperty("voice", voice.id)
        engine.setProperty("rate", 170)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
    finally:
        engine.stop()


def _load_emotion(name):
    """Read the official library from cache; never download during feedback."""
    from huggingface_hub import snapshot_download
    from reachy_mini.motion.recorded_move import RecordedMoves

    # RecordedMoves itself falls back to a network download on a cache miss.
    # Require a cached snapshot first so English feedback remains responsive.
    snapshot_download(EMOTIONS_DATASET, repo_type="dataset", local_files_only=True)
    move = RecordedMoves(EMOTIONS_DATASET).get(EMOTION_NAMES[name])
    if not math.isfinite(move.duration) or not 0 < move.duration <= 10:
        raise ValueError("Expected an official emotion clip lasting at most 10 seconds")
    return move


def _gesture(robot, move, stop=None):
    """Play one complete official emotion, then restore head, antennas and body."""
    import numpy as np

    pose = robot.get_current_head_pose().copy()
    joints, antenna_positions = robot.get_current_joint_positions()
    antennas = np.asarray(antenna_positions, dtype=float)
    body_yaw = float(joints[0])  # SDK joint order: body, six head motors.
    if antennas.shape != (2,) or not np.isfinite(antennas).all() or not math.isfinite(body_yaw):
        raise ValueError("Invalid robot joint positions")
    if stop is not None and stop.is_set():
        return 0
    robot.enable_motors()
    try:
        # The official sound is mixed into the speech stream at a lower gain.
        # Disable SDK audio here to avoid playing that sound a second time.
        robot.play_move(move, initial_goto_duration=EMOTION_TRANSITION_SECONDS, sound=False)
    finally:
        robot.goto_target(head=pose, antennas=antennas, duration=1.0, body_yaw=body_yaw)
    return 1


def _mix_emotion_audio(samples, rate, move):
    """Mix the official sound under speech, including its complete audio tail."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if move.sound_path is None:
        return samples, False
    effect, effect_rate = sf.read(move.sound_path, dtype="float32", always_2d=True)
    effect = effect.mean(axis=1)
    if not 0 < len(effect) / effect_rate <= 15 or not np.isfinite(effect).all():
        raise ValueError("Invalid official emotion audio")
    if effect_rate != rate:
        divisor = math.gcd(rate, effect_rate)
        effect = resample_poly(effect, rate // divisor, effect_rate // divisor).astype("float32")
    # SDK first moves to the clip's opening pose, then starts its trajectory.
    offset = round(EMOTION_TRANSITION_SECONDS * rate)
    mixed = np.zeros(max(len(samples), offset + len(effect)), dtype="float32")
    mixed[:len(samples)] = np.asarray(samples) * (1 - EMOTION_SOUND_GAIN)
    mixed[offset:offset + len(effect)] += effect * EMOTION_SOUND_GAIN
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak  # Leave headroom without hard clipping.
    return mixed, True


def _speak(media, samples, sample_rate, on_started=None):
    """Feed short chunks to the robot; wait until the final chunk has played."""
    from scipy.signal import resample_poly

    output_rate = media.get_output_audio_samplerate()
    if output_rate <= 0:
        raise RuntimeError("Reachy speaker is unavailable")
    if output_rate != sample_rate:
        divisor = math.gcd(output_rate, sample_rate)
        samples = resample_poly(samples, output_rate // divisor, sample_rate // divisor).astype("float32")
    chunk_size = max(1, output_rate // 10)
    try:
        media.start_playing()
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset:offset + chunk_size]
            media.push_audio_sample(chunk)
            if offset == 0 and on_started is not None:
                on_started()
            time.sleep(len(chunk) / output_rate)
        time.sleep(0.5)
    finally:
        media.stop_playing()


def _speak_with_gestures(robot, name, samples, rate):
    """Start motion with the first audio chunk and join it before disconnecting."""
    started, stop = threading.Event(), threading.Event()
    motion = {"gesture_completed": False, "gesture_plays": 0, "warnings": [],
              "emotion_dataset": EMOTIONS_DATASET, "emotion_name": EMOTION_NAMES[name],
              "emotion_sound_included": False}
    try:
        move = _load_emotion(name)
    except Exception as exc:
        motion["warnings"].append(f"Official emotion unavailable: {exc}. "
                                  "Cache the official emotions library before the session; see README.")
        _speak(robot.media, samples, rate)
        return motion
    try:
        samples, motion["emotion_sound_included"] = _mix_emotion_audio(samples, rate, move)
    except Exception as exc:
        motion["warnings"].append(f"Emotion sound unavailable; continuing with speech and motion: {exc}")
    motion["audio_duration_seconds"] = len(samples) / rate

    def animate():
        started.wait()
        if stop.is_set():
            return
        try:
            motion["gesture_plays"] = _gesture(robot, move, stop)
            motion["gesture_completed"] = motion["gesture_plays"] > 0
        except Exception as exc:
            motion["warnings"].append(f"Gesture failed: {exc}")

    thread = threading.Thread(target=animate, name="reachy-gesture", daemon=True)
    thread.start()
    try:
        _speak(robot.media, samples, rate, on_started=started.set)
    finally:
        stop.set()
        started.set()  # Release the waiter even if audio failed before starting.
        # Once playback starts, finish the short official clip and restore pose
        # before disconnecting, even if speech ends first or fails. Avoid SDK
        # cancel_move here because it also controls the shared audio pipeline.
        # The parent process still enforces the overall 45-second deadline.
        thread.join()
    return motion


def _perform_feedback(plan, folder):
    import numpy as np
    import soundfile as sf
    from reachy_mini import ReachyMini
    from reachy_mini.media.device_detection import Gst, get_audio_device

    # SDK falls back to the computer speaker if the USB audio device is absent.
    # Require the robot's device before creating its local media backend.
    Gst.init(None)
    if get_audio_device("Sink") is None:
        raise RuntimeError("Reachy Mini USB speaker was not detected")
    audio_path = Path(folder) / "speech.wav"
    _synthesize_speech(plan["text"], audio_path)
    samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    duration = len(samples) / rate
    if not 0 < duration <= 30 or not np.isfinite(samples).all():
        raise ValueError("Speech must contain finite audio lasting at most 30 seconds")
    with ReachyMini(connection_mode="localhost_only", spawn_daemon=False,
                   timeout=5, media_backend="local", automatic_body_yaw=False) as robot:
        motion = _speak_with_gestures(robot, plan["gesture"], samples, rate)
    return {"status": "partial" if motion["warnings"] else "completed", "error": None,
            **motion, "speech_duration_seconds": duration}


def give_block_feedback(output_dir, **settings):
    """Bound SDK/TTS hangs in a child process, log failures and let play continue."""
    report = {"status": "failed", "error": None}
    try:
        plan = asdict(build_feedback(**settings))
        report["plan"] = plan
        print(f"[Reachy] {plan['text']}")
        with tempfile.TemporaryDirectory(prefix="face-reachy-") as folder:
            request = Path(folder) / "request.json"
            result = Path(folder) / "result.json"
            request.write_text(json.dumps(plan), encoding="utf-8")
            worker = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(request), str(result)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=FEEDBACK_TIMEOUT, check=False)
            if worker.returncode or not result.exists():
                raise RuntimeError(f"Feedback worker exited: {worker.stderr[-500:]}")
            report.update(json.loads(result.read_text(encoding="utf-8")))
    except Exception as exc:
        report["error"] = str(exc)
    if report["error"]:
        print(f"[Reachy] Feedback failed; continuing game: {report['error']}")
    for warning in report.get("warnings", []):
        print(f"[Reachy] {warning}; continuing game")
    try:
        path = Path(output_dir) / "reachy_feedback.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[Reachy] Could not save feedback log: {exc}")
    return report


if __name__ == "__main__":
    # Internal isolation worker, not a separate user-facing launch mode.
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Enable feedback with: python src/face2_api.py SESSION GROUP reachy")
    request, result = map(Path, sys.argv[2:])
    try:
        report = _perform_feedback(json.loads(request.read_text(encoding="utf-8")), request.parent)
    except Exception as exc:
        report = {"status": "failed", "error": str(exc)}
    result.write_text(json.dumps(report), encoding="utf-8")
