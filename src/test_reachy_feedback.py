"""Offline checks: no robot movement, speaker playback, camera or game server.

Run: python src/test_reachy_feedback.py
"""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reachy_feedback import (build_feedback, give_block_feedback, start_local_daemon,
                            _synthesize_speech, _gesture, _speak, _perform_feedback,
                            _speak_with_gestures, _mix_emotion_audio, _load_emotion, EMOTIONS_DATASET, EMOTION_NAMES)


class FeedbackTests(unittest.TestCase):
    def plan(self, **changes):
        settings = dict(accuracy=0.92, accuracy_high=True, fatigue=False,
                        current_game=1, current_level=110, next_game=1, next_level=120)
        settings.update(changes)
        return build_feedback(**settings)

    def test_four_outcomes_and_unknown_fatigue(self):
        for high, fatigue, gesture in [(True, False, "celebrate"), (False, False, "encourage"),
                                      (True, True, "calm"), (False, True, "calm"), (True, None, "celebrate")]:
            with self.subTest(high=high, fatigue=fatigue):
                plan = self.plan(accuracy_high=high, fatigue=fatigue)
                self.assertEqual(plan.gesture, gesture)
                self.assertEqual("relax" in plan.text, fatigue is True)
                self.assertNotIn("you are tired", plan.text.lower())
                self.assertIs(plan.fatigue, fatigue)

    def test_describes_difficulty_and_variation_separately(self):
        self.assertIn("increase", self.plan().text)
        lower = self.plan(current_level=120, next_level=410).text
        self.assertIn("decrease", lower)  # Numeric 410 > 120 is NOT harder.
        self.assertIn("different task variation", lower)
        unchanged = self.plan(next_level=110).text
        self.assertIn("stay the same", unchanged)
        self.assertNotIn("different task variation", unchanged)

    def test_switch_does_not_compare_difficulty_across_games(self):
        text = self.plan(next_game=2, next_level=990).text
        self.assertIn("switch to Target Tracker", text)
        self.assertNotIn("increase", text)
        self.assertNotIn("decrease", text)

    def test_final_block_has_no_next_game_promise(self):
        text = self.plan(final_block=True, next_game=4).text
        self.assertIn("last block", text)
        self.assertNotIn("Next", text)
        self.assertNotIn("switch", text)
        self.assertEqual(self.plan(final_block=True).gesture, "thanks")

    def test_invalid_accuracy_is_not_spoken(self):
        for value in [float("nan"), float("inf"), -1, 92]:
            with self.assertRaises(ValueError):
                self.plan(accuracy=value)

    def test_timeout_logs_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as folder, patch("reachy_feedback.importlib.util.find_spec", return_value=Mock()), \
                patch("reachy_feedback.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 45)), \
                redirect_stdout(io.StringIO()):
            result = give_block_feedback(folder, accuracy=.8, accuracy_high=True, fatigue=None,
                                         current_game=1, current_level=110, next_game=1, next_level=110)
            self.assertEqual(json.loads((Path(folder) / "reachy_feedback.json").read_text()), result)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timed out", result["error"])

    def test_worker_result_is_logged_with_final_settings(self):
        def run(command, **kwargs):
            plan = json.loads(Path(command[-2]).read_text())
            self.assertEqual(plan["next_level"], 120)
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(kwargs["timeout"], 45)
            Path(command[-1]).write_text(json.dumps({"status": "completed", "error": None}))
            return SimpleNamespace(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as folder, patch("reachy_feedback.importlib.util.find_spec", return_value=Mock()), \
                patch("reachy_feedback.subprocess.run", side_effect=run), redirect_stdout(io.StringIO()):
            result = give_block_feedback(folder, accuracy=.9, accuracy_high=True, fatigue=False,
                                         current_game=1, current_level=110, next_game=1, next_level=120)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["next_level"], 120)

    def test_existing_daemon_is_reused(self):
        with patch("reachy_feedback.importlib.util.find_spec", return_value=Mock()), \
                patch("reachy_feedback._daemon_listening", return_value=True), \
                patch("reachy_feedback.subprocess.Popen") as launch:
            self.assertTrue(start_local_daemon("unused"))
        launch.assert_not_called()

    def test_launches_only_local_usb_daemon_and_leaves_it_running(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder, patch("reachy_feedback.importlib.util.find_spec", return_value=Mock()), \
                patch("reachy_feedback._daemon_listening", side_effect=[False, True]), \
                patch("reachy_feedback.subprocess.Popen", return_value=process) as launch, redirect_stdout(io.StringIO()):
            self.assertTrue(start_local_daemon(folder))
        command = launch.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "reachy_mini.daemon.app.main"])
        self.assertIn("127.0.0.1", command)
        self.assertNotIn("--sim", command)
        process.terminate.assert_not_called()

    def test_startup_timeout_terminates_only_owned_process_and_fails_open(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder, patch("reachy_feedback.importlib.util.find_spec", return_value=Mock()), \
                patch("reachy_feedback._daemon_listening", return_value=False), \
                patch("reachy_feedback.time.monotonic", side_effect=[0, 31]), \
                patch("reachy_feedback.subprocess.Popen", return_value=process), redirect_stdout(io.StringIO()):
            self.assertFalse(start_local_daemon(folder))
        process.terminate.assert_called_once()

    def test_missing_dependencies_continue_without_launch(self):
        with patch("reachy_feedback.importlib.util.find_spec", return_value=None), \
                patch("reachy_feedback.subprocess.Popen") as launch, redirect_stdout(io.StringIO()):
            self.assertFalse(start_local_daemon("unused"))
        launch.assert_not_called()

    def test_tts_selects_english_and_writes_file_without_speaking_on_host(self):
        for language in ("en-US", "en_US", b"\x05en-us"):
            engine = Mock()
            engine.getProperty.return_value = [SimpleNamespace(id="other", languages=["zh_CN"]),
                                               SimpleNamespace(id="english", languages=[language])]
            with patch.dict("sys.modules", {"pyttsx3": SimpleNamespace(init=lambda: engine)}):
                _synthesize_speech("Next block", Path("speech.wav"))
            engine.setProperty.assert_any_call("voice", "english")
            engine.save_to_file.assert_called_once_with("Next block", "speech.wav")
            engine.say.assert_not_called()
            engine.stop.assert_called_once()

    def test_no_english_voice_fails_explicitly(self):
        engine = Mock()
        engine.getProperty.return_value = [SimpleNamespace(id="other", languages=["zh_CN"])]
        with patch.dict("sys.modules", {"pyttsx3": SimpleNamespace(init=lambda: engine)}):
            with self.assertRaisesRegex(RuntimeError, "English"):
                _synthesize_speech("Next block", Path("speech.wav"))
        engine.save_to_file.assert_not_called()
        engine.stop.assert_called_once()

    def test_official_move_uses_sdk_and_restores_all_original_positions(self):
        robot = Mock()
        pose = np.eye(4)
        pose[:3, 3] = [.002, .001, .01]
        robot.get_current_head_pose.return_value = pose
        robot.get_current_joint_positions.return_value = ([0.25] + [0] * 6, [0.1, -0.1])
        move = Mock()
        self.assertEqual(_gesture(robot, move), 1)
        robot.play_move.assert_called_once_with(move, initial_goto_duration=0.8, sound=False)
        robot.enable_motors.assert_called_once_with()
        restored = robot.goto_target.call_args.kwargs
        np.testing.assert_array_equal(restored["head"], pose)
        np.testing.assert_array_equal(restored["antennas"], [0.1, -0.1])
        self.assertEqual(restored["body_yaw"], .25)

    def test_speech_end_before_motion_start_skips_official_move(self):
        robot = Mock()
        robot.get_current_head_pose.return_value = np.eye(4)
        robot.get_current_joint_positions.return_value = ([0] * 7, [0.1, -0.1])
        stop = threading.Event()
        stop.set()
        self.assertEqual(_gesture(robot, Mock(), stop), 0)
        robot.play_move.assert_not_called()
        robot.enable_motors.assert_not_called()

    def test_official_playback_failure_still_restores_pose(self):
        robot = Mock()
        robot.get_current_head_pose.return_value = np.eye(4)
        robot.get_current_joint_positions.return_value = ([.2] + [0] * 6, [0, 0])
        robot.play_move.side_effect = OSError("motor offline")
        with self.assertRaisesRegex(OSError, "motor offline"):
            _gesture(robot, Mock())
        robot.goto_target.assert_called_once()
        self.assertEqual(robot.goto_target.call_args.kwargs["body_yaw"], .2)

    def test_official_library_uses_exact_names_and_cached_snapshot(self):
        snapshot = Mock()
        library = Mock()
        library.get.return_value = Mock(duration=3)
        recorded = Mock(return_value=library)
        with patch.dict("sys.modules", {
                "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
                "reachy_mini.motion.recorded_move": SimpleNamespace(RecordedMoves=recorded)}):
            for name, official in EMOTION_NAMES.items():
                self.assertIs(_load_emotion(name), library.get.return_value)
                library.get.assert_called_with(official)
        snapshot.assert_called_with(EMOTIONS_DATASET, repo_type="dataset", local_files_only=True)
        recorded.assert_called_with(EMOTIONS_DATASET)

    def test_missing_cache_does_not_trigger_sdk_network_fallback(self):
        recorded = Mock()
        with patch.dict("sys.modules", {
                "huggingface_hub": SimpleNamespace(snapshot_download=Mock(side_effect=OSError("not cached"))),
                "reachy_mini.motion.recorded_move": SimpleNamespace(RecordedMoves=recorded)}):
            with self.assertRaisesRegex(OSError, "not cached"):
                _load_emotion("calm")
        recorded.assert_not_called()

    def test_missing_emotion_preserves_speech_and_reports_failure(self):
        with patch("reachy_feedback._load_emotion", side_effect=OSError("not cached")), \
                patch("reachy_feedback._speak") as speak, patch("reachy_feedback._gesture") as gesture:
            result = _speak_with_gestures(Mock(), "calm", [], 16000)
        speak.assert_called_once()
        gesture.assert_not_called()
        self.assertFalse(result["gesture_completed"])
        self.assertEqual(result["emotion_name"], "calming1")
        self.assertIn("not cached", result["warnings"][0])

    def test_motion_overlaps_speech_and_finishes_before_return(self):
        entered, restored = threading.Event(), threading.Event()
        events = []
        def gesture(robot, name, stop):
            events.append("motion")
            entered.set()
            if not stop.wait(2):
                raise RuntimeError("Speech never signalled motion to stop")
            events.append("restored")
            restored.set()
            return 3
        def speak(media, samples, rate, on_started):
            events.append("speech_started")
            on_started()
            self.assertTrue(entered.wait(1))
            self.assertFalse(restored.is_set())
            events.append("speech_finished")
        with patch("reachy_feedback._load_emotion", return_value=Mock(duration=3, sound_path=None)), patch("reachy_feedback._gesture", side_effect=gesture), patch("reachy_feedback._speak", side_effect=speak):
            result = _speak_with_gestures(Mock(), "celebrate", [], 16000)
        self.assertEqual(events, ["speech_started", "motion", "speech_finished", "restored"])
        self.assertTrue(result["gesture_completed"])
        self.assertEqual(result["gesture_plays"], 3)

    def test_motion_failure_preserves_speech_and_is_reported(self):
        attempted = threading.Event()
        speech_finished = []
        def gesture(*args):
            attempted.set()
            raise OSError("motor offline")
        def speak(media, samples, rate, on_started):
            on_started()
            self.assertTrue(attempted.wait(1))
            speech_finished.append(True)
        with patch("reachy_feedback._load_emotion", return_value=Mock(duration=3, sound_path=None)), patch("reachy_feedback._gesture", side_effect=gesture), patch("reachy_feedback._speak", side_effect=speak):
            result = _speak_with_gestures(Mock(), "calm", [], 16000)
        self.assertEqual(speech_finished, [True])
        self.assertFalse(result["gesture_completed"])
        self.assertIn("motor offline", result["warnings"][0])

    def test_audio_start_failure_releases_waiting_thread_without_motion(self):
        with patch("reachy_feedback._load_emotion", return_value=Mock(duration=3, sound_path=None)), patch("reachy_feedback._gesture") as gesture, \
                patch("reachy_feedback._speak", side_effect=OSError("audio offline")):
            with self.assertRaisesRegex(OSError, "audio offline"):
                _speak_with_gestures(Mock(), "calm", [], 16000)
        gesture.assert_not_called()

    def test_audio_disconnect_waits_for_active_motion_before_propagating(self):
        entered, restored = threading.Event(), threading.Event()
        def gesture(robot, name, stop):
            entered.set()
            if stop.wait(2):
                restored.set()
            return 1
        def speak(media, samples, rate, on_started):
            on_started()
            self.assertTrue(entered.wait(1))
            raise OSError("audio disconnected")
        with patch("reachy_feedback._load_emotion", return_value=Mock(duration=3, sound_path=None)), patch("reachy_feedback._gesture", side_effect=gesture), patch("reachy_feedback._speak", side_effect=speak):
            with self.assertRaisesRegex(OSError, "audio disconnected"):
                _speak_with_gestures(Mock(), "calm", [], 16000)
        self.assertTrue(restored.is_set())

    def test_audio_resamples_streams_small_chunks_and_waits(self):
        media = Mock()
        media.get_output_audio_samplerate.return_value = 48000
        samples = np.ones(2400, dtype="float32")  # 0.15 s at 16 kHz.
        with patch("reachy_feedback.time.sleep") as sleep:
            started = Mock()
            _speak(media, samples, 16000, on_started=started)
        started.assert_called_once_with()
        chunks = [call.args[0] for call in media.push_audio_sample.call_args_list]
        self.assertEqual(sum(map(len, chunks)), 7200)
        self.assertTrue(all(len(chunk) <= 4800 for chunk in chunks))
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .65)
        media.stop_playing.assert_called_once()

    def test_official_sound_is_quieter_and_starts_after_pose_transition(self):
        speech = np.full(20, 0.5, dtype="float32")
        effect = np.full((10, 1), 0.5, dtype="float32")
        with patch("soundfile.read", return_value=(effect, 10)):
            mixed, included = _mix_emotion_audio(speech, 10, Mock(sound_path="official.ogg"))
        self.assertTrue(included)
        np.testing.assert_allclose(mixed[:8], .4)  # Speech alone during 0.8 s transition.
        np.testing.assert_allclose(mixed[8:18], .5)  # .4 speech + .1 official sound.
        np.testing.assert_allclose(mixed[18:], .4)

    def test_mixing_resamples_stereo_and_preserves_sound_tail(self):
        speech = np.zeros(10, dtype="float32")
        effect = np.full((20, 2), .5, dtype="float32")
        with patch("soundfile.read", return_value=(effect, 20)):
            mixed, included = _mix_emotion_audio(speech, 10, Mock(sound_path="official.ogg"))
        self.assertTrue(included)
        self.assertEqual(mixed.shape, (18,))  # .8 s lead + full 1 s effect.
        self.assertEqual(mixed.dtype, np.float32)
        self.assertGreater(float(np.max(np.abs(mixed[10:]))), 0)

    def test_mixing_prevents_clipping_and_rejects_invalid_effect(self):
        speech = np.ones(20, dtype="float32")
        with patch("soundfile.read", return_value=(np.ones((10, 1)), 10)):
            mixed, _ = _mix_emotion_audio(speech, 10, Mock(sound_path="official.ogg"))
        self.assertLessEqual(float(np.max(np.abs(mixed))), .980001)
        with patch("soundfile.read", return_value=(np.full((10, 1), np.nan), 10)):
            with self.assertRaisesRegex(ValueError, "Invalid official emotion audio"):
                _mix_emotion_audio(speech, 10, Mock(sound_path="official.ogg"))

    def test_no_sidecar_preserves_original_speech(self):
        speech = np.ones(10, dtype="float32")
        mixed, included = _mix_emotion_audio(speech, 10, Mock(sound_path=None))
        self.assertIs(mixed, speech)
        self.assertFalse(included)

    def test_unreadable_sidecar_keeps_speech_and_reports_warning(self):
        with patch("reachy_feedback._load_emotion", return_value=Mock(sound_path="missing.ogg")), \
                patch("reachy_feedback._mix_emotion_audio", side_effect=OSError("missing audio")), \
                patch("reachy_feedback._speak") as speak:
            result = _speak_with_gestures(Mock(), "calm", [0.1], 16000)
        self.assertEqual(speak.call_args.args[1], [0.1])
        self.assertFalse(result["emotion_sound_included"])
        self.assertIn("missing audio", result["warnings"][0])

    def test_playback_failure_stops_audio(self):
        media = Mock()
        media.get_output_audio_samplerate.return_value = 16000
        media.push_audio_sample.side_effect = OSError("USB disconnected")
        with self.assertRaisesRegex(OSError, "USB disconnected"):
            _speak(media, np.ones(100, dtype="float32"), 16000)
        media.stop_playing.assert_called_once()

    def test_missing_robot_speaker_never_falls_back_to_computer(self):
        sdk = Mock()
        with patch.dict("sys.modules", {"reachy_mini": SimpleNamespace(ReachyMini=sdk),
                        "reachy_mini.media.device_detection": SimpleNamespace(Gst=Mock(), get_audio_device=lambda kind: None),
                        "soundfile": Mock()}), patch("reachy_feedback._synthesize_speech") as speech:
            with self.assertRaisesRegex(RuntimeError, "USB speaker"):
                _perform_feedback({}, "unused")
        sdk.assert_not_called()
        speech.assert_not_called()


class SessionIntegrationTests(unittest.TestCase):
    def run_session(self, enabled=True, face_group=True, startup=True):
        tree = ast.parse(Path(__file__).with_name("face2_api.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_session")
        source = ast.parse(Path(__file__).with_name("face2_script.py").read_text())
        accuracy_fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_accuracy")
        events = []
        blocks = [{"identifier": f"id-{i}", "block_ID": i, "game_FK": 1 if i == 1 else 2,
                   "startLevel": 110 if i == 1 else 220} for i in range(1, 21)]
        def get(url):
            if "/session/" in url:
                data = {"identifier": "session"}
            elif "/game/GetResult" in url:
                data = {"accuracy": .85}
            else:
                data = {"isGameReady": False}
            return SimpleNamespace(json=lambda: data)
        def put(url, json):
            events.append(("save" if "SaveExtendedResult" in url else "ready", dict(json)))
        def feedback(output_dir, **settings):
            events.append(("feedback", settings))
        namespace = dict(
            SESSION_EXTERNAL_ID="TEST_1", WEBSERVER="http://test", USE_ROTATION=face_group,
            REQUEST_INTERVAL=.5, REACHY_ENABLED=enabled, start_local_daemon=Mock(return_value=startup),
            FATIGUE_N_SEGMENTS=4, FATIGUE_CUTOFF=1, ECG_DEVICE_ADDRESS="unused",
            GAME_IDENTIFIERS={i: f"game-{i}" for i in range(1, 5)},
            requests=SimpleNamespace(get=get, put=put), time=SimpleNamespace(time=lambda: 0, sleep=Mock()),
            get_new_block=Mock(side_effect=blocks), check_game_status=Mock(return_value=True),
            make_sync_objects=Mock(return_value=(Mock(), Mock())), VideoRecorder=Mock(), ECGRecorder=Mock(),
            os=SimpleNamespace(path=os.path, makedirs=Mock()), random=SimpleNamespace(choice=min),
            parse_subject_id=Mock(return_value="TEST"), select_model_for_subject=Mock(return_value="model.pth"),
            get_block_fatigue_4x=Mock(side_effect=[None] + [True] * 19),
            make_decision=Mock(side_effect=[(2, 990, False)] + [(1, 880, False)] * 19),
            give_block_feedback=feedback,
        )
        exec(compile(ast.Module(body=[function, accuracy_fn], type_ignores=[]), "face2_api.py", "exec"), namespace)
        with redirect_stdout(io.StringIO()):
            namespace["_run_session"]()
        if not enabled:
            namespace["start_local_daemon"].assert_not_called()
        return events

    def test_feedback_precedes_save_and_ready_with_final_sent_settings(self):
        events = self.run_session()
        feedback = [data for kind, data in events if kind == "feedback"]
        self.assertEqual(len(feedback), 20)
        self.assertIsNone(feedback[0]["fatigue"])
        self.assertFalse(feedback[0]["accuracy_high"])  # .85 below Sound Sweeps .90
        self.assertTrue(feedback[1]["accuracy_high"])   # .85 above Target Tracker .75
        self.assertEqual(feedback[0]["next_level"], 110)  # Actual send_level, not decision's 990.
        self.assertEqual(feedback[1]["next_level"], 990)  # Stored level of destination game.
        self.assertEqual(feedback[9]["next_game"], 3)  # Exploration override, not decision's game 1.
        self.assertTrue(feedback[-1]["final_block"])
        block_number = 0
        for i, (kind, data) in enumerate(events):
            if kind == "feedback":
                block_number += 1
                self.assertEqual(events[i - 1], ("ready", {"isReady": False, "sessionIdentifier": f"id-{block_number}"}))
                self.assertEqual(events[i + 1][0], "save")
                self.assertEqual(events[i + 1][1]["NextGameIdentifier"], f"game-{data['next_game']}")
                self.assertEqual(events[i + 1][1]["NextLevel"], data["next_level"])
                self.assertEqual(events[i + 2][0], "ready")
                self.assertTrue(events[i + 2][1]["isReady"])

    def test_control_feedback_ignores_fatigue_and_disabled_run_has_no_feedback(self):
        self.assertTrue(all(data["fatigue"] is None for kind, data in self.run_session(face_group=False) if kind == "feedback"))
        failed = self.run_session(startup=False)
        self.assertFalse(any(kind == "feedback" for kind, _ in failed))
        self.assertEqual(sum(kind == "save" for kind, _ in failed), 20)
        events = self.run_session(enabled=False)
        self.assertFalse(any(kind == "feedback" for kind, _ in events))
        self.assertEqual(sum(kind == "save" for kind, _ in events), 20)


if __name__ == "__main__":
    unittest.main()
