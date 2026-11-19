# FACE-HMI

Adaptive brain-wellness game controller for the **FACE** study — the
facial-expression-based personalization engine (FPE) used in the
**FACE Phase II trial** ([NCT07130669](https://clinicaltrials.gov/study/NCT07130669)).
The controller sits between the Brain Wellness Games web app and the participant,
recording facial video (and optionally ECG) during each game block, estimating
**fatigue** from the video with a fine-tuned deep model, and using accuracy +
fatigue to decide the next game and difficulty level.

Two optional features run through the same API entry point: `monitor` shows the
model's frame-set predictions in a separate window, and `reachy` provides English
robotic feedback with official Reachy Mini emotions and sound effects between
blocks. Both are off by default and can be enabled independently.

The trial randomizes participants to two arms, which the controller selects with its
`GROUP_ARM_ID` argument:

- **FACE group** (`GROUP_ARM_ID = 1`) — registered as *unsupervised cognitive training
  **with** personalized engine* (experimental). Difficulty, novelty, and game
  switching adapt to **both** task accuracy and model-derived fatigue.
- **Control group** (`GROUP_ARM_ID = 0`) — registered as *unsupervised cognitive
  training **without** personalized engine* (active comparator). Adaptation uses
  **accuracy only**; fatigue is ignored and games advance on a fixed rotation.

---

## Demo (4× speed)

[![FACE HMI demonstration at 4× speed](docs/media/FE-HMI-4x.gif)](https://github.com/ymliu17/FACE-HMI/raw/refs/heads/main/docs/media/FE-HMI-4x.mp4)

Click the animated preview to download the [full-resolution MP4](https://github.com/ymliu17/FACE-HMI/raw/refs/heads/main/docs/media/FE-HMI-4x.mp4).

---

## Table of contents

- [Demo (4× speed)](#demo-4-speed)
- [Study](#study)
- [How it works](#how-it-works)
- [Adaptive decision logic](#adaptive-decision-logic)
- [Fatigue estimation](#fatigue-estimation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running a session](#running-a-session)
  - [Online](#online-azure)
  - [Local](#local-docker)
- [Fatigue monitor](#fatigue-monitor)
- [Reachy Mini robotic feedback](#reachy-mini-robotic-feedback)
- [Updating an existing installation](#updating-an-existing-installation)
- [Recorded output](#recorded-output)
- [Fatigue models](#fatigue-models)
- [Fine-tuning per subject](#fine-tuning-per-subject)

---

## Study

| | |
|---|---|
| Registration | [NCT07130669](https://clinicaltrials.gov/study/NCT07130669) — *FACE Phase II (a Stage II Trial)* |
| Full title | A Facial Expression-based Personalization Engine (FPE) for Monitoring and Modulating Real-time Effective Engagement in Cognitive Training in Older Adults at Risk for AD/ADRD |
| Sponsor / PI | Stanford University — Feng Lin |
| Design | Interventional, randomized, parallel, double-masked (participant + investigator) |
| Enrollment | 80 (estimated) |
| Population | Subjective cognitive decline (SCD), MCI, mild behavioral impairment |
| Intervention | 6-week unsupervised computerized cognitive training |
| Status | Recruiting; started 2025-10-31, estimated completion 2028-08-31 |

Registry details above are as of the 2026-04-24 record update. Earlier-phase (R21)
pilot data also lives in this repo under `Pilot/`.

## How it works

For each of up to 20 blocks in a session, [`src/face2_api.py`](src/face2_api.py):

1. Polls the web API for the next game block (game id + start level).
2. Signals the game app that the device is ready and waits for the block to start
   (the ready signal is re-sent every 10 s for up to 2 min, since the game app may
   not be listening when the first one arrives).
3. Records facial video (and ECG via the PLUX sensor, when connected) for the block.
   Both recorders start on a shared barrier so the streams are aligned.
4. Reads the game's **accuracy**, then estimates a per-block **fatigue flag** from
   the video (`get_block_fatigue_4x`). If `monitor` is enabled, its window updates
   as each frame set is processed — see [Fatigue estimation](#fatigue-estimation).
5. Runs `make_decision(...)`, applies exploration overrides, and looks up the
   final next game's saved level.
6. If `reachy` is enabled and its local service started successfully, speaks the
   result and next-block changes while playing an official emotion and its sound.
   The device stays not ready until feedback completes or fails.
7. Saves the decision back to the API so the game app serves the next block, then
   immediately re-signals device-ready (the freshly loaded game app polls only once
   at startup).

Fatigue estimation and the recorder threads live in
[`src/face2_script.py`](src/face2_script.py) (`VideoRecorder`, `ECGRecorder`,
`get_block_fatigue_4x`, `make_decision`, and the adaptation helpers).

## Adaptive decision logic

All branching happens in `make_decision`
([`src/face2_script.py`](src/face2_script.py)). Raw accuracy is first turned
into a boolean by per-game thresholds in `evaluate_accuracy`
([`src/face2_script.py`](src/face2_script.py)): Sound Sweeps ≥ 0.90,
Target Tracker ≥ 0.75, Mixed Signals ≥ 0.80, Delayed Task Switching ≥ 0.80.

Levels are 3-digit encoded — hundreds = **novelty**, tens = **difficulty/staircase**,
ones = sublevel (unused). Both digits are clamped to 0–9.

Levels are tracked **per game** (`game_levels` in
[`src/face2_api.py`](src/face2_api.py)), so each game keeps its own
progression across switches: the decision is computed from this game's saved level,
and the level sent to the server is the *next* game's saved level — or the server's
`startLevel` if that game has not been played yet.

### Difficulty (per block)

**FACE group** — `single_block_change(fatigue, acc, level)`
([`src/face2_script.py`](src/face2_script.py)):

| accuracy | fatigue | action |
|---|---|---|
| high | tired | difficulty +1 **and** change novelty |
| high | not tired | difficulty +1 |
| low | tired | change novelty |
| low | not tired | no change |

**Control group** — `single_block_increase(acc, level)`
([`src/face2_script.py`](src/face2_script.py)), fatigue not used:

| accuracy | action |
|---|---|
| high | difficulty +1 |
| low | change novelty |

Novelty changes pick a random new hundreds digit (any value 0–9 except the current
one).

**Both groups:**

- Two consecutive low-accuracy blocks trigger a difficulty step-down of 1 via
  `two_blocks_change` ([`src/face2_script.py`](src/face2_script.py)).
- On a game's **entry block** (`game_streak == 1`) the novelty change is suppressed
  (`allow_novelty=False`). A novelty-carrying 3-digit level is mishandled at the
  game-switch handoff and would surface as a spurious difficulty jump, so a low-acc
  entry block changes nothing.

### Game switching

- **FACE group** — after a **3-block** streak on a game, the next game is chosen by
  the fatigue × accuracy quadrant (`three_blocks_change`,
  [`src/face2_script.py`](src/face2_script.py)):

  | fatigue | accuracy | next game |
  |---|---|---|
  | tired | high | 4 — Delayed Task Switching |
  | not tired | high | 2 — Target Tracker |
  | tired | low | 3 — Mixed Signals |
  | not tired | low | 1 — Sound Sweeps |

- **Control group** — after a **5-block** streak, advance by a fixed round-robin
  `game % 4 + 1` (1 → 2 → 3 → 4 → 1), independent of fatigue/accuracy.

- **Both groups** — an exploration overlay in
  [`src/face2_api.py`](src/face2_api.py) overrides the choice with a random
  unplayed game after 10 total blocks (if ≥ 2 of the 4 games remain unplayed) and
  after 15 (if ≥ 1 remains), so every game gets seen within a session.

## Fatigue estimation

`get_block_fatigue_4x` ([`src/face2_script.py`](src/face2_script.py)):

1. Splits the block video into `n_segments` (default 4) consecutive windows.
2. Uniformly selects **16 video frames per window**, in chronological order, and
   runs RetinaFace on those frames. Saves only the selected crops, at their
   original resolution without labels, as lossless PNGs under
   `frame_sets/set-01/01.png` through `frame_sets/set-04/16.png`.
3. Reads these exact saved sequences, resizes them to 112×112, and
   runs the recurrent model, taking `argmax` → 0 or 1 (`infer_score_on_faces`).
4. Sums the per-segment predictions and compares to `cutoff` (default 1) — so with
   the defaults, **any** segment predicting fatigue marks the block as fatigued.

Only a window containing fewer than 16 video frames repeats its last frame to
reach 16 inputs. The old every-5-second extraction pass is no longer used by this
function. This changes the sampling cadence, so predictions can differ from the
previous version. As before, an invalid/missing face detection falls back to the
full camera frame; the manifest records this with `crop_box: null`.

Empty windows are skipped. If no segment yields a prediction the
function returns `None`, and the session driver defaults that block to **not
fatigued**.

`frame_sets/manifest.json` records each input's sequence position (filename), video
frame index, timestamp, crop box and original size, plus segment predictions and
the block result. Directory input can replay these saved sequences; legacy
directories containing `<second>.jpg` files remain readable with the old sampler.

`get_block_fatigue` (single-window variant) is retained but unused by the driver.

## Repository layout

| Path | Purpose |
|---|---|
| [`src/face2_api.py`](src/face2_api.py) | Session driver — the entry point you run |
| [`src/face2_script.py`](src/face2_script.py) | Recorders, fatigue inference, decision logic |
| [`src/face_frame_sets.py`](src/face_frame_sets.py) | Select and save the exact frames used for inference |
| [`src/fatigue_monitor.py`](src/fatigue_monitor.py) | Optional prediction window and its separate viewer process |
| [`src/reachy_feedback.py`](src/reachy_feedback.py) | Optional USB robot feedback, official emotions and sound/voice mixing |
| [`src/requirements-reachy.txt`](src/requirements-reachy.txt) | Optional robot dependencies for the existing FACE environment |
| [`src/face2_monitor.py`](src/face2_monitor.py) | Compatibility shortcut for the API with `monitor` enabled |
| [`src/ft_orientation.py`](src/ft_orientation.py) | Per-subject fine-tuning on Orientation data |
| [`src/ft_train.py`](src/ft_train.py) | Base fatigue model training |
| [`src/trainer/`](src/trainer/), [`src/loader/`](src/loader/), [`src/models/`](src/models/), [`src/utils/`](src/utils/) | Training/data/model modules |
| `pre_model/` | Fatigue model checkpoints (`model_9.pth` default, `model_<subject>.pth` per subject) |
| `Orientation/` | Per-subject orientation recordings used for fine-tuning |
| `Pilot/` | Session recordings written at run time (`ses-<id>/…`) |
| `results_orientation/` | Fine-tuning logs and per-run best checkpoints |
| `PLUX-API-Python3/` | Vendor SDK for the PLUX ECG device |
| [`README_Docker.txt`](README_Docker.txt) | Detailed Docker setup for the local API |

Note: [`.gitignore`](.gitignore) ignores everything by default and re-includes only
`src/`, `preprocess/`, `PLUX-API-Python3/`, the top-level docs and the two public
`docs/media/FE-HMI-4x` demo assets. Session data, model checkpoints and participant
recordings remain untracked.

## Installation

```bash
pip install -r requirement.txt   # use pip3 on macOS
```

Some dependencies may not be listed; install any that surface as import errors.
On the FACE laptop a preconfigured `face` conda/venv environment is available.
Use that existing environment for the optional features as well:

- **Monitor:** requires Tkinter and Pillow; `python -m tkinter` checks the GUI.
- **Reachy:** requires Python 3.11+ and the additional dependencies described in
  [robotic feedback setup](#one-time-setup). Install them in the same FACE
  environment; a separate robot environment is not needed.

The PLUX SDK is loaded from `PLUX-API-Python3/<platform>_<pyver>` based on your OS
and Python version. If the matching build is absent the import fails with a warning
and the session runs **video-only**.

## Configuration

Constants at the top of [`src/face2_api.py`](src/face2_api.py):

| Constant | Meaning |
|---|---|
| `WEBSERVER` | API base URL — Azure or `http://localhost:8080` (currently set to localhost) |
| `ECG_DEVICE_ADDRESS` | PLUX sensor address, printed on the back of the sensor |
| `FATIGUE_N_SEGMENTS` / `FATIGUE_CUTOFF` | Segment count (4) and fatigue threshold (1) |
| `DEFAULT_MODEL_PATH` / `MODEL_DIR` | Fallback checkpoint and checkpoint directory |
| `REQUEST_INTERVAL` | Poll interval for API status checks (0.5 s) |
| `GAME_IDENTIFIERS` | Game index (1–4) → server GUID |

Camera settings are at the top of [`src/face2_script.py`](src/face2_script.py):
`CAM_INDEX` (0), `CAM_WIDTH`/`CAM_HEIGHT` (1920×1080), `CAM_FPS` (30),
`CAM_USE_MJPG`, and `WRITER_FOURCC` (`mp4v`; use `avc1` on macOS if playback fails).
The capture backend is chosen per platform (AVFoundation / DirectShow / V4L2).

## Running a session

Command form:

```bash
python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID> [monitor] [reachy]
# GROUP_ARM_ID: 0 = control group, 1 = FACE group
# monitor: prediction window; reachy: USB robot feedback
```

Choose the desired mode (run one command):

| Mode | Command |
|---|---|
| Standard session | `python src/face2_api.py FT19_2 1` |
| Monitor only | `python src/face2_api.py FT19_2 1 monitor` |
| Robot only | `python src/face2_api.py FT19_2 1 reachy` |
| Monitor and robot | `python src/face2_api.py FT19_2 1 monitor reachy` |

`monitor` and `reachy` may appear in either order. Use the literal option `reachy`
for robotic feedback. Existing API endpoints, payloads and adaptation rules are
unchanged. If using an older checkout, apply the
[source update](#updating-an-existing-installation) before using the robot option.

`SESSION_EXTERNAL_ID` must use the format `subidx_sesidx` (e.g. `FT19_2`) — the
subject part selects the fatigue model and names the output directory.

Set `WEBSERVER` and `ECG_DEVICE_ADDRESS` before running (see
[Configuration](#configuration)).

### Online (Azure)

1. Set `WEBSERVER = "https://brainwellnessgamesmasterapi.azurewebsites.net"`.
2. In a browser open <https://brainwellnessgames.azurewebsites.net/>, start
   **Master Game → Updated UI**, enter the Session ID, and submit the session.
3. From the repo root: `python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID>`.
4. Wait for the script to be ready, then click **Start** in the browser.

### Local (Docker)

1. Set `WEBSERVER = "http://localhost:8080"`.
2. Start the local API container in Docker Desktop (Containers → ▶). Wait for
   `Database restored successfully!` in the container logs. See
   [`README_Docker.txt`](README_Docker.txt) for full details.
3. In a browser open <http://localhost:8082/>, start **Master**, enter the Session
   ID, and submit the session.
4. (On the FACE laptop: `conda activate face`.) Then
   `python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID>`.
5. Wait for the script to be ready, then click **Start** in the browser.

On Windows, [`RUN_FACE_API.bat`](RUN_FACE_API.bat) currently launches the standard
session without either option. To enable the monitor or robot, use the terminal
commands above after activating the FACE environment.

Run from the **repo root** — model paths and the `Pilot/` output directory are
resolved relative to the working directory.

## Fatigue monitor

Monitoring is **off by default**. Add `monitor` after the session and group
arguments to open the window when the first completed block starts fatigue
processing:

```bash
python src/face2_api.py FT19_2 1 monitor
```

Without `monitor`, recording, frame saving, model inference and game decisions
still run normally, with no monitoring window or temporary monitor snapshots.

Monitoring is implemented in the separate `fatigue_monitor.py` module and enabled
by the session driver. API endpoints, request payloads, fatigue function signatures/
return types, and decision rules are unchanged. `face2_monitor.py` remains a
compatibility shortcut that explicitly supplies `monitor` to the same driver;
only one entry point needs to run.

After each block, the window updates as each time segment finishes inference:

- Each frame set shows the **exact 16 frames supplied to the model**, in model
  order, including any repeated frames used to pad a short sequence. With four
  valid frame sets, this is 64 images per block. The viewer shows resized model
  inputs; the saved crops retain their original resolution and have no labels.
- The set's prediction (`Fatigue: 0` or `Fatigue: 1`) is printed over its frames.
  This is one prediction for the entire sequence, not individual frame predictions
  or a confidence/probability score.
- The block summary shows the final fatigue flag, number of positive/valid sets,
  and configured cutoff. Missing sets are marked unavailable; a block with no
  predictions remains **Unavailable** in the monitor even though the existing
  controller falls back to not fatigued for its decision.

The latest block remains visible while the next game runs, then is replaced when
the next block starts processing. Scrollbars allow viewing all frame sets. The
viewer runs in a separate process: closing it or a display failure does not stop
recording, inference, or API calls. If closed, it reopens when the next completed
block starts processing. It closes when the session driver exits.

The viewer requires **Tkinter** and the existing **Pillow** dependency in the same
Python environment as FACE HMI. `python -m tkinter` can check Tk availability.
Monitor snapshots are local temporary files, removed when the monitored session
exits; original face crops are never annotated on disk. No web service or port is
added. The monitor supports the driver's `get_block_fatigue_4x` inference path.

## Reachy Mini robotic feedback

Use one local USB setup on Windows or macOS. Connect and power Reachy Mini, then
add `reachy` to the normal API command. The optional `monitor` window is independent:

```bash
python src/face2_api.py FT19_2 1 reachy
python src/face2_api.py FT19_2 1 monitor reachy
```

Without `reachy`, no robot service, speech or movement is started. The game API
request format and adaptation rules are unchanged.

### One-time setup

Activate your existing **FACE Python environment** (Python 3.11+) and install:

```bash
python -m pip install -r src/requirements-reachy.txt
```

For the configured macOS workstation, activate `~/.venvs/face`:

```bash
source ~/.venvs/face/bin/activate
python -m pip install -r src/requirements-reachy.txt
python src/face2_api.py FT19_2 1 monitor reachy
```

On Windows, activate your existing FACE environment and use the same install and
API commands. The API runs the robot daemon and feedback worker with the **same
Python executable** as the game; no additional environment is created. The
optional requirements pin Reachy Mini SDK 1.11.0, pyttsx3 2.99 and soundfile 0.13.1,
require NumPy >=2.2.5 and SciPy >=1.15, and keep setuptools below 82 for the
existing Torch/NeuroKit environment. NumPy can be updated in the FACE environment.

Install an English system text-to-speech voice (Windows Settings → Time & language
→ Speech) and complete the robot's initial USB/driver setup using the
[official installation guide](https://huggingface.co/docs/reachy_mini/en/SDK/installation).

FACE reuses a local SDK daemon or launches the official SDK daemon automatically
on `127.0.0.1:8000` before recording starts. A newly launched daemon uses the SDK's
default wake-up behavior. An existing daemon should already have the robot awake
and ready for interaction. Feedback enables the robot motors at their
current positions. A successfully launched daemon stays running after the FACE
session and can be managed with Reachy Mini Control. The daemon's output is saved
to `Pilot/ses-<server-session-identifier>/reachy_daemon.log` when FACE launches it.
This identifier comes from the session API and may differ from the external ID
used for recordings. Startup waits at most 30 seconds, plus bounded cleanup; if
it fails, the terminal reports the error and the session continues without robot
feedback. Avoid running
another robot conversation or movement app during the session.

Speech is generated offline using **pyttsx3**, which uses the installed system
English voice, and is played through the SDK's local USB audio backend on
**Reachy's speaker**. This does not use the voice model of Reachy's separate
conversation app and requires no cloud API key. If the Reachy USB speaker cannot
be detected, feedback fails instead of selecting the computer speaker.

### Between-block behavior

After each block, FACE computes accuracy/fatigue and selects the **final** next
game and level, including exploration overrides and stored per-game levels.
Then Reachy speaks English while playing a preset from Pollen Robotics'
[official emotions library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library),
using SDK `RecordedMoves` and `play_move`:

| Result | Official emotion | Feedback |
|---|---|---|
| High accuracy, no positive fatigue flag | `success1` | “Nice work!” and accuracy |
| Lower accuracy, no positive fatigue flag | `helpful1` | “Thank you for your effort.” and accuracy |
| Positive fatigue prediction | `calming1` | Accuracy and “Take a moment to relax.” |
| Final block | `grateful1` | Accuracy and thanks for completing the session |

The existing per-game accuracy thresholds determine high/low accuracy. The
control arm uses accuracy only; unavailable fatigue is not treated as a measured
negative result. Final-block gratitude takes priority over the other emotion
choices. Motion begins with the first speech audio chunk and plays **once**.
The original recorded head, antenna and body motions are used, replacing the
handwritten movement loops. The official sound effect is mixed with English
speech in a single audio stream (20% effect gain, 80% speech gain, with peak
headroom). Its start is delayed by the 0.8-second transition into the recorded
pose. SDK `play_move(sound=False)` prevents duplicate sound playback; the official
audio remains audible through the mixed stream. The complete effect is retained
even if it lasts longer than speech. Before the next block, FACE waits for
the complete mixed audio and emotion playback, including restoration of the
original head, antenna and body positions. Restoration follows the motion clip
and can finish while a longer speech track is still playing. Motor failures are
logged while speech continues.
If an effect file is missing or unreadable, speech and motion continue and a
warning is logged.

The SDK daemon normally caches this library at startup. On a new computer,
prepare the cache once while online, using the same FACE environment:

```bash
python -c "from reachy_mini.motion.recorded_move import preload_dataset; assert preload_dataset('pollen-robotics/reachy-mini-emotions-library') is not None"
```

During block feedback, library loading uses the local cache. If it is missing,
FACE reports the problem and continues with speech; it does not start a download
or substitute a handwritten motion. No new environment or dependency is needed.

For the same game, speech explains whether difficulty increases, decreases or
stays the same, and mentions changed task variations/settings. A game switch
names the new game without comparing level numbers across games. Block 20 ends
with thanks instead of announcing another block.

The device stays **not ready** until feedback ends. Feedback runs before
`SaveExtendedResult`, so the next block cannot start during the announcement.
SDK and speech work runs in a bounded child process (45 seconds maximum per
block). Disconnection, missing dependencies, speech failure and timeouts are
printed in the terminal and **do not stop the game**. Each attempted block
feedback writes `reachy_feedback.json` with its text, final settings and outcome,
including the official dataset, emotion name, playback count, completion status
and whether its sound was included.
Speech files are temporary. Both monitor and robot are controlled by the API's
options; no second user-facing process needs to be launched.

| Situation | Behavior |
|---|---|
| Robot startup fails | Report in the terminal; continue the session without robot feedback |
| Emotion cache is unavailable | Continue with English speech; log the missing emotion |
| Emotion sound cannot be read | Continue with speech and movement; log the sound error |
| Movement fails | Continue speech; log the movement error |
| Speech/connection fails or feedback exceeds 45 seconds | Log failure and let the game continue |
| Monitor window is closed | Game continues; the window reopens at the next block's fatigue processing |

Offline checks cover the monitor, frame sampling, feedback wording, official
emotion selection, sound mixing, failure handling and API ordering:

```bash
python -B -m unittest discover -s src -p "test_*.py"
```

These checks do not move the robot or play sound. Windows USB audio and motion
still require an on-device check.

## Updating an existing installation

Monitor and robotic feedback are included in this repository's `main` branch.
For an existing Git checkout, update the source and install the optional robot
dependencies in your existing FACE environment:

```bash
git pull --ff-only origin main
python -m pip install -r src/requirements-reachy.txt
```

For manual updates, copy the matching files together as described below.

If the existing installation already includes the monitor and exact input-frame
saving, copy these three files from the updated source package into `src/`:

- `face2_api.py`
- `reachy_feedback.py`
- `requirements-reachy.txt`

For an older installation, also replace/add `face2_script.py`,
`fatigue_monitor.py` and `face_frame_sets.py` (six files in total). Keep model
weights and the matching PLUX driver. Preserve machine-specific server, camera
and ECG settings when copying the driver files. Install the optional Reachy
dependencies and cache the official emotion library as described above.

The `test_*.py` files are useful for regression checks but are not needed to run a
session. A new computer also needs the original FACE dependencies, model weights
and game server; this source update alone is not a complete installation.

## Recorded output

Per session, under `Pilot/ses-<SESSION_EXTERNAL_ID>/`, where `<n>` is the
server-assigned `block_ID` (not a 1-based counter):

```
block-<n>.mp4            # facial video for block n
block-<n>/ecg_raw.csv    # ECG samples for block n (only if PLUX connected)
block-<n>/frame_sets/set-01/01.png ... 16.png  # segment 1: original-resolution crops
block-<n>/frame_sets/set-02/01.png ... 16.png  # segment 2
block-<n>/frame_sets/set-03/01.png ... 16.png  # segment 3
block-<n>/frame_sets/set-04/01.png ... 16.png  # segment 4
block-<n>/frame_sets/manifest.json           # source frames, crop metadata and predictions
block-<n>/reachy_feedback.json               # optional robot feedback text, settings and outcome
```

For a normal block, four sets contain 64 saved input images. Model resizing and
monitor labels do not modify these originals. Reprocessing replaces only the
generated `frame_sets` directory after successful extraction. Previously saved
top-level JPGs and ECG data are preserved; video inference does not read those old
JPGs or create a new every-5-second image collection.

## Fatigue models

Checkpoints live in `pre_model/`. At session start the controller picks a model by
subject id parsed from `SESSION_EXTERNAL_ID` (format `subidx_sesidx`):

- If `pre_model/model_<subidx>.pth` exists, it is used (subject-specific).
- Otherwise it falls back to the default `pre_model/model_9.pth`.

The choice is printed as a `[Model]` line at session start.

## Fine-tuning per subject

Build a subject-specific fatigue model from Orientation recordings with
[`src/ft_orientation.py`](src/ft_orientation.py). Each subject is fine-tuned from
every candidate base model and the best-scoring result (by eval F1) is kept:

```bash
python src/ft_orientation.py --subjects FT19
# defaults: bases = pre_model/model_1.pth + model_9.pth,
#           data_dir = Orientation, output = pre_model/model_<subject>.pth
```

Details:

- Continuous fatigue labels are binarized at the subject's **median**; if the median
  equals the maximum (which would leave zero positives) the midpoint of the label
  range is used instead.
- The seed is reset to 42 before each base so runs are comparable.
- Per-run logs and the trainer's best checkpoint go to
  `results_orientation/orientation-<subject>-<base>/`; the winning checkpoint is
  copied to `pre_model/model_<subject>.pth`.

Useful flags: `--pretrained_models`, `--data_dir`, `--output_dir`, `--log_dir`,
`--epochs` (20), `--lr` (1e-4), `--batch_size` (1), `--subjects` (default: all
subjects in `data_dir`). Omit `--subjects` to fine-tune everyone.
