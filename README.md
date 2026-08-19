# FACE-HMI

Adaptive brain-wellness game controller for the **FACE** study (R21). The controller
sits between the Brain Wellness Games web app and the participant, recording facial
video (and optionally ECG) during each game block, estimating **fatigue** from the
video with a fine-tuned deep model, and using accuracy + fatigue to decide the next
game and difficulty level.

The study has two arms:

- **FACE group** (`GROUP_ARM_ID = 1`) — difficulty, novelty, and game switching adapt
  to **both** task accuracy and model-derived fatigue.
- **Control group** (`GROUP_ARM_ID = 0`) — adaptation uses **accuracy only**; fatigue
  is ignored and games advance on a fixed rotation.

---

## Table of contents

- [How it works](#how-it-works)
- [Adaptive decision logic](#adaptive-decision-logic)
- [Fatigue estimation](#fatigue-estimation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running a session](#running-a-session)
  - [Online](#online-azure)
  - [Local](#local-docker)
- [Recorded output](#recorded-output)
- [Fatigue models](#fatigue-models)
- [Fine-tuning per subject](#fine-tuning-per-subject)

---

## How it works

For each of up to 20 blocks in a session, [`src/face2_api.py`](src/face2_api.py):

1. Polls the web API for the next game block (game id + start level).
2. Signals the game app that the device is ready and waits for the block to start
   (the ready signal is re-sent every 10 s for up to 2 min, since the game app may
   not be listening when the first one arrives).
3. Records facial video (and ECG via the PLUX sensor, when connected) for the block.
   Both recorders start on a shared barrier so the streams are aligned.
4. Estimates a per-block **fatigue flag** from the video using a fine-tuned model
   (`get_block_fatigue_4x`) — see [Fatigue estimation](#fatigue-estimation).
5. Reads the block **accuracy** returned by the game and runs `make_decision(...)`
   to choose the next game and difficulty level.
6. Saves the decision back to the API so the game app serves the next block, then
   immediately re-signals device-ready (the freshly loaded game app polls only once
   at startup).

Fatigue estimation and the recorder threads live in
[`src/face2_script.py`](src/face2_script.py) (`VideoRecorder`, `ECGRecorder`,
`get_block_fatigue_4x`, `make_decision`, and the adaptation helpers).

## Adaptive decision logic

All branching happens in `make_decision`
([`src/face2_script.py:606`](src/face2_script.py#L606)). Raw accuracy is first turned
into a boolean by per-game thresholds in `evaluate_accuracy`
([`src/face2_script.py:593`](src/face2_script.py#L593)): Sound Sweeps ≥ 0.90,
Target Tracker ≥ 0.75, Mixed Signals ≥ 0.80, Delayed Task Switching ≥ 0.80.

Levels are 3-digit encoded — hundreds = **novelty**, tens = **difficulty/staircase**,
ones = sublevel (unused). Both digits are clamped to 0–9.

Levels are tracked **per game** (`game_levels` in
[`src/face2_api.py:95`](src/face2_api.py#L95)), so each game keeps its own
progression across switches: the decision is computed from this game's saved level,
and the level sent to the server is the *next* game's saved level — or the server's
`startLevel` if that game has not been played yet.

### Difficulty (per block)

**FACE group** — `single_block_change(fatigue, acc, level)`
([`src/face2_script.py:469`](src/face2_script.py#L469)):

| accuracy | fatigue | action |
|---|---|---|
| high | tired | difficulty +1 **and** change novelty |
| high | not tired | difficulty +1 |
| low | tired | change novelty |
| low | not tired | no change |

**Control group** — `single_block_increase(acc, level)`
([`src/face2_script.py:521`](src/face2_script.py#L521)), fatigue not used:

| accuracy | action |
|---|---|
| high | difficulty +1 |
| low | change novelty |

Novelty changes pick a random new hundreds digit (any value 0–9 except the current
one).

**Both groups:**

- Two consecutive low-accuracy blocks trigger a difficulty step-down of 1 via
  `two_blocks_change` ([`src/face2_script.py:558`](src/face2_script.py#L558)).
- On a game's **entry block** (`game_streak == 1`) the novelty change is suppressed
  (`allow_novelty=False`). A novelty-carrying 3-digit level is mishandled at the
  game-switch handoff and would surface as a spurious difficulty jump, so a low-acc
  entry block changes nothing.

### Game switching

- **FACE group** — after a **3-block** streak on a game, the next game is chosen by
  the fatigue × accuracy quadrant (`three_blocks_change`,
  [`src/face2_script.py:575`](src/face2_script.py#L575)):

  | fatigue | accuracy | next game |
  |---|---|---|
  | tired | high | 4 — Delayed Task Switching |
  | not tired | high | 2 — Target Tracker |
  | tired | low | 3 — Mixed Signals |
  | not tired | low | 1 — Sound Sweeps |

- **Control group** — after a **5-block** streak, advance by a fixed round-robin
  `game % 4 + 1` (1 → 2 → 3 → 4 → 1), independent of fatigue/accuracy.

- **Both groups** — an exploration overlay in
  [`src/face2_api.py:221`](src/face2_api.py#L221) overrides the choice with a random
  unplayed game after 10 total blocks (if ≥ 2 of the 4 games remain unplayed) and
  after 15 (if ≥ 1 remains), so every game gets seen within a session.

## Fatigue estimation

`get_block_fatigue_4x` ([`src/face2_script.py:415`](src/face2_script.py#L415)):

1. Runs RetinaFace over the block video and saves one face crop every **5 s** as
   `<second>.jpg` (`extract_frames`).
2. Splits the block into `n_segments` (default 4) equal time windows.
3. For each segment, samples a 16-frame sequence of crops (resized to 112×112) and
   runs the recurrent model, taking `argmax` → 0 or 1 (`infer_score_on_faces`).
4. Sums the per-segment predictions and compares to `cutoff` (default 1) — so with
   the defaults, **any** segment predicting fatigue marks the block as fatigued.

Segments with no usable crops are skipped. If no segment yields a prediction the
function returns `None`, and the session driver defaults that block to **not
fatigued**.

`get_block_fatigue` (single-window variant) is retained but unused by the driver.

## Repository layout

| Path | Purpose |
|---|---|
| [`src/face2_api.py`](src/face2_api.py) | Session driver — the entry point you run |
| [`src/face2_script.py`](src/face2_script.py) | Recorders, fatigue inference, decision logic |
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
`src/`, `preprocess/`, `PLUX-API-Python3/`, and the top-level docs. Data, model
checkpoints, and recordings are deliberately untracked.

## Installation

```bash
pip install -r requirement.txt   # use pip3 on macOS
```

Some dependencies may not be listed; install any that surface as import errors.
On the FACE laptop a preconfigured `face` conda/venv environment is available.

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
python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID>
# GROUP_ARM_ID: 0 = control group, 1 = FACE group
```

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

On Windows, [`RUN_FACE_API.bat`](RUN_FACE_API.bat) wraps the run command.

Run from the **repo root** — model paths and the `Pilot/` output directory are
resolved relative to the working directory.

## Recorded output

Per session, under `Pilot/ses-<SESSION_EXTERNAL_ID>/`, where `<n>` is the
server-assigned `block_ID` (not a 1-based counter):

```
block-<n>.mp4            # facial video for block n
block-<n>/ecg_raw.csv    # ECG samples for block n (only if PLUX connected)
block-<n>/<sec>.jpg      # face crops extracted for fatigue inference
```

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
