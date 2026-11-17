"""Local-only fatigue viewer. Importing this module opens no windows.

The session process publishes atomic snapshots; a separate Python process owns
Tk's event loop. Neither closing the viewer nor a display error stops inference.
"""

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


_active_monitor = None


def publish(event, **data):
    """Best-effort observation hook; a no-op outside a monitored session."""
    if _active_monitor is not None:
        try:
            _active_monitor.update(event, **data)
        except Exception as exc:
            # A broken monitor must never change a model result or API request.
            print(f"[Fatigue monitor] Display update failed: {exc}")


class _SnapshotMonitor:
    def __init__(self, state_path, session_id, open_on_block=False):
        self.state_path = Path(state_path)
        self.session_id = session_id
        self.process = None
        self.open_on_block = open_on_block
        self.state = {"session": session_id, "status": "Waiting for a completed block", "segments": []}
        self._save()

    def _save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def update(self, event, **data):
        if self.process is not None and self.process.poll() is not None and event != "begin":
            return
        if event == "begin":
            self.state = {
                "session": self.session_id,
                "block": Path(data["face_outdir"]).name,
                "model": Path(data["model_dir"]).name,
                "n_segments": data["n_segments"],
                "cutoff": data["cutoff"],
                "status": "Extracting faces / loading model",
                "segments": [],
            }
        elif event == "segment":
            faces = data["faces"]
            frames = []
            if faces is not None:
                # Use the exact tensor used for inference, including its order
                # and repeated frames. Do not resample or rerun the model.
                pixels = faces.detach().cpu().permute(0, 2, 3, 1).numpy()
                pixels = (pixels * 255).clip(0, 255).astype("uint8")
                for frame in pixels:
                    frames.append({
                        "size": [frame.shape[1], frame.shape[0]],
                        "rgb": base64.b64encode(frame.tobytes()).decode("ascii"),
                    })
            self.state["segments"].append({
                "index": data["index"], "start": data["start"], "end": data["end"],
                "prediction": data["prediction"], "frames": frames,
            })
            self.state["status"] = f"Processing frame sets: {data['index'] + 1}/{self.state['n_segments']}"
        elif event == "end":
            self.state["fatigued"] = data["fatigued"]
            self.state["status"] = "Block processing complete"
        else:
            raise ValueError(f"Unknown monitor event: {event}")
        self._save()
        # Open only after a completed game block starts fatigue processing.
        # Reuse the window, or reopen it on the next block if it was closed.
        if event == "begin" and self.open_on_block and (self.process is None or self.process.poll() is not None):
            self.process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--viewer", str(self.state_path)
            ])


@contextmanager
def enabled_monitor(session_id):
    """Enable per-block viewing; open lazily and clean up when the API exits."""
    global _active_monitor
    previous = _active_monitor
    monitor = None
    with tempfile.TemporaryDirectory(prefix="face-fatigue-monitor-") as directory:
        try:
            monitor = _SnapshotMonitor(Path(directory) / "state.json", session_id, open_on_block=True)
            _active_monitor = monitor
        except Exception as exc:
            print(f"[Fatigue monitor] Could not open viewer; session continues: {exc}")
        try:
            yield
        finally:
            _active_monitor = previous
            process = monitor.process if monitor is not None else None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def prediction_label(prediction):
    if prediction is None:
        return "Unavailable"
    return "Fatigue (1)" if prediction else "Not fatigued (0)"


def render_frame_set(segment):
    """Render a sequence with its set-level prediction printed on each face."""
    from PIL import Image, ImageDraw

    frames = segment["frames"]
    columns, tile, gap, header = 8, 112, 8, 36
    rows = max(1, (len(frames) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * (tile + gap) + gap, header + rows * (tile + gap)), "#17212f")
    draw = ImageDraw.Draw(sheet)
    prediction = segment["prediction"]
    color = "#b63749" if prediction == 1 else "#19624d" if prediction == 0 else "#596273"
    title = (f"Frame set {segment['index'] + 1} | {segment['start']:.2f}-{segment['end']:.2f} s"
             f" | Set prediction: {prediction_label(prediction)}")
    draw.text((gap, 10), title, fill="white")
    for i, frame in enumerate(frames):
        face = Image.frombytes("RGB", tuple(frame["size"]), base64.b64decode(frame["rgb"]))
        face = face.resize((tile, tile))
        x, y = gap + (i % columns) * (tile + gap), header + (i // columns) * (tile + gap)
        sheet.paste(face, (x, y))
        draw.rectangle((x, y, x + 25, y + 16), fill="#17212f")
        draw.text((x + 3, y + 2), str(i + 1), fill="white")
        draw.rectangle((x, y + tile - 21, x + tile - 1, y + tile - 1), fill=color)
        label = "Fatigue: 1" if prediction == 1 else "Fatigue: 0" if prediction == 0 else "Unavailable"
        draw.text((x + 5, y + tile - 16), label, fill="white")
    if not frames:
        draw.text((gap, header + 20), "No usable faces in this time window - prediction unavailable", fill="#ccd3df")
    return sheet


def _run_viewer(state_path):
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("FACE HMI - Fatigue monitor")
    root.geometry("1020x820")
    root.minsize(600, 400)
    title = ttk.Label(root, text="FACE HMI", font=("Arial", 16, "bold"), padding=12)
    title.pack(anchor="w")
    status = ttk.Label(root, padding=(12, 0, 12, 8), wraplength=950)
    status.pack(anchor="w")
    summary = ttk.Label(root, padding=(12, 0, 12, 8), font=("Arial", 12, "bold"), wraplength=950)
    summary.pack(anchor="w")
    ttk.Label(root, text="Each label is a frame-set prediction, not a separate prediction for each frame.\n"
              "The latest block remains visible until the next block is processed. Closing this window keeps the session running; it reopens for the next block.",
              padding=(12, 0, 12, 12), wraplength=950).pack(anchor="w")

    area = ttk.Frame(root)
    area.pack(fill="both", expand=True)
    area.rowconfigure(0, weight=1)
    area.columnconfigure(0, weight=1)
    canvas = tk.Canvas(area, background="#17212f", highlightthickness=0)
    vertical = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
    horizontal = ttk.Scrollbar(area, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    images = []  # Keep PhotoImage objects alive for Tk.
    last_stamp = None
    last_block = None

    def refresh():
        nonlocal last_stamp, last_block
        try:
            path = Path(state_path)
            stamp = path.stat().st_mtime_ns
            if stamp != last_stamp:
                state = json.loads(path.read_text(encoding="utf-8"))
                title.configure(text=f"Session {state['session']}  |  {state.get('block', 'Waiting for block')}")
                status.configure(text=f"{state['status']}  |  Model: {state.get('model', '-')}")
                segments = state["segments"]
                valid = [s["prediction"] for s in segments if s["prediction"] is not None]
                if "fatigued" in state:
                    result = prediction_label(state["fatigued"])
                    detail = (f"{sum(valid)} positive / {len(valid)} valid sets; cutoff = {state['cutoff']}")
                    if state["fatigued"] is None:
                        detail = "No valid predictions; the existing controller uses its not-fatigued fallback."
                    summary.configure(text=f"Block result: {result}  |  {detail}")
                else:
                    summary.configure(text="Block result: pending")
                canvas.delete("all")
                images.clear()
                y = 0
                for segment in segments:
                    image = ImageTk.PhotoImage(render_frame_set(segment), master=root)
                    images.append(image)
                    canvas.create_image(0, y, image=image, anchor="nw")
                    y += image.height() + 8
                canvas.configure(scrollregion=canvas.bbox("all") or (0, 0, 0, 0))
                if state.get("block") != last_block:
                    canvas.yview_moveto(0)
                last_block = state.get("block")
                last_stamp = stamp
        except FileNotFoundError:
            root.destroy()
            return
        except Exception as exc:
            status.configure(text=f"Display update unavailable: {exc}")
        root.after(200, refresh)

    root.after(0, refresh)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--viewer":
        raise SystemExit("Start a monitored session with: python src/face2_api.py SESSION_ID GROUP_ARM_ID monitor")
    try:
        _run_viewer(sys.argv[2])
    except Exception as exc:
        print(f"[Fatigue monitor] Viewer unavailable; session continues: {exc}", file=sys.stderr)
