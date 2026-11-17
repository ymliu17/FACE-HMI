"""Save only the original-resolution images selected for segment inference."""

import json
import math
import os
from pathlib import Path
import shutil
import tempfile


FORMAT = "face-hmi-frame-sets-v1"


def sample_indices(first, stop, count=16):
    """Uniform chronological samples; pad only when the video has too few frames."""
    if count < 1:
        raise ValueError("Frame count must be positive")
    available = stop - first
    if available <= 0:
        return []
    if available < count:
        return list(range(first, stop)) + [stop - 1] * (count - available)
    if count == 1:
        return [first]
    return [first + round(i * (available - 1) / (count - 1)) for i in range(count)]


def _crop(frame, detector):
    """Keep the controller's full-frame fallback when no valid face is found."""
    detections = detector.detect(frame)
    bbox = getattr(detections[0], "bbox", None) if detections else None
    if bbox is not None and len(bbox) >= 4 and all(math.isfinite(float(v)) for v in bbox[:4]):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2], [x1, y1, x2, y2]
    return frame, None


def read_frame_sets(face_outdir):
    path = Path(face_outdir) / "frame_sets" / "manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"Unrecognized frame set manifest: {path}")
    return data


def extract_frame_sets(video_path, face_outdir, detector, n_segments=4, seq_len=16):
    """Decode selected video frames once and save lossless, unannotated crops.

    The stable frame_sets directory is replaced only after extraction succeeds.
    ECG, source video, and historical top-level JPGs are never touched.
    """
    import cv2
    from PIL import Image

    if n_segments < 1 or seq_len < 1:
        raise ValueError("n_segments and seq_len must be positive")
    output = Path(face_outdir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "frame_sets"
    if destination.exists() and read_frame_sets(output) is None:
        raise FileExistsError(f"Refusing to replace an unrecognized directory: {destination}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(total_frames) or total_frames < 0:
            raise ValueError(f"Invalid video timing metadata: {video_path}")
        total_frames = int(total_frames)
        data = {"format": FORMAT, "source_video": str(video_path), "fps": fps,
                "total_frames": total_frames, "n_segments": n_segments,
                "sequence_length": seq_len, "segments": []}
        selected = {}
        for i in range(n_segments):
            first = total_frames * i // n_segments
            stop = total_frames * (i + 1) // n_segments
            indices = sample_indices(first, stop, seq_len)
            segment = {"index": i, "start": first / fps, "end": stop / fps,
                       "prediction": None, "frames": []}
            for position, index in enumerate(indices, 1):
                record = {"file": f"set-{i + 1:02d}/{position:02d}.png",
                          "frame_index": index, "time_sec": index / fps}
                segment["frames"].append(record)
                selected.setdefault(index, []).append(record)
            data["segments"].append(segment)

        with tempfile.TemporaryDirectory(prefix=".frame-sets-", dir=output) as directory:
            staging = Path(directory) / "frame_sets"
            staging.mkdir()
            last = max(selected, default=-1)
            for index in range(last + 1):
                if not cap.grab():
                    raise OSError(f"Video ended before selected frame {index}: {video_path}")
                if index not in selected:
                    continue
                success, frame = cap.retrieve()
                if not success or frame is None:
                    raise OSError(f"Cannot decode selected frame {index}: {video_path}")
                crop, bbox = _crop(frame, detector)
                original = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                first_file = None
                for record in selected[index]:
                    path = staging / record["file"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if first_file is None:
                        original.save(path)
                        first_file = path
                    else:
                        shutil.copyfile(first_file, path)
                    record["crop_box"] = bbox
                    record["size"] = list(original.size)
            (staging / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            backup = Path(directory) / "previous"
            if destination.exists():
                destination.rename(backup)
            try:
                staging.rename(destination)
            except Exception:
                if backup.exists():
                    backup.rename(destination)
                raise
        return data
    finally:
        cap.release()


def save_predictions(face_outdir, data, model_dir, cutoff, fatigued):
    """Pair each saved sequence with the prediction obtained from that sequence."""
    data.update(model=str(model_dir), cutoff=cutoff, fatigued=fatigued)
    path = Path(face_outdir) / "frame_sets" / "manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)
