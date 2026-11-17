"""Run with: python src/test_face_frame_sets.py (OpenCV, NumPy, Pillow)."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from face_frame_sets import extract_frame_sets, read_frame_sets, sample_indices, save_predictions


class FrameSetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "block-42"
        self.detector = Mock()
        self.detector.detect.return_value = [SimpleNamespace(bbox=[7, 9, 53, 43])]

    def video(self, count=160):
        # MJPEG is broadly available; expected pixels are read back from the video
        # so assertions compare against decoded originals, not pre-codec pixels.
        path = Path(self.directory.name) / "sample.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(count):
            frame = np.full((48, 64, 3), (i % 256, 70, 180), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_uniform_sampling_and_short_sequence_padding(self):
        indices = sample_indices(0, 900)
        self.assertEqual(len(indices), 16)
        self.assertEqual((indices[0], indices[-1]), (0, 899))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(sample_indices(10, 13), [10, 11] + [12] * 14)
        self.assertEqual(sample_indices(4, 4), [])

    def test_exact_saved_originals_no_intermediate_jpgs(self):
        path = self.video()
        result = extract_frame_sets(path, self.output, self.detector)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 64)
        self.assertEqual(list(self.output.glob("*.jpg")), [])
        self.assertEqual(self.detector.detect.call_count, 64)
        cap = cv2.VideoCapture(str(path))
        decoded = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            decoded.append(frame)
        cap.release()
        for i, segment in enumerate(result["segments"]):
            indices = [r["frame_index"] for r in segment["frames"]]
            self.assertEqual(indices, sample_indices(i * 40, (i + 1) * 40))
            for record in segment["frames"]:
                with Image.open(self.output / "frame_sets" / record["file"]) as saved:
                    self.assertEqual(saved.size, (46, 34))
                    expected = cv2.cvtColor(decoded[record["frame_index"]][9:43, 7:53], cv2.COLOR_BGR2RGB)
                    np.testing.assert_array_equal(np.asarray(saved), expected)
        result["segments"][1]["prediction"] = 1
        save_predictions(self.output, result, "model_TEST.pth", 1, True)
        saved = read_frame_sets(self.output)
        self.assertIs(saved["fatigued"], True)
        self.assertEqual(saved["segments"][1]["prediction"], 1)

    def test_short_video_keeps_padding_and_fallback_color(self):
        path = self.video(8)
        self.detector.detect.return_value = []
        result = extract_frame_sets(path, self.output, self.detector)
        self.assertEqual(self.detector.detect.call_count, 8)
        for i, segment in enumerate(result["segments"]):
            self.assertEqual([r["frame_index"] for r in segment["frames"]], [i * 2] + [i * 2 + 1] * 15)
            for record in segment["frames"]:
                self.assertIsNone(record["crop_box"])
                with Image.open(self.output / "frame_sets" / record["file"]) as image:
                    self.assertEqual(image.size, (64, 48))
                    # Red > blue: fallback must not swap the camera's BGR colors.
                    self.assertGreater(image.getpixel((30, 20))[0], image.getpixel((30, 20))[2])

    def test_failed_reprocessing_preserves_previous_inputs(self):
        path = self.video()
        extract_frame_sets(path, self.output, self.detector)
        manifest = self.output / "frame_sets" / "manifest.json"
        before = manifest.read_bytes()
        self.detector.detect.side_effect = RuntimeError("detector failed")
        with self.assertRaisesRegex(RuntimeError, "detector failed"):
            extract_frame_sets(path, self.output, self.detector)
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 64)
        self.assertEqual(list(self.output.glob(".frame-sets-*")), [])

    def test_reprocessing_replaces_generated_sets_only(self):
        path = self.video()
        extract_frame_sets(path, self.output, self.detector)
        (self.output / "ecg_raw.csv").write_text("preserve ECG")
        (self.output / "0.jpg").write_bytes(b"preserve historical image")
        result = extract_frame_sets(path, self.output, self.detector, n_segments=2)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 32)
        self.assertFalse((self.output / "frame_sets" / "set-04").exists())
        self.assertEqual((self.output / "ecg_raw.csv").read_text(), "preserve ECG")
        self.assertEqual((self.output / "0.jpg").read_bytes(), b"preserve historical image")

    def test_invalid_or_truncated_video_does_not_publish_partial_sets(self):
        path = self.video()
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [20, 160]
        cap.grab.return_value = False
        with patch.object(cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(OSError, "ended before"):
                extract_frame_sets(path, self.output, self.detector)
        cap.release.assert_called_once()
        self.assertFalse((self.output / "frame_sets").exists())


if __name__ == "__main__":
    unittest.main()
