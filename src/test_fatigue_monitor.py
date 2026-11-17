"""Monitor regression checks without a camera, server, checkpoint, or GPU.

Run: python src/test_fatigue_monitor.py
Pillow and NumPy are needed for the frame rendering check. Inference dependencies
are mocked; the actual get_block_fatigue_4x function is compiled from its source.
"""

import ast
import base64
from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import fatigue_monitor as monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state_path = Path(self.directory.name) / "state.json"
        self.observer = monitor._SnapshotMonitor(self.state_path, "TEST_1")

    def begin(self):
        self.observer.update("begin", face_outdir="Pilot/ses-TEST_1/block-42",
                             model_dir="model_TEST.pth", n_segments=4, cutoff=1)

    def inference(self, sequences, predictions, cutoff=1, saved=None, video=False):
        # Avoid loading the hardware SDK and model stack just to test the
        # integration hooks. Execute the real function body with fake dependencies.
        tree = ast.parse(Path(__file__).with_name("face2_script.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "get_block_fatigue_4x")
        namespace = {
            "torch": SimpleNamespace(load=Mock(return_value={"model": Mock()}),
                                     device=lambda name: name,
                                     cuda=SimpleNamespace(is_available=lambda: False)),
            "get_faces_segment": Mock(side_effect=sequences),
            "read_frame_sets": Mock(return_value=saved),
            "extract_frame_sets": Mock(return_value=saved),
            "_load_saved_frame_set": Mock(side_effect=sequences),
            "save_predictions": Mock(),
            "RetinaFace": Mock(),
            "infer_score_on_faces": Mock(side_effect=predictions),
            "_publish_fatigue_monitor": monitor.publish,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "face2_script.py", "exec"), namespace)
        infer = namespace["get_block_fatigue_4x"]
        self.assertEqual(str(inspect.signature(infer)),
                         "(input_path, face_outdir, model_dir, n_segments=4, cutoff=1)")
        with redirect_stdout(io.StringIO()):
            result = infer("block.mp4" if video else "faces", "block-42", "model.pth", cutoff=cutoff)
        return result, namespace

    def test_video_and_saved_directory_use_exact_saved_sequences(self):
        for video in [True, False]:
            with self.subTest(video=video):
                saved = {"n_segments": 4, "segments": [
                    {"index": i, "start": i * 32.5, "end": (i + 1) * 32.5,
                     "frames": [{"file": f"set-{i + 1:02d}/{j:02d}.png"} for j in range(1, 17)]}
                    for i in range(4)
                ]}
                sequences = [object() for _ in range(4)]
                result, namespace = self.inference(sequences, [0, 1, 0, 0], saved=saved, video=video)
                self.assertIs(result, True)
                namespace["get_faces_segment"].assert_not_called()
                self.assertEqual(namespace["extract_frame_sets"].call_count, int(video))
                self.assertEqual([s["prediction"] for s in saved["segments"]], [0, 1, 0, 0])
                for i, call in enumerate(namespace["infer_score_on_faces"].call_args_list):
                    self.assertIs(call.args[2], sequences[i])
                namespace["save_predictions"].assert_called_once_with("block-42", saved, "model.pth", 1, True)

    def test_disabled_monitor_does_not_inspect_faces(self):
        with patch.object(monitor, "_active_monitor", None):
            monitor.publish("segment", faces=object())
            result, namespace = self.inference([object()] * 4, [0, 0, 1, 0])
        self.assertIs(result, True)
        self.assertEqual(namespace["infer_score_on_faces"].call_count, 4)

    def test_observer_failure_cannot_change_inference(self):
        failing = Mock()
        failing.update.side_effect = RuntimeError("closed display")
        with patch.object(monitor, "_active_monitor", failing):
            result, _ = self.inference([object()] * 4, [0, 0, 1, 0], cutoff=2)
        self.assertIs(result, False)

    def test_segment_identity_bounds_and_missing_segment(self):
        observer = Mock()
        sequences = [object(), None, object(), object()]
        with patch.object(monitor, "_active_monitor", observer):
            result, namespace = self.inference(sequences, [0, 1, 0])
        self.assertIs(result, True)
        calls = observer.update.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["begin"] + ["segment"] * 4 + ["end"])
        for i, call in enumerate(calls[1:5]):
            self.assertEqual((call.kwargs["index"], call.kwargs["start"], call.kwargs["end"]),
                             (i, i * 30, (i + 1) * 30))
            self.assertIs(call.kwargs["faces"], sequences[i])
        self.assertEqual([call.kwargs["prediction"] for call in calls[1:5]], [0, None, 1, 0])
        self.assertIs(calls[-1].kwargs["fatigued"], True)
        self.assertEqual(namespace["infer_score_on_faces"].call_count, 3)

    def test_no_faces_remains_unavailable(self):
        with patch.object(monitor, "_active_monitor", self.observer):
            result, _ = self.inference([None] * 4, [])
        state = json.loads(self.state_path.read_text())
        self.assertIsNone(result)
        self.assertIsNone(state["fatigued"])
        self.assertEqual(len(state["segments"]), 4)
        self.assertTrue(all(s["prediction"] is None for s in state["segments"]))

    def test_tensor_order_padding_and_rendering(self):
        import numpy as np

        # Distinct RGB values plus repeated final frames exercise exact ordering.
        pixels = np.zeros((16, 112, 112, 3), dtype=np.float32)
        pixels[0, :, :, 0] = 1
        pixels[1, :, :, 1] = 1
        pixels[2:, :, :, 2] = 1
        tensor = Mock()
        tensor.detach.return_value.cpu.return_value.permute.return_value.numpy.return_value = pixels
        self.begin()
        self.observer.update("segment", index=0, start=0, end=30, faces=tensor, prediction=1)
        state = json.loads(self.state_path.read_text())
        segment = state["segments"][0]
        colors = [base64.b64decode(f["rgb"])[:3] for f in segment["frames"]]
        self.assertEqual(colors, [b"\xff\x00\x00", b"\x00\xff\x00"] + [b"\x00\x00\xff"] * 14)
        sheet = monitor.render_frame_set(segment)
        self.assertEqual(sheet.size, (968, 276))
        self.assertEqual(sheet.getpixel((40, 70)), (255, 0, 0))
        self.assertEqual(sheet.getpixel((160, 70)), (0, 255, 0))
        # The overlay is drawn on the display sheet, leaving the frame data intact.
        self.assertEqual(sheet.getpixel((10, 130)), (182, 55, 73))
        self.assertEqual(base64.b64decode(segment["frames"][0]["rgb"])[-3:], b"\xff\x00\x00")

    def test_new_block_clears_old_predictions(self):
        self.begin()
        self.observer.update("segment", index=0, start=0, end=30, faces=None, prediction=None)
        self.observer.update("end", fatigued=None)
        self.observer.update("begin", face_outdir="block-43", model_dir="model.pth", n_segments=4, cutoff=1)
        state = json.loads(self.state_path.read_text())
        self.assertEqual(state["block"], "block-43")
        self.assertEqual(state["segments"], [])
        self.assertNotIn("fatigued", state)

    def test_closed_viewer_stops_snapshot_writes(self):
        self.observer.process = Mock()
        self.observer.process.poll.return_value = 0
        before = self.state_path.read_bytes()
        self.observer.update("segment", index=0, start=0, end=30, faces=None, prediction=None)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_context_cleans_up_after_session_error(self):
        process = Mock()
        process.poll.return_value = None
        previous = monitor._active_monitor
        with patch.object(monitor.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "session error"):
                with monitor.enabled_monitor("TEST_1"):
                    monitor.publish("begin", face_outdir="block-42", model_dir="model.pth", n_segments=4, cutoff=1)
                    state_path = monitor._active_monitor.state_path
                    self.assertTrue(state_path.exists())
                    raise RuntimeError("session error")
        self.assertIs(monitor._active_monitor, previous)
        self.assertFalse(state_path.exists())
        process.terminate.assert_called_once()

    def test_viewer_start_failure_allows_session(self):
        with patch.object(monitor.subprocess, "Popen", side_effect=OSError("no display")):
            with redirect_stdout(io.StringIO()), monitor.enabled_monitor("TEST_1"):
                monitor.publish("begin", face_outdir="block-42", model_dir="model.pth", n_segments=4, cutoff=1)
                self.assertIsNotNone(monitor._active_monitor)
                self.assertIsNone(monitor._active_monitor.process)

    def test_api_main_opens_on_block_reuses_and_reopens_window(self):
        tree = ast.parse(Path(__file__).with_name("face2_api.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        first, second = Mock(), Mock()
        first.poll.return_value = None
        second.poll.return_value = None
        previous = monitor._active_monitor
        state_paths = []
        with patch.object(monitor.subprocess, "Popen", side_effect=[first, second]) as start:
            def session():
                self.assertIsNotNone(monitor._active_monitor)
                state_paths.append(monitor._active_monitor.state_path)
                start.assert_not_called()  # No window during recording.
                for block in [42, 43]:
                    monitor.publish("begin", face_outdir=f"block-{block}", model_dir="model.pth", n_segments=4, cutoff=1)
                    monitor.publish("end", fatigued=False)
                self.assertEqual(start.call_count, 1)  # Reuse the existing window.
                first.poll.return_value = 0  # User closed it during the next game.
                monitor.publish("begin", face_outdir="block-44", model_dir="model.pth", n_segments=4, cutoff=1)
                self.assertEqual(start.call_count, 2)
                self.assertEqual(monitor._active_monitor.state["block"], "block-44")
            namespace = {"enabled_monitor": monitor.enabled_monitor,
                         "SESSION_EXTERNAL_ID": "TEST_1", "_run_session": session,
                         "MONITOR_ENABLED": True}
            exec(compile(ast.Module(body=[main], type_ignores=[]), "face2_api.py", "exec"), namespace)
            namespace["main"]()
        self.assertIs(monitor._active_monitor, previous)
        second.terminate.assert_called_once()
        self.assertFalse(state_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
