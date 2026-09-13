import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("accel_calibrate.py")
spec = importlib.util.spec_from_file_location("accel_calibrate", MODULE_PATH)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class AccelCalibrationTests(unittest.TestCase):
    def test_six_faces_derive_only_from_input(self):
        rows = [
            ("+x", 11.0, 1.0, 2.0), ("-x", -9.0, 1.0, 2.0),
            ("+y", 1.0, 12.0, 2.0), ("-y", 1.0, -8.0, 2.0),
            ("+z", 1.0, 2.0, 10.0), ("-z", 1.0, 2.0, -10.0),
        ]
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as stream:
            writer = csv.writer(stream)
            writer.writerow(("face", "x", "y", "z"))
            writer.writerows(rows)
            path = stream.name
        try:
            result = cal.calculate(cal.load_samples(path), 10.0)
        finally:
            Path(path).unlink()
        self.assertAlmostEqual(result["bias"][0], 1.0)
        self.assertAlmostEqual(result["bias"][1], 2.0)
        self.assertAlmostEqual(result["bias"][2], 0.0)
        self.assertAlmostEqual(result["scale"][0], 1.0)
        self.assertAlmostEqual(result["scale"][1], 1.0)
        self.assertAlmostEqual(result["scale"][2], 1.0)

    def test_missing_face_rejected(self):
        with self.assertRaises(ValueError):
            cal.calculate({face: [(0.0, 0.0, 0.0)] for face in cal.FACES[:-1]}, 9.8)

    def test_degenerate_axis_rejected(self):
        samples = {face: [(0.0, 0.0, 0.0)] for face in cal.FACES}
        with self.assertRaises(ValueError):
            cal.calculate(samples, 9.8)


if __name__ == "__main__":
    unittest.main()
