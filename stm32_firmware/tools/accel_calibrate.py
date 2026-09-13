#!/usr/bin/env python3
"""Derive neutral accelerometer bias/scale from labeled six-face samples.

The tool does not contain sensor-specific physical constants. It only derives
parameters from the supplied measurements and a caller-provided gravity value.
Input CSV columns: face,x,y,z; faces must be +x,-x,+y,-y,+z,-z.
"""
import argparse
import csv
import json
import math
from collections import defaultdict

FACES = ("+x", "-x", "+y", "-y", "+z", "-z")


def load_samples(path):
    samples = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            face = row["face"].strip().lower()
            if face not in FACES:
                raise ValueError(f"unknown face: {face}")
            vector = tuple(float(row[name]) for name in ("x", "y", "z"))
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("non-finite sample")
            samples[face].append(vector)
    missing = [face for face in FACES if not samples[face]]
    if missing:
        raise ValueError("missing faces: " + ",".join(missing))
    return samples


def calculate(samples, gravity):
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be a positive finite input")
    means = {
        face: tuple(sum(row[index] for row in values) / len(values) for index in range(3))
        for face, values in samples.items()
    }
    bias = []
    scale = []
    for axis, positive, negative in ((0, "+x", "-x"), (1, "+y", "-y"), (2, "+z", "-z")):
        plus = means[positive][axis]
        minus = means[negative][axis]
        span = plus - minus
        if not math.isfinite(span) or abs(span) < 1e-9:
            raise ValueError(f"degenerate span on axis {axis}")
        bias.append((plus + minus) / 2.0)
        scale.append((2.0 * gravity) / abs(span))
    return {"bias": bias, "scale": scale, "gravity_input": gravity}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--gravity", type=float, required=True,
                        help="gravity magnitude in the same units as the CSV")
    parser.add_argument("--output", type=argparse.FileType("w"), default="-")
    args = parser.parse_args(argv)
    result = calculate(load_samples(args.csv_path), args.gravity)
    json.dump(result, args.output, indent=2, sort_keys=True)
    args.output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
