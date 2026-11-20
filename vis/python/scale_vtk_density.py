#!/usr/bin/env python
"""Scale the density and pressure of a shared-IC legacy VTK (athenak_bin_to_vtk.py
layout) by one factor, keeping the temperature and the velocities.

Used to run the same turbulent realization at a different mean density, e.g. the
n_H = 10 snapshot at n_H = 1 with --factor 0.1.

Usage:
    scale_vtk_density.py in.vtk out.vtk --factor 0.1
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("--factor", type=float, required=True,
                        help="multiplies density and pressure")
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        raw = f.read()

    out = bytearray()
    pos = 0
    ncell = None

    def line():
        nonlocal pos
        end = raw.index(b"\n", pos)
        text = raw[pos:end]
        pos = end + 1
        return text

    # header up to CELL_DATA
    while True:
        text = line()
        out += text + b"\n"
        if text.startswith(b"CELL_DATA"):
            ncell = int(text.split()[1])
            break
    # SCALARS name / LOOKUP_TABLE / data, or VECTORS name / data
    while pos < len(raw):
        text = line()
        if not text.strip():
            out += text + b"\n"
            continue
        out += text + b"\n"
        kind, name = text.split()[0], text.split()[1]
        if kind == b"SCALARS":
            out += line() + b"\n"  # LOOKUP_TABLE default
            n = ncell
        elif kind == b"VECTORS":
            n = 3 * ncell
        else:
            raise ValueError(f"unexpected section {text!r}")
        data = np.frombuffer(raw, dtype=">f4", count=n, offset=pos)
        pos += 4 * n
        if kind == b"SCALARS" and name in (b"density", b"pressure"):
            data = (data.astype(np.float64) * args.factor).astype(">f4")
        out += data.tobytes()
        # the writer ends each data block with a newline
        if pos < len(raw) and raw[pos:pos + 1] == b"\n":
            out += b"\n"
            pos += 1

    with open(args.outfile, "wb") as f:
        f.write(bytes(out))


if __name__ == "__main__":
    main()
