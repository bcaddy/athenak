#!/usr/bin/env python
"""Convert an AthenaK binary dump to the Athena 4.2 legacy VTK that Athena++'s
read_vtk problem generator consumes.

The point is a shared initial condition: both codes read this one file, so the
comparison does not depend on their turbulence drivers producing the same
realization. AthenaK's own VTK writer emits one SCALARS block per variable with
its own labels, which read_vtk cannot parse, hence this converter.

Output layout: BINARY, DATASET STRUCTURED_POINTS, CELL_DATA, big-endian
float32, with SCALARS density, SCALARS pressure and VECTORS velocity.

Usage:
    athenak_bin_to_vtk.py in.bin out.vtk [--gamma 1.6666666666666667]
"""
import argparse

import numpy as np

import bin_convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", help="AthenaK .bin dump")
    parser.add_argument("outfile", help="VTK file to write")
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0 / 3.0,
        help="adiabatic index, to turn internal energy into pressure",
    )
    parser.add_argument(
        "--zero-velocity",
        action="store_true",
        help="write zero velocities, holding the background fixed so that only "
        "the chemistry evolves (AthenaK kinematic, Athena++ active = fixed)",
    )
    args = parser.parse_args()

    data = bin_convert.read_binary_as_athdf(args.infile)

    # read_binary_as_athdf merges the MeshBlocks, so each variable arrives as a
    # single (nk, nj, ni) array and the coordinates as 1D face arrays.
    def field(name):
        return np.asarray(data[name])

    dens = field("dens")
    if args.zero_velocity:
        velx = vely = velz = np.zeros_like(dens)
    else:
        velx, vely, velz = field("velx"), field("vely"), field("velz")
    eint = field("eint")
    pres = eint * (args.gamma - 1.0)

    nk, nj, ni = dens.shape
    x1f, x2f, x3f = field("x1f"), field("x2f"), field("x3f")
    dx1 = float(x1f[1] - x1f[0])
    dx2 = float(x2f[1] - x2f[0])
    dx3 = float(x3f[1] - x3f[0])

    be = ">f4"  # legacy VTK is big-endian
    with open(args.outfile, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(
            f"AthenaK snapshot, time={float(data['Time']):.8e}\n".encode("ascii")
        )
        f.write(b"BINARY\n")
        f.write(b"DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {ni + 1} {nj + 1} {nk + 1}\n".encode("ascii"))
        f.write(
            f"ORIGIN {float(x1f[0]):.8e} {float(x2f[0]):.8e} "
            f"{float(x3f[0]):.8e}\n".encode("ascii")
        )
        f.write(f"SPACING {dx1:.8e} {dx2:.8e} {dx3:.8e}\n".encode("ascii"))
        f.write(f"CELL_DATA {ni * nj * nk}\n".encode("ascii"))

        f.write(b"SCALARS density float\nLOOKUP_TABLE default\n")
        dens.astype(be).tofile(f)
        f.write(b"\n")

        f.write(b"SCALARS pressure float\nLOOKUP_TABLE default\n")
        pres.astype(be).tofile(f)
        f.write(b"\n")

        f.write(b"VECTORS velocity float\n")
        vel = np.stack([velx, vely, velz], axis=-1)
        vel.astype(be).tofile(f)
        f.write(b"\n")

    print(
        f"wrote {args.outfile}: {ni}x{nj}x{nk} cells, "
        f"time={float(data['Time']):.6e}, "
        f"density {dens.min():.4e} to {dens.max():.4e}, "
        f"pressure {pres.min():.4e} to {pres.max():.4e}"
    )


if __name__ == "__main__":
    main()
