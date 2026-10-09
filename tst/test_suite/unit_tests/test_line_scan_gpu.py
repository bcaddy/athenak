"""
Unit tests for the line scans in utils/scan/line_scan.hpp. The line_scan problem
generator runs prefix and suffix sums in every direction on several fields and exits
with an error if any cell of the scan differs from a host reference. With AMR the scans
are checked again after the mesh has been refined. Every case has at least 7 MeshBlocks
so it can run on 7 ranks.
"""

# Modules
import pytest
import test_suite.testutils as testutils


def uniform(mesh, meshblock, outflow=()):
    """Input file and flags for a uniform mesh: mesh and meshblock sizes, and the axes
    (1, 2, 3) with outflow instead of periodic boundaries."""
    flags = [f"mesh/nx{d + 1}={n}" for d, n in enumerate(mesh)]
    flags += [f"meshblock/nx{d + 1}={n}" for d, n in enumerate(meshblock)]
    flags += [f"mesh/{side}x{d}_bc=outflow" for d in outflow for side in ("i", "o")]
    return "inputs/line_scan_uniform.athinput", flags


def refined(name, flags=()):
    """Input file and flags for one of the mesh refinement inputs."""
    return f"inputs/line_scan_{name}.athinput", list(flags)


# name: (input file, flags)
cases = {
    "uniform_1d": uniform((128, 1, 1), (16, 1, 1), outflow=(1,)),
    "uniform_2d_nx1_16": uniform((64, 48, 1), (16, 8, 1)),
    "uniform_2d_nx1_32": uniform((128, 32, 1), (32, 8, 1)),
    "uniform_2d_nx1_64": uniform((256, 16, 1), (64, 4, 1)),
    # Longer than the GPU vector length but not a multiple of it (32 on CUDA, 64 on
    # HIP), so the vector scan ends in a partial chunk
    "uniform_2d_nx1_48": uniform((192, 16, 1), (48, 4, 1)),
    "uniform_2d_nx1_96": uniform((384, 16, 1), (96, 4, 1)),
    # Longer than any GPU vector, and than 128 threads, so the vector length is the
    # backend maximum
    "uniform_2d_nx1_256": uniform((1024, 16, 1), (256, 4, 1)),
    "uniform_3d_nx1_4": uniform((16, 8, 8), (4, 4, 4), outflow=(2,)),
    "uniform_3d_nx1_24": uniform((48, 16, 16), (24, 8, 8)),
    "uniform_3d_non_pow2": uniform((24, 36, 40), (8, 12, 20), outflow=(1, 2, 3)),
    "smr_1d": refined("smr1d"),
    "smr_2d": refined("smr2d"),
    "smr_2d_swapped_bcs": refined(
        "smr2d",
        [
            "mesh/ix1_bc=periodic",
            "mesh/ox1_bc=periodic",
            "mesh/ix2_bc=outflow",
            "mesh/ox2_bc=outflow",
        ],
    ),
    "smr_3d": refined("smr3d"),
    "smr_2d_two_regions_nx12": refined("smr2d_two_regions"),
    "smr_2d_two_regions_nx20": refined(
        "smr2d_two_regions",
        ["mesh/nx1=240", "mesh/nx2=80", "meshblock/nx1=20", "meshblock/nx2=20"],
    ),
    "amr_2d": refined("amr"),
    "amr_3d": refined(
        "amr",
        [
            "mesh/nx1=32",
            "mesh/nx2=32",
            "mesh/nx3=32",
            "mesh_refinement/num_levels=3",
            "time/nlim=4",
        ],
    ),
}


def run_line_scan(case, nranks=None):
    """Run the line_scan problem generator on the named case, with MPI on nranks ranks
    if nranks is given."""
    input_file, flags = cases[case]
    if nranks is None:
        testutils.run(input_file, flags=flags)
    else:
        testutils.mpi_run(input_file, flags=flags, threads=nranks)


@pytest.mark.parametrize("case", cases.keys())
def test_line_scan_gpu(case):
    """GPU test for the line scans."""
    run_line_scan(case)
