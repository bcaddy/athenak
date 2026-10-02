"""
Unit tests for the block local line scans in utils/scan/line_scan.hpp.
The line_scan problem generator runs prefix and suffix sums in every direction and
exits with an error if any entry differs from a host reference. Each case overrides the
mesh and meshblock sizes in the input file.
"""

# Modules
import pytest
import test_suite.testutils as testutils

input_file = "inputs/line_scan_block_local_unittest.athinput"

# (mesh nx1, nx2, nx3), (meshblock nx1, nx2, nx3)
shapes = {
    "1d_nx1_16": ((64, 1, 1), (16, 1, 1)),
    "2d_nx1_16": ((32, 16, 1), (16, 8, 1)),
    "2d_nx1_32": ((64, 16, 1), (32, 8, 1)),
    "2d_nx1_64": ((128, 8, 1), (64, 4, 1)),
    "3d_nx1_4": ((16, 8, 8), (4, 4, 4)),
    "3d_nx1_24": ((48, 8, 8), (24, 8, 8)),
    "3d_non_pow2": ((16, 24, 40), (8, 12, 20)),
}


def run_line_scan(shape):
    """Run the line_scan problem generator on the mesh named by shape."""
    mesh, meshblock = shapes[shape]
    flags = [f"mesh/nx{d + 1}={n}" for d, n in enumerate(mesh)]
    flags += [f"meshblock/nx{d + 1}={n}" for d, n in enumerate(meshblock)]
    testutils.run(input_file, flags=flags)


@pytest.mark.parametrize("shape", shapes.keys())
def test_line_scan_gpu(shape):
    """GPU test for the block local line scans."""
    run_line_scan(shape)
