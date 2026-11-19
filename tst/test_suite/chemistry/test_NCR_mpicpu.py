"""
Test for chemistry using the NCR network: the thermal/chemical equilibrium test. A
smoke subset (log-log interpolation, Z = 1) of the GPU tests on several MeshBlocks and
MPI ranks, see test_NCR_gpu.py.
"""

# Modules
import pytest
import test_suite.chemistry.test_NCR_gpu as ncr_tests


@pytest.mark.parametrize("iprob", [0, 1])
def test_ncr_equilibrium_mpicpu(iprob):
    """MPI-CPU Test for the NCR equilibrium test problem."""
    ncr_tests.run_ncr_equilibrium(iprob, interp_mode=1, z=1.0, mpi=True)
