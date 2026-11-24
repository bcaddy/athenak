"""
Test for chemistry using the GOW17 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_GOW17_gpu as gow17_tests


@pytest.mark.parametrize("ode_solver", gow17_tests.ode_solvers)
def test_gow17_uniform_big_step_mpicpu(ode_solver):
    """GPU Test for GOW17 uniform test problem."""
    gow17_tests.run_gow17_uniform_big_step(ode_solver, mpi=True)


def test_pdr_static_mpicpu():
    """MPI-CPU Test for the static PDR test problem."""
    gow17_tests.run_pdr_regression("PDR_static", "pdr_static.npz", mpi=True)


def test_pdr_moving_mpicpu():
    """MPI-CPU Test for the moving PDR test problem."""
    gow17_tests.run_pdr_regression("PDR_moving", "pdr_moving.npz", mpi=True)


def test_pdr_moving_amr_mpicpu():
    """MPI-CPU Test for the moving PDR test problem with AMR. Uses 8 ranks since the
    mesh starts with 14 MeshBlocks."""
    gow17_tests.run_pdr_regression(
        "PDR_moving_amr", "pdr_moving_amr.npz", mpi=True, mpi_ranks=8
    )


def test_pdr_moving_adiabatic_mpicpu():
    """MPI-CPU Test for the adiabatic moving PDR test problem."""
    gow17_tests.run_pdr_adiabatic_regression(
        "PDR_moving", "pdr_moving_adiabatic.npz", mpi=True
    )


def test_pdr_moving_amr_adiabatic_mpicpu():
    """MPI-CPU Test for the adiabatic moving PDR test problem with AMR. Uses 8 ranks
    since the mesh starts with 14 MeshBlocks."""
    gow17_tests.run_pdr_adiabatic_regression(
        "PDR_moving_amr", "pdr_moving_amr_adiabatic.npz", mpi=True, mpi_ranks=8
    )


def test_decaying_turbulence_mpicpu():
    """MPI-CPU Test for the decaying turbulence test problem. Uses 8 ranks since the
    mesh starts with 8 MeshBlocks."""
    gow17_tests.run_decaying_turbulence_regression(mpi=True)
