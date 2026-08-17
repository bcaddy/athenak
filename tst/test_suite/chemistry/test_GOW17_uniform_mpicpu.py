"""
Test for chemistry using the GOW17 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_GOW17_uniform_gpu as gow17_uniform


@pytest.mark.skip(
    reason="GOW17 MPI test disabled until the ODE solver is fixed with larger grids"
)
@pytest.mark.parametrize("ode_solver", gow17_uniform.ode_solvers)
def test_gow17_uniform_mpicpu(ode_solver):
    """GPU Test for GOW17 uniform test problem."""
    gow17_uniform.run_gow17_uniform(ode_solver, mpi=True)
