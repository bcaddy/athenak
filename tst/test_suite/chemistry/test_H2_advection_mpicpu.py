"""
Test for chemistry using the H2 network with advecting gaussian initial conditions. Runs
tests for different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_H2_advection_gpu as h2_advection


@pytest.mark.parametrize("ode_solver", h2_advection.ode_solvers)
def test_h2_advection_mpicpu(ode_solver):
    """MPI-CPU Test for H2 advection test problem."""
    h2_advection.run_h2_advection(ode_solver, mpi=True)


@pytest.mark.parametrize("ode_solver", h2_advection.ode_solvers)
def test_h2_advection_amr_mpicpu(ode_solver):
    """MPI-CPU Test for H2 advection AMR test problem."""
    h2_advection.run_h2_advection_amr(ode_solver, mpi=True)
