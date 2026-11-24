"""
Test for chemistry using the GOW17 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.chemistry.test_GOW17_gpu as gow17_tests


@pytest.mark.parametrize("ode_solver", gow17_tests.ode_solvers)
def test_gow17_uniform_big_step_cpu(ode_solver):
    """CPU Test for GOW17 uniform test problem."""
    gow17_tests.run_gow17_uniform_big_step(ode_solver)


@pytest.mark.parametrize("ode_solver", gow17_tests.ode_solvers)
def test_gow17_cfl_dependence_cpu(ode_solver):
    """CPU Test for if the results of network depend on the CFL number"""
    gow17_tests.run_gow17_cfl_dependence(ode_solver)


@pytest.mark.parametrize("ode_solver", gow17_tests.ode_solvers)
def test_gow17_uniform_time_series_cpu(ode_solver):
    """CPU Test for GOW17 time series problem."""
    gow17_tests.run_gow17_uniform_time_series(ode_solver)


def test_pdr_static_cpu():
    """CPU Test for the static PDR test problem."""
    gow17_tests.run_pdr_regression("PDR_static", "pdr_static.npz")


def test_pdr_moving_cpu():
    """CPU Test for the moving PDR test problem."""
    gow17_tests.run_pdr_regression("PDR_moving", "pdr_moving.npz")


def test_pdr_moving_amr_cpu():
    """CPU Test for the moving PDR test problem with AMR."""
    gow17_tests.run_pdr_regression("PDR_moving_amr", "pdr_moving_amr.npz")


def test_pdr_moving_adiabatic_cpu():
    """CPU Test for the adiabatic moving PDR test problem."""
    gow17_tests.run_pdr_adiabatic_regression("PDR_moving", "pdr_moving_adiabatic.npz")


def test_pdr_moving_amr_adiabatic_cpu():
    """CPU Test for the adiabatic moving PDR test problem with AMR."""
    gow17_tests.run_pdr_adiabatic_regression(
        "PDR_moving_amr", "pdr_moving_amr_adiabatic.npz"
    )


def test_decaying_turbulence_cpu():
    """CPU Test for the decaying turbulence test problem."""
    gow17_tests.run_decaying_turbulence_regression()
