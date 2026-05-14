"""
Test for chemistry using the H2 network with advecting gaussian initial conditions. Runs
tests for different ODE solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import pathlib
import numpy as np

ode_solvers = ["forward_euler"]
input_file = "inputs/H2_advection_test.athinput"


class constants:
    km = 1.0e5  # cm/s
    mh = 1.6733e-24  # hydrogen mass
    kb = 1.380658e-16  # boltzmann's constant
    mu = 1.4  # mean molecular weight
    cv = 1.65 * kb
    density_cgs = 2.108884e-24
    n_H = 100.0
    pc_cgs = 3.0856775809623245e18
    km_s_cgs = 1.0e5


def H2_advection_analytical_solutions(x, t, mu, sigma):
    kcr = 2.0e-16 * 3
    kgr = 3.0e-17
    a1 = kcr + 2.0 * constants.n_H * kgr
    a2 = kcr
    fH0 = np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    fH = (fH0 - a2 / a1) * np.exp(-a1 * t) + a2 / a1
    fH2 = 0.5 * (1.0 - fH)
    return fH, fH2


def assert_constant(arr, value, field_name):
    assert np.all(arr == value), (
        f"The {field_name} is either not constant or not the correct value."
    )


def H2_advection_verify_state(
    state, t, mu, sigma, e_int_fiducial, n_dt_fiducial, H2_L1_limit, H_L1_limit
):
    # Verify the correct number of time steps
    assert state["cycle"] == n_dt_fiducial, (
        f"The number of time steps ({state['cycle']}) does not match the fiducial value of {n_dt_fiducial}."
    )

    # Verify the constant fields
    assert_constant(state["dens"], 1.11083e+02, "density")
    assert_constant(state["eint"], e_int_fiducial, "internal energy")
    assert_constant(state["velx"], 0.2, "velx")
    assert_constant(state["vely"], 0.0, "vely")
    assert_constant(state["velz"], 0.0, "velz")

    # Compute the analytical answers for the variable fields
    H_fiducial, H2_fiducial = H2_advection_analytical_solutions(
        state["x1v"], t, mu, sigma
    )

    # Compute the L1 norms of the variable fields
    l1_H2 = np.sum(np.abs(H2_fiducial - state["s_00_chem_H2"])) / H2_fiducial.size
    l1_H = np.sum(np.abs(H_fiducial - state["s_01_chem_H"])) / H_fiducial.size

    # Assert correctness
    assert l1_H2 < H2_L1_limit, (
        f"The L1 error for H2 ({l1_H2}) has exceeded its limit of {H2_L1_limit}."
    )
    assert l1_H < H_L1_limit, (
        f"The L1 error for H ({l1_H}) has exceeded its limit of {H_L1_limit}."
    )

    return l1_H2, l1_H


def run_h2_advection(ode_solver):
    """Run the H2 advecting Gaussian state test and compare to the analytic results.
    Parameterized over the different ODE solvers. This function is called by both the CPU
    and GPU tests."""
    try:
        results = testutils.run(input_file, [f"chemistry/{ode_solver}"])
        assert results, f"H2 uniform test run failed for {ode_solver} solver."

        # Load the data
        root_path = pathlib.Path("./tab")
        initial_state = athena_read.tab(root_path / "H2_advection.hydro_w.00000.tab")
        final_state = athena_read.tab(root_path / "H2_advection.hydro_w.00001.tab")

        # Verify the states
        _, _ = H2_advection_verify_state(
            initial_state,
            t=0.0,
            mu=0.5,
            sigma=0.1,
            e_int_fiducial=1.66625e+02,
            n_dt_fiducial=0,
            H2_L1_limit=1.1*3.2697262080917544e-07,
            H_L1_limit=1.1*5.160076015742856e-07,
        )
        l1_H2, l1_H = H2_advection_verify_state(
            final_state,
            t=5 * constants.pc_cgs / constants.km_s_cgs,
            mu=1.5,
            sigma=0.1,
            e_int_fiducial=1.29430e+2,
            n_dt_fiducial=565,
            H2_L1_limit=1.1*0.0012869569746452175,
            H_L1_limit=1.1*0.0025741334805404496,
        )

    finally:
        pass
        testutils.cleanup()


@pytest.mark.parametrize("ode_solver", ode_solvers)
def test_h2_uniform_gpu(ode_solver):
    """GPU Test for H2 advection test problem."""
    run_h2_advection(ode_solver)
