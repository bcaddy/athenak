"""
Test for chemistry using the NCR network: the thermal/chemical equilibrium test of
Section 7.1 in Kim et al. 2023, ApJS 264, 10. Port of the TIGRESS-NCR regression test
tst/regression/scripts/tests/photchem/equil.py, run for every interpolation mode
(<chemistry>/NCR_interp_mode) and metallicity (Z_g = Z_d = Z) that it runs.

- iprob=0, thermal and chemical equilibrium as a function of n_H, is compared with the
  unshielded reference solutions at a relative tolerance of 1e-2 (P/k_B, x_H+, x_H2),
  for n_H < 1e3 cm^-3 as TIGRESS does. Only those 100 rows of TIGRESS's 161-row grid
  are run, n_H = 10^-2 to 10^2.95 cm^-3 in 0.05 dex. Higher on that grid, near
  n_H = 10^5.65 cm^-3, the NCR equations have no stable equilibrium (a Hopf-unstable
  window, see docs/Chemistry.md) and BDF takes a very large number of steps there.
- iprob=1, chemical equilibrium as a function of T at n_H = 1 cm^-3, is compared with
  the TIGRESS exact-mode (NCR_interp_mode = 0) solution at TIGRESS's per-mode
  tolerances (x_HI, x_H2, x_e). x_e is not an AthenaK output, so it is recovered from
  the pressure: P = n_H k_B T (1.1 + x_e - x_H2) with the target T.

The reference data, test_suite/chemistry/data/ncr_equilibrium.npz, is in physical units:
  - iprob=0: tigress-ncr tst/regression/data/ref_photchem_solutions/
    noshld_zg{log10 Z}_zd{log10 Z}_chi0.0_xi0.00.txt, the first 100 rows,
    keys iprob0_zg{log10 Z:.1f}_{nH, pok, xH2, xHII, xe}
  - iprob=1: a converged rerun of TIGRESS-NCR (commit
    118ffe56acd7712ace476e1e687a234a96fd5577) photchem_equil, built with
    `configure.py -ncr --prob=photchem_equil` and run with equil.py's arguments and
    interp_mode=0. TIGRESS's stored photchem_equil_iprob1_*.tab references are not
    converged: SolveToEquilibrium stops once the elapsed time reaches the dt = 1e5 the
    problem generator passes, while the lagged-x_e iteration is still moving, leaving
    x_e off by up to 5-8e-4 at Z >= 1, 4e-3 at Z = 0.1, and 1e-2 at Z = 0.01 (x_H+
    likewise; x_HI and x_H2 by up to 1.3e-4). The rerun drops that elapsed-time
    stop and iterates until the per-iteration relative changes of x_H+, x_e, and x_HI
    (relative to max(x_HI, 1e-2), as x_HI = 1 - x_H+ - 2 x_H2 carries ~1e-16 absolute
    round-off) are below 1e-13, with niter_max = 1e6; the physics is unchanged. At
    Z = 0.01 and T < 8e3 K the iteration ends in a period-2 round-off cycle with
    changes of ~1e-12, stable to 4e6 iterations. The output,
    photchem_equil_exact_iprob1_zg{log10 Z}_zd{log10 Z}.block0.out1.00000.tab, is
    converted with n_H = rho * density_to_nh and P/k_B = press * pressure_cgs / k_B in
    TIGRESS's ism units (m_H = 1.6738234e-24, k_B = 1.38065e-16, mu_H = 1.4) and the
    target temperatures T = 10^(1 + 7 k / 160), keys
    iprob1_zg{log10 Z:.1f}_{T_target, nH, pok, xHI, xH2, xe}
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import pathlib
import numpy as np

# The units in inputs/NCR_equilibrium_test.athinput and AthenaK's constants, used to
# convert the output to physical units
length_cgs = 3.08567758e18
mass_cgs = 6.882615081124725e31
time_cgs = 3.08567758e13
density_cgs = mass_cgs / length_cgs**3
pressure_cgs = mass_cgs / length_cgs / time_cgs**2
k_boltzmann_cgs = 1.3806488e-16  # units::Units::k_boltzmann_cgs
hydrogen_mass_cgs = 1.6733e-24  # units::Units::hydrogen_mass_cgs
mu_H = 1.4  # <chemistry>/mu_H default
gamma = 1.666666666666667
x_He = 0.1  # NCRNetwork::xHe
temp_hot1 = 3.5e4  # <chemistry>/NCR_temp_hot1

# The metallicities and interpolation modes TIGRESS tests
metallicities = [0.01, 0.1, 1.0, 10.0**0.5]
interp_modes = [0, 1, 2, 3]  # exact, log-log, NQT2, NQT1

# iprob=0 relative tolerance, as TIGRESS
iprob0_rtol = 1e-2
# iprob=1 relative tolerances per interp_mode, as TIGRESS (equil.py tol_exact,
# tol_loglog, tol_nqt2, tol_nqt1), against the exact-mode reference
iprob1_rtol = {
    0: {"xHI": 2e-4, "xH2": 2e-4, "xe": 3e-4},
    1: {"xHI": 3e-4, "xH2": 3e-4, "xe": 3e-4},
    2: {"xHI": 5e-3, "xH2": 5e-3, "xe": 5e-3},
    3: {"xHI": 2e-2, "xH2": 2e-2, "xe": 2e-2},
}
# The grid of each iprob, as command line overrides of the 161-cell mesh of
# inputs/NCR_equilibrium_test.athinput. iprob=0 runs only the compared rows, n_H < 1e3
# cm^-3: the first 100 rows of TIGRESS's grid, n_H = 10^-2 to 10^2.95 in 0.05 dex.
# iprob=1 runs TIGRESS's full 161-row T grid. mpi_nx1 is the MeshBlock size of the MPI
# tests, which divides nx1 into at least as many MeshBlocks (20 and 23) as MPI ranks
grids = {
    0: {
        "nx1": 100,
        "mpi_nx1": 5,
        "args": ["problem/log_nH_min=-2.0", "problem/log_nH_max=2.95"],
    },
    1: {"nx1": 161, "mpi_nx1": 7, "args": ["chemistry/NCR_chem_only=true"]},
}
# The n_H grid must match the reference, as TIGRESS checks
nH_rtol = 1e-8
# Above temp_hot1 the reference x_H2 is a 2^-58 round-off artifact and AthenaK's H2
# decays rather than being zeroed, so there x_H2 is only checked to be negligible
xH2_hot_max = 5e-14


def assert_rel_close(name, test, ref, rtol):
    """Assert that every element of test is within rtol of ref, relatively."""
    assert test.shape == ref.shape, (
        f"{name}: shape {test.shape} does not match the reference {ref.shape}."
    )
    rel = np.abs(test / ref - 1.0)
    idx = np.argmax(rel)
    assert rel[idx] <= rtol, (
        f"{name}: maximum relative difference {rel[idx]:.3e} > {rtol:g} at row {idx}, "
        f"test = {test[idx]:.10e}, reference = {ref[idx]:.10e}."
    )


def load_output():
    """Load the final tab output, sorted by x1v, and convert it to physical units.
    Returns a dict of n_H, P/k_B, x_H2, and x_H+."""
    data_path = sorted(pathlib.Path("./tab").glob("NCR_equilibrium.hydro_w.*.tab"))[-1]
    data = athena_read.tab(data_path)
    order = np.argsort(data["x1v"])

    # The species columns are s_NN_chem_<name>
    def species(name):
        keys = [key for key in data if key.endswith(f"_chem_{name}")]
        assert len(keys) == 1, f"Found {len(keys)} columns for {name} in {data_path}."
        return data[keys[0]][order]

    return {
        "nH": data["dens"][order] * density_cgs / (mu_H * hydrogen_mass_cgs),
        "pok": (gamma - 1.0) * data["eint"][order] * pressure_cgs / k_boltzmann_cgs,
        "xH2": species("H2"),
        "xH+": species("H+"),
    }


def run_ncr_equilibrium(iprob, interp_mode, z, mpi=False):
    """Run inputs/NCR_equilibrium_test.athinput with the given iprob, NCR_interp_mode,
    and metallicity Z = Z_g = Z_d, and compare it with the TIGRESS reference
    solution. This function is called by the CPU, MPI-CPU, and GPU tests."""
    RUN = testutils.mpi_run if mpi else testutils.run

    try:
        grid = grids[iprob]
        mb_nx1 = grid["mpi_nx1"] if mpi else grid["nx1"]
        cli_args = [
            f"mesh/nx1={grid['nx1']}",
            f"meshblock/nx1={mb_nx1}",
            *grid["args"],
            f"problem/iprob={iprob}",
            f"chemistry/NCR_interp_mode={interp_mode}",
            # Formatted as TIGRESS's equil.py did when making the reference
            f"chemistry/NCR_Z_g={z:f}",
            f"chemistry/NCR_Z_d={z:f}",
        ]

        input_file = "inputs/NCR_equilibrium_test.athinput"
        results = RUN(input_file, cli_args)
        assert results, (
            f"NCR equilibrium run failed for iprob={iprob}, "
            f"interp_mode={interp_mode}, Z={z:g}."
        )

        test = load_output()
        ref = np.load("../../test_suite/chemistry/data/ncr_equilibrium.npz")
        key = f"iprob{iprob}_zg{np.log10(z):.1f}"
        label = f"iprob={iprob}, interp_mode={interp_mode}, Z={z:g}"

        assert_rel_close(f"{label} nH", test["nH"], ref[f"{key}_nH"], nH_rtol)
        if iprob == 0:
            assert_rel_close(
                f"{label} P/k", test["pok"], ref[f"{key}_pok"], iprob0_rtol
            )
            assert_rel_close(
                f"{label} xHII", test["xH+"], ref[f"{key}_xHII"], iprob0_rtol
            )
            assert_rel_close(
                f"{label} xH2", test["xH2"], ref[f"{key}_xH2"], iprob0_rtol
            )
        else:
            rtol = iprob1_rtol[interp_mode]
            T = ref[f"{key}_T_target"]
            xHI = 1.0 - test["xH+"] - 2.0 * test["xH2"]
            xe = test["pok"] / (test["nH"] * T) - (1.0 + x_He) + test["xH2"]

            assert_rel_close(f"{label} xHI", xHI, ref[f"{key}_xHI"], rtol["xHI"])
            assert_rel_close(f"{label} xe", xe, ref[f"{key}_xe"], rtol["xe"])

            cold = T < temp_hot1
            assert_rel_close(
                f"{label} xH2 (T < temp_hot1)",
                test["xH2"][cold],
                ref[f"{key}_xH2"][cold],
                rtol["xH2"],
            )
            xH2_hot = np.abs(test["xH2"][~cold]).max()
            assert xH2_hot < xH2_hot_max, (
                f"{label} xH2 (T >= temp_hot1): maximum |xH2| {xH2_hot:.3e} >= "
                f"{xH2_hot_max:g}."
            )
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("z", metallicities, ids=lambda z: f"Z{z:.3g}")
@pytest.mark.parametrize("interp_mode", interp_modes)
@pytest.mark.parametrize("iprob", [0, 1])
def test_ncr_equilibrium_gpu(iprob, interp_mode, z):
    """GPU Test for the NCR equilibrium test problem."""
    run_ncr_equilibrium(iprob, interp_mode, z)
