//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file NCR_equilibrium.cpp
//! \brief Problem generator for the NCR thermal and chemical equilibrium tests.
//! Port of the TIGRESS-NCR photchem_equil problem generator, see Section 7.1 in
//! Kim et al. 2023, ApJS 264, 10. Each cell is an independent one-zone problem
//! along x1:
//!   - iprob=0: thermal and chemical equilibrium as a function of n_H, which is
//!     log-spaced from 10^log_nH_min to 10^log_nH_max
//!   - iprob=1: chemical equilibrium at a fixed temperature, log-spaced from
//!     10^log_tgas_min to 10^log_tgas_max at n_H = nH. Requires
//!     <chemistry>/NCR_chem_only = true, which pins the temperature
//! Unlike TIGRESS there is no equilibrium pre-solve, equilibrium is reached by
//! integrating for a long time.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::NCREquilibrium()
//! \brief Problem Generator for the NCR equilibrium tests. The position along
//! the log-spaced grid comes from the global cell index, computed from the cell
//! center, so that runs with any number of MeshBlocks or MPI ranks have the
//! same initial conditions as TIGRESS's single block. All cells start with
//! x_H+ = 0.01 and x_H2 = 0 at rest.

void ProblemGenerator::NCREquilibrium(ParameterInput* pin, const bool restart) {
  if (restart) return;

  // ----- Check the setup -----
  const std::string network = pin->GetString("chemistry", "network");
  if (network != "NCR") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "NCR_equilibrium requires <chemistry>/network = NCR, found "
              << network << std::endl;
    exit(EXIT_FAILURE);
  }
  const int iprob = pin->GetInteger("problem", "iprob");
  if (iprob != 0 && iprob != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<problem>/iprob = " << iprob
              << " is not a valid option, choose from {0, 1}" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto& indcs = pmy_mesh_->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  auto& size = pmbp->pmb->mb_size;
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // The root level grid, used to find each cell's global index along x1
  const Real mesh_x1min = pmy_mesh_->mesh_size.x1min;
  const int mesh_nx1 = pmy_mesh_->mesh_indcs.nx1;
  const Real mesh_dx1 = (pmy_mesh_->mesh_size.x1max - mesh_x1min) / mesh_nx1;

  // ----- Get the input parameters from the input file -----
  // iprob=0, the n_H grid
  const Real log_nH_min = pin->GetOrAddReal("problem", "log_nH_min", -2.0);
  const Real log_nH_max = pin->GetOrAddReal("problem", "log_nH_max", 6.0);
  // iprob=1, n_H and the temperature grid
  const Real nH_fixed = pin->GetOrAddReal("problem", "nH", 1.0);
  const Real log_tgas_min = pin->GetOrAddReal("problem", "log_tgas_min", 1.0);
  const Real log_tgas_max = pin->GetOrAddReal("problem", "log_tgas_max", 8.0);

  // The network settings and the chemistry radiation, whose constant field is
  // the one the network sees with <chemistry>/radiation_integrator = constant
  const chemistry::NCRSettings settings =
      pmbp->pchemistry->GetNetworkSettings<chemistry::NCRSettings>();
  const chemistry::Radiation& chem_rad = pmbp->pchemistry->GetRadiation();
  const chemistry::Radiation::ConstantRadiationFunc<chemistry::NCRNetwork>
      rad_func(chem_rad);

  // iprob=1 holds T fixed, which needs chem_only mode, and sets the initial
  // energy from x_e at the target temperature, which needs the radiation field
  // to be known here
  if (iprob == 1 && !settings.chem_only) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "NCR_equilibrium with <problem>/iprob = 1 requires "
              << "<chemistry>/NCR_chem_only = true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (iprob == 1 && chem_rad.integrator != "constant") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "NCR_equilibrium with <problem>/iprob = 1 requires "
              << "<chemistry>/radiation_integrator = constant" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Unit conversions
  const Real density_cgs = pmbp->punit->density_cgs();
  const Real energy_density_cgs = pmbp->punit->pressure_cgs();
  const Real hydrogen_mass_cgs = pmbp->punit->hydrogen_mass_cgs;
  const Real k_boltzmann_cgs = pmbp->punit->k_boltzmann_cgs;
  const Real mu_H = pmbp->pchemistry->mu_H;
  const Real n_H_to_d = hydrogen_mass_cgs * mu_H / density_cgs;
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;

  // Initial abundances, as TIGRESS
  const Real xH_plus_init = 0.01;
  const Real xH2_init = 0.0;

  // Initial T/mu for iprob=0, in K. TIGRESS sets P = rho k_B (T/mu) / m_H with
  // rho = 1.4 m_H n_H, i.e. its own mu convention rather than AthenaK's
  // temperature_cgs. It only sets the starting point.
  const Real tgas_mu_init = 1.0e3;

  // Assign values
  const int chem_start = pmbp->pchemistry->get_chemistry_scalars_first_idx();
  par_for(
      "pgen_NCR_equilibrium", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks,
      ke, js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // The global cell index along x1 and its fraction of the grid
        Real& x1min = size.d_view(m).x1min;
        Real& x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        const Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
        const int i_global =
            static_cast<int>(Kokkos::floor((x1v - mesh_x1min) / mesh_dx1));
        const Real ridx =
            static_cast<Real>(i_global) / static_cast<Real>(mesh_nx1 - 1);

        // Density, pressure, and temperature
        Real n_H, e_int;
        if (iprob == 0) {
          n_H = Kokkos::pow(10.0,
                            log_nH_min + ridx * (log_nH_max - log_nH_min));
          const Real pressure_cgs = 1.4 * n_H * k_boltzmann_cgs * tgas_mu_init;
          e_int = pressure_cgs / ((gamma - 1.0) * energy_density_cgs);
        } else {
          n_H = nH_fixed;
          const Real tgas = Kokkos::pow(
              10.0, log_tgas_min + ridx * (log_tgas_max - log_tgas_min));

          // x_e at the target temperature with the constant radiation field
          const Real xe = chemistry::NCRNetwork::ElectronAbundance(
              settings, rad_func(m, k, j, i), n_H, tgas, xH2_init,
              xH_plus_init);

          const Real cv = chemistry::Thermo::CvCold(
              xH2_init, chemistry::NCRNetwork::xHe, xe, gamma);
          e_int = n_H * cv * tgas / energy_density_cgs;
        }

        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = n_H * n_H_to_d;
        w0(m, IVX, k, j, i) = 0.0;
        w0(m, IVY, k, j, i) = 0.0;
        w0(m, IVZ, k, j, i) = 0.0;
        w0(m, IEN, k, j, i) = e_int;

        // Assign chemistry values to this cell
        w0(m, chem_start + chemistry::NCRNetwork::IH2, k, j, i) = xH2_init;
        w0(m, chem_start + chemistry::NCRNetwork::IH_plus, k, j, i) =
            xH_plus_init;
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
}
