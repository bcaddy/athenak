//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file PDR_moving.cpp
//! \brief Problem generator for the moving PDR test with the GOW17 network and
//! six-ray radiation. A cold slab embedded in warm gas moves at a uniform
//! velocity. Port of the Athena++ chem_pdr_moving problem generator, see
//! Section 4.2.2 in Gong et al. 2023, arXiv:2305.04965

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
//! \fn ProblemGenerator::PDRMoving()
//! \brief Problem Generator for the moving PDR test. The cold slab occupies
//! 5 <= x1 <= 45 with n_H = 100, warm gas with n_H = n_H_warm fills the rest. Both
//! phases have the same internal energy density, that of the cold slab at 40 K
//! (as in Athena++), so they start in pressure equilibrium.

void ProblemGenerator::PDRMoving(ParameterInput* pin, const bool restart) {
  if (restart) return;

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

  // ----- Get the input parameters from the input file -----
  // Two phase initial condition, the cold phase is fixed as in Athena++
  
  const Real n_H_cold = 100.0;  // n_H of the cold slab, in cm^-3
  const Real T_cold = 40.0;     // temperature of the cold slab, in K

  // n_H of the warm gas surrounding the slab, in cm^-3
  const Real n_H_warm = pin->GetOrAddReal("problem", "n_H_warm", 0.1);
  
  const Real xc_start = 5.0;  // inner x1 edge of the cold slab, in pc
  const Real xc_end = 45.0;   // outer x1 edge of the cold slab, in pc
  
  // x1 velocity of all the gas, in km/s (code velocity units)
  const Real vx = pin->GetReal("problem", "vx_kms");

  // Convert n_H to code density
  const Real n_H_to_d = pmbp->punit->hydrogen_mass_cgs *
                        pmbp->pchemistry->mu_H / pmbp->punit->density_cgs();

  // Internal energy density of the cold slab, fully molecular gas with xHe =
  // 0.1
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real cv = chemistry::Thermo::CvCold(0.5, 0.1, 0.0, gamma);
  const Real eint = n_H_cold * T_cold * cv / pmbp->punit->pressure_cgs();

  // Chemistry values
  const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
  HostArray1D<Real> initial_chemistry_h("initial_chemistry_h",
                                        chemistry::GOW17Network::neqs - 1);
  for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
    // Determine the name in the parameter file
    const auto name = chemistry::GOW17Network::species_names[i];
    const auto init_name = std::string("init_") + std::string(name);

    // Get the value and save it
    initial_chemistry_h(i) =
        pin->GetOrAddReal("problem", init_name, init_default);
  }

  // Copy intializing data to the device
  DvceArray1D<Real> initial_chemistry_d("initial_chemistry_d",
                                        chemistry::GOW17Network::neqs - 1);
  Kokkos::deep_copy(initial_chemistry_d, initial_chemistry_h);

  // Assign values
  const int chem_start = pmbp->pchemistry->get_chemistry_scalars_first_idx();
  par_for(
      "pgen_PDR_moving", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
      je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Compute the location
        Real& x1min = size.d_view(m).x1min;
        Real& x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        const Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

        // Cold slab or warm gas
        const Real n_H = (x1 >= xc_start && x1 <= xc_end) ? n_H_cold : n_H_warm;

        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = n_H * n_H_to_d;
        w0(m, IVX, k, j, i) = vx;
        w0(m, IVY, k, j, i) = 0.0;
        w0(m, IVZ, k, j, i) = 0.0;
        w0(m, IEN, k, j, i) = eint;

        // Assign chemistry values to this cell
        for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
          w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
        }
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
}
