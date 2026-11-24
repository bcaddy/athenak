//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file GOW17_decaying_turbulence.cpp
//! \brief Decaying turbulence with the GOW17 network, in a single run.
//!
//! The gas starts uniform in density and temperature with a seeded random velocity
//! field on the largest scales of the box, then decays with chemistry on. It is the
//! single-run form of the AthenaK/Athena++ comparison of jeonggyukim/athenak-photchem
//! (tests/turb-box-athenak-vs-athenapp), which drove the turbulence in a separate
//! hydro-only run. There is no driver, so the problem works with adaptive mesh
//! refinement and with any radiation integrator.
//!
//! The velocity field is reproducible across compilers, architectures, MPI layouts
//! and mesh refinement:
//!   - The random amplitudes come from RanSt, which is integer arithmetic up to one
//!     multiply by a constant and so bit-identical everywhere, unlike the std::
//!     distributions, whose algorithms are implementation-defined, or RanGaussianSt,
//!     which uses log and sqrt.
//!   - The field is a function of the global position only, so it does not depend
//!     on how the mesh is split into MeshBlocks or ranks, or on refinement level.
//!   - The normalization to v_rms is analytic and done on the host, not a reduction
//!     over cells whose order would vary.
//! What remains is sin and cos of each cell's phase, which can differ in the last
//! bit between the host and device math libraries; the hydro differs between them
//! at that level anyway.

#include <cmath>
#include <cstdlib>
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
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GOW17DecayingTurbulence()
//! \brief Uniform gas with a seeded velocity field
//!   v(x) = sum_k [a_k cos(k.x) + b_k sin(k.x)],
//! over the wavevectors k = 2 pi n / L with nlow <= |n| <= nhigh, one of each +-n
//! pair. Every component of a_k and b_k is uniform in (-1, 1) times |n|^-(expo+2)/2,
//! the weighting of the turbulence driver, and the field is scaled so that its rms
//! over the box is v_rms. The modes are orthogonal on the periodic box, so the mean
//! of |v|^2 is sum_k (|a_k|^2 + |b_k|^2) / 2 exactly.

void ProblemGenerator::GOW17DecayingTurbulence(ParameterInput *pin,
                                               const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pchemistry == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "GOW17_decaying_turbulence needs a <chemistry> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;

  // ----- Get the input parameters from the input file -----
  const Real n_H = pin->GetReal("problem", "n_H");        // in cm^-3
  const Real T_init = pin->GetReal("problem", "T_init");  // in K
  const Real v_rms = pin->GetReal("problem", "v_rms");    // in km/s, code velocity
  const int nlow = pin->GetOrAddInteger("problem", "nlow", 1);
  const int nhigh = pin->GetOrAddInteger("problem", "nhigh", 2);
  const Real expo = pin->GetOrAddReal("problem", "expo", 5.0 / 3.0);
  const int seed = pin->GetOrAddInteger("problem", "seed", 1);

  // Chemistry values
  const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  Real x_H2 = 0.0;
  for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
    const auto name = chemistry::GOW17Network::species_names[i];
    const auto init_name = std::string("init_") + std::string(name);
    initial_chemistry.h_view(i) = pin->GetOrAddReal("problem", init_name, init_default);
    if (name == "H2") x_H2 = initial_chemistry.h_view(i);
  }
  initial_chemistry.template modify<HostMemSpace>();
  initial_chemistry.template sync<DevMemSpace>();
  auto initial_chemistry_d = initial_chemistry.d_view;

  // Uniform density and internal energy. The heat capacity is the network's own, so
  // the gas starts at T_init as the network sees it; the electrons from the trace
  // initial ions are negligible.
  const Real dens = n_H * pmbp->punit->hydrogen_mass_cgs * pmbp->pchemistry->mu_H /
                    pmbp->punit->density_cgs();
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real xHe = chemistry::GOW17Network::GetSettings(pin, pmbp).xHe;
  const Real eint = n_H * T_init * chemistry::Thermo::CvCold(x_H2, xHe, 0.0, gamma) /
                    pmbp->punit->pressure_cgs();

  // ----- Draw the modes on the host -----
  // One wavevector of each +-n pair, the one whose first nonzero component is
  // positive, with nlow <= |n| <= nhigh
  auto is_mode = [nlow, nhigh](const int nx, const int ny, const int nz) {
    const bool upper_half =
        (nx > 0) || (nx == 0 && ny > 0) || (nx == 0 && ny == 0 && nz > 0);
    const int nsq = nx * nx + ny * ny + nz * nz;
    return upper_half && nsq >= nlow * nlow && nsq <= nhigh * nhigh;
  };
  int nmodes = 0;
  for (int nx = 0; nx <= nhigh; ++nx) {
    for (int ny = -nhigh; ny <= nhigh; ++ny) {
      for (int nz = -nhigh; nz <= nhigh; ++nz) {
        if (is_mode(nx, ny, nz)) nmodes++;
      }
    }
  }

  // Wavevectors are k = 2 pi n / L, with phases measured from the domain's lower corner
  const Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  const Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  const Real x3min_mesh = pmy_mesh_->mesh_size.x3min;
  const Real two_pi_lx = 2.0 * M_PI / (pmy_mesh_->mesh_size.x1max - x1min_mesh);
  const Real two_pi_ly = 2.0 * M_PI / (pmy_mesh_->mesh_size.x2max - x2min_mesh);
  const Real two_pi_lz = 2.0 * M_PI / (pmy_mesh_->mesh_size.x3max - x3min_mesh);

  // Per mode: k (3 entries), then a (3) and b (3), one velocity component each
  DualArray2D<Real> modes("turbulence_modes", nmodes, 9);
  RNG_State rng_state;
  rng_state.idum = -seed;  // a non-positive idum seeds RanSt
  Real mean_vsq = 0.0;
  int m = 0;
  for (int nx = 0; nx <= nhigh; ++nx) {
    for (int ny = -nhigh; ny <= nhigh; ++ny) {
      for (int nz = -nhigh; nz <= nhigh; ++nz) {
        if (!is_mode(nx, ny, nz)) continue;
        const int nsq = nx * nx + ny * ny + nz * nz;
        const Real weight = std::pow(static_cast<Real>(nsq), -(expo + 2.0) / 4.0);
        modes.h_view(m, 0) = nx * two_pi_lx;
        modes.h_view(m, 1) = ny * two_pi_ly;
        modes.h_view(m, 2) = nz * two_pi_lz;
        for (int c = 3; c < 9; ++c) {
          // Uniform in (-1, 1). 2 RanSt is exact, so the result is the same whether or
          // not the compiler fuses it with the subtraction
          modes.h_view(m, c) = weight * (2.0 * RanSt(&rng_state) - 1.0);
          mean_vsq += 0.5 * SQR(modes.h_view(m, c));
        }
        m++;
      }
    }
  }

  // Scale to v_rms
  const Real norm = v_rms / std::sqrt(mean_vsq);
  for (int mode = 0; mode < nmodes; ++mode) {
    for (int c = 3; c < 9; ++c) modes.h_view(mode, c) *= norm;
  }
  modes.template modify<HostMemSpace>();
  modes.template sync<DevMemSpace>();
  auto modes_d = modes.d_view;

  // ----- Assign values -----
  const int chem_start = pmbp->pchemistry->get_chemistry_scalars_first_idx();
  par_for(
      "pgen_GOW17_decaying_turbulence", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
      ks, ke, js, je, is, ie, KOKKOS_LAMBDA(int mb, int k, int j, int i) {
        const Real x1 = CellCenterX(i - is, indcs.nx1, size.d_view(mb).x1min,
                                    size.d_view(mb).x1max);
        const Real x2 = CellCenterX(j - js, indcs.nx2, size.d_view(mb).x2min,
                                    size.d_view(mb).x2max);
        const Real x3 = CellCenterX(k - ks, indcs.nx3, size.d_view(mb).x3min,
                                    size.d_view(mb).x3max);

        Real v[3] = {0.0, 0.0, 0.0};
        for (int mode = 0; mode < nmodes; ++mode) {
          const Real phase = modes_d(mode, 0) * (x1 - x1min_mesh) +
                             modes_d(mode, 1) * (x2 - x2min_mesh) +
                             modes_d(mode, 2) * (x3 - x3min_mesh);
          const Real cos_phase = Kokkos::cos(phase);
          const Real sin_phase = Kokkos::sin(phase);
          for (int c = 0; c < 3; ++c) {
            v[c] += modes_d(mode, 3 + c) * cos_phase + modes_d(mode, 6 + c) * sin_phase;
          }
        }

        w0(mb, IDN, k, j, i) = dens;
        w0(mb, IVX, k, j, i) = v[0];
        w0(mb, IVY, k, j, i) = v[1];
        w0(mb, IVZ, k, j, i) = v[2];
        w0(mb, IEN, k, j, i) = eint;

        for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
          w0(mb, chem_start + s, k, j, i) = initial_chemistry_d(s);
        }
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
}
