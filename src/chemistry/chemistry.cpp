//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry.cpp
//! \brief implementation of Chemistry class constructor and assorted other
//! functions
#include "chemistry/chemistry.hpp"

#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

namespace chemistry {
//----------------------------------------------------------------------------------------
// Constructor, initializes data structures and parameters
//----------------------------------------------------------------------------------------
Chemistry::Chemistry(MeshBlockPack* ppack, ParameterInput* pin)
    : pmy_pack(ppack),
      is_hydro_enabled(pin->DoesBlockExist("hydro")),
      is_mhd_enabled(pin->DoesBlockExist("mhd")),
      nscalars_chemistry(SetupGetNumChemistryScalars(ppack, pin, -1, false)),
      chemistry_scalars_first_idx(ComputeChemistryScalarsStartIndex()) {
  // print a message telling users that this module isn't ready yet
  std::string const red = "\033[31m";
  std::string const reset = "\033[0m";
  std::cerr << red
            << "The chemistry module has been enabled. Chemistry is not fully "
               "implemented yet and using it may lead to unpredictable results."
            << reset << std::endl;
}

//----------------------------------------------------------------------------------------
// Destructor, primarily frees memory
//----------------------------------------------------------------------------------------
Chemistry::~Chemistry() {}

// ================
// Member Functions
// ================
TaskStatus Chemistry::UpdateChemistry(Driver* d, int stage) {
  // ------ Collect variables that we'll need -----
  // The primitive grid
  auto w0 = GetW0();
  // The time at the beginning of this timestep
  Real const t_start = pmy_pack->pmesh->time;
  // The timestep
  Real const dt = pmy_pack->pmesh->dt;

  // ----- Variables for the Forward Euler Solver -----
  // For reporting if the forward euler solver didn't converge within
  // max_iterations
  Kokkos::View<bool> forward_euler_failure("forward_euler_failure", false);
  static constexpr Real small_ = 1024. * std::numeric_limits<float>::min();
  // TODO convert these into runtime arguments
  // The floor for chemical abundances
  Real const yfloor = 1.0e-3;
  // The CFL number for the forward euler subcycling
  Real const cfl_cool_subcycle = 0.1;
  // The maximum number of forward euler iterations
  Real const max_iterations = 1e5;

  // ----- Get the unit conversions we'll need -----
  Real const time_cgs = pmy_pack->punit->time_cgs();
  Real const energy_density_cgs = pmy_pack->punit->pressure_cgs();

  // ----- Get all the loop limits and generate the parallel policy ------
  // NOLINTNEXTLINE(whitespace/braces)
  auto const [start_limit, end_limit] = LoopLimitsAllCells();
  int const species_start_idx = chemistry_scalars_first_idx;
  auto const policy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
      DevExeSpace(), start_limit, end_limit);

  Kokkos::parallel_for(
      "write_to_chem_scalars", policy,
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        // Create the chemisty object
        H2Network chemistry_network(w0(mb_idx, IDN, k, j, i), time_cgs,
                                    energy_density_cgs);

        // ------ Thread local arrays for ODE stuff ------
        // TODO move these into the chemistry network
        Real y_raw[H2Network::neqs], f_raw[H2Network::neqs];
        Kokkos::View<Real[H2Network::neqs],
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            y(y_raw), f(f_raw);

        // ------ Load cell values ------
        // Internal energy
        y[H2Network::IIE] = w0(mb_idx, IEN, k, j, i);

        // Chemistry scalars. The loop is based off of the chemical
        // network's number of equations since that's known at compile time,
        // enabling more loop optimizations
        int grid_idx = species_start_idx;
        for (int s_idx = 1; s_idx < H2Network::neqs; s_idx++) {
          y[s_idx] = w0(mb_idx, grid_idx, k, j, i);
          grid_idx += 1;
        }

        // ------ Solve the ODEs ------
        int icount = 0;
        Real t_now = t_start;
        Real t_end = t_start + dt;
        while (t_now < t_end) {
          // Evaluate the ODEs
          chemistry_network.evaluate_function(t_now, dt, y, f);

          // Compute the chemistry time scale
          Real dt_subcycle;
          {
            // calculate chemistry timescale
            dt_subcycle = Kokkos::reduction_identity<Real>::min();
            for (int s_idx = 0; s_idx < H2Network::neqs; s_idx++) {
              // put floor in species abundance
              Real const yf = Kokkos::max(y[s_idx], yfloor);

              // Compute the value to reduce
              dt_subcycle = Kokkos::min(dt_subcycle,
                                        Kokkos::abs(yf / (f[s_idx] + small_)));
            }
            dt_subcycle = cfl_cool_subcycle * dt_subcycle;

            // If t_now + dt_subcycle is greater than t_end then lower the
            // timestep accordingly
            dt_subcycle = Kokkos::min(dt_subcycle, t_end - t_now);
          }

          // Advance one subcycle
          {
            for (int s_idx = 0; s_idx < H2Network::neqs; s_idx++) {
              y[s_idx] += f[s_idx] * dt_subcycle;
            }
          }

          // Update timing
          t_now += dt_subcycle;
          icount++;

          // check if convergence is established within max_iterations.  If not,
          // trigger a failure
          if (icount > max_iterations) {
            forward_euler_failure() = true;
          }
        }

        // ------ Write cell values back out ------
        // Internal energy
        w0(mb_idx, IEN, k, j, i) = y[H2Network::IIE];

        // Chemistry scalars
        grid_idx = species_start_idx;
        for (int s_idx = 1; s_idx < H2Network::neqs; s_idx++) {
          w0(mb_idx, grid_idx, k, j, i) = y[s_idx];
          grid_idx += 1;
        }
      });

  // Get the failure flag and check for failure
  bool forward_euler_failure_h;
  Kokkos::deep_copy(forward_euler_failure_h, forward_euler_failure);
  if (forward_euler_failure_h) {
    std::cerr << "The Forwared Euler ODE solver failed to converge within "
              << max_iterations << "cycles." << std::endl;
    return TaskStatus::fail;
  }

  return TaskStatus::complete;
}

std::string Chemistry::GetSpeciesNames(int const& scalar_idx) {
  // Only the first time this is called create the mapping between species names
  // and grid index
  static std::map<int, std::string> species_names_map;
  if (species_names_map.size() == 0) {
    // std::vector of scalar names
    std::vector<std::string> species_names = {
        "chem_species_1", "chem_species_2", "chem_species_3"};

    // Create the mapping
    int name_idx = 0;
    for (size_t i = get_chemistry_scalars_first_idx();
         i < get_chemistry_scalars_last_idx() + 1; i++) {
      species_names_map[i] = species_names[name_idx];
      name_idx++;
    }
  }

  // Verify that this is a chemistry scalar
  if (scalar_idx < get_chemistry_scalars_first_idx() ||
      scalar_idx > get_chemistry_scalars_last_idx()) {
    std::stringstream msg;
    msg << "Attempted to output the field at index " << scalar_idx
        << " as a passive scalar for the chemistry module but it is not one of "
           "the scalars managed by the chemistry module.";
    throw std::runtime_error(msg.str());
  }

  // Return the proper name
  return species_names_map[scalar_idx];
}

DvceArray5D<Real> Chemistry::GetU0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->u0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->u0;
  }
}

DvceArray5D<Real> Chemistry::GetW0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->w0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->w0;
  }
}

int Chemistry::ComputeChemistryScalarsStartIndex() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->nhydro + nscalars_pre_chemistry;
  } else if (is_mhd_enabled) {
    return pmy_pack->pmhd->nmhd + nscalars_pre_chemistry;
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }
}

std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>>
Chemistry::LoopLimitsAllCells() {
  Kokkos::Array<int, 4> const start = {
      0,                             // meshblock start
      pmy_pack->pmesh->mb_indcs.ks,  // k start
      pmy_pack->pmesh->mb_indcs.js,  // j start
      pmy_pack->pmesh->mb_indcs.is   // i start
  };
  Kokkos::Array<int, 4> const end = {
      pmy_pack->nmb_thispack,            // meshblock end
      pmy_pack->pmesh->mb_indcs.ke + 1,  // k end
      pmy_pack->pmesh->mb_indcs.je + 1,  // j end
      pmy_pack->pmesh->mb_indcs.ie + 1   // i end
  };
  return {start, end};
}

}  // namespace chemistry
