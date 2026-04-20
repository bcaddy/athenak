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
#include <vector>
#include <tuple>

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
TaskStatus Chemistry::TestKernel(Driver* d, int stage) {
  // Get the conserved grid
  auto u0 = GetU0();

  // Get all the loop limits
  auto const[start_4, end_4] = LoopLimitsAllCells();
  int const field_start_idx = chemistry_scalars_first_idx;
  int const field_stop_idx = get_chemistry_scalars_last_idx() + 1;

  Kokkos::parallel_for(
      "write_to_chem_scalars",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), start_4, end_4),
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        for (int field_idx = field_start_idx; field_idx < field_stop_idx;
             field_idx++) {
          u0(mb_idx, field_idx, k, j, i) = 13. * u0(mb_idx, IDN, k, j, i);
        }
      });

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
