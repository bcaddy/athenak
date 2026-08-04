
# ODE Solvers

*NOTE: This is still an area of active development and as such is subject to change. If you see any discrepancies please open an issue.*

AthenaK provides ODE solvers for solving systems of ODEs. This is still a work in progress and more ODE solvers will be added.

These ODE solvers require that the ODEs they're solving have a consistent interface. This is discussed in more detail in the [Developer Documentation](#developer-documentation)

## How to Use

### Input File

ODE solvers are chosen and initialized in the module that they are used and so the parameters for them should be defined within the block of a specific physics module. This is to accommodate using different ODE solvers with different settings in the various modules in AthenaK. For example, to use the forward euler solver in the chemistry module and in another module those sections of your input file might look like this:

```
<chemistry>
network    = H2            # Chemistry network to be used
ode_solver = forward_euler # ODE solver to be used
fe_cfl     = 0.02          # The CFL number for the forward euler ODE solver

<other physics module>
ode_solver        = forward_euler # ODE solver to be used
fe_n_subcycle_max = 1e7           # maximum number of substeps for the forward euler ODE solver
```

### Forward Euler

The simple forward Euler solver uses the explicit first-order differentiation formula to solve the ODEs by sub-cycling. Since this is an explicit method it may not be suitable for stiff ODEs. This solver is primarily intended for testing due to its simplicity and isn't necessarily expected to be performant or highly accurate.

Runtime Parameters:

| Option               | Type   | Default | Description                                        |
| -------------------- | ------ | ------- | -------------------------------------------------- |
| fe_cfl               | Real   | 0.1     | cfl number for subcycling                          |
| fe_n_subcycle_max    | int    | 1e5     | maximum number of substeps                         |
| fe_yfloor            | Real   | 1e-12   | y value floor for calculating subcycling timescale |

*`fe` stands for Forward Euler.*

## Developer Documentation

To facilitate easy swapping between different ODE solvers both the solvers and ODE system classes must have a very strict APIs. The forward euler solver in [forward_euler.hpp](../../blob/master/src/ode_solvers/forward_euler.hpp) and H2 network in [H2.hpp](../../blob/master/src/chemistry/network/H2.hpp) can serve as a templates but here's a description of the APIs in more detail.

### ODE Solver API

The ODE solvers should be contained in a class that is templated on the ODE system they're solving. This enables inlining which should provide a significant performance boost on GPUs. The ODE solver's interface consists of three methods:

- A `static` `GetSettings` method that is called from the host to get any parameters needed from the input file. This should return a struct that contains all the runtime parameters for the ODE solver
- A constructor with the following signature `ODESolverName(ODESettings const settings, T& ode_system, Real const t_start, Real const dt)`
  - `ODESettings const settings`: the struct that contains the runtime parameters for the solver
  - `T& ode_system`: The ODE object to evolve
  - `Real const t_start`: The start time
  - `Real const dt`: How much time to evolve the ODEs
- A `SolveODE()` method that evolves the system of ODEs by time `dt`

### ODE System API

The ODE system should be contained in a class with at minimum the following members:

- A `neqs` variable that contains the total number of equations in the system. This should probably be declared as `static constexpr int` since it's used to define loop limits
- A `y` member variable. It should be an array of type `Real` with length `neqs` that contains the current state of the values being updated by the solver. This must support accesses `operator()` and `operator[]`, the `RegisterArray` class provides this syntax if a Kokkos View doesn't work.
- A `f` member variable. It should be an array of type `Real` with length `neqs` that contains the result of evaluating the ODE
- An `evaluate_function()` method that computes new values of `f` from the values of `y`

Specific ODE solvers might require other methods and they are noted below.

### Code Example

See the `Chemistry::UpdateChemistryTask` and `Chemistry::UpdateChemistry` methods in  [chemistry.cpp](../../blob/master/src/chemistry/chemistry.cpp) for a complete in example. A simpliefied version of the most salient sections is below.

```cpp
// First we select which ODE system to use (the H2 network) and which ODE solver
// to use (forward euler) and get their settings from the input file
TaskStatus Chemistry::UpdateChemistryTask(Driver* d, int stage) {
  const std::string network = my_pin->GetString("chemistry", "network");
  const std::string ode_solver = my_pin->GetString("chemistry", "ode_solver");

  if (network == "H2") {
    auto h2_settings = H2Network::GetSettings(my_pin);
    if (ode_solver == "forward_euler") {
      auto fe_settings = ode_solvers::ForwardEuler<H2Network>::GetSettings(
          my_pin, "chemistry");
      UpdateChemistry<ode_solvers::ForwardEuler<H2Network>, H2Network>(
          fe_settings, h2_settings);
    }
  }

  return TaskStatus::complete;
}

// Now we actually use the solver in a kernel to solve the system of ODEs
template <typename ODE_Solver_t, typename Network_t, typename ODESettings,
          typename NetworkSettings>
void Chemistry::UpdateChemistry(ODESettings const& ode_settings,
                                NetworkSettings const& network_settings) {
  // ------ Collect variables that we'll need -----
  // The primitive grid
  auto w0 = GetW0();
  // The time at the beginning of this timestep
  Real const t_start = pmy_pack->pmesh->time;
  // The timestep
  Real const dt = pmy_pack->pmesh->dt;

  // ----- Variables for the ODE solver -----
  // For reporting if the ODE solver doesn't converge
  DvceArray0D<bool> chemisty_ode_failure("chemisty_ode_failure", false);

  Kokkos::parallel_for(
      "Chemistry_ODE_Solve", policy,
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        // Create the chemisty object
        Network_t chem_net(network_settings, w0(mb_idx, IDN, k, j, i));

        // ------ Load cell values ------
        // Load values into the `y` array
        int grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs; s_idx++) {
          chem_net.y(s_idx) = w0(mb_idx, grid_idx, k, j, i);
          grid_idx += 1;
        }

        // ------ Solve the ODEs ------
        ODE_Solver_t ode_solver(ode_settings, chem_net, t_start, dt);
        ode_solver.SolveODE();

        // check if the ODE solver failed
        if (ode_solver.failed) {
          chemisty_ode_failure() = ode_solver.failed;
        }

        // ------ Write cell values back out ------
        // Write the values back out
        int grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs; s_idx++) {
          w0(mb_idx, grid_idx, k, j, i) = chem_net.y(s_idx);
          grid_idx += 1;
        }
      });

  // Get the failure flag and check for failure
  bool chemisty_ode_failure_h;
  Kokkos::deep_copy(chemisty_ode_failure_h, chemisty_ode_failure);
  if (chemisty_ode_failure_h) {
    std::cerr << "The ODE solver failed to converge." << std::endl;
  }
}
```
