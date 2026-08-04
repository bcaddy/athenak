### Welcome to the AthenaK code repository.

AthenaK is a complete re-write of the 
[Athena++](https://github.com/PrincetonUniversity/athena)
adaptive mesh refinement (AMR) code for astrophysical MHD based on the 
[Kokkos](https://github.com/kokkos/kokkos)
library for performance portability.  AthenaK runs on any hardware supported by Kokkos, including CPUs, GPUs from most vendors, and ARM processors.

Note that Athena++ is itself an extension of the original C-version of
[Athena](https://github.com/PrincetonUniversity/Athena-Cversion).

AthenaK was written to enable calculations on modern exascale HPC systems that rely on heterogeneous architectures such as GPUs.  As such, it only implements a limited number of features that are useful for large-scale calculations.  Perhaps the most important limitation compared to Athena++ is that only Cartesian coordinates are implemented.

AthenaK was developed in conjunction with the 
[Parthenon AMR framework](https://github.com/parthenon-hpc-lab/parthenon) and borrows many features
from that effort. It is also closely related to the 
[AthenaPK MHD code](https://github.com/parthenon-hpc-lab/athenapk), which is another implementation
of Athena based on the Parthenon framework.

## Documentation:

Links in the sidebar provide further information on how build, compile, and run AthenaK, as well as giving further details on the physics modules currently implemented.
