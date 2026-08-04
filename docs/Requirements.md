`AthenaK` is designed to use as few external libraries as possible, enabling so-called "_hermetic builds_". The only requirements are:

* the `Kokkos` Core library (can be cloned automatically with the source code)
* `cmake` (version 3.0 or later)
* a `C++17` (or later) compliant compiler

Additional compilers and software provided by the vendor may be required to build `AthenaK` on heterogeneous architectures such as GPU clusters.  For example, the CUDA toolkit is needed to build on NVidia GPUs.  See the 
[build](https://github.com/IAS-Astrophysics/athenak/wikis/Build) or 
[Notes for Specific Machines](https://github.com/IAS-Astrophysics/athenak/wikis/Notes-for-Specific-Machines) links for examples.

Additional, optional requirements are

* a `MPI` library (for distributed memory parallelism).  On GPUs, device-aware MPI libraries are required.
* `python3` for running many of the data analysis scripts included with the code