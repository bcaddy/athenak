Due to the large number of options required to manage building `AthenaK` and `Kokkos` on different architectures, the code no longer uses a configure script (as in Athena++), but instead adopts `cmake` (version 3.0 or later) to manage builds.  In-source builds are not allowed; you must create a new build directory in which `cmake` can be run, for example using the `-B` option:

    $ cmake -B {build_dir} [options]

Alternatively you can create a new directory manually in which to run `cmake`:

    $ mkdir {build_dir}
    $ cd {build_dir}
    $ cmake [options] ../

Before running `cmake` it is important to ensure the correct compilers, device toolkits (if required), and `MPI` libraries (if required) are properly loaded (see [Requirements](https://github.com/IAS-Astrophysics/athenak/wikis/Requirements)).

`Kokkos` provides many options for controlling `cmake`, see their [documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

### Frequently used cmake options

#### Building for CPUs

By default (with no options), `cmake` will build an x86 compatible CPU executable.  Sometimes better performance can be achieved by instructing `Kokkos` to optimize for the native architecture of the machine:

    $ cmake -D Kokkos_ARCH_NATIVE=ON

If the specific architecture of the machine is known, it can be useful to set it explicitly using the `-D Kokkos_ARCH_XXX=On` option, where `XXX` specifies the target architecture (see the [Architecture Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#architectures)).  

For example, to build for Intel Skylake processor using `icpc` compiler:

    $ cmake -D CMAKE_CXX_COMPILER=icpc -D Kokkos_ARCH_SKX=On

Build for Mac M1 processor using `clang-15` compiler:

    $ cmake -D CMAKE_CXX_COMPILER=clang++-mp-15 -D CMAKE_C_COMPILER=clang-mp-15 -D Kokkos_ARCH_ARMV81=On

#### Building for GPUs

Each specific device from each vendor has a corresponding architecture flag that must be set by `cmake`.

For example, to build for an NVIDIA V100 GPU:

    $  cmake -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_VOLTA70=On -D CMAKE_CXX_COMPILER=${path_to_code}/kokkos/bin/nvcc_wrapper

Note that additional options are required to specify a `CUDA` backend and `nvcc` compiler

To build for NVIDIA A100 GPU:

    $  cmake -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_AMPERE80=On -D CMAKE_CXX_COMPILER=${path_to_code}/kokkos/bin/nvcc_wrapper

To build for AMD MI250x GPU:

    $  cmake3 -D Kokkos_ENABLE_HIP=On -D Kokkos_ARCH_ZEN3=On -D Kokkos_ARCH_VEGA90A=On -D CMAKE_CXX_COMPILER=CC -D CMAKE_EXE_LINKER_FLAGS="-L${ROCM_PATH}/lib -lamdhip64" -D CMAKE_CXX_FLAGS=-I${ROCM_PATH}/include

#### Building with custom problem generators

A variety of problems can be initialized using the default problem generators compiled automatically with the code.  The source code for each is located in the /src/pgen/tests directory.  A variety of other problem generators are also included in the code in the /src/pgen direcory.  To compile the code with these, use the `-D PROBLEM={name}` option with cmake, where `{name}` is the name of the file containing the desired problem generator in /src/pgen/.

You can also build the code with any new, custom problem generator file you add to the /src/pgen directory the same way.

#### To build with MPI

Use the `-D Athena_ENABLE_MPI=ON` option.

#### To build with specific compiler options

Additional compiler options can be included using `-D CMAKE_CXX_FLAGS="options"`.  For example

    $ cmake -D CMAKE_CXX_COMPILER=icpc -D Kokkos_ARCH_BDW=On -D CMAKE_CXX_FLAGS="-O3 -inline-forceinline -qopenmp-simd -qopt-prefetch=4 -diag-disable 3180 " -D CMAKE_C_FLAGS="-O3 -finline-functions"

#### Building in debug mode

Include the `-D CMAKE_BUILD_TYPE=Debug` option
