The software stack on every machine has its own idiosyncrasies, especially on GPU clusters. Below are settings that are known to work (as of the latest update to this page) on the following open-science clusters:

- [OLCF: Frontier (MI250X GPU nodes)](#olcf-frontier-mi250x-gpu-nodes)
- [ALCF: Aurora (PVC GPU nodes)](#alcf-aurora-pvc-gpu-nodes)
- [ALCF: Polaris (A100 GPU nodes)](#alcf-polaris-a100-gpu-nodes)
- [NERSC: Perlmutter (A100 GPU nodes)](#nersc-perlmutter-a100-gpu-nodes)
- [TACC: Stampede2 (Skylake and Ice Lake nodes)](#tacc-stampede2-skylake-and-ice-lake-nodes)
- [TACC: Stampede2 (Knights Landing nodes)](#tacc-stampede2-knights-landing-nodes)
- [NCSA: DeltaAI (GH200 GPU nodes)](#ncsa-deltaai-gh200-gpu-nodes)
- [Flatiron Institute: Rusty (A100 GPU nodes)](#flatiron-institute-rusty-a100-gpu-nodes)
- [Princeton: Della (A100 GPU nodes)](#princeton-della-a100-gpu-nodes)
- [Princeton: Stellar (Cascade Lake nodes)](#princeton-stellar-cascade-lake-nodes)
- [IAS: Apollo (A100 GPU nodes)](#ias-apollo-a100-gpu-nodes)
- [PSU: Roar Collab (A100 GPU nodes)](#psu-roar-collab-a100-gpu-nodes)

In each of the following instructions, the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

-------------------------------------------------------------------------------------------------------------
### OLCF: Frontier (MI250X GPU nodes)

For reference, consult the [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#).

##### Building and compiling

```bash
module restore
module load PrgEnv-cray \
            craype-accel-amd-gfx90a \
            cmake \
            cray-python \
            amd-mixed/5.3.0 \
            cray-mpich/8.1.23 \
            cce/15.0.1
export MPICH_GPU_SUPPORT_ENABLED=1

cd ${athenak}
cmake -Bbuild -DAthena_ENABLE_MPI=ON -DKokkos_ARCH_ZEN3=ON -DKokkos_ARCH_VEGA90A=ON \
      -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_EXE_LINKER_FLAGS="-L${ROCM_PATH}/lib -lamdhip64" \
      -DCMAKE_CXX_FLAGS=-I${ROCM_PATH}/include
cd ${build}
make -j
```

##### Running

Frontier uses the Slurm batch scheduler.  Jobs should be run in the $PROJWORK directory.  A simple example Slurm script is given below.

```bash
#!/bin/bash
#SBATCH -A AST179
#SBATCH -J mad_64_8
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 64

module restore
module load PrgEnv-cray craype-accel-amd-gfx90a cmake cray-python \
            amd-mixed/5.3.0 cray-mpich/8.1.23 cce/15.0.1
export MPICH_GPU_SUPPORT_ENABLED=1

# Can get an increase in write performance when disabling collective buffering for MPI IO
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable"

cd /lustre/orion/ast179/proj-shared/mad_64_8
srun -N 64 -n 512 -c 1 --gpus-per-node=8 --gpu-bind=closest athena -i mad_64_8.athinput
```
##### Full example with bundled jobs

The below script can be used to run any number of jobs, with each using the same executable and same number of nodes but with different input files and command-line arguments. Each job will combine stdout and stderr and write them to its own file. This script will automatically check for existing restarts and continue any such jobs, starting from the beginning only in cases where no restart files can be found.

```bash
#! /bin/bash

#SBATCH --job-name <overall_job_name>
#SBATCH --account <project>
#SBATCH --partition batch
#SBATCH --nodes <total_num_nodes>
#SBATCH --time <hours>:<minutes>:<seconds>
#SBATCH --output <overall_output_file>
#SBATCH --mail-user <email>
#SBATCH --mail-type END,FAIL

# Parameters
nodes_per_job=<nodes_per_job>
ranks_per_job=<ranks_per_job>
gpus_per_node=8
run_dir=<run_dir>
executable=<athenak_executable>
input_dir=<directory_with_athinput_files>
output_dir=<directory_to_write_terminal_output_for_each_job>
names=(<first_job_name> <second_job_name> <...>)
arguments=("<first_job_command_line_arguments>" "<second_job_command_line_arguments>" "<...>")

# Set environment
cd $run_dir
module restore
module load PrgEnv-cray craype-accel-amd-gfx90a cmake cray-python amd-mixed/5.3.0 cray-mpich/8.1.23 cce/15.0.1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable"

# Check parallel values
num_jobs=${#names[@]}
num_nodes=$((num_jobs * nodes_per_job))
if [ $num_nodes -gt $SLURM_JOB_NUM_NODES ]; then
  echo "Insufficient nodes requested."
  exit
fi
gpus_per_job=$((nodes_per_job * gpus_per_node))
if [ $ranks_per_job -gt $gpus_per_job ]; then
  echo "Insufficient GPUs requested."
  exit
fi

# Check for restart files
restart_lines=()
for ((n = 0; n < $num_jobs; n++)); do
  name=${names[$n]}
  test_file=$(find $name/rst -maxdepth 1 -name "$name.*.rst" -print -quit)
  if [ -n "$test_file" ]; then
    restart_files=$(ls -t $name/rst/$name.*.rst)
    restart_file=(${restart_files[0]})
    restart_line="-r $restart_file"
    printf "\nrestarting $name from $restart_file\n\n"
  else
    restart_line="-i $input_dir/$name.athinput"
    printf "\nstarting $name from beginning\n\n"
  fi
  restart_lines+=("$restart_line")
done

# Run code
for ((n = 0; n < $num_jobs; n++)); do
  name=${names[$n]}
  mpi_options="--nodes $nodes_per_job --ntasks $ranks_per_job --cpus-per-task 1 --gpus-per-node $gpus_per_node --gpu-bind=closest"
  athenak_options="-d $name ${restart_lines[$n]} ${arguments[$n]}"
  output_file=$output_dir/$name.out
  time srun -u $mpi_options $executable $athenak_options &> $output_file && echo $name &
  sleep 10
done
wait
```
##### Notes

There are many separate filesystems on this machine, and often rearrangements of the path are equivalent. Some of the most useful are:
  - `$HOME` ('/ccs/home/<user>'): 50 GB, backed up, files retained; good for source code and scripts
  - `$MEMBERWORK/<project>` (`/lustre/orion/<project>/scratch/<user>`): 50 TB, not backed up, 90-day purge; good for miscellaneous simulation outputs
  - `$PROJWORK/<project>` (`/lustre/orion/<project>/proj-shared`): 50 TB, not backed up, 90-day purge; good for simulation outputs shared with other project members
  - `/hpss/prod/<project>/users/<user>`: 100 TB, not backed up, files retained; needs `hsi`, `htar`, or Globus to access; good for personal storage
  - `/hpss/prod/<project>/proj-shared`: 100 TB, not backed up, files retained; needs `hsi`, `htar`, or Globus to access; good for storage shared with other project members

Andes can be used for visualization (try `module load python texlive` to get everything needed for the plotting scripts to work). Its `/gpfs/alpine/<project>` directory structure mirrors `/lustre/orion/<project>`. Files can only be transferred between them with something like Globus (the `OLCF DTN` endpoint works for both). Small files can be easily transferred through the shared home directory.

-------------------------------------------------------------------------------------------------------------
### ALCF: Aurora (PVC GPU nodes)

Intel Data Center GPU Max 1550, formerly Ponte Vecchio (PVC).

Refer to [Aurora- Early User Notes and Known Issues](https://docs.alcf.anl.gov/aurora/known-issues/) page of the ALCF User Guides frequently for updates. The most important items to note right now are:

- Job stability beyond 2048 nodes is a serious issue for most applications. Stay below this node count, for now. Major instability causes/symptoms include:
  - Ping failures (network)
  - Node hangs (node-local hardware: memory controller, voltage regulation, etc.)
  - Kernel panics
- Run job I/O on the [Flare](https://docs.alcf.anl.gov/aurora/data-management/lustre/flare/) Lustre filesystem, project directory `/lus/flare/projects/RadBlackHoleAcc/`

<!-- cmake -DAthena_ENABLE_MPI=OFF -DCMAKE_CXX_COMPILER=icpx -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON -DKokkos_ENABLE_OPENMP=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${build} -Bbuild -->

##### Building and compiling

It is recommended to bump Kokkos to at least version 4.5.01 (2024-12-23), since SYCL support was promoted from Experimental in that previous minor version release. Assuming that the AthenaK repository clone has its submodules properly initialized, execute this once:
```bash
cd ${athenak}
cd kokkos
git fetch
git checkout 4.5.01
```

To build using ahead-of-time compilation (AOT compilation):
```bash
module use cmake

cd ${athenak}
cmake -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=icpx -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_SYCL=ON \
	  -DKokkos_ARCH_INTEL_PVC=ON -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${build} \
      [-DPROBLEM=gr_torus] \
      -Bbuild
	  
cd ${build}
make -j104 all

# installs the kokkos include/, lib64/, and bin/
cmake --install . --prefix `pwd`
```

To build with just-in-time (JIT) compilation (**not recommended**), replace `-DKokkos_ARCH_INTEL_PVC=ON` with `-DKokkos_ARCH_INTEL_GEN=ON`. 

##### Running

Aurora uses the PBS job scheduler, and the officially-supported MPI distribution is Aurora MPICH. 

Launch an interactive job via:
```bash
qsub -l select=2 -l walltime=00:30:00 -A RadBlackHoleAcc -l filesystems=home:flare -I -q debug
```
See the next section on ALCF Polaris, which uses a similar MPICH (Cray) + PBS job scheduler + Cray PALS launcher setup, for a non-interactive batch jobscript example.

The resource (CPU core and multi-tile GPU) affinity options on Aurora are quite complicated. Consult the relevant [Aurora user-guides page](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/) for a detailed walkthrough. Binding 12 MPI ranks to the 12 GPU tiles on a node is best accomplished via a shared wrapper script located at `/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh`

**not yet updated:**
```bash
export MPICH_GPU_SUPPORT_ENABLED=1

cd ${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NDEPTH=16
NTHREADS=1

mpiexec -np ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} -d ${NDEPTH} --cpu-bind numa --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh ./athena -i ../../inputs/grmhd/gr_fm_torus_sane_8_4.athinput

```

-------------------------------------------------------------------------------------------------------------
### ALCF: Polaris (A100 GPU nodes)

For reference, consult the Polaris link in the [ALCF User Guide](https://docs.alcf.anl.gov/)

##### Building and compiling
The simplest way to build and compile on Polaris is:
```bash
module use /soft/modulefiles
module load PrgEnv-gnu
module load spack-pe-base cmake
module load cudatoolkit-standalone/12.4.0
module load craype-x86-milan

export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

cd ${athenak}
cmake -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE80=On \
      -DMPI_CXX_COMPILER=CC -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_ZEN3=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${build} \
      -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=CC \
      [-DPROBLEM=gr_torus] \
      -Bbuild

cd ${build}
gmake -j8 all

# installs the kokkos include/, lib64/, and bin/
cmake --install . --prefix `pwd`
```

##### Running

Polaris uses the PBS batch scheduler. An example PBS job script for Polaris can be found [here](https://docs.alcf.anl.gov/running-jobs/example-job-scripts/#gpu-mpi-examples).

Important: all multi-GPU jobs assume that you have a local copy of the script https://github.com/argonne-lcf/GettingStarted/blob/master/Examples/Polaris/affinity_gpu/set_affinity_gpu_polaris.sh in your home directory. This is required to enforce that only 1 distinct GPU device per node is visible to each MPI rank, since the PBS scheduler does not offer such a built-in option.

The following example selects 16 nodes; all jobs with >10 nodes must be routed to `prod` queue. It is strongly suggested to specify only the filesystems that are truly required for a particular job's input/output to avoid the job being stuck in the queue if Grand and/or Eagle are down for maintenance.

```bash
#!/bin/bash -l
#PBS -l select=16:ncpus=64:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:grand:eagle
#PBS -q prod
#PBS -A RadBlackHoleAcc

cd ${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NDEPTH=16
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

module use /soft/modulefiles
module load PrgEnv-gnu
module load cudatoolkit-standalone/12.4.0
module load craype-x86-milan
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
mpiexec -np ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} -d ${NDEPTH} --cpu-bind numa --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads ~/set_affinity_gpu_polaris.sh ./athena -i ../../inputs/grmhd/gr_fm_torus_sane_8_4.athinput
```

Some more options that may be useful:
```bash
#PBS -joe
#PBS -o example.out
#PBS -M <email address>
#PBS -m be
```
By default, the stderr of the job gets put into `<script name>.e<job ID>` and the stdout is written to `<script name>.o<job ID>`. The first option here merges the two streams to `<script name>.o<job ID>`. The second option overrides the default naming scheme to instead call this file `example.out`. The final two options should enable email notifications and the beginning and end of jobs.

##### Full example with bundled jobs

The below script can be used to run up to 10 jobs (limited only by `--suffix-length`), with each using the same executable and same number of nodes. Each job will combine stdout and stderr and write them to its own file. This script will also create hostfiles, which will contain a record of exactly which nodes were assigned to each job. Additionally, it will automatically check for existing restarts and continue any such jobs, starting from the beginning only in cases where no restart files can be found.

```bash
#! /bin/bash

#PBS -N <overall_job_name>
#PBS -A <project>
#PBS -q prod
#PBS -l select=<total_num_nodes>:ncpus=64:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand
#PBS -l walltime=<hours>:<minutes>:<seconds>
#PBS -j oe
#PBS -o <overall_output_file>
#PBS -M <email>
#PBS -m ae

# Parameters
nodes_per_job=<nodes_per_job>
executable=<executable>
names=(<first_job_name> <second_job_name> <...>)
input_dir=<directory_with_athinput_files>
data_dir=<directory_containing_output_directories_for_each_job>
output_dir=<directory_to_write_terminal_output_for_each_job>
arguments="<command_line_arguments>"
affinity_script=<path_to_script>/set_affinity_gpu_polaris.sh
host_name=<directory_to_use_for_temp_hostfiles>/hostfile_

# Set environment
cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load PrgEnv-gnu
module load cudatoolkit-standalone/12.4.0
module load craype-x86-milan
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

# Calculate parallel values
num_jobs=${#names[@]}
num_nodes=$((num_jobs * nodes_per_job))
max_nodes=`wc -l < $PBS_NODEFILE`
if [ $num_nodes -gt $max_nodes ]; then
  echo "Insufficient nodes requested."
  exit
fi
ranks_per_node=$(nvidia-smi -L | wc -l)
ranks_per_job=$((nodes_per_job * ranks_per_node))
depth=16
split --lines=$nodes_per_job --numeric-suffixes --suffix-length=1 $PBS_NODEFILE $host_name

# Check for restart files
restart_lines=()
for ((n = 0; n < $num_jobs; n++)); do
  name=${names[$n]}
  test_file=$(find $data_dir/$name/rst -maxdepth 1 -name "$name.*.rst" -print -quit)
  if [ -n "$test_file" ]; then
    restart_files=$(ls -t $data_dir/$name/rst/$name.*.rst)
    restart_file=(${restart_files[0]})
    restart_line="-r $restart_file"
    printf "\nrestarting $name from $restart_file\n\n"
  else
    restart_line="-i $input_dir/$name.athinput"
    printf "\nstarting $name from beginning\n\n"
  fi
  restart_lines+=("$restart_line")
done

# Run code
for ((n = 0; n < $num_jobs; n++)); do
  name=${names[$n]}
  mpi_options="-n $ranks_per_job --ppn $ranks_per_node -d $depth --cpu-bind numa --hostfile $host_name$n"
  athenak_options="-d $data_dir/$name ${restart_lines[$n]} $arguments"
  output_file=$output_dir/$name.out
  time mpiexec $mpi_options $affinity_script $executable $athenak_options &> $output_file &
done
wait
```

-------------------------------------------------------------------------------------------------------------
### NERSC: Perlmutter (A100 GPU nodes)

##### Building and compiling

```bash
module purge
module load cpe/23.12
module load PrgEnv-gnu
module load cudatoolkit/12.2
module load craype-accel-nvidia80
module load craype-x86-milan
module load xpmem
module load gpu/1.0

cd ${athena}
cmake -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
      -DKokkos_ARCH_AMPERE80=On -DKokkos_ARCH_ZEN3=ON \
      -DPROBLEM=problem \
      -Bbuild

cd ${build}
make -j4
```

Note that on Perlmutter it is important to limit the number of cores used for compiling using '-j4'.

##### Running

Perlmutter uses the Slurm scheduler.  Jobs should be run in the $PSCRATCH directory.  An example run script is given below.

```bash
#!/bin/bash
#SBATCH -A <account_number>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --license=SCRATCH

module purge
module load PrgEnv-gnu
module load cudatoolkit/11.7
module load craype-accel-nvidia80
module load cpe/23.03
module load craype-x86-milan
module load xpmem
module load gpu/1.0
module load gsl/2.7

cd $PSCRATCH/working_dir
srun ./athena -i turb.athinput
```

-------------------------------------------------------------------------------------------------------------
### TACC: Stampede2 (Skylake and Ice Lake nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

```bash
module purge
module load intel/19.1.1 impi/19.0.9 cmake/3.20.2

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D Kokkos_ARCH_SKX=On \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Slurm script

```bash
#SBATCH --account <account>
#SBATCH --partition <partition>
#SBATCH --nodes <nodes>
#SBATCH --ntasks <tasks>
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH --time <time>

module purge
module load intel/19.1.1 impi/19.0.9 cmake/3.20.2

ibrun $athenak/$build/src/athena <athenak options>"
```

This can be submitted with `sbatch <script>`.

##### Notes

The Skylake nodes on `<partition> = skx-normal` have 48 CPU cores per node, so generally `<tasks_per_node> = 48` and `<tasks> = <nodes> * 48`. For Ice Lake, `<partition> = icx-normal`, and there are 80 cores per node.

The `skx-dev` and `skx-large` queues are also available.

Source code, executables, input files, and scripts can be placed in `$HOME` or `$WORK`. Scratch space appropriate for large I/O is under `$SCRATCH`, though this is regularly purged. Archival should use space on Ranch.

-------------------------------------------------------------------------------------------------------------

### TACC: Stampede2 (Knights Landing nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

```bash
module purge
module load intel/19.1.1 impi/19.0.9 cmake/3.20.2

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D Kokkos_ARCH_KNL=On \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Slurm script

```bash
#SBATCH --account <account>
#SBATCH --partition normal
#SBATCH --nodes <nodes>
#SBATCH --ntasks <tasks>
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH --time <time>

module purge
module load intel/19.1.1 impi/19.0.9 cmake/3.20.2

ibrun $athenak/$build/src/athena <athenak options>"
```

This can be submitted with `sbatch <script>`.

##### Notes

The Knights Landing nodes have 68 CPU cores per node, so generally `<tasks_per_node> = 68` and `<tasks> = <nodes> * 68`.

The `development`, `large`, `long`, and `flat-quadrant` queues are also available.

Source code, executables, input files, and scripts can be placed in `$HOME` or `$WORK`. Scratch space appropriate for large I/O is under `$SCRATCH`, though this is regularly purged. Archival should use space on Ranch.

-------------------------------------------------------------------------------------------------------------
### NCSA: DeltaAI (GH200 GPU nodes)

The top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

As of 4/11/2026, the default modules work well. Therefore, no module changes are necessary.

```bash
cmake -D Athena_ENABLE_MPI=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ARCH_HOPPER90=ON \
      -D Kokkos_ARCH_ARMV9_GRACE=ON \
      -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
      [-D PROBLEM=<problem> \]
      -B $build

cd $build
make -j8
```

##### SLURM script

```bash
#!/bin/bash
#SBATCH -A <allocation>
#SBATCH -N <nodes>
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=<up to 4>
#SBATCH -J <job name>
#SBATCH --partition=ghx4
#SBATCH --gpu-bind=none
#SBATCH -t <time>

# Setup for problem & define any environment variables here
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"

srun -n <tasks> $athena --kokkos-map-device-id-by=mpi_rank <athenak options>

# perform any cleanup or short post-processing here
```

##### Notes

Each node is an Nvidia GH200 superchip with four sockets, each containing a 72-core ARM chip with 120 GB of RAM and a H100 GPU with 96 GB of RAM. Source code and input files should be stored in your home directory or a shared projects directory. The HDD and NVME work directories affiliated with your project (`/work/hdd/<project>/<user>` and `/work/nvme/<project>/<user>`) should be used as scratch space. Avoid I/O operations to and from your home directory while running.

-------------------------------------------------------------------------------------------------------------

### Flatiron Institute: Rusty (A100 GPU nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

```bash
module purge
module load modules/2.1-20230203 slurm cuda/11.8.0 openmpi/cuda-4.0.7
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

cmake \
  -D CMAKE_CXX_COMPILER=$athenak/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Slurm script

```bash
#SBATCH --partition gpu
#SBATCH --constraint a100,ib
#SBATCH --nodes <nodes>
#SBATCH --ntasks <tasks>
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-task 1
#SBATCH --time <time>

module purge
module load modules/2.1-20230203 slurm cuda/11.8.0 openmpi/cuda-4.0.7
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores --gpu-bind=single:2 \
  bash -c "unset CUDA_VISIBLE_DEVICES; \
  $athenak/$build/src/athena <athenak options>"
```

This can be submitted with `sbatch <script>`.

##### Notes

There are 4 GPUs per node, so generally `<tasks> = <nodes> * 4`. The code does not necessarily use all 16 CPUs per task, but this forces the tasks to be bound appropriately to NUMA nodes. The `--gpu-bind=single:2` statement may appear odd (we want 1 task per GPU, not 2), but this is a workaround for a bug in the current Slurm version installed.

Scratch space appropriate for large I/O is under `/mnt/ceph/users/`.

-------------------------------------------------------------------------------------------------------------

### Princeton: Della (A100 GPU nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

```bash
module purge
module load nvhpc/24.11 openmpi/cuda-12.6/nvhpc-24.11/4.1.6 cudatoolkit/11.8

cmake \
  -D CMAKE_CXX_COMPILER=$athenak/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Slurm script

```bash
#SBATCH --nodes <nodes>
#SBATCH --ntasks <tasks>
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 128G
#SBATCH --gres gpu:4
#SBATCH --time <time>

module purge
module load nvhpc/21.5 openmpi/cuda-11.3/nvhpc-21.5/4.1.1 cudatoolkit/11.4

srun -n $SLURM_NTASKS \
  $athenak/$build/src/athena \
  --kokkos-map-device-id-by=mpi_rank \
  <athenak options>
```

This can be submitted with `sbatch <script>`.

##### Notes

There are 4 GPUs per node, so generally `<tasks> = <nodes> * 4`. The memory per CPU should be large enough to place the tasks on different NUMA nodes; it can probably be slightly larger than 128G.

Scratch space appropriate for large I/O is under `/scratch/gpfs/`.

-------------------------------------------------------------------------------------------------------------

### Princeton: Stellar (Cascade Lake nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation

```bash
module purge
module load intel/2022.2.0 intel-mpi/intel/2021.7.0

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D Kokkos_ARCH_SKX=On \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Slurm script

```bash
#SBATCH --nodes <nodes>
#SBATCH --ntasks <tasks>
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH --cpus-per-task 1
#SBATCH --time <time>

module purge
module load intel/2022.2.0 intel-mpi/intel/2021.7.0

srun $athenak/$build/src/athena <athenak options>"
```

This can be submitted with `sbatch <script>`.

##### Notes

The Cascade Lake nodes have 96 CPU cores per node, so generally `<tasks_per_node> = 96` and `<tasks> = <nodes> * 96`.

Scratch space appropriate for large I/O is under `/scratch/gpfs/`.

-------------------------------------------------------------------------------------------------------------

### IAS: Apollo (A100 GPU nodes)

Here the top-level `AthenaK` directory will be denoted `$athenak`, and the relative build directory will be denoted `$build`.

##### Configuration and compilation:

```bash
module load nvhpc/21.5
module load openmpi/cuda-11.3/nvhpc-21.5/4.1.1
module load cudatoolkit/11.4

cmake \
  -D Kokkos_ENABLE_CUDA=On \
  -D Kokkos_ARCH_AMPERE80=On \
  -D CMAKE_CXX_COMPILER=<$athenak/kokkos/bin/nvcc_wrapper> \
  -D Athena_ENABLE_MPI=On \
  [-D PROBLEM=<problem>] \
  -B $build

cd $build
make -j 4
```

##### Example Slurm script:

```bash
#!/bin/bash

#SBATCH --job-name=<job_name>            # a short name for job
#SBATCH --nodes=1                		     # node count
#SBATCH --ntasks-per-node=4                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=1                       # gpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                   # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4                               # number of gpus per node
#SBATCH --time=24:00:00                          # total run time limit (HH:MM:SS)
#SBATCH --output=%x.%j.out                     # output stream
#SBATCH --error=%x.%j.err                        # error stream

module purge
module load nvhpc/21.5
module load openmpi/cuda-11.3/nvhpc-21.5/4.1.1
module load cudatoolkit/11.4

cd <job_directory>
srun -n 4 $athenak/$build/src/athena  --kokkos-map-device-id-by=mpi_rank <athenak options>
```

### PSU: Roar Collab (A100 GPU nodes)

Currently this configuration only supports single-GPU runs, and it assumes the build directory is located in `athenak/build`.

#### Configuration and compilation
```bash
module load cuda/11.5.0
module load cmake/3.21.4

cmake \
    -DKokkos_ENABLE_CUDA=On \
    -DKokkos_ARCH_AMPERE80=On \
    -DCMAKE_CXX_COMPILER=$PWD/../kokkos/bin/nvcc_wrapper \
    -DPROBLEM=<problem>
```

#### Example Slurm script:
```bash
#! /bin/bash
#SBATCH -A @ALLOCATION@
#SBATCH -t @WALLTIME@
#SBATCH -N @NODES@
#SBATCH -J @SIMULATION_NAME@
#SBATCH --mail-type=ALL
#SBATCH --mail-user=@EMAIL@
#SBATCH -o @RUNDIR@/@SIMULATION_NAME@.out
#SBATCH -e @RUNDIR@/@SIMULATION_NAME@.err
#SBATCH --gpus=@NGPUS@

echo "Preparing:"
set -x                          # Output commands
set -e                          # Abort on errors

cd @RUNDIR@

module load cuda/11.5.0
module load cmake/3.21.4

echo "Checking:"
pwd
hostname
date

echo "Environment:"
env | sort > ENVIRONMENT
echo ${SLURM_NODELIST} > NODES

# set up for problem & define any environment variables here

./athena -i @PARFILE@
```
