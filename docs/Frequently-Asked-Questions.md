* [Why does the code run fine with one task but crash with MPI?](https://github.com/IAS-Astrophysics/athenak/wiki/Frequently-Asked-Questions#why-does-the-code-run-fine-with-one-task-but-crash-with-mpi)
* [I'm not getting the speedup I expect on GPUs versus CPUs. Why?](https://github.com/IAS-Astrophysics/athenak/wiki/Frequently-Asked-Questions#im-not-getting-the-speedup-i-expect-on-gpus-versus-cpus-why)
* [Why am I getting poor scaling?](https://github.com/IAS-Astrophysics/athenak/wiki/Frequently-Asked-Questions#why-am-i-getting-poor-scaling)
* [What are the conventions for primitive and conserved variables?](https://github.com/IAS-Astrophysics/athenak/wiki/Frequently-Asked-Questions#what-are-the-conventions-for-primitive-and-conserved-variables)
* [I keep getting errors about `NoOp ConsToPrim` and `NoOp PrimToCons` in my custom pgen.](https://github.com/IAS-Astrophysics/athenak/wiki/Frequently-Asked-Questions#i-keep-getting-errors-about-noop-constoprim-and-noop-primtocons-in-my-custom-pgen)

## Compiling

### Why does the code run fine with one task but crash with MPI?
This most frequently happens on GPUs and is usually one of a few different things:
* Your MPI installation is not GPU-aware. Consult your computing facility's documentation or support staff to see if there is a CUDA-aware MPI version available or if one can be installed. If not, all MPI calls need to be performed on the CPU. Right now there is no built-in option to do this in AthenaK, though it is being tested.
* Your MPI installation is GPU-aware, but the machine is not properly configured for GPU-to-GPU communication (e.g., NCSA Delta). This can also be circumvented by moving MPI calls to the CPU.
* Your MPI installation is GPU-aware, but it is incompatible with your particular GPU toolkit (CUDA, ROCm, etc.).
* Your MPI installation is GPU-aware, but your batch script is not configured correctly. For systems using SLURM, try setting `--gpu-bind=none`. Additionally, some versions of MPI need specific environment variables set. For example, if you are running into issues with MPICH, try setting `MPICH_GPU_SUPPORT_ENABLED=1`.

## Performance

### I'm not getting the speedup I expect on GPUs versus CPUs. Why?
GPUs achieve most of their performance through vectorization and distributing a job across many threads. If your job is too small, it will not saturate the GPU properly, and you will not see significant performance gains compared to CPUs.

If you are certain that your job size is not an issue, consider if there are other bottlenecks that could be degrading performance (e.g., excessive or poorly configured I/O).

Lastly, we have sometimes seen that some hardware/compiler combinations conspire together in ways that can cause Kokkos's scheduling heuristics to select sub-optimal options that degrade performance. Certain compiler versions on OLCF Frontier, for example, have been observed to do this. If you suspect this is a problem on a specific machine, please consult your machine's computing support or open a [discussion](https://github.com/IAS-Astrophysics/athenak/discussions).

### Why am I getting poor scaling?
Here are a few common reasons for poor scaling:
* Your job is too small.
* Your job is not being distributed evenly across MPI tasks. The smallest unit of work in AthenaK is a single meshblock, so if the number of blocks does not evenly divide across the number of tasks, some tasks will receive more blocks than others. You can check the number of blocks before launching a batch job by running locally with `./athena -m -i <parfile>`.
* Your computing environment is not properly configured for your machine. Consult your HPC center's documentation or support staff if you believe this might be an issue.

Where strong scaling is an issue, you may need to temper your expectations, unfortunately. The theoretical compute-to-memory ratio on a GPU is much higher than on most comparable CPU machines. Therefore, the difference between a load that saturates a GPU's compute capabilities and a load that fills all available memory is typically much smaller than on a typical CPU machine, so strong scaling is necessarily worse. For most problems on most machines, we generally find that AthenaK can usually maintain 80% efficiency or better on a job that uses 8x the minimal number of resources (e.g., if the job needs 1 GPU, you should get at least 80% of the theoretical speedup for 8 GPUs).

## Physics

### What are the conventions for primitive and conserved variables?
Primitive variables:
* Newtonian hydro and MHD: the density $\rho$ (`IDN` or `dens`), the velocity $\mathbf{v}$ (`IVX`, `IVY`, and `IVZ` or `velx`, `vely`, and `velz`), the internal energy density $u$ (`IEN` or `eint`), and any passive scalars (`s_00`, `s_01`, etc.). Internal energy density will not be included when using an isothermal EOS.
* Relativistic hydro and MHD: the rest-mass density $\rho$ (`IDN` or `dens`), the three-velocity weighted by the Lorentz factor $W v^i$ (or $\Gamma v^i$, depending on your notation) in the Eulerian frame (`IVX`, `IVY`, and `IVZ` or `velx`, `vely`, and `velz`), the internal energy density $u$ in the rest frame (`IEN` or `eint`), and passive scalars $Y_x$ as measured in the rest frame (`s_00`, `s_01`, etc.).
* Dynamical GRMHD: the same as the other relativistic solvers, except that the pressure $P$ (`IPR` or `press`) is output instead of internal energy density, and the temperature $T$ (`temperature`) is appended to primitive outputs. The electron fraction, if present, will be stored as the passive scalar `s_00`.

In all cases, MHD can optionally append the cell-centered magnetic fields in the Eulerian frame ($\mathbf{B}$ or $B^i$) to the primitive variables during output (`bcc1`, `bcc2`, and `bcc3`), though these are kept separate in the code. For dynamical GRMHD, these will be densitized by the volume form, i.e., it outputs $\tilde{B}^i = \sqrt{\gamma} B^i$.

Conserved variables:
* Newtonian hydro and MHD: the density $\rho$ (`IDN` or `dens`), the momentum density $\rho \mathbf{v}$ (`IM1`, `IM2`, and `IM3` or `mom1`, `mom2`, and `mom3`), the total energy density $E$ (`IEN` or `ener`), and the mass fractions for each passive scalar $\rho Y_x$ (`r_00`, `r_01`, etc.). The cell-centered magnetic fields can be appended if requested, as in the case of the primitive variables.
* Special relativistic hydro and MHD: the lab-frame density $D=\rho \Gamma$ (`IDN` or `dens`), the total momentum density $S_i$ (`IM1`, `IM2`, and `IM3` or `mom1`, `mom2`, and `mom3`), the lab-frame energy density without mass energy $\tau = E - \rho \Gamma$ (`IEN` or `ener`), and mass fractions for each passive scalar $DY_x$ (`r_00`, `r_01`, etc.).
* Stationary GR hydro and GRMHD: the coordinate-frame density $\rho u^0$ (`IDN` or `dens`), the total momentum density $T^t_i$ (`IM1`, `IM2`, and `IM3` or `mom1`, `mom2`, and `mom3`), the energy density without mass energy $\tau = T^t_t - \rho u^0$ (`IEN` or `ener`), and mass fractions for each passive scalar $\rho u^0 Y_x$ (`r_00`, `r_01`, etc.). Note that the stationary GR(M)HD only supports Minkowski and Cartesian Kerr-Schild spacetimes where $\sqrt{-g} = 1$, so the conserved variables are not explicitly weighted by the spacetime volume.
* Dynamical GRMHD: all variables are measured relative to the normal observer and densitized by the volume form $\sqrt{\gamma}$. Thus the conserved variables are the lab-frame density $\tilde{D} = \sqrt{\gamma} \rho W$ (`IDN` or `dens`), the momentum density $\tilde{S}_i = \sqrt{\gamma} \alpha T^t_i$ (`IM1`, `IM2`, and `IM3` or `mom1`, `mom2`, and `mom3`), the energy density without mass energy $\tilde{\tau} = \sqrt{\gamma}\alpha^2 T^{00} - \tilde{D}$ (`IEN` or `ener`), and the mass fractions for each advected scalar `\tilde{D} Y_x$ (`r_00`, `r_01`, etc.)

As for the primitive variables, MHD can optionally append the cell-centered magnetic fields in the Eulerian frame during output. For dynamical GRMHD, these will be again be densitized by the volume form.

Regarding the different definitions for stationary GRMHD and dynamical GRMHD, the densitized forms for all variables are equivalent except for the energy density; for the stationary GRMHD solver, the HARM-like convention $E = T^t_t$ is used, while dynamical GRMHD uses the 3+1 projection $E = n_\mu n_\nu T^{\mu\nu}$.

### I keep getting errors about `NoOp ConsToPrim` and `NoOp PrimToCons` in my custom pgen.
There are one of three possibilities:
1. You are calling the hydro versions of `PrimToCons` or `ConsToPrim` functions while using MHD. Check that your argument list contains the face-centered and cell-centered magnetic fields.
2. You are calling the MHD versions of `PrimToCons` or `ConsToPrim` functions while using hydro. Check that you are not trying to pass magnetic fields into these functions.
3. You are using `DynGRMHD` and calling the `PrimToCons` and `ConsToPrim` functions supplied by the `EOS` object inside `MHD`. Though the `MHD` object is available for `DynGRMHD`, these functions are explicitly disabled because they are designed for the stationary GRMHD solver, which uses a different formulation of the GRMHD equations. Try calling `pdyngr->ConToPrim`, `pyngr->ConToPrimBC`, or `pdyngr->ConToPrimInit` instead.