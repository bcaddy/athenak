# Adaptive Mesh Refinement
In addition to static refinement, AthenaK also supports adaptive mesh refinement (AMR), or updating the mesh at fixed intervals to modify where refinement regions are located according to some criteria.

As in the case of SMR, AMR is controlled through the `mesh_refinement` block:
```
<mesh_refinement>
refinement = adaptive    # enable adaptive mesh refinement
num_levels = 2           # the number of refinement levels in the mesh, including the base level
refinement_interval = 1  # check refinement criteria and load balance every n cycles
max_nmb_per_rank = 512   # the maximum number of mesh blocks per MPI rank
```

You then also set one or more refinement criteria:
```
<amr_criterion0>
# Refine on slopes of the rest-mass density > 0.5
method = slope
variable = mhd_w_d
value_max = 0.5

<amr_criterion1>
# Refine where the lab-frame density is < 1e-6
method = min_max
variable = mhd_u_d
value_min = 1e-6

<amr_criterion2>
# Refine a cube with half-width 5 around the origin
method = location
location_x1 = 0.0
location_x2 = 0.0
location_x3 = 0.0
location_rad = 5.0
```

The `<refined_region#>` blocks used for SMR can be used to seed the mesh with an initial refinement structure. However, the AMR criteria will not preserve these structures on the next refinement cycle. If parts of your mesh need to remain refined no matter what, use the `location` criterion.

## Built-in refinement criteria
The `method` parameter can take the following arguments:
* `min_max`: Refine based on the minimum or maximum of `variable`.
* `slope`: Refine based on the slope of `variable`.
* `second_deriv`: Refine based on the second derivative of `variable.
* `location`: Refine a cube centered on the specified location with a given side half-length. This is useful if part of the mesh should always be refined independent of the refinement criteria.
* `user`: Refine based on a user-specified refinement criterion in the problem generator.

Be aware that presently the only supported refinement variables for `min_max`, `slope`, or `second_deriv` are `hyd_w_d`, `hyd_u_d`, `mhd_w_d`, `mhd_u_d`, and `rad_coord_e` (see [issue #658](https://github.com/IAS-Astrophysics/athenak/issues/658)).

## Custom refinement criteria
If `method=user`, the user must supply an appropriate custom refinement function inside the problem generator. We will use the Z4c linear wave test problem (`src/pgen/tests/z4c_linear_wave.cpp`) as an example.

First, the refinement function must be a `void` function that takes a pointer to the `MeshBlockPack` as a parameter, e.g.,:
```c++
void RefinementCondition(MeshBlockPack* pmbp);
```

The `ProblemGenerator` class has a member variable, `user_ref_func`, which is a function pointer to the refinement function. The problem generator therefore must assign this function:
```c++
void ProblemGenerator::Z4cLinearWave(ParameterInput* pin, const bool restart) {
  ...
  user_ref_func = RefinementCondition;
  ...
}
```

The refinement function then must loop over all mesh blocks and set their refinement flag (`refine_flag.d_view(m)`) to refine (`1`), coarsen (`-1`), or do nothing (`0`). For example, the Z4c linear wave criterion marks a block for refinement if the metric component $\gamma_{xy}$ is positive at any point, otherwise it marks it to be coarsened:
```c++
void RefinementCondition(MeshBlockPack* pmbp) {                                           
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;                                      
  int I_Z4C_GXY  = pmbp->pz4c->I_Z4C_GXY;                                                 
  int nmb           = pmbp->nmb_thispack;                                                 
  auto &indcs       = pmbp->pmesh->mb_indcs;                                              
  int &is = indcs.is, nx1 = indcs.nx1;                                                    
  int &js = indcs.js, nx2 = indcs.nx2;                                                    
  int &ks = indcs.ks, nx3 = indcs.nx3;                                                    
  const int nkji = nx3 * nx2 * nx1;                                                       
  const int nji  = nx2 * nx1;                                                             
  int mbs           = pmbp->pmesh->gids_eachrank[global_variable::my_rank];               
  auto &u0       = pmbp->pz4c->u0;                                                        
                                                                                          
  par_for_outer("Z4c_AMR::GXYMAX", DevExeSpace(), 0, 0, 0, (nmb - 1),                     
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {                                      
    Real team_dmax;                                                                       
    Kokkos::parallel_reduce(                                                              
      Kokkos::TeamThreadRange(tmember, nkji),                                             
      [=](const int idx, Real &dmax) {                                                    
        int k = (idx) / nji;                                                              
        int j = (idx - k * nji) / nx1;                                                    
        int i = (idx - k * nji - j * nx1) + is;                                           
        j += js;                                                                          
        k += ks;                                                                          
        dmax = fmax(u0(m, I_Z4C_GXY, k, j, i), dmax);                                     
      },                                                                                  
      Kokkos::Max<Real>(team_dmax));                                                      
                                                                                          
    if (team_dmax > 0) {                                                                  
      refine_flag.d_view(m + mbs) = 1;                                                    
    } else {                                                                              
      refine_flag.d_view(m + mbs) = -1;                                                   
    }                                                                                     
  });                                                                                     
                                                                                          
  // sync host and device                                                                 
  refine_flag.template modify<DevExeSpace>();                                             
  refine_flag.template sync<HostMemSpace>();                                              
}
```

## Common problems
TODO