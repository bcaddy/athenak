Static mesh refinement (SMR), sometimes also called fixed mesh refinement (FMR), can be enabled by adding a `<mesh_refinement>` block to the parameter file with `refinement = static` and defining one or more `<refined_region#>` blocks. Consider the following example:

```
<mesh_refinement>
refinement = static

<refined_region1>
level = 1
x1min = -32.0
x1max =  32.0
x2min = -32.0
x2max =  32.0
x3min = -32.0
x3max =  32.0
```

This indicates that the Cartesian box spanning $[-32,32]^3$ will be refined once relative to the base level, i.e., the mesh spacing will be half of the base level. If we set instead `level = 3`, this box would be refined three times, or have have 1/8 the spacing of the base level. We can add additional `<refined_region#>` blocks to add refinement in other places of the grid. The `level` parameter will always be relative to the base level, even if your additional refinement regions are nested inside others.

Some things to remember when using mesh refinement:
* Refinement in AthenaK happens per mesh block. If the refinement boundary passes through part of a mesh block, the entire mesh block will be refined. This means that the box defined by `<refined_region#>` is the *minimum* region that will be refined. It is imperative when using SMR to choose mesh block sizes and refinement regions carefully in order to avoid unintentionally over-refining large regions of the mesh.
* Due to finite-precision effects when calculating mesh block boundaries and reading in refinement boundaries, the actual applied refinement box can sometimes differ very slightly from what is in the parameter file. If you find your mesh is over-refining relative to what you expect, consider slightly shrinking the refinement boundary.
* AthenaK enforces a strict 2:1 refinement condition; i.e., the refinement in a single mesh block cannot differ from any of its neighbors by more than one level. For example, if a single refinement region is specified at the center of your computational domain with three levels of refinement, you will actually have three nested refinement regions, with the innermost covering the desired region and the outer two dictated by what obeys the 2:1 balancing condition.

You can check the generated mesh by running AthenaK with the `-m` option:
```
./athena -m -i parfile.athinput
```

This generates a file called `mesh_structure.dat`, which can be plotted using the `vis/python/plot_mesh.py` script located in the repository.