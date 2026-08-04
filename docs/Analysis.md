As discussed in the [Outputs](https://github.com/IAS-Astrophysics/athenak/wiki/Outputs) section, AthenaK supports multiple output formats. Depending on the format, there may be multiple ways to analyze the data.

# Repository analysis tools
The AthenaK repository contains a number of simple scripts for reading data, all inside the `vis/python` directory:

## `plot_slice.py`
This script can be used to read a `.bin` output, slice it (for 3D data), and plot an output variable. Here is an example:
```bash
python3 plot_slice.py -d y \ # Slice along the y axis
                      -l 0.0 \ # Slice location
                      --cmap viridis \ # Matplotlib colormap
                      --norm log --vmin 1e-10 --vmax 1.0 \ # Matplotlib colorbar normalization
                      --horizon \ # Plot horizon location
                      gr_torus.mhd_w_bcc.00010.bin \ # positional argument: input file
                      dens \ # positional argument: variable to plot
                      show # positional argument: show or output filename
```
This produces an output like the following:

<img width="684" height="512" alt="torus" src="https://github.com/user-attachments/assets/00d40112-cc35-4547-9d52-6963bc86e61d" />

Additional arguments can be seen by using the `-h` option or reading the instructions at the beginning of the script.

## `plot_tab.py`
This script can be used for quickly looking at tabulated data, including animations. Consider the following example:
```bash
python3 plot_tab.py -i tab/Sod.mhd_w_bcc.00000.tab \ # input file (only stem is used if -n is specified)
                    -v dens \ # variable to plot
                    -n 51 # number of frames to read
```
This will generate an animation starting using `Sod.mhd_w_bcc.[00000..00050].tab` as inputs. The last frame will look something like the following:
<img width="640" height="443" alt="st_rho" src="https://github.com/user-attachments/assets/9ac8675c-9d5d-40da-bb7c-7090bad0d427" />

## `plot_hst.py`
TODO

# External analysis tools
AthenaK `.vtk` outputs can be read by any tool that reads legacy VTK files. `.hst` and `.tab` files are written as whitespace-separated ASCII files, so they can easily be read using numpy or similar tools. The `.bin` format is a bespoke format, so there are not currently plugins for major visualization suites. However, there are still some ways to read these files, either by converting them to a different format or using custom analysis tools.

## Converting to Athena++ XDMF files
The `make_athdf.py` script can be used to convert `.bin` files into Athena++ `.athdf` files, which can be read into visualization software like VisIt or Paraview via their XDMF readers. In theory this also makes it possible to use many Athena++ tools (see [the Athena++ wiki](https://github.com/PrincetonUniversity/athena/wiki/Analysis-Tools) for more information), though be aware that AthenaK does have some small differences in naming conventions that can sometimes lead to incompatibilities.

Example:
```bash
python3 make_athdf.py bin/tov.mhd_w_bcc
```
This will glob all `*.bin` files with the stem `tov.mhd_w_bcc`, such as `tov.mhd_w_bcc.00000.bin`, `tov.mhd_w_bcc.00001.bin`, etc., and write out the corresponding `.athdf` and `.athdf.xmdf` files.

## AthenaK-compatible libraries and tools
Some users have developed and shared their own analysis tools:
* [plot-tools](https://github.com/jfields7/plot-tools): a collection of scripts maintained by Jacob Fields. Many of them are targeted toward numerical relativity applications, such as a utility for compact object trackers or the ejecta analysis scripts, though the basic file reading and plotting scripts are compatible with any dataset.
* [wongutils](https://github.com/gnwong/wongutils): a suite of tools developed by George Wong. These tools are excellent for 3D GRMHD data in Kerr-Schild spacetimes, such as accretion disks, and they include robust utilities for coordinate transformations, swapping between observer-frame and fluid-frame quantities, etc.

If you are interested in sharing your own analysis tools, please let us know!