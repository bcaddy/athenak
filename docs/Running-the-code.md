After building the code, an executable named `athena` will be created in the `/build/src` subdirectory.  To run it, an input file must also be specified on the command line, e.g.

    $ athena -i example.athinput

Examples of input files can be found in the `/inputs` subdirectory.  Input files that work with the built-in test problems can be found in `/inputs/tests`.

When the code runs, all output files are created in the current working directory by default. The code will output simulation progress and some diagnosis messages to stdout.

### Command-Line Options
A variety of command-line options have been implemented in `AthenaK`. The list of the options as well as the code configuration is given by the `-h` switch:

    $ athena -h
    Athena v0.1
    Usage: athena [options] [block/par=value ...]
    Options:
      -i <file>       specify input file [athinput]
      -r <file>       restart with this file
      -d <directory>  specify run dir [current dir]
      -n              parse input file and quit
      -c              show configuration and quit
      -m              output mesh structure and quit
      -t hh:mm:ss     wall time limit for final output
      -h              this help

### Overriding Parameters in Input File
Some parameter values specified in the input file can be overridden by specifying new values on the command-line using the following format: `<parameter_block>/<parameter>=<value>`. For example, if you want to extend the simulation time limit when restarting,

    $ athena -r example.00010.rst time/tlim=100

Some parameters, such as the sizes of the Mesh and MeshBlock, cannot be changed in this way.

### Resuming Cluster Jobs from Latest Restart File

Large jobs on clusters often require submitting requests to continue from a checkpoint. The following snippet can help in making a submission script (e.g. for Slurm) that does this automatically. This uses `$input_dir/$name.athinput` but overrides the `job/basename` field to be `$name`. Only the `# Parameters` section needs to be modified from job to job, and generally at most the `tlim` entry needs to be modified with each restart. Additional command-line overrides can be specified with additional lines to the `arguments` variable.

```
# Parameters
name=<name of job and input file>
bin_name=<name of executable>
input_dir=<path to directory containing input file>
data_dir=<path to directory for writing outputs>
arguments="\
  time/tlim=<tlim> \
"

# Check for restart file
test_file=$(find $data_dir/rst -maxdepth 1 -name "$name.*.rst" -print -quit)
if [ -n "$test_file" ]; then
  restart_files=$(ls -t $data_dir/rst/$name.*.rst)
  restart_file=(${restart_files[0]})
  restart_line="-r $restart_file"
  printf "\nrestarting from $restart_file\n\n"
else
  restart_line="-i $input_dir/$name.athinput job/basename=$name"
  printf "\nstarting from beginning\n\n"
fi

# Run code
<srun or equivalent> <options> $bin_name -d $data_dir $restart_line $arguments
```