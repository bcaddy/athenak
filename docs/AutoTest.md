# Test suite

## Purpose
An extensive set of automatic tests are provided in the `/tst` directory. They check that basic features (such as AMR) and all the physics modules implemented in the code are functioning correctly by running, e.g. convergence tests of linear waves, shock tube tests, or comparisons to specific known analytic solutions (e.g. Bondi flows).

These tests are useful for developers to check that any changes they make do not negatively impact the rest of the code.  All tests are run for every merge request as part of the CI infrastructure.  Users are encouraged to run the tests manually before making a merge request, and any new substantive module added to the code should include a new automatic test for that feature.  New tests must be kept simple: to keep the full test suite manageable the total run time for all tests on any specific hardware configuration should be less than about 10 minutes.

The tests are managed by the `/tst/run_test_suite.py` script.  It is designed to execute specific subsets of tests based on the hardware configuration (CPU, MPI-enabled CPU, or GPU). It uses `pytest` to run tests and provides flexibility for developers to target specific test cases.

---

## Usage

### Command-line Arguments
- `--style`: Runs style checks.
- `--cpu`: Runs tests for CPU-only configurations. 
- `--mpicpu`: Runs tests for MPI-enabled CPU configurations.
- `--gpu`: Runs tests for GPU configurations.
- `--test`: Runs a specific set of test(s) by name. This accepts a space seperated list of test names and/or directories and will run all the tests specified and all the tests found in the directories recursively. The test name(s) must include one of the following suffixes:
  - `_cpu` for CPU tests.
  - `_mpicpu` for MPI CPU tests.
  - `_gpu` for GPU tests.

### Example Commands
- Run style checks:
  ```bash
  python run_test_suite.py --style

- Run CPU tests:
  ```bash
  python run_test_suite.py --cpu <cpu_flags>

- Run MPI+CPU tests:
  ```bash
  python run_test_suite.py --mpicpu <mpicpu_flags>

- Run GPU tests:
  ```bash
  python run_test_suite.py --gpu <gpu_flags>

- Run multiple test types:
  ```bash
  python run_test_suite.py --gpu <gpu_flags> --cpu <cpu_flags>

- Run a specific test:
  ```bash
  python run_test_suite.py --test test_suite/subdirectory/test_example_cpu.py

- Run all tests in a directory:
  ```bash
  python run_test_suite.py --test test_suite/subdirectory

- Run CPU and GPU tests in a directory:
  ```bash
  python run_test_suite.py --test test_suite/subdirectory --gpu <gpu_flags> --cpu <cpu_flags>

- Run a subset of CPU tests:
  ```bash
  python run_test_suite.py --test test_suite/subdirectory/test_example_cpu.py test_suit/subdirectory_2 --cpu <cpu_flags>


## Behavior

The `run_test_suite.py` script performs the following actions based on the provided arguments:

1. **No Target Device Specified**:
   - If no arguments are provided, the script prints an error message and displays the help menu using `parser.format_help()`.
   - The script then exits with `sys.exit(1)`.

2. **Style Checks**:
   - If the `--style` argument is provided, the script runs style checks using `pytest` on the `test_suite/style` directory.

3. **CPU Tests**:
   - If the `--cpu` argument is provided, the script:
     - Calls `testutils.clean_make()` to clean and build the project with CPU-specific flags.
     - Runs all tests in the `test_suite` directory that contain `_cpu` in their name using `pytest`.

4. **MPI CPU Tests**:
   - If the `--mpicpu` argument is provided, the script:
     - Calls `testutils.clean_make()` to clean and build the project with MPI-enabled CPU flags (`Athena_ENABLE_MPI=ON`).
     - Runs all tests in the `test_suite` directory that contain `_mpicpu` in their name using `pytest`.

5. **GPU Tests**:
   - If the `--gpu` argument is provided, the script:
     - Calls `testutils.clean_make()` to clean and build the project with GPU-specific flags (`Kokkos_ENABLE_CUDA=On`).
     - Runs all tests in the `test_suite` directory that contain `_gpu` in their name using `pytest`.

6. **Subset of Tests Execution**:
   - If the `--test` argument is provided, the script:
     - For each file name provided it validates the test name(s) to ensure it contains one of the following suffixes: `_cpu`, `_mpicpu`, or `_gpu`.
     - For each directory name provided it finds all tests within the directory and verifies they have the proper suffices.
     - If the test name is invalid, the script exits with an error message.
     - If valid, the script runs the specified test(s) using `pytest`. If multiple tests or directories are specified then it will run all tests of all types. If any combination of `--cpu`, `--gpu`, or `--mpicpu` are passed as well then it will run only those tests.
     - Note the test name must include the full path to the file or directory, usually `test_suite/subdirectory/testname.py`

7. **Log File Management**:
   - At the beginning of the script, the log file (`test_log.txt`) is removed if it exists to ensure a fresh log file is created during the script's execution.


## Adding Tests to `test_suite`

### Test Naming Convention
- To ensure your test is executed by the script, name your test files appropriately:
  - Use `_cpu` in the filename for CPU tests.
  - Use `_mpicpu` in the filename for MPI CPU tests.
  - Use `_gpu` in the filename for GPU tests.

#### Example:
- `test_example_cpu.py` for CPU tests.
- `test_example_mpicpu.py` for MPI CPU tests.
- `test_example_gpu.py` for GPU tests.

### Test File Location
- Place your test files in a new subdirectory inside the `/tst/test_suite` directory.  The new directory also must contain an `__init__.py` file. Any script that follows the above naming convention and is located in a subdirectory with an `__init__.py` will be run automaically by the `run_test_suite.py` script. 

### Writing Tests

It is best to use one of the existing test scripts as a template for writing new tests. New tests must follow `pytest` conventions, for example the name of the function that runs the test must begin with `test_`.  Tests that compare to known analytic solutions, or check convergence of solutions, are preferred to regression tests that compare to pre-computed solutions.