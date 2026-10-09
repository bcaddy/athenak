"""
Unit tests for the line scans in utils/scan/line_scan.hpp, on several numbers of MPI
ranks.
"""

# Modules
import pytest
import test_suite.unit_tests.test_line_scan_gpu as line_scan


@pytest.mark.parametrize("nranks", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("case", line_scan.cases.keys())
def test_line_scan_mpicpu(case, nranks):
    """MPI-CPU test for the line scans."""
    line_scan.run_line_scan(case, nranks=nranks)
