"""
Unit tests for the line scans in utils/scan/line_scan.hpp.
"""

# Modules
import pytest
import test_suite.unit_tests.test_line_scan_gpu as line_scan


@pytest.mark.parametrize("case", line_scan.cases.keys())
def test_line_scan_cpu(case):
    """CPU test for the line scans."""
    line_scan.run_line_scan(case)
