"""
Unit tests for the block local line scans in utils/scan/line_scan.hpp.
"""

# Modules
import pytest
import test_suite.unit_tests.test_line_scan_gpu as line_scan


@pytest.mark.parametrize("shape", line_scan.shapes.keys())
def test_line_scan_cpu(shape):
    """CPU test for the block local line scans."""
    line_scan.run_line_scan(shape)
