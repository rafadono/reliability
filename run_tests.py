"""
Test runner script with coverage reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run fast tests only
    python run_tests.py --cov        # Run with coverage
"""

import subprocess
import sys


def run_tests(args=None):
    """Run pytest with optional arguments."""
    cmd = ["pytest", "tests/", "-v"]

    if args:
        if "--fast" in args:
            cmd.append("-m not slow")
        if "--cov" in args:
            cmd.extend(["--cov=src", "--cov-report=html"])

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    returncode = run_tests(sys.argv[1:])
    sys.exit(returncode)
