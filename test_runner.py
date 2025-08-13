
#!/usr/bin/env python3
"""
Test Runner Script for Crypto Trading Bot

This script runs the comprehensive test suite and generates detailed reports.
Usage: python test_runner.py [--coverage] [--html-report]
"""

import unittest
import sys
import os
import argparse
import time
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False
    print("Warning: coverage.py not installed. Install with: pip install coverage")


class ColoredTextTestResult(unittest.TextTestResult):
    """Enhanced test result with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write(f"✅ {self.getDescription(test)} ... ")
            self.stream.writeln("PASS")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f"❌ {self.getDescription(test)} ... ")
            self.stream.writeln("ERROR")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f"❌ {self.getDescription(test)} ... ")
            self.stream.writeln("FAIL")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f"⏭️  {self.getDescription(test)} ... ")
            self.stream.writeln(f"SKIP ({reason})")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    
    resultclass = ColoredTextTestResult
    
    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)


def discover_tests():
    """Discover and return all test modules."""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover tests in current directory
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    return suite


def run_tests_with_coverage(test_suite, html_report=False):
    """Run tests with coverage measurement."""
    if not HAS_COVERAGE:
        print("Coverage measurement requires 'coverage' package. Running tests without coverage...")
        return run_tests_without_coverage(test_suite)
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Generate coverage report
    print(f"\n{'='*60}")
    print("COVERAGE REPORT")
    print(f"{'='*60}")
    
    # Console report
    cov.report()
    
    # HTML report if requested
    if html_report:
        try:
            cov.html_report(directory='htmlcov')
            print(f"\n📊 HTML coverage report generated in 'htmlcov' directory")
            print(f"   Open 'htmlcov/index.html' in your browser to view")
        except Exception as e:
            print(f"Failed to generate HTML report: {e}")
    
    return result


def run_tests_without_coverage(test_suite):
    """Run tests without coverage measurement."""
    runner = ColoredTextTestRunner(verbosity=2)
    return runner.run(test_suite)


def generate_test_report(result, start_time, end_time):
    """Generate a detailed test report."""
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("DETAILED TEST REPORT")
    print(f"{'='*60}")
    print(f"Execution Time: {duration:.2f} seconds")
    print(f"Tests Run: {result.testsRun}")
    print(f"✅ Successes: {getattr(result, 'success_count', result.testsRun - len(result.failures) - len(result.errors))}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🔥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    # Detailed failure/error reports
    if result.failures:
        print(f"\n{'='*30} FAILURES {'='*30}")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print("-" * 50)
            print(traceback)
    
    if result.errors:
        print(f"\n{'='*30} ERRORS {'='*31}")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print("-" * 50)
            print(traceback)
    
    return result.wasSuccessful()


def check_test_dependencies():
    """Check if all required dependencies for testing are available."""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import unittest.mock
    except ImportError:
        missing_deps.append('unittest.mock (should be built-in)')
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests.")
        return False
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run Crypto Trading Bot Test Suite')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage measurement')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--pattern', default='test_*.py', help='Test file pattern (default: test_*.py)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 Crypto Trading Bot Test Suite")
    print("=" * 40)
    
    # Check dependencies
    if not check_test_dependencies():
        sys.exit(1)
    
    # Discover tests
    print("🔍 Discovering tests...")
    test_suite = discover_tests()
    test_count = test_suite.countTestCases()
    
    if test_count == 0:
        print("❌ No tests found!")
        sys.exit(1)
    
    print(f"📋 Found {test_count} tests")
    
    # Run tests
    print(f"\n🏃 Running tests...")
    start_time = time.time()
    
    if args.coverage:
        result = run_tests_with_coverage(test_suite, args.html_report)
    else:
        result = run_tests_without_coverage(test_suite)
    
    end_time = time.time()
    
    # Generate report
    success = generate_test_report(result, start_time, end_time)
    
    # Exit with appropriate code
    if success:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
