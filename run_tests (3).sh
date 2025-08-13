
#!/bin/bash

# Crypto Trading Bot Test Runner Script
# Usage: ./run_tests.sh [coverage|quick|integration]

set -e

echo "🚀 Crypto Trading Bot Test Suite"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Function to run basic tests
run_basic_tests() {
    echo "🏃 Running basic test suite..."
    python3 test_runner.py
}

# Function to run tests with coverage
run_coverage_tests() {
    echo "🏃 Running tests with coverage..."
    python3 test_runner.py --coverage --html-report
}

# Function to run quick tests (unit tests only)
run_quick_tests() {
    echo "🏃 Running quick tests (unit tests only)..."
    python3 -m unittest discover -s . -p "test_*.py" -k "not integration" -v
}

# Function to run integration tests only
run_integration_tests() {
    echo "🏃 Running integration tests..."
    python3 -m unittest discover -s . -p "test_*.py" -k "integration" -v
}

# Function to check test environment
check_environment() {
    echo "🔍 Checking test environment..."
    
    # Check for required files
    required_files=("config.json" "test_trading_bot.py")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "❌ Required file missing: $file"
            exit 1
        fi
    done
    
    # Check Python packages
    echo "📦 Checking Python packages..."
    python3 -c "import pandas, unittest" 2>/dev/null || {
        echo "❌ Required Python packages not found"
        echo "Please install: pandas"
        exit 1
    }
    
    echo "✅ Environment check passed"
}

# Main script logic
case "${1:-basic}" in
    "coverage")
        check_environment
        run_coverage_tests
        ;;
    "quick")
        check_environment
        run_quick_tests
        ;;
    "integration")
        check_environment
        run_integration_tests
        ;;
    "basic"|"")
        check_environment
        run_basic_tests
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [coverage|quick|integration|basic]"
        echo ""
        echo "Options:"
        echo "  basic       Run all tests (default)"
        echo "  coverage    Run tests with coverage report"
        echo "  quick       Run only unit tests (fast)"
        echo "  integration Run only integration tests"
        echo "  help        Show this help message"
        exit 0
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "✨ Test execution completed!"
