#!/bin/bash
# Database Test Runner Script
# Runs database tests with reporting and optionally starts required services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/reports/database"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
USE_DOCKER=false
TEST_CATEGORY=""
VERBOSE=false
COVERAGE=false
BENCHMARKS=false

# Help message
show_help() {
    echo "Database Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --docker        Start PostgreSQL in Docker before running tests"
    echo "  -c, --category CAT  Run specific test category:"
    echo "                        schema, data, crud, relationships,"
    echo "                        performance, concurrency, all (default)"
    echo "  -v, --verbose       Verbose output"
    echo "  --coverage          Run with coverage reporting"
    echo "  --benchmarks        Include benchmark tests"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d -c schema           # Run schema tests with Docker DB"
    echo "  $0 -c crud --coverage     # Run CRUD tests with coverage"
    echo "  $0 --benchmarks           # Run all tests including benchmarks"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker)
            USE_DOCKER=true
            shift
            ;;
        -c|--category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --benchmarks)
            BENCHMARKS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Start Docker PostgreSQL if requested
if [ "$USE_DOCKER" = true ]; then
    echo -e "${BLUE}Starting PostgreSQL container...${NC}"

    docker run -d \
        --name nerdlearn-test-db \
        -e POSTGRES_USER=test \
        -e POSTGRES_PASSWORD=test \
        -e POSTGRES_DB=nerdlearn_test \
        -p 5433:5432 \
        postgres:15 \
        || true  # Ignore if already running

    # Wait for PostgreSQL to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    for i in {1..30}; do
        if docker exec nerdlearn-test-db pg_isready -U test > /dev/null 2>&1; then
            echo -e "${GREEN}PostgreSQL is ready!${NC}"
            break
        fi
        sleep 1
    done

    export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5433/nerdlearn_test"
    export TEST_SYNC_DATABASE_URL="postgresql://test:test@localhost:5433/nerdlearn_test"
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add test path based on category
case $TEST_CATEGORY in
    schema)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_schema_integrity.py"
        ;;
    data)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_data_integrity.py"
        ;;
    crud)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_crud_operations.py"
        ;;
    relationships)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_relationships.py"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_performance.py"
        ;;
    concurrency)
        PYTEST_CMD="$PYTEST_CMD tests/database/test_concurrency.py"
        ;;
    all|"")
        PYTEST_CMD="$PYTEST_CMD tests/database/"
        ;;
    *)
        echo -e "${RED}Unknown category: $TEST_CATEGORY${NC}"
        exit 1
        ;;
esac

# Add options
PYTEST_CMD="$PYTEST_CMD --tb=short"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=app.models --cov-report=html:$REPORTS_DIR/coverage"
fi

if [ "$BENCHMARKS" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not benchmark'"
fi

# Add report generation
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTEST_CMD="$PYTEST_CMD --html=$REPORTS_DIR/report_$TIMESTAMP.html --self-contained-html"
PYTEST_CMD="$PYTEST_CMD --json-report --json-report-file=$REPORTS_DIR/report_$TIMESTAMP.json"

# Run tests
echo -e "${BLUE}Running database tests...${NC}"
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

cd "$PROJECT_ROOT"

# Execute
if eval "$PYTEST_CMD"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    EXIT_CODE=0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    EXIT_CODE=1
fi

echo ""
echo -e "${BLUE}Reports saved to: $REPORTS_DIR${NC}"
echo "  - HTML: report_$TIMESTAMP.html"
echo "  - JSON: report_$TIMESTAMP.json"

# Cleanup Docker if we started it
if [ "$USE_DOCKER" = true ]; then
    read -p "Stop and remove test database container? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker stop nerdlearn-test-db
        docker rm nerdlearn-test-db
        echo -e "${GREEN}Container removed${NC}"
    fi
fi

exit $EXIT_CODE
