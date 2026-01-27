# Database Testing Guide for VS Code

This guide explains how to run the NerdLearn database tests in VS Code, including setup, configuration, and interpreting results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [VS Code Setup](#vs-code-setup)
3. [Running Tests](#running-tests)
4. [Using the Test Runner Script](#using-the-test-runner-script)
5. [VS Code Test Explorer](#vs-code-test-explorer)
6. [Running with Docker](#running-with-docker)
7. [Understanding Test Reports](#understanding-test-reports)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Python Environment

Ensure you have Python 3.11+ installed:

```bash
python --version
# Should show Python 3.11.x or higher
```

### 2. Install Python Dependencies

From the project root, install the required packages:

```bash
# Install API dependencies
pip install -r apps/api/requirements.txt

# Install test dependencies
pip install -r apps/api/requirements-test.txt

# Install additional test packages
pip install pytest-html pytest-json-report pytest-asyncio aiosqlite
```

If `requirements-test.txt` doesn't exist, install these manually:

```bash
pip install pytest pytest-asyncio pytest-cov pytest-html pytest-json-report aiosqlite
```

### 3. Database Options

You have two options for running tests:

#### Option A: In-Memory SQLite (Default - No Setup Required)
Tests will automatically use SQLite in-memory database. This is the simplest option.

#### Option B: PostgreSQL (Recommended for Full Testing)
For tests that require PostgreSQL-specific features:

```bash
# Using Docker
docker run -d \
  --name nerdlearn-test-db \
  -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=nerdlearn_test \
  -p 5433:5432 \
  postgres:15

# Set environment variable
export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5433/nerdlearn_test"
```

---

## VS Code Setup

### 1. Install Required Extensions

Open VS Code and install these extensions:

| Extension | ID | Purpose |
|-----------|-----|---------|
| Python | `ms-python.python` | Python language support |
| Pylance | `ms-python.vscode-pylance` | Python IntelliSense |
| Python Test Explorer | `littlefoxteam.vscode-python-test-adapter` | Visual test runner |

To install from command line:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
```

### 2. Configure Python Interpreter

1. Open Command Palette: `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: `Python: Select Interpreter`
3. Choose your Python 3.11+ interpreter

### 3. Configure pytest in VS Code

Create or update `.vscode/settings.json`:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests/database",
        "-v",
        "--tb=short"
    ],
    "python.testing.cwd": "${workspaceFolder}",
    "python.envFile": "${workspaceFolder}/.env.test"
}
```

### 4. Create Test Environment File (Optional)

Create `.env.test` in the project root:

```bash
# For SQLite (default)
TEST_DATABASE_URL=sqlite+aiosqlite:///:memory:
TEST_SYNC_DATABASE_URL=sqlite:///:memory:

# For PostgreSQL (uncomment if using Docker)
# TEST_DATABASE_URL=postgresql+asyncpg://test:test@localhost:5433/nerdlearn_test
# TEST_SYNC_DATABASE_URL=postgresql://test:test@localhost:5433/nerdlearn_test
```

---

## Running Tests

### Method 1: VS Code Terminal

Open the integrated terminal (`Ctrl+`` ` or `View > Terminal`) and run:

```bash
# Run all database tests
pytest tests/database/ -v

# Run specific test file
pytest tests/database/test_schema_integrity.py -v

# Run specific test class
pytest tests/database/test_crud_operations.py::TestUserCRUD -v

# Run specific test
pytest tests/database/test_crud_operations.py::TestUserCRUD::test_create_user -v

# Run with coverage
pytest tests/database/ -v --cov=apps/api/app/models --cov-report=html

# Run only fast tests (exclude benchmarks)
pytest tests/database/ -v -m "not benchmark"

# Run with HTML report
pytest tests/database/ -v --html=reports/database/report.html --self-contained-html
```

### Method 2: VS Code Test Explorer UI

1. Click the **Testing** icon in the Activity Bar (flask icon on left sidebar)
2. Click **Refresh** to discover tests
3. You'll see a tree of all database tests organized by file and class
4. Click the **Play** button next to any test, class, or file to run it
5. Green checkmarks = passed, red X = failed

![Test Explorer](https://code.visualstudio.com/assets/docs/python/testing/test-explorer.png)

### Method 3: Run Tests from Editor

When viewing a test file:
1. You'll see **Run Test | Debug Test** links above each test function
2. Click **Run Test** to execute that specific test
3. Results appear in the **Python Test Log** output panel

---

## Using the Test Runner Script

We've provided a convenient shell script for running tests:

### Make Script Executable (First Time Only)

```bash
chmod +x scripts/run_db_tests.sh
```

### Basic Usage

```bash
# Run all tests with SQLite
./scripts/run_db_tests.sh

# Run with Docker PostgreSQL
./scripts/run_db_tests.sh -d

# Run specific category
./scripts/run_db_tests.sh -c schema
./scripts/run_db_tests.sh -c data
./scripts/run_db_tests.sh -c crud
./scripts/run_db_tests.sh -c relationships
./scripts/run_db_tests.sh -c performance
./scripts/run_db_tests.sh -c concurrency

# Verbose output
./scripts/run_db_tests.sh -v

# With coverage
./scripts/run_db_tests.sh --coverage

# Include benchmark tests
./scripts/run_db_tests.sh --benchmarks

# Combine options
./scripts/run_db_tests.sh -d -c crud -v --coverage
```

### Script Options Reference

| Option | Description |
|--------|-------------|
| `-d, --docker` | Start PostgreSQL in Docker before running tests |
| `-c, --category` | Run specific test category (schema, data, crud, relationships, performance, concurrency, all) |
| `-v, --verbose` | Verbose output |
| `--coverage` | Generate coverage report |
| `--benchmarks` | Include performance benchmark tests |
| `-h, --help` | Show help message |

---

## VS Code Test Explorer

### Configuring Test Discovery

If tests aren't appearing in Test Explorer, check:

1. **Python extension is activated**: Look for Python version in status bar
2. **pytest is installed**: `pip install pytest pytest-asyncio`
3. **Correct workspace**: Ensure you opened the NerdLearn folder
4. **Refresh tests**: Click refresh button in Test Explorer

### Running Tests by Category

You can filter tests in Test Explorer:
- Type in the search box to filter by name
- Use markers like `@pytest.mark.requires_db` to group tests

### Debugging Tests

1. Set breakpoints by clicking left of line numbers
2. Right-click a test in Test Explorer
3. Select **Debug Test**
4. Use Debug toolbar to step through code

---

## Running with Docker

### Start PostgreSQL Container

```bash
# Start container
docker run -d \
  --name nerdlearn-test-db \
  -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=nerdlearn_test \
  -p 5433:5432 \
  postgres:15

# Verify it's running
docker ps

# Check logs if needed
docker logs nerdlearn-test-db
```

### Set Environment Variables

**Option 1: Terminal**
```bash
export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5433/nerdlearn_test"
export TEST_SYNC_DATABASE_URL="postgresql://test:test@localhost:5433/nerdlearn_test"
pytest tests/database/ -v
```

**Option 2: VS Code launch.json**

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/database/", "-v"],
            "env": {
                "TEST_DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5433/nerdlearn_test",
                "TEST_SYNC_DATABASE_URL": "postgresql://test:test@localhost:5433/nerdlearn_test"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

### Stop Container When Done

```bash
docker stop nerdlearn-test-db
docker rm nerdlearn-test-db
```

---

## Understanding Test Reports

### HTML Reports

After running tests with `--html` flag, open the report:

```bash
# Generate report
pytest tests/database/ -v --html=reports/database/report.html --self-contained-html

# Open in browser (Linux)
xdg-open reports/database/report.html

# Open in browser (Mac)
open reports/database/report.html

# Open in browser (Windows)
start reports/database/report.html
```

The HTML report shows:
- **Summary**: Total, passed, failed, skipped counts
- **Environment**: Python version, platform, plugins
- **Results Table**: Each test with status, duration, and error details

### JSON Reports

For programmatic access:

```bash
pytest tests/database/ -v --json-report --json-report-file=reports/database/report.json
```

### Coverage Reports

```bash
# Generate coverage
pytest tests/database/ --cov=apps/api/app/models --cov-report=html:reports/coverage

# Open coverage report
open reports/coverage/index.html
```

---

## Troubleshooting

### Common Issues

#### 1. "No tests discovered"

**Solution:**
```bash
# Verify pytest can find tests
pytest tests/database/ --collect-only

# Check for import errors
python -c "import tests.database.test_schema_integrity"
```

#### 2. "ModuleNotFoundError: No module named 'app'"

**Solution:** Add the project to Python path:
```bash
# In terminal
export PYTHONPATH="${PYTHONPATH}:${PWD}/apps/api"

# Or in .vscode/settings.json
{
    "python.analysis.extraPaths": ["./apps/api"]
}
```

#### 3. "Database connection failed"

**Solution:**
```bash
# Check if using SQLite (should work without setup)
echo $TEST_DATABASE_URL

# If empty, tests will use SQLite automatically
unset TEST_DATABASE_URL

# For PostgreSQL, verify container is running
docker ps | grep nerdlearn-test-db
```

#### 4. "pytest-asyncio error"

**Solution:**
```bash
pip install pytest-asyncio

# Ensure asyncio_mode is set
# Check tests/database/conftest.py has proper async fixtures
```

#### 5. Tests hanging or timing out

**Solution:**
```bash
# Run with timeout
pytest tests/database/ -v --timeout=30

# Run specific test to isolate issue
pytest tests/database/test_schema_integrity.py::TestTableExistence -v
```

### Getting Help

If you encounter issues:

1. Check the test output for specific error messages
2. Run a single test to isolate the problem
3. Verify all dependencies are installed
4. Check the GitHub Issues for known problems

### Debug Mode

For detailed debugging:

```bash
# Maximum verbosity
pytest tests/database/ -vvv --tb=long

# Show print statements
pytest tests/database/ -v -s

# Stop on first failure
pytest tests/database/ -v -x

# Run last failed tests
pytest tests/database/ -v --lf
```

---

## Quick Reference Card

```bash
# === SETUP ===
pip install pytest pytest-asyncio pytest-cov pytest-html aiosqlite

# === RUN ALL TESTS ===
pytest tests/database/ -v

# === RUN BY CATEGORY ===
pytest tests/database/test_schema_integrity.py -v    # Schema tests
pytest tests/database/test_data_integrity.py -v      # Data integrity
pytest tests/database/test_crud_operations.py -v     # CRUD operations
pytest tests/database/test_relationships.py -v       # Relationships
pytest tests/database/test_performance.py -v         # Benchmarks
pytest tests/database/test_concurrency.py -v         # Concurrency

# === WITH REPORTS ===
pytest tests/database/ -v --html=report.html --self-contained-html

# === WITH COVERAGE ===
pytest tests/database/ -v --cov=apps/api/app/models --cov-report=html

# === USING SCRIPT ===
./scripts/run_db_tests.sh -d -c all -v --coverage
```

---

## Next Steps

1. Run the tests to verify everything works: `pytest tests/database/ -v`
2. Set up the VS Code Test Explorer for visual test running
3. Configure pre-commit hooks to run tests before commits
4. Check the CI workflow (`.github/workflows/database-tests.yml`) for automated testing
