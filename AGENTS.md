# Workspace Rules & Environment Execution Guidelines

## 1. Python Environment & Dependency Management

- **Virtual Environment:** Always use the dedicated workspace virtual environment located at `./venv/bin/python` and `./venv/bin/pip`.
- **Package Installation Allowlist:**
  - All package installations MUST target the local virtual environment directly using the prefix-matchable command shape:
    ```bash
    ./venv/bin/pip install <package_name>
    ```
  - Global `pip install` or installing outside `./venv` is strictly disallowed.
  - Do not use shell activation chaining (e.g. `source venv/bin/activate && pip install ...`) because it defeats command auto-approval prefix matching. Always invoke the binary directly: `./venv/bin/pip install ...`.

## 2. Command Execution Guidelines

- **Standard Prefix:** Use `./venv/bin/python` and `./venv/bin/pip` for all script runs, tests, benchmarks, and installations.
- **Auto-Approval Consistency:** Keep `toolAction` and `toolSummary` consistent across runs to ensure seamless execution in the development environment.
