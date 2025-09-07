# Contributing to AITERM

Thank you for your interest in contributing to AITERM! We welcome contributions of all kinds, from bug reports and documentation improvements to new features.

## Setting Up for Development

To get started with modifying the code, you'll want to install the project in an "editable" mode. This allows you to make changes to the source code and have them immediately reflected when you run the `aiterm` command.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/aiterm.git
    cd aiterm
    ```

2.  **Create and Activate a Virtual Environment**
    It's strongly recommended to work within a Python virtual environment to avoid conflicts with other projects or your system's Python installation.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    On Windows, the activation command is:
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

3.  **Install in Editable Mode**
    Install the package in editable (`-e`) mode along with the extra dependencies needed for development and testing.
    ```bash
    pip install -e ".[dev,test]"
    ```
    Now, any changes you make in the `src/aiterm/` directory will be live when you run the `aiterm` command from your terminal.

## Code Style

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. We use `black` for code formatting and `ruff` for linting. Before submitting a pull request, please ensure your code is formatted correctly:

```bash
# Run from the project root directory
black .
ruff check . --fix
```

## Submitting Pull Requests

1.  Fork the repository and create a new branch for your feature or bug fix.
2.  Make your changes and commit them with a clear, descriptive message.
3.  Ensure your code passes all tests (`pytest`).
4.  Push your branch to your fork and open a pull request against the `main` branch of the original repository.

We will review your contribution as soon as possible. Thank you!
