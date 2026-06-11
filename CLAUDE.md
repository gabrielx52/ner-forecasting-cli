## Code Style & Python Conventions

### PEP Standards
* **PEP 8 Compliance**: Code must follow standard PEP 8 formatting rules.
* **Line Length**: Limit all lines to exactly 88 characters (matching Black/Ruff standards).
* **Indentation**: Always use 4 spaces for indentation. Never use tabs.
* **PEP 484 Typing**: Require strict type hints for all function arguments and return signatures.
* **PEP 257 Docstrings**: Include descriptive docstrings for all public classes, methods, and functions.
* **Modern Syntax (PEP 585/604)**: Use native collections (`list`, `dict`) and union operators (`|`) for types.

### Naming Conventions
* **Functions & Variables**: Use `snake_case`.
* **Classes**: Use `PascalCase`.
* **Constants**: Use `UPPER_CASE`.
* **Protected Members**: Use a leading underscore `_single_leading_underscore`.
* **Variable Names**: Never use single letter or abreviated variable names, they should always be explicit and descriptive. 

### Strict Restrictions
* **No Legacy Types**: Never import `List`, `Dict`, or `Optional` from the `typing` module.
* **No Emojis**: Never insert emojis or decorative unicode checkmarks into code or logs.
* **File Operations**: Always use `pathlib.Path` instead of legacy `os.

### Tech Stack
* **Runtime**: Python 3.14.6
* **Testing**: `pytest`, `pytest-mock`, `coverage`
* **Linting/Formatting**: `ruff`
* **Package Manager**: `uv`