# Contributing to This Project

Thank you for considering contributing to this project! Your help is greatly appreciated.

## How to Contribute

### Reporting Issues
If you encounter a bug, have a feature request, or need clarification, please open an issue in the [GitHub Issues](https://github.com/MarkusPic/intensity_duration_frequency_analysis/issues) section. When reporting a bug, include:

- A clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Any relevant error messages or logs

### Submitting Code Changes
1. **Fork the repository** and clone it locally.
2. **Create a new branch** for your feature or fix:
   ```sh
   git checkout -b feature-name
   ```
3. **Make your changes** and ensure they follow the project's coding style.
4. **Run tests** to verify your changes:
   ```sh
   pytest tests
   ```
5. **Commit your changes** with a meaningful message:
   ```sh
   git commit -m "Describe your change briefly"
   ```
6. **Push the branch** to your fork:
   ```sh
   git push origin feature-name
   ```
7. **Create a Pull Request (PR)** from your branch to the `main` branch.

#### Code Style
This project follows standard Python best practices ([PEP 8](https://pep8.org)).

Please ensure your code is typed where applicable using type hints.

Make sure that every function has a docstring with types. We use the [Google style docstring format](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings). Here is an [example for a Google style docstring](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).


### Writing Tests
If your contribution includes code changes, please add or update tests to maintain high code quality. We use `pytest` for testing. All tests are located in the tests folder:
```sh
pytest tests
```
Ensure all tests pass before submitting your PR.

### Reviewing Pull Requests
We encourage contributors to review open PRs and provide constructive feedback.

### Donate
If you find this project useful, consider supporting it by [buying me a coffee](https://www.buymeacoffee.com/MarkusP).

### Other Ways to Contribute

* Writing tutorials or examples using [GitHub Gists](https://docs.github.com/en/get-started/writing-on-github/editing-and-sharing-content-with-gists/creating-gists) and referencing them in [discussions](https://github.com/MarkusPic/intensity_duration_frequency_analysis/discussions/categories/show-and-tell). 
* Fixing typos and improving the documentation by opening issues or creating pull requests.

## Getting Help
If you have any questions, feel free to open a discussion in the [GitHub Discussions](https://github.com/MarkusPic/intensity_duration_frequency_analysis/discussions) or comment on an open issue.


## Ground Rules
The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**. Everyone must abide by our
[Code of Conduct](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/CODE_OF_CONDUCT.md)
and we encourage all to read it carefully.

## Miscellaneous
For more information on contributing to open source projects,
[GitHub's own guide](https://opensource.guide/how-to-contribute)
is a great starting point if you are new to version control. Also, checkout the
[Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software
contributions.

Thank you for contributing!
