# Contributing to Hybrid RAG

Thanks for your interest in improving this project. This guide covers how to set up your environment, run tests, and submit changes.

## Getting started

1. Fork the repository and clone your fork locally.
2. 2. Create a virtual environment and install dependencies:
  
   3.    ```bash
            python -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt -r requirements-dev.txt
            ```

         3. Copy `.env.example` to `.env` and fill in the required API keys and service URLs.
     
         4. ## Running tests
     
         5. Run the full test suite with pytest before opening a pull request:
     
         6. ```bash
            pytest
            ```

            Please add or update tests under `tests/` for any behavior change.

            ## Code style

            This project uses `ruff` for linting. Run it locally with:

            ```bash
            ruff check .
            ```

            Keep functions small and favor clear names over comments where possible.

            ## Submitting changes

            1. Create a feature branch for your change.
            2. 2. Make sure tests and linting pass locally.
               3. 3. Open a pull request with a clear description of the motivation and approach.
                  4. 4. Link any related issues in the pull request description.
                    
                     5. ## Reporting issues
                    
                     6. If you find a bug or have a feature request, please open an issue describing the problem, the expected behavior, and steps to reproduce it when applicable.
                     7. 
