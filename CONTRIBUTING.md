# Contributing to Reliability Analysis Platform

First off, thank you for considering contributing to the Reliability Analysis Platform! It's people like you that make open source tools great.

## Development Setup

The easiest way to get started is to use Docker, but for active development, running locally is often faster.

### Prerequisites
- Python 3.12+
- Node.js 20+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/rafadono/reliability.git
cd reliability
```

### 2. Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff
   ```
4. Run the development server:
   ```bash
   uvicorn app:app --reload
   ```

### 3. Frontend Setup (Vue 3)

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## Code Style & Linting

### Python (Backend)
We use `ruff` for all Python formatting and linting. Before committing your code, please run:

```bash
ruff format .
ruff check --fix .
```

Our GitHub Actions CI workflow will automatically check and format your code using Ruff when you push or open a PR.

### JavaScript/Vue (Frontend)
Make sure your frontend code follows the existing style patterns and builds successfully (`npm run build`).

## Running Tests

All new backend features should include appropriate unit tests. Run the test suite using pytest:

```bash
# from the root or backend directory
python -m pytest tests/
```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. You may merge the Pull Request in once you have the sign-off of other developers, or if you do not have permission to do that, you may request the reviewer to merge it for you.

## License

By contributing to this repository, you agree that your contributions will be licensed under its MIT License.
