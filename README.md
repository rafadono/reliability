# Reliability Analysis Platform

[![CI](https://github.com/rafadono/reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/rafadono/reliability/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Vue 3](https://img.shields.io/badge/vue-3.x-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

Modern, fast reliability engineering analysis platform built with Vue 3 and FastAPI.

## Features

- **Independent Modular Filtering**: Each chart/module features its own independent filter options (equipment → type → failure mode cascade), allowing users to analyze and compare different assets and analysis types side-by-side.
- **Pareto Analysis**: 80/20 split analysis at equipment, type, and failure mode levels.
- **Maintenance Jackknife**: Frequency vs Total Downtime scatter plots to identify acute vs chronic issues.
- **Criticality Matrix**: Dynamic risk matrix plotting failure probability vs average downtime (MTTR).
- **Weibull & Proactive Analysis**: Fitting, Optimal PM Interval, and Conditional Reliability.
- **APM Metrics**: Bad actors ranking (MTBF, MTTR, Availability) and Reliability Growth (Crow-AMSAA).
- **Event Plot Timeline**: Visual failure tracking across assets over time.
- **Interactive Dashboard**: Real-time charts and metrics.
- **Maintainability Analysis**: Switch seamlessly between Time Between Failures (TBX) and Repair Times (TTX).
- **Report Export**: Download full interactive dashboards as high-quality A3 PDF reports.
- **REST API**: Full-featured API with automatic documentation.

---

## Quick Start (with Docker - Recommended)

The simplest way to run the entire stack is using Docker Compose:

### Installation via Docker (Recommended)
1. Run `docker-compose up --build -d`
2. Access the frontend at `http://localhost:5173` and the backend API at `http://localhost:8000`.

### Important Note on AI Models and GPU/CPU
This project uses Hugging Face AI models for semantic text analysis. 
By default, the `docker-compose.yml` is configured to build using the **CPU version** of the AI frameworks (`USE_GPU: 0`). 
If you are running this project on a powerful server or PC with an **Nvidia GPU**, you should enable GPU acceleration to make the text analysis exponentially faster:
1. Open `docker-compose.yml`.
2. Under the `backend` build section, change `USE_GPU: 0` to `USE_GPU: 1`.
3. Rebuild your containers: `docker-compose up --build`.

*(Note: Automatic detection of a GPU during a Docker build is technically impossible because Docker isolates the build environment from the host's physical hardware. Therefore, this toggle must be changed manually).*

### Access the Application
* **Frontend Dashboard**: http://localhost:5173
* **FastAPI Interactive Docs (Swagger)**: http://localhost:8000/docs
* **FastAPI Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### Data & Filters
- `POST /api/upload` - Upload CSV data file
- `GET /api/filters` - Get available filter options hierarchically
- `POST /api/filters/set` - Apply active filters cascade
- `GET /api/data/available-filters` - Get all unique filter options
- `GET /api/data/reset-filters` - Reset all filter selections

### Statistics & Overview
- `GET /api/stats/summary` - Summary statistics for active filters

### Reliability & Maintenance Analytics
- `POST /api/analysis/pareto` - Pareto failure frequency & 80/20 split
- `POST /api/analysis/jackknife-plot` - Maintenance Jackknife plot (failure frequency vs total downtime scatter & averages)
- `POST /api/analysis/fit` - Fits Weibull distributions (for reliability & maintainability curves)
- `POST /api/analysis/bad-actors` - APM Bad Actor rankings (MTBF, MTTR, Availability)
- `POST /api/analysis/growth` - Reliability Growth tracking (Crow-AMSAA)
- `POST /api/analysis/kijima-fit` - Fits Kijima imperfect repair virtual age models (Kijima I & II)
- `POST /api/analysis/event-plot` - Event timeline dataset generation per asset
- `POST /api/analysis/optimal-pm` - Calculates optimal Preventive Maintenance interval
- `POST /api/analysis/conditional-reliability` - Computes conditional mission reliability
- `POST /api/analysis/kpi-trend` - Historically tracks Monthly KPI trends (MTBF, MTTR, Availability)
- `POST /api/analysis/download-model` - Pre-downloads Hugging Face AI models to local cache
- `POST /api/analysis/comment-mining` - Runs zero-shot semantic text classification on log comments

Full interactive documentation (Swagger UI): http://localhost:8000/docs (when API is running)

---

## Option 2: Local Development Setup

If running locally without Docker:

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## Debugging and Logs (Docker)

If you need to inspect the status or debug issues:

```bash
# View backend service logs
docker-compose logs backend

# View frontend service logs  
docker-compose logs frontend

# Follow both logs in real time
docker-compose logs -f

# Stop and clean up all containers
docker-compose down
```

### Common Troubleshooting:
* **Port 5173 is busy**: Vite will automatically try the next port (e.g. `5174`). Check frontend logs to confirm the port.
* **Backend not responding**: Wait 10–15 seconds after starting the container for all Python and C++ package initializations to complete.
* **Clean rebuild**: If experiencing caching issues with Node.js modules or packages:
  ```bash
  docker-compose build --no-cache
  docker-compose up
  ```

---

## Data Format

The platform supports semicolon-separated CSV files (`;`) and automatically handles **both English and Spanish headers**.

Example format:
```csv
Equipment;Type;mdf;TTX;Censored;Date;Comment
Motor A;Mechanical;Bearing;100;0;01/01/2026;Mechanical failure of bearing due to wear
Pump B;Hydraulic;Seal;120;1;01/02/2026;Operational decision failure
```

### Supported Columns and Spanish Auto-Mappings:
- **Equipment** (or `Equipo`): Asset identifier.
- **Type** (or `Tipo`): Failure classification.
- **mdf** (or `Modo de Falla`, `failure mode`): Specific failure mechanism.
- **TTX** (or `Duracion`, `duración`, `duration`, `downtime`): Repair or downtime duration (hours).
- **Censored** (or `Censurado`): Censoring indicator (`0` for failure event, `1` for censored/operational stop).
- **Date** (or `Fecha`): Start date of the event (`dd/mm/yyyy` format).
- **Time** (or `Hora`): Optional start time of the event (`hh:mm:ss`).
- **Comment** (or `Comentario`): Event descriptions used for NLP text mining analysis.
- **Days** (or `Dias`, `días`): Alternate column for days to failure (if TTX is not present).

---

## Technical Stack

**Backend:**
- FastAPI - Async Python web framework
- Pandas & NumPy - Data analysis and structures
- SciPy - Parametric fitting optimization
- reliability - Reliability engineering calculations

**Frontend:**
- Vue 3 - Reactive user interface (Composition API)
- Vite - Production build tool
- Chart.js - Canvas charts
- Tailwind CSS - Component styling

---

## Development & Verification

### Running Tests
Ensure all backend assertions are passing:
```bash
pytest tests/
```

### Formatting and Linting
The codebase is structured around PEP 8 styles using Ruff:
```bash
ruff check .
```

---

## Future Review & Unused Features

The following features and functions are implemented and tested, but currently not active in the main UI/API workflow:
- **General Distribution Fitting**: `ReliabilityFitter.fit` in `models.py` uses the `reliability` package's `Fit_Everything` to fit 10+ probability distributions. Kept for future development if non-Weibull parametric fittings are needed.
- **Kijima Virtual Age Model**: `KijimaFitter` and `kijima_model.py` are fully implemented and tested core analytical tools, left for future API/UI integration.
- **Unused Core Helper Functions & Metrics**: `kolmogorov_smirnov_test`, `ks_test_weibull_pit`, and `r2_mrl` in `metrics.py` are preserved as they could be useful for validation/goodness-of-fit once broader fitting algorithms are enabled.
- **DataProcessor Helper Methods**: `get_equipment_types` and `calculate_tbf` in `data_processing.py` are kept for utility in custom script pipelines.

---

## Contributing & Development Guide

First off, thank you for considering contributing to the Reliability Analysis Platform! It's people like you that make open source tools great.

### Development Setup

The easiest way to get started is to use Docker, but for active development, running locally is often faster.

#### Prerequisites
- Python 3.12+
- Node.js 20+
- Git

#### 1. Clone the repository

```bash
git clone https://github.com/rafadono/reliability.git
cd reliability
```

#### 2. Backend Setup (FastAPI)

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

#### 3. Frontend Setup (Vue 3)

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

### Code Style & Linting

#### Python (Backend)
We use `ruff` for all Python formatting and linting. Before committing your code, please run:

```bash
ruff format .
ruff check --fix .
```

Our GitHub Actions CI workflow will automatically check and format your code using Ruff when you push or open a PR.

#### JavaScript/Vue (Frontend)
Make sure your frontend code follows the existing style patterns and builds successfully (`npm run build`).

### Running Tests

All new backend features should include appropriate unit tests. Run the test suite using pytest:

```bash
# from the root or backend directory
python -m pytest tests/
```

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. You may merge the Pull Request in once you have the sign-off of other developers, or if you do not have permission to do that, you may request the reviewer to merge it for you.

### License

By contributing to this repository, you agree that your contributions will be licensed under its MIT License.

