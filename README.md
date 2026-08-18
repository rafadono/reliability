# RelAI

[![CI](https://github.com/rafadono/reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/rafadono/reliability/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Vue 3](https://img.shields.io/badge/vue-3.x-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

AI-assisted reliability & maintenance engineering platform, built with Vue 3 and FastAPI.

## Key Capabilities

* **Reliability Workbench**: Interactive flow and pipeline builder to visually connect data sources, filters, Weibull fittings, Pareto charts, and RAM simulations.
* **Quantitative Analysis**: Fits lifetime data to Weibull distributions and imperfect repair virtual age models (Kijima I & II). Includes Pareto analysis, Jackknife charts, and criticality matrices.
* **International Standards Compliance**:
  * **RCM (SAE JA1011/12)**: Guided 7-question failure mode evaluation.
  * **FMECA (IEC 60812)**: Risk Priority Number (RPN) scoring matrix.
  * **RCA (IEC 62740)**: Automated 5 Whys and Ishikawa (Fishbone) diagrams.
  * **FTA (IEC 61025)**: Graphical Fault Tree logic gate builder.
  * **RAM Assurance (ISO 20815)**: Plant availability simulator modeling logistics delays and maintenance efficiency.
* **AI Copilot**: Vendor-agnostic LLM assistant (Gemini, OpenAI, Ollama, Mock) for RCM recommendations, comment mining, and an interactive chat grounded in the currently loaded dataset (record counts, active filters).

---

## Quick Start (Docker - Recommended)

Run the entire stack containing both the backend and frontend:

```bash
docker-compose up --build -d
```

* **Frontend Dashboard**: http://localhost:5180
* **FastAPI Interactive Swagger Docs**: http://localhost:8008/docs
* **FastAPI ReDoc**: http://localhost:8008/redoc

### GPU Acceleration for AI Models
The Hugging Face models used for text mining run on CPU by default. If you have an Nvidia GPU, change `USE_GPU: 0` to `USE_GPU: 1` under the backend build section in `docker-compose.yml`, then rebuild: `docker-compose up --build -d`.

---

## Local Development Setup

If running locally without Docker:

### Backend (FastAPI)
1. Go to backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn app:app --reload
   ```

### Frontend (Vue 3 / Vite)
1. Go to frontend directory:
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

---

## Configuration (.env)

Customize the LLM provider by creating a `.env` file in the root directory:

* `LLM_PROVIDER`: Selected LLM vendor (`mock`, `gemini`, `openai`, `ollama`).
* `LLM_MODEL`: Model name (e.g. `gemini-1.5-flash`, `gpt-4o`, `llama3`).
* `GEMINI_API_KEY`: API key for Google Gemini.
* `OPENAI_API_KEY`: API key for OpenAI.
* `OLLAMA_BASE_URL`: API URL for local Ollama server (default: `http://localhost:11434`).
* `VITE_SHOW_SIDEBAR`: Set to `true` to re-enable the legacy left sidebar navigation (default: `false`, hidden — it currently duplicates the top tab bar; see `frontend/src/featureFlags.js`).

---

## Data Format

The platform ingests semicolon-separated (`;`) CSV files with English or Spanish headers.

```csv
Equipment;Type;mdf;TTX;Censored;Date;Comment
Motor A;Mechanical;Bearing;100;0;01/01/2026;Mechanical failure of bearing due to wear
Pump B;Hydraulic;Seal;120;1;01/02/2026;Operational decision failure
```

* **Equipment** (or `Equipo`): Asset identifier.
* **Type** (or `Tipo`): Failure classification.
* **mdf** (or `Modo de Falla`, `failure mode`): Specific failure mechanism.
* **TTX** (or `Duracion`, `duración`, `duration`, `downtime`): Downtime or duration (hours).
* **Censored** (or `Censurado`): Status (`0` for failure, `1` for operational/censored stop).
* **Date** (or `Fecha`): Start date (`dd/mm/yyyy`).
* **Comment** (or `Comentario`): Short text used for NLP mining.
* **Plant** (or `Planta`, `Site`): Optional plant/site identifier, used by the Plant-level filter cascade.

On upload, the platform reports a small data-quality summary (rows dropped for an unparseable date, duplicate rows removed) and the cleaned dataset can be downloaded back out at any time via `GET /api/data/export`.

---

## Testing & Quality

* Run backend tests:
  ```bash
  pytest tests/
  ```
* Lint check:
  ```bash
  ruff check .
  ```
