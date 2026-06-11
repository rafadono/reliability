# Reliability Analysis Platform

Modern, fast reliability engineering analysis platform built with Vue 3 and FastAPI.

## Features

- **Independent Modular Filtering**: Each chart/module features its own independent filter options (equipment → type → failure mode cascade), allowing users to analyze and compare different assets and analysis types side-by-side.
- **Pareto Analysis**: 80/20 split analysis at equipment, type, and failure mode levels
- **Maintenance Jackknife**: Frequency vs Total Downtime scatter plots to identify acute vs chronic issues
- **Criticality Matrix**: Dynamic risk matrix plotting failure probability vs average downtime (MTTR)
- **Weibull & Proactive Analysis**: Fitting, Optimal PM Interval, and Conditional Reliability
- **APM Metrics**: Bad actors ranking (MTBF, MTTR, Availability) and Reliability Growth (Crow-AMSAA)
- **Event Plot Timeline**: Visual failure tracking across assets over time
- **Interactive Dashboard**: Real-time charts and metrics
- **Maintainability Analysis**: Switch seamlessly between Time Between Failures (TBX) and Repair Times (TTX)
- **Report Export**: Download full interactive dashboards as high-quality A3 PDF reports
- **REST API**: Full-featured API with automatic documentation

## Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)
- OR: Python 3.12+, Node.js 20+

### Option 1: Docker (Recommended - Simplest)

```bash
docker-compose up --build
```

**Wait for services to start (about 30 seconds), then open:**

### http://localhost:5173

That's it! You should see the Reliability Analysis dashboard.

---

**Optional - API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 2: Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
reliability/
├── backend/           # FastAPI application
│   ├── app.py        # Main API server
│   ├── src/reliability_analysis/
│   │   ├── core/     # Core analysis modules
│   │   ├── analysis/ # Pareto, Jackknife

│   │   └── utils/    # Helpers and configuration
│   └── requirements.txt
├── frontend/         # Vue 3 application
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── tests/           # Unit tests
├── docker-compose.yml
└── Dockerfile
```

## API Endpoints

- `POST /api/upload` - Upload CSV data
- `GET /api/filters` - Get available filters
- `POST /api/analysis/pareto` - Pareto analysis
- `POST /api/analysis/jackknife` - Jackknife CI calculation
- `GET /api/analysis/summary` - Summary statistics

Full API documentation: http://localhost:8000/docs (when running)

## Data Format

Upload CSV files with these columns:

```csv
Fecha,Hora,Equipo,Tipo,Modo de Falla,Duracion
01/01/2024,08:00:00,Motor A,Mechanical,Bearing,2.5
15/01/2024,14:30:00,Pump B,Hydraulic,Seal,4.0
```

- **Equipo**: Equipment identifier
- **Tipo**: Failure type classification
- **Mdf**: Failure mode (specific failure mechanism)
- **Dias**: Days to failure (numeric)
- **Censurado**: Censoring flag (0=complete, 1=censored)

## Development

### Running Tests

```bash
pytest tests/
```

### Building Production Images

```bash
docker-compose build
```

### Environment Variables

Create `.env` (optional):

```
VITE_API_URL=http://backend:8000
PYTHONUNBUFFERED=1
```

## Technical Stack

**Backend:**
- FastAPI - Async Python web framework
- Pandas - Data analysis
- NumPy - Numerical computing
- Uvicorn - ASGI server

**Frontend:**
- Vue 3 - Reactive UI framework
- Vite - Fast build tool
- Tailwind CSS - Utility-first styling
- Chart.js - Interactive charts

## Performance Notes

- Dockerized setup uses `uv` for fast dependency installation
- Multi-stage Docker builds minimize image size
- Jackknife CI uses Leave-One-Out resampling (O(n²) complexity)
- API endpoints cached where applicable

## Troubleshooting

**Port 8000 already in use:**
```bash
docker-compose down
docker-compose up --build
```

**Node modules issues:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**API not responding:**
Check backend logs: `docker-compose logs backend`

## Contributing

This project follows industry standard practices:
- Python code follows PEP 8
- Vue components follow Composition API patterns
- Tests required for new analysis features
- Docker builds are multi-stage for optimization

## License

MIT License - See LICENSE file for details

---

**Need Help?** Check the API documentation at http://localhost:8000/docs after starting the service.
