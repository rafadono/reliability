"""
Configuración centralizada de la aplicación.

Utiliza variables de entorno con valores por defecto.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

APP_NAME = "Reliability Analysis"
APP_VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
STREAMLIT_HEADLESS = os.getenv("STREAMLIT_HEADLESS", "true").lower() == "true"

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
SUPPORTED_FILE_FORMATS = ["csv"]
CSV_ENCODING = "latin-1"
CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

EXCLUDED_MODELS = [
    "Weibull_2P", "Weibull_CR", "Weibull_Mixture", "Weibull_DS",
    "Gamma_2P", "Loglogistic_2P", "Gamma_3P", "Lognormal_3P",
    "Loglogistic_3P", "Gumbel_2P", "Exponential_2P", "Beta_2P"
]

KIJIMA_MODELS = [1, 2]  # Kijima I y II

PLOT_WIDTH = 1000
PLOT_HEIGHT = 600
PLOT_GRID_POINTS = 200
