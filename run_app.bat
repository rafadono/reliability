@echo off
REM -------------------------------------------------------------
REM 0. Movernos al directorio donde está este .bat (unidad + carpeta)
REM -------------------------------------------------------------
cd /d %~dp0

REM -------------------------------------------------------------
REM 1. Comprobar/crear el venv
REM -------------------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual en .venv...
    python -m venv .venv || (
        echo ERROR: no se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

REM -------------------------------------------------------------
REM 2. Activar el entorno virtual
REM -------------------------------------------------------------
call .venv\Scripts\activate.bat

REM -------------------------------------------------------------
REM 3. Actualizar pip e instalar dependencias
REM -------------------------------------------------------------
echo Actualizando pip...
pip install --upgrade pip

echo Instalando dependencias...
pip install -r requirements.txt || (
    echo ERROR: fallo en pip install.
    pause
    exit /b 1
)

REM -------------------------------------------------------------
REM 4. Ejecutar Streamlit vía main.py (auto-reload on save)
REM -------------------------------------------------------------
echo Iniciando Streamlit...
python -m streamlit run main.py --server.runOnSave true

pause