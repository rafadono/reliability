@echo off
cd /d %~dp0
call .venv\Scripts\activate
python -m streamlit run main.py --server.runOnSave true
pause