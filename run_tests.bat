@echo off
call venv\Scripts\activate.bat
python -m pytest tests/ -v
pause