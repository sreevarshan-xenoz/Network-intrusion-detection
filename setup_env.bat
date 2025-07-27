@echo off
echo Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing development dependencies...
pip install -r requirements-dev.txt
echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
pause