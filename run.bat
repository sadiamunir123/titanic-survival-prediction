@echo off
echo Installing requirements (first time only)...
pip install -r requirements.txt
echo.
echo Starting the app. Your browser will open automatically.
python app.py
pause
