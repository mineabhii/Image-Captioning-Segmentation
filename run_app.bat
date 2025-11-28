@echo off
echo Starting Image Captioning and Segmentation Web Application...
echo =====================================================

cd /d "C:\Users\admin\Image_Captioning_and_Segmentation"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Flask application...
echo Open your browser and go to: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server

python app.py

pause