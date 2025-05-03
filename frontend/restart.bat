@echo off
echo Stopping any running Node/Python processes...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo Starting DeepCleanse application...
npm start 