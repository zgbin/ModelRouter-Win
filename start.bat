@echo off
chcp 65001 >nul 2>&1
title ModelRouter v4.1
cd /d "%~dp0"
python main.py
pause
