@echo off
title RAG Self-Correction Inspector (Port 8502)
echo ========================================================
echo   Avvio Self-Correction Inspector su http://localhost:8502
echo ========================================================
echo.
uv run streamlit run app_self_correction_ui.py --server.port 8502
pause
