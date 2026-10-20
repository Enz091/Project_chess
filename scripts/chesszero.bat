@echo off
rem Lanceur UCI pour Cute Chess / Arena : utilise le Python du venv du projet, d'où qu'on l'appelle.
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\uci.py" %*
