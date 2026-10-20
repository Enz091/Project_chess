@echo off
rem Prépare la liste des moteurs de Cute Chess (ChessZero + Stockfish) puis lance Cute Chess.
rem A lancer par double-clic depuis l'explorateur. "lancer_cutechess.bat setup" prépare sans lancer.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "FWD=%ROOT:\=/%"
set "CUTE=%ROOT%\tools\cutechess\cutechess-1.5.1-win64\cutechess.exe"
set "CONF_DIR=%APPDATA%\cutechess"
set "CONF=%CONF_DIR%\engines.json"

if not exist "%CONF_DIR%" mkdir "%CONF_DIR%"
if exist "%CONF%" (
    findstr /c:"ChessZero" "%CONF%" >nul && goto :launch
    copy /y "%CONF%" "%CONF%.bak" >nul
    echo Ancienne liste de moteurs sauvegardee dans engines.json.bak
)
(
echo [
echo     {
echo         "command": "%FWD%/scripts/chesszero.bat",
echo         "name": "ChessZero - MCTS 200",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%"
echo     },
echo     {
echo         "command": "%FWD%/tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe",
echo         "name": "Stockfish 19",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%/tools/stockfish/stockfish"
echo     }
echo ]
) > "%CONF%"
echo Moteurs ajoutes a Cute Chess : %CONF%

:launch
if /i "%~1"=="setup" goto :eof
start "" "%CUTE%"
