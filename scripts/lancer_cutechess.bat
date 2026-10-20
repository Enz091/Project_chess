@echo off
rem Prepare la liste des moteurs de Cute Chess (profils ChessZero + Stockfish) puis lance Cute Chess.
rem A lancer par double-clic depuis l'explorateur. "lancer_cutechess.bat setup" prepare sans lancer.
rem Les profils sont un meme moteur lance par un petit .bat par profil (chesszero_50.bat, _200, _800, _pendule) :
rem --simulations N (force fixe) ou --use-clock (l'IA gere sa pendule). Cute Chess ne transmet pas les "arguments" du JSON. Les accents sont recopies tels quels (UTF-8) dans le JSON.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "FWD=%ROOT:\=/%"
set "CUTE=%ROOT%\tools\cutechess\cutechess-1.5.1-win64\cutechess.exe"
set "CONF_DIR=%APPDATA%\cutechess"
set "CONF=%CONF_DIR%\engines.json"

if not exist "%CONF_DIR%" mkdir "%CONF_DIR%"
if exist "%CONF%" (
    findstr /c:"800 sim" "%CONF%" >nul && goto :launch
    copy /y "%CONF%" "%CONF%.bak" >nul
    echo Ancienne liste de moteurs sauvegardee dans engines.json.bak
)
(
echo [
echo     {
echo         "name": "ChessZero - 50 sim",
echo         "command": "%FWD%/scripts/chesszero_50.bat",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%"
echo     },
echo     {
echo         "name": "ChessZero - 200 sim (référence)",
echo         "command": "%FWD%/scripts/chesszero_200.bat",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%"
echo     },
echo     {
echo         "name": "ChessZero - 800 sim",
echo         "command": "%FWD%/scripts/chesszero_800.bat",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%"
echo     },
echo     {
echo         "name": "ChessZero - pendule",
echo         "command": "%FWD%/scripts/chesszero_pendule.bat",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%"
echo     },
echo     {
echo         "name": "Stockfish 19",
echo         "command": "%FWD%/tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe",
echo         "protocol": "uci",
echo         "workingDirectory": "%FWD%/tools/stockfish/stockfish"
echo     }
echo ]
) > "%CONF%"
echo Moteurs ajoutes a Cute Chess : %CONF%

:launch
if /i "%~1"=="setup" goto :eof
start "" "%CUTE%"
