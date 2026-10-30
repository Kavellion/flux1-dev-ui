@echo off
REM FLUX.1 dev UI one-click installer for Windows.
REM
REM Python is INCLUDED in this package, in the python_embeded folder,
REM Python 3.12, ComfyUI portable style. There is nothing to install
REM first, no venv is created, and no Python on your system is
REM touched or needed -- if you have no Python at all, this still
REM works.
REM
REM What it does:
REM   1. Bootstraps pip into the bundled Python, first run only,
REM      using the bundled get-pip.py.
REM   2. Installs PyTorch with CUDA support from the cu130 index,
REM      then the rest of the requirements -- Diffusers 0.42, Gradio,
REM      and the rest -- into python_embeded Lib site-packages.
REM   3. Creates images_out for finished pictures.
REM
REM What it does NOT do:
REM   - Touch any Python you already have. This app lives entirely
REM     in this folder and uses only the bundled Python 3.12.
REM   - Download the model weights. Diffusers downloads the model,
REM     about 17 GB on the default quantized path, from Hugging
REM     Face on the first Generate, into the Hugging Face cache
REM     on your user profile -- not into this folder.
REM   - Accept the model license for you or save any token. The
REM     FLUX.1 dev license acceptance happens on the Hugging Face
REM     model page in your browser, and the read token goes into
REM     the app page itself -- see README.md before the first run.
REM
REM Requirements: Windows, an NVIDIA GPU, and about 25 GB free
REM disk for the first download plus room for pictures.

setlocal
cd /d "%~dp0"

if exist python_embeded\python.exe goto HavePython
echo.
echo The bundled Python in python_embeded is missing.
echo Re-download or fully re-extract this package, then run
echo this file again.
echo.
pause
exit /b 1

:HavePython
echo Bundled Python:
python_embeded\python.exe --version

if exist python_embeded\Scripts\pip.exe goto HavePip
echo Setting up pip for the bundled Python - first run only...
python_embeded\python.exe python_embeded\get-pip.py --no-warn-script-location
if not errorlevel 1 goto HavePip
echo.
echo Pip setup failed. Check your internet connection,
echo then run this again.
echo.
pause
exit /b 1

:HavePip
python_embeded\python.exe -m pip install --no-warn-script-location --upgrade pip

echo Installing PyTorch with CUDA support - this is the big download...
python_embeded\python.exe -m pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu130
if not errorlevel 1 goto TorchOk
echo.
echo PyTorch install failed. Check your internet connection and
echo that your GPU driver is up to date, then run this again.
echo.
pause
exit /b 1

:TorchOk
echo Installing the remaining packages...
python_embeded\python.exe -m pip install --no-warn-script-location -r requirements.txt
if not errorlevel 1 goto ReqOk
echo.
echo Package install failed. Run this again -- pip resumes
echo downloads it already finished.
echo.
pause
exit /b 1

:ReqOk
if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start FLUX.1 dev UI.
echo.
echo Before the first Generate, accept the FLUX.1 dev license
echo on its Hugging Face model page and have a read token ready
echo to paste into the app. The first Generate then downloads
echo about 17 GB of model weights -- that is the one-time
echo download, not a stuck render.
echo.
pause
