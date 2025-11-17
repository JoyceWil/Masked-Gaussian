@echo off
echo === Activating Visual Studio 2019 build tools ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

echo === Activating Conda Environment: r2_gaussian ===
call C:\Users\guheg\anaconda3\Scripts\activate.bat
call conda activate r2_gaussian

echo === Ready for CUDA Extension Build ===
cmd
