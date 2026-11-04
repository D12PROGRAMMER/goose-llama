@echo off
rem Builds Goose with goose-llama's compiler patch. Expects ..\..\goose to be a clone of
rem https://github.com/aardappel/goose at 74c641c with goose\goose-llama.patch applied.
rem Needs VS 2022 Build Tools, CMake and Ninja.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=C:\Program Files\CMake\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
cd /d "%~dp0..\..\goose" || exit /b 1
if not exist build\build.ninja cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGOOSE_BUILTIN_LIBS=all || exit /b 1
cmake --build build --config Release --target goose || exit /b 1
