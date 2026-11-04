@echo off
rem Builds llama.exe from llama.goose with the patched Goose and clang (simd fn versions need clang).
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
"%~dp0..\goose\build\goose.exe" build "%~dp0llama.goose" --cc clang -o "%~dp0llama.exe"
