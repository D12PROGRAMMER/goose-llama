@echo off
rem i9-13900KF: pin to the 8 P-cores (logical 0-15); fixed shards stall on E-cores
start "" /affinity FFFF /wait /b "%~dp0llama.exe" %*
