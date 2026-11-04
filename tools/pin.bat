@echo off
set "A=%~1"
set "E=%~2"
shift
shift
set R=
:loop
if "%~1"=="" goto run
set R=%R% %1
shift
goto loop
:run
start "" /affinity %A% /wait /b %E% %R%
