@echo off
rem findBeeQueen launcher (double-click on Windows). Copy this file to the Desktop if you like.
title findBeeQueen
wsl.exe -d Ubuntu-24.04 --cd ~ -- bash -lc "cd ~/work/findBeeQueen && ./start.sh %*"
pause
