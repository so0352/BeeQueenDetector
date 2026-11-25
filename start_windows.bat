@echo off
rem BeeQueenDetector launcher (double-click on Windows). Copy this file to the Desktop if you like.
title BeeQueenDetector
wsl.exe -d Ubuntu-24.04 --cd ~ -- bash -lc "cd ~/work/BeeQueenDetector && ./start.sh %*"
pause
