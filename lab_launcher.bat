:: echo off
title Jupyter lab
mode con:cols=80 lines=100
echo Start python enviroment
call C:\programs\projx_venv\Scripts\activate.bat
echo Start Jupyter lab
cd C:\git\digger
jupyter lab