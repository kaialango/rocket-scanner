@echo off
cd /d C:\Users\kaihu\LPPFS\rocket_scanner
echo 当前目录: %CD%
python openai_launchpad.py
echo.
echo ========== CSV 文件列表 ==========
dir *.csv
echo =================================
echo 程序执行完毕，按任意键退出...
pause > nul