@echo off
setlocal enabledelayedexpansion

set /a "time1=%time:~0,2%*3600%%100*60%%100+%time:~3,2%*60%%100+%time:~6,2%"

sd -m allInOnePixelModel_v1.ckpt -p "a cat"

set /a "time2=%time:~0,2%*3600%%100*60%%100+%time:~3,2%*60%%100+%time:~6,2%"

set /a "diff=!time2!-!time1!"
echo Elapsed time: !diff!
pause