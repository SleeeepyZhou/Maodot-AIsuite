@echo off

cd ..
set datetime=%date:~0,10% %time:~0,5%
git status
pause
echo Press any key to continue after adding files...
git add .
git commit -m "Test commit at %datetime%"
git push

cd ..
cd .\Maodot\modules\ai_suite
git pull
pause