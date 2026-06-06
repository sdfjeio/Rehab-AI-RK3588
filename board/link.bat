@echo off
set PATH=C:\msys64\home\bin;%PATH%
cd /d E:\RK\board
aarch64-none-linux-gnu-g++ -o rehab_app src/image_utils.o src/file_utils.o src/postprocess.o src/yolov8-pose.o src/main_board.o -Wl,--allow-shlib-undefined -Wl,--rpath-link=lib -L lib -l rknnrt -l rga -l drm -l pthread -l stdc++ -l m -l c 2>&1
echo EXIT CODE: %ERRORLEVEL%
dir rehab_app 2>&1
