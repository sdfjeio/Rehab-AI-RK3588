#!/bin/bash
# AI Rehab Launcher — stops gdm3, runs rehab_app, restarts gdm3 on exit
# Runs detached from the desktop session so it survives gdm3 shutdown.

if command -v zenity &>/dev/null; then
    zenity --question --title="AI 康复训练" \
        --text="即将关闭桌面进入康复训练模式。\n\n触摸屏幕操作，退出训练后桌面会自动恢复。" \
        --width=400 --height=160 2>/dev/null
    if [ $? -ne 0 ]; then
        exit 0
    fi
fi

# Detach with setsid so stopping gdm3 doesn't kill this script
setsid /bin/bash -c '
    systemctl stop gdm3 2>/dev/null
    sleep 1
    cd /data/rehab && exec ./rehab_app
    systemctl start gdm3 2>/dev/null
' &> /dev/null < /dev/null &

exit 0
