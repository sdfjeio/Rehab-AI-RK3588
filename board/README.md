# AI 康复训练系统 — RK3588

基于 RK3588 的实时康复训练交互系统。摄像头采集人体姿态，NPU (YOLOv8s-Pose) 检测 17 个 COCO 关键点，DTW 评估深蹲动作规范性。触摸屏 UI 支持患者自主操作。

## 硬件

- RK3588 开发板 (ARM64)
- DSI MIPI 显示屏 (1024x600) + FT5x06 触摸屏
- MIPI CSI 摄像头
- Ubuntu 22.04 (GNOME 桌面)

## 依赖 (板端安装)

```bash
sudo apt install build-essential librknnrt-dev librga-dev libdrm-dev
```

## 编译 & 部署

```bash
# 板载编译
cd /data/rehab
make clean && make

# 安装桌面启动器（使得 GNOME 菜单出现图标）
sudo ./scripts/install.sh
```

## 使用

**桌面模式**: GNOME 菜单点击 "AI 康复训练" 图标，确认后自动关闭桌面进入全屏模式。

**命令行模式**:
```bash
sudo systemctl stop gdm3
cd /data/rehab && ./rehab_app
# 退出后: sudo systemctl start gdm3
```

## 项目结构

```
board/
├── src/                 # 源码
│   ├── main_board.cc    # 主程序 (DRM显示/V4L2摄像头/主循环)
│   ├── postprocess.cc   # YOLOv8-pose 后处理 (NMS/DFL/关键点解码)
│   ├── yolov8-pose.cc   # RKNN 模型加载与推理
│   ├── image_utils.c    # 图像预处理 (letterbox)
│   ├── file_utils.c     # 模型文件读取
│   └── ui.cpp           # 触摸 UI (4 屏状态机 + 按钮渲染)
├── include/             # 项目头文件
├── vendor/              # 第三方头文件 (RGA/DRM/RKNN)
├── model/               # RKNN 模型 + 标签
├── tests/               # DRM 测试程序
├── scripts/             # launch.sh / install.sh
├── assets/              # .desktop 文件 + 图标
└── Makefile             # 板载原生编译
```

## 触摸 UI 流程

- **WELCOME** — 标题 + 开始/退出按钮
- **EXERCISE SELECT** — 深蹲 / 手臂上举 / 侧平举 / 前平举
- **TRAINING** — 摄像头 + 骨骼叠加 + HUD + 暂停/结束按钮
- **RESULTS** — DTW 评分 + 关节角度汇总 + 重新开始/返回菜单
