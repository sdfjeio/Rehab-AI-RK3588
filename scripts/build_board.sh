#!/bin/bash
# ============================================================================
# build_board.sh — RK3588 板端一键编译脚本
# ============================================================================
# 用法:
#   在板端项目根目录执行:  bash scripts/build_board.sh
#   或:                    bash scripts/build_board.sh --clean
#
# 前置条件:
#   - RKNN SDK (librknnrt.so + rknn_api.h) 已安装或位于 sdk/ 目录
#   - OpenCV 已安装 (sudo apt install libopencv-dev)
#   - cmake >= 3.14
#   - g++ (aarch64)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build_board"

# 解析参数
CLEAN=0
for arg in "$@"; do
    case $arg in
        --clean|-c) CLEAN=1 ;;
        --help|-h)
            echo "用法: $0 [--clean] [--help]"
            echo "  --clean  清理后重新构建"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  AI 智能康复交互系统 - 板端编译"
echo "============================================"
echo "  项目目录: $PROJECT_DIR"
echo "  构建目录: $BUILD_DIR"
echo "============================================"

# 检查 SDK
RKNN_SDK=""
if [ -d "$PROJECT_DIR/sdk/rknn" ]; then
    RKNN_SDK="$PROJECT_DIR/sdk/rknn"
    echo "[OK] RKNN SDK 已找到: $RKNN_SDK"
elif [ -f "/usr/lib/aarch64-linux-gnu/librknnrt.so" ]; then
    echo "[OK] librknnrt.so 在系统路径"
else
    echo "[WARN] 未找到 RKNN SDK — 将编译桌面模式 (无 NPU 推理)"
    echo "       请将 RKNN SDK 放到 sdk/rknn/ 或安装到系统路径"
fi

# 清理
if [ $CLEAN -eq 1 ]; then
    echo "[CLEAN] 清理构建目录..."
    rm -rf "$BUILD_DIR"
fi

# 配置
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[CMAKE] 配置中..."

# 如果有 RKNN SDK, 启用板端模式
CMAKE_EXTRA=""
if [ -n "$RKNN_SDK" ] || [ -f "/usr/lib/aarch64-linux-gnu/librknnrt.so" ]; then
    CMAKE_EXTRA="-DRK3588=ON"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    $CMAKE_EXTRA

# 编译
echo "[MAKE] 编译中... (使用 $(nproc) 核心)"
make -j$(nproc)

echo ""
echo "============================================"
echo "  编译完成!"
echo "  可执行文件: $BUILD_DIR/rehab_app"
echo "============================================"
echo ""
echo "运行:"
echo "  cd $PROJECT_DIR"
echo "  $BUILD_DIR/rehab_app -a m01"
echo ""
echo "板端自启动 (可选):"
echo "  将以下行加入 /etc/rc.local 或创建 systemd 服务:"
echo "  $BUILD_DIR/rehab_app -a m01 --no-audio &"
