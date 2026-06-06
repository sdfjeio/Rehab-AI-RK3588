# ============================================================================
# deploy.ps1 — Windows → RK3588 一键部署 (通过 ADB)
# ============================================================================
# 用法:
#   .\deploy.ps1                    # 推送源码并编译
#   .\deploy.ps1 -SkipBuild         # 仅推送模型和模板
#   .\deploy.ps1 -DevicePath /data/rehab  # 自定义板端路径
#
# 前置条件:
#   - ADB 已添加到 PATH
#   - 板端已通过 USB/WiFi 连接 (adb devices 可见)
#   - 板端已安装 cmake, g++, opencv (或使用预编译二进制)
# ============================================================================

param(
    [string]$DevicePath = "/home/user/rehab",
    [switch]$SkipBuild = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host @"
AI 智能康复交互系统 - ADB 部署工具

用法: .\deploy.ps1 [选项]

选项:
  -DevicePath <路径>   板端项目根目录 (默认: /home/user/rehab)
  -SkipBuild           仅推送文件, 不编译
  -Help                显示此帮助

示例:
  .\deploy.ps1                              # 标准部署
  .\deploy.ps1 -DevicePath /tmp/rehab       # 指定板端路径
  .\deploy.ps1 -SkipBuild                   # 仅推送模型
"@
    exit 0
}

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# ----------------------------------------------------------------------
# 1. 检查 ADB 连接
# ----------------------------------------------------------------------
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI 智能康复交互系统 - 部署工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/5] 检查 ADB 连接..." -ForegroundColor Yellow
$devices = adb devices 2>&1 | Select-String -Pattern "device$"
if (-not $devices) {
    Write-Host "[ERROR] 未检测到 ADB 设备! 请确认:" -ForegroundColor Red
    Write-Host "  1. 板端已通过 USB 连接并开启 USB 调试"
    Write-Host "  2. adb 已添加到 PATH"
    Write-Host "  3. adb devices 能看到设备"
    exit 1
}
Write-Host "  $devices" -ForegroundColor Green

# ----------------------------------------------------------------------
# 2. 创建板端目录
# ----------------------------------------------------------------------
Write-Host "[2/5] 创建板端目录..." -ForegroundColor Yellow
adb shell "mkdir -p $DevicePath/models $DevicePath/templates $DevicePath/src $DevicePath/include $DevicePath/3rdparty $DevicePath/scripts" 2>&1 | Out-Null
Write-Host "  目录已创建: $DevicePath" -ForegroundColor Green

# ----------------------------------------------------------------------
# 3. 推送源码
# ----------------------------------------------------------------------
Write-Host "[3/5] 推送源码文件..." -ForegroundColor Yellow

$files = @(
    @{Src="src/main.cpp";           Dst="$DevicePath/src/"},
    @{Src="src/cv_pipeline.cpp";    Dst="$DevicePath/src/"},
    @{Src="src/audio_engine.cpp";   Dst="$DevicePath/src/"},
    @{Src="src/llm_pipeline.cpp";   Dst="$DevicePath/src/"},
    @{Src="include/app_config.h";   Dst="$DevicePath/include/"},
    @{Src="include/cv_pipeline.h";  Dst="$DevicePath/include/"},
    @{Src="include/audio_engine.h"; Dst="$DevicePath/include/"},
    @{Src="include/llm_pipeline.h"; Dst="$DevicePath/include/"},
    @{Src="3rdparty/json.hpp";      Dst="$DevicePath/3rdparty/"},
    @{Src="CMakeLists.txt";         Dst="$DevicePath/"},
    @{Src="scripts/build_board.sh"; Dst="$DevicePath/scripts/"}
)

foreach ($f in $files) {
    $localPath = Join-Path $ProjectRoot $f.Src
    if (Test-Path $localPath) {
        adb push $localPath $f.Dst 2>&1 | Out-Null
        Write-Host "  OK: $($f.Src)" -ForegroundColor Gray
    } else {
        Write-Host "  SKIP (not found): $($f.Src)" -ForegroundColor DarkYellow
    }
}

# ----------------------------------------------------------------------
# 4. 推送模型与模板
# ----------------------------------------------------------------------
Write-Host "[4/5] 推送模型与模板..." -ForegroundColor Yellow

# 模型
$modelDir = Join-Path $ProjectRoot "models"
if (Test-Path $modelDir) {
    $rknnFiles = Get-ChildItem $modelDir -Filter "*.rknn"
    foreach ($f in $rknnFiles) {
        adb push $f.FullName "$DevicePath/models/" 2>&1 | Out-Null
        Write-Host "  OK: models/$($f.Name)" -ForegroundColor Gray
    }
}

# 模板
$tplDir = Join-Path $ProjectRoot "templates"
if (Test-Path $tplDir) {
    $jsonFiles = Get-ChildItem $tplDir -Filter "*.json"
    foreach ($f in $jsonFiles) {
        adb push $f.FullName "$DevicePath/templates/" 2>&1 | Out-Null
        Write-Host "  OK: templates/$($f.Name)" -ForegroundColor Gray
    }
}

# ----------------------------------------------------------------------
# 5. 编译或完成
# ----------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "[5/5] 板端编译..." -ForegroundColor Yellow
    Write-Host "  在板端执行 (请等待)..."
    adb shell "cd $DevicePath && bash scripts/build_board.sh" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  编译成功!" -ForegroundColor Green
    } else {
        Write-Host "  编译遇到问题, 请手动在板端检查" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "[5/5] 跳过编译 (--SkipBuild)" -ForegroundColor Yellow
}

# ----------------------------------------------------------------------
# 完成
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  部署完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "在板端运行:" -ForegroundColor White
Write-Host "  adb shell" -ForegroundColor Gray
Write-Host "  cd $DevicePath" -ForegroundColor Gray
Write-Host "  ./build_board/rehab_app -a m01" -ForegroundColor Gray
Write-Host ""
Write-Host "或直接:" -ForegroundColor White
Write-Host "  adb shell `"cd $DevicePath && ./build_board/rehab_app -a m01`"" -ForegroundColor Gray
