#pragma once

// ============================================================================
// app_config.h — 集中式配置常量
// ============================================================================
// 所有可调节的超参数、路径、阈值均集中于此文件, 方便快速调试。
// 板端部署时定义 RK3588_DEPLOY 宏即可切换到板端路径。
// ============================================================================

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Config {

// ---------------------------------------------------------------------------
// 模型与资源路径
// ---------------------------------------------------------------------------
#ifdef RK3588_DEPLOY
inline const std::string MODEL_DIR    = "/home/user/rehab/models/";
inline const std::string TEMPLATE_DIR = "/home/user/rehab/templates/";
#else
inline const std::string MODEL_DIR    = "../models/";
inline const std::string TEMPLATE_DIR = "../templates/";
#endif

inline const std::string POSE_MODEL = MODEL_DIR + "yolov8s-pose-int8.rknn";

// ---------------------------------------------------------------------------
// 摄像头
// ---------------------------------------------------------------------------
constexpr int    CAMERA_INDEX = 0;
constexpr int    CAMERA_WIDTH = 640;
constexpr int    CAMERA_HEIGHT = 480;
constexpr double CAMERA_FPS   = 30.0;

// ---------------------------------------------------------------------------
// 模型输入 (YOLOv8-Pose 640×640)
// ---------------------------------------------------------------------------
constexpr int MODEL_INPUT_W   = 640;
constexpr int MODEL_INPUT_H   = 640;
constexpr int MODEL_CHANNELS  = 3;
constexpr int NUM_KEYPOINTS   = 17;
constexpr int OUTPUT_CHANNELS = 56;   // 4 bbox + 1 conf + 17×3 kpt
constexpr int NUM_ANCHORS     = 8400; // 80×80 + 40×40 + 20×20

// 各尺度特征图的 stride 和网格尺寸
constexpr int STRIDES[3]   = { 8, 16, 32 };
constexpr int GRID_W[3]    = { 80, 40, 20 };
constexpr int GRID_H[3]    = { 80, 40, 20 };

// ---------------------------------------------------------------------------
// 检测阈值
// ---------------------------------------------------------------------------
constexpr float OBJ_THRESHOLD   = 0.45f;
constexpr float KPT_THRESHOLD   = 0.50f;
constexpr float NMS_THRESHOLD   = 0.55f;
constexpr int   MAX_DETECTIONS  = 3;

// ---------------------------------------------------------------------------
// 骨骼绘制
// ---------------------------------------------------------------------------
constexpr int   KPT_RADIUS        = 4;
constexpr int   LINE_THICKNESS    = 2;
constexpr float KPT_CONF_MIN_DRAW = 0.40f;

// BGR 颜色 — 左侧 / 右侧 / 中线
inline const cv::Scalar COLOR_LEFT   { 0, 255, 0 };    // 绿色
inline const cv::Scalar COLOR_RIGHT  { 255, 0, 0 };    // 蓝色
inline const cv::Scalar COLOR_CENTER { 255, 255, 255 }; // 白色
inline const cv::Scalar COLOR_FACE   { 128, 128, 128 }; // 灰色
inline const cv::Scalar COLOR_HUD    { 0, 255, 255 };   // 黄色
inline const cv::Scalar COLOR_ALERT  { 0, 0, 255 };     // 红色

// 关键点绘制 — 按置信度分级着色
inline const cv::Scalar COLOR_KPT_HIGH { 0, 255, 0 };    // 高置信度 > 0.75
inline const cv::Scalar COLOR_KPT_MED  { 0, 255, 255 };  // 中置信度 > 0.50
inline const cv::Scalar COLOR_KPT_LOW  { 0, 0, 255 };    // 低置信度

// ---------------------------------------------------------------------------
// DTW 评分
// ---------------------------------------------------------------------------
constexpr int   ANGLE_HISTORY_MAX    = 150;
constexpr float DTW_INTERVAL_SEC     = 3.0f;
constexpr float ANGLE_NORM_RANGE     = 180.0f;

// ---------------------------------------------------------------------------
// 音频反馈
// ---------------------------------------------------------------------------
constexpr float FEEDBACK_COOLDOWN_SEC = 5.0f;
constexpr int   FEEDBACK_MIN_FRAMES   = 45;

// ---------------------------------------------------------------------------
// UI 布局
// ---------------------------------------------------------------------------
constexpr int DISPLAY_WIDTH  = 1280;
constexpr int DISPLAY_HEIGHT = 480;
constexpr int HUD_FONT       = cv::FONT_HERSHEY_SIMPLEX;

} // namespace Config

