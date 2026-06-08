// ============================================================================
// cv_pipeline.cpp — 视觉管线实现
// ============================================================================
// 管线: 摄像头采集 → 预处理 → NPU 推理 → 后处理 → 骨骼绘制 → 角度计算
//
// 板端编译: 定义 RK3588_DEPLOY 并链接 librknnrt.so
// 桌面调试: 不定义该宏, 使用模拟关键点进行骨架绘制测试
// ============================================================================

#include "cv_pipeline.h"
#include "app_config.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// ---------------------------------------------------------------------------
// RKNN SDK (板端可用)
// ---------------------------------------------------------------------------
#ifdef RK3588_DEPLOY
#include "rknn_api.h"
#endif

using json = nlohmann::json;

// ============================================================================
// 骨骼连接定义 (COCO 17 关键点)
// ============================================================================
struct Bone {
    int a;
    int b;
    cv::Scalar color;
};

static const std::vector<Bone> BONES = {
    // ---- 躯干中线 ----
    { (int)Kpt::NOSE,          (int)Kpt::LEFT_EYE,       Config::COLOR_FACE   },
    { (int)Kpt::NOSE,          (int)Kpt::RIGHT_EYE,      Config::COLOR_FACE   },
    { (int)Kpt::LEFT_EYE,      (int)Kpt::LEFT_EAR,       Config::COLOR_FACE   },
    { (int)Kpt::RIGHT_EYE,     (int)Kpt::RIGHT_EAR,      Config::COLOR_FACE   },

    // ---- 肩-髋连线 ----
    { (int)Kpt::LEFT_SHOULDER, (int)Kpt::RIGHT_SHOULDER, Config::COLOR_CENTER  },
    { (int)Kpt::LEFT_HIP,      (int)Kpt::RIGHT_HIP,      Config::COLOR_CENTER  },
    { (int)Kpt::LEFT_SHOULDER, (int)Kpt::LEFT_HIP,       Config::COLOR_LEFT    },
    { (int)Kpt::RIGHT_SHOULDER,(int)Kpt::RIGHT_HIP,      Config::COLOR_RIGHT   },

    // ---- 左臂 ----
    { (int)Kpt::LEFT_SHOULDER, (int)Kpt::LEFT_ELBOW,     Config::COLOR_LEFT    },
    { (int)Kpt::LEFT_ELBOW,    (int)Kpt::LEFT_WRIST,     Config::COLOR_LEFT    },

    // ---- 右臂 ----
    { (int)Kpt::RIGHT_SHOULDER,(int)Kpt::RIGHT_ELBOW,    Config::COLOR_RIGHT   },
    { (int)Kpt::RIGHT_ELBOW,   (int)Kpt::RIGHT_WRIST,    Config::COLOR_RIGHT   },

    // ---- 左腿 ----
    { (int)Kpt::LEFT_HIP,      (int)Kpt::LEFT_KNEE,      Config::COLOR_LEFT    },
    { (int)Kpt::LEFT_KNEE,     (int)Kpt::LEFT_ANKLE,     Config::COLOR_LEFT    },

    // ---- 右腿 ----
    { (int)Kpt::RIGHT_HIP,     (int)Kpt::RIGHT_KNEE,     Config::COLOR_RIGHT   },
    { (int)Kpt::RIGHT_KNEE,    (int)Kpt::RIGHT_ANKLE,    Config::COLOR_RIGHT   },
};

// ============================================================================
// 工具函数
// ============================================================================

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

// ---- 以下函数仅 NPU 推理路径使用 ----
#ifdef RK3588_DEPLOY

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 计算两矩形 IoU
static float bboxIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width,  b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float iou = inter / (areaA + areaB - inter + 1e-6f);
    return iou;
}

// 非极大值抑制
static void nms(std::vector<Detection>& dets, float thresh, int maxKeep) {
    // 按置信度降序
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.objScore > b.objScore;
        });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> kept;
    kept.reserve(maxKeep);

    for (size_t i = 0; i < dets.size() && kept.size() < (size_t)maxKeep; ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (bboxIoU(dets[i].bbox, dets[j].bbox) > thresh) {
                suppressed[j] = true;
            }
        }
    }
    dets.swap(kept);
}
#endif // RK3588_DEPLOY

// ============================================================================
// PIMPL 内部状态
// ============================================================================
struct CVPipeline::Impl {
    // ---- 摄像头 ----
    cv::VideoCapture cap;
    cv::Mat          rawFrame;

    // ---- RKNN ----
#ifdef RK3588_DEPLOY
    rknn_context       rkCtx     = 0;
    rknn_input_output_num ioNum;
    rknn_tensor_attr   inputAttr;
    rknn_tensor_attr   outputAttr;
#endif
    bool              npuReady  = false;
    std::string       modelPath;
    std::vector<uint8_t> modelBuffer;

    // ---- 预处理缓冲区 ----
    cv::Mat letterboxBuffer;      // 640x640 BGR
    cv::Mat rgbBuffer;            // 640x640 RGB (float or uint8)
    float   scaleX = 1.0f;       // letterbox 缩放比
    float   scaleY = 1.0f;
    int     padX   = 0;          // letterbox 填充偏移
    int     padY   = 0;

    // ---- 推理结果 ----
    std::vector<Detection> detections;

    // ---- 关节角度 ----
    JointAngles           angles;
    std::deque<float>     kneeAngleHistory;   // 左膝角度历史

    // ---- DTW ----
    float  currentDTWScore = 0.0f;

    // ---- 帧计数 ----
    int    frameCount = 0;

    // ---- 输出帧 ----
    cv::Mat displayFrame;

    // ---- 调试 ----
    bool verbose = true;
};

// ============================================================================
// 构造 / 析构
// ============================================================================

CVPipeline::CVPipeline(const std::string& rknnModelPath)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->modelPath = rknnModelPath;
}

CVPipeline::~CVPipeline() {
    if (pImpl_->cap.isOpened()) {
        pImpl_->cap.release();
    }
#ifdef RK3588_DEPLOY
    if (pImpl_->npuReady) {
        rknn_destroy(pImpl_->rkCtx);
    }
#endif
}

// ============================================================================
// init — 加载模型 + 打开摄像头
// ============================================================================
bool CVPipeline::init(int cameraIndex) {
#ifdef RK3588_DEPLOY
    // 1. 读取模型文件
    FILE* fp = fopen(pImpl_->modelPath.c_str(), "rb");
    if (!fp) {
        std::cerr << "[CV] 错误: 无法打开模型文件 " << pImpl_->modelPath << "\n";
        return false;
    }
    fseek(fp, 0, SEEK_END);
    size_t modelSize = ftell(fp);
    rewind(fp);
    pImpl_->modelBuffer.resize(modelSize);
    size_t readBytes = fread(pImpl_->modelBuffer.data(), 1, modelSize, fp);
    fclose(fp);
    if (readBytes != modelSize) {
        std::cerr << "[CV] 错误: 模型文件读取不完整\n";
        return false;
    }
    std::cout << "[CV] 模型已加载: " << pImpl_->modelPath
              << " (" << modelSize / 1024 << " KB)\n";

    // 2. 初始化 RKNN
    int ret = rknn_init(&pImpl_->rkCtx,
                        pImpl_->modelBuffer.data(),
                        modelSize, 0, nullptr);
    if (ret < 0) {
        std::cerr << "[CV] 错误: rknn_init 失败, ret=" << ret << "\n";
        return false;
    }
    pImpl_->npuReady = true;

    // 3. 查询输入/输出属性
    ret = rknn_query(pImpl_->rkCtx, RKNN_QUERY_IN_OUT_NUM,
                     &pImpl_->ioNum, sizeof(pImpl_->ioNum));
    if (ret < 0) {
        std::cerr << "[CV] 错误: rknn_query IO 数量失败\n";
        return false;
    }
    std::cout << "[CV] NPU 就绪 — 输入: " << pImpl_->ioNum.n_input
              << ", 输出: " << pImpl_->ioNum.n_output << "\n";

    // 输入属性
    memset(&pImpl_->inputAttr, 0, sizeof(pImpl_->inputAttr));
    pImpl_->inputAttr.index = 0;
    ret = rknn_query(pImpl_->rkCtx, RKNN_QUERY_INPUT_ATTR,
                     &pImpl_->inputAttr, sizeof(pImpl_->inputAttr));
    if (ret < 0) {
        std::cerr << "[CV] 错误: 查询输入属性失败\n";
        return false;
    }

    // 输出属性
    memset(&pImpl_->outputAttr, 0, sizeof(pImpl_->outputAttr));
    pImpl_->outputAttr.index = 0;
    ret = rknn_query(pImpl_->rkCtx, RKNN_QUERY_OUTPUT_ATTR,
                     &pImpl_->outputAttr, sizeof(pImpl_->outputAttr));
    if (ret < 0) {
        std::cerr << "[CV] 错误: 查询输出属性失败\n";
        return false;
    }
    std::cout << "[CV] 输出张量: " << pImpl_->outputAttr.n_elems
              << " floats (" << pImpl_->outputAttr.n_elems * 4 / 1024
              << " KB)\n";
#else
    std::cout << "[CV] 桌面模式 (NPU 未启用) — 使用模拟关键点\n";
    pImpl_->npuReady = false;
#endif

    // 4. 打开摄像头
    pImpl_->cap.open(cameraIndex);
    if (!pImpl_->cap.isOpened()) {
        std::cerr << "[CV] 错误: 无法打开摄像头 index=" << cameraIndex << "\n";
        return false;
    }
    pImpl_->cap.set(cv::CAP_PROP_FRAME_WIDTH,  Config::CAMERA_WIDTH);
    pImpl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::CAMERA_HEIGHT);
    pImpl_->cap.set(cv::CAP_PROP_FPS, Config::CAMERA_FPS);

    std::cout << "[CV] 摄像头就绪 — " << Config::CAMERA_WIDTH
              << "×" << Config::CAMERA_HEIGHT << " @ index " << cameraIndex
              << "\n";

    // 预分配缓冲区
    pImpl_->letterboxBuffer = cv::Mat(Config::MODEL_INPUT_H,
                                      Config::MODEL_INPUT_W, CV_8UC3);
    pImpl_->rgbBuffer = cv::Mat(Config::MODEL_INPUT_H,
                                Config::MODEL_INPUT_W, CV_8UC3);

    return true;
}

bool CVPipeline::isReady() const {
#ifdef RK3588_DEPLOY
    return pImpl_->npuReady && pImpl_->cap.isOpened();
#else
    return pImpl_->cap.isOpened();
#endif
}

#ifdef RK3588_DEPLOY
// ============================================================================
// preprocess — letterbox + BGR→RGB (仅 NPU 推理路径使用)
// ============================================================================
static void preprocess(const cv::Mat& src, cv::Mat& dstBGR, cv::Mat& dstRGB,
                       float& scaleX, float& scaleY, int& padX, int& padY) {
    int srcW = src.cols;
    int srcH = src.rows;
    int dstW = Config::MODEL_INPUT_W;
    int dstH = Config::MODEL_INPUT_H;

    // Letterbox: 等比缩放 + 居中填充
    float r = std::min((float)dstW / srcW, (float)dstH / srcH);
    int newW = (int)(srcW * r);
    int newH = (int)(srcH * r);
    padX = (dstW - newW) / 2;
    padY = (dstH - newH) / 2;
    scaleX = r;
    scaleY = r;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));
    cv::copyMakeBorder(resized, dstBGR, padY, padY, padX, padX,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // BGR → RGB
    cv::cvtColor(dstBGR, dstRGB, cv::COLOR_BGR2RGB);
}
#endif // RK3588_DEPLOY

// ============================================================================
// decodeYOLOv8Pose — 解析 RKNN 输出为 Detection 列表 (仅 NPU 推理路径使用)
// ============================================================================
#ifdef RK3588_DEPLOY
static void decodeYOLOv8Pose(const float* output, int origW, int origH,
                              float scaleX, float scaleY, int padX, int padY,
                              float objThresh, std::vector<Detection>& dets,
                              bool applySigmoid) {
    dets.clear();

    const int A = Config::NUM_ANCHORS;     // 8400

    // 遍历三尺度特征图
    int anchorOffset = 0;
    for (int level = 0; level < 3; ++level) {
        int stride = Config::STRIDES[level];
        int gridW  = Config::GRID_W[level];
        int gridH  = Config::GRID_H[level];
        int num    = gridW * gridH;

        for (int i = 0; i < num; ++i) {
            int idx = anchorOffset + i;
            int gy  = i / gridW;
            int gx  = i % gridW;

            // 读取 bbox + conf
            float cx_raw = output[0 * A + idx];
            float cy_raw = output[1 * A + idx];
            float w_raw  = output[2 * A + idx];
            float h_raw  = output[3 * A + idx];
            float obj    = output[4 * A + idx];

            if (applySigmoid) {
                cx_raw = sigmoid(cx_raw);
                cy_raw = sigmoid(cy_raw);
                w_raw  = sigmoid(w_raw);
                h_raw  = sigmoid(h_raw);
                obj    = sigmoid(obj);
            }

            if (obj < objThresh) continue;

            // Decode bbox
            float cx = (cx_raw * 2.0f - 0.5f + (float)gx) * (float)stride;
            float cy = (cy_raw * 2.0f - 0.5f + (float)gy) * (float)stride;
            float w  = std::pow(w_raw * 2.0f, 2.0f) * (float)stride;
            float h  = std::pow(h_raw * 2.0f, 2.0f) * (float)stride;

            // 映射回原始图像坐标 (去除 letterbox)
            float x1 = (cx - w * 0.5f - (float)padX) / scaleX;
            float y1 = (cy - h * 0.5f - (float)padY) / scaleY;
            float bw = w / scaleX;
            float bh = h / scaleY;

            // 裁剪到画面内
            x1 = clampf(x1, 0.0f, (float)origW - 1.0f);
            y1 = clampf(y1, 0.0f, (float)origH - 1.0f);
            bw = clampf(bw, 1.0f, (float)origW - x1);
            bh = clampf(bh, 1.0f, (float)origH - y1);

            Detection det;
            det.bbox    = cv::Rect2f(x1, y1, bw, bh);
            det.objScore = obj;

            // 解码关键点
            int validKpts = 0;
            for (int k = 0; k < 17; ++k) {
                int chX = 5 + k * 3 + 0;
                int chY = 5 + k * 3 + 1;
                int chC = 5 + k * 3 + 2;

                float kx = output[chX * A + idx];
                float ky = output[chY * A + idx];
                float kc = output[chC * A + idx];

                if (applySigmoid) {
                    kx = sigmoid(kx);
                    ky = sigmoid(ky);
                    kc = sigmoid(kc);
                }

                // Decode 关键点坐标
                float px = (kx * 2.0f - 0.5f + (float)gx) * (float)stride;
                float py = (ky * 2.0f - 0.5f + (float)gy) * (float)stride;

                // 映射回原始图像
                px = (px - (float)padX) / scaleX;
                py = (py - (float)padY) / scaleY;

                det.kpts[k]      = cv::Point2f(px, py);
                det.kptScores[k] = kc;

                if (kc >= Config::KPT_THRESHOLD) validKpts++;
            }

            // 至少需要 5 个有效关键点
            if (validKpts >= 5) {
                dets.push_back(det);
            }
        }
        anchorOffset += num;
    }
}
#endif // RK3588_DEPLOY

// ============================================================================
// drawSkeleton — 在帧上绘制骨骼与关键点
// ============================================================================
static void drawSkeleton(cv::Mat& frame, const Detection& det) {
    int W = frame.cols;
    int H = frame.rows;

    // 1. 绘制骨骼连线
    for (const auto& bone : BONES) {
        const cv::Point2f& pa = det.kpts[bone.a];
        const cv::Point2f& pb = det.kpts[bone.b];
        float ca = det.kptScores[bone.a];
        float cb = det.kptScores[bone.b];

        if (ca < Config::KPT_CONF_MIN_DRAW ||
            cb < Config::KPT_CONF_MIN_DRAW) continue;

        cv::Point paI(static_cast<int>(pa.x), static_cast<int>(pa.y));
        cv::Point pbI(static_cast<int>(pb.x), static_cast<int>(pb.y));

        // 检查坐标有效性
        if (paI.x < 0 || paI.x >= W || paI.y < 0 || paI.y >= H) continue;
        if (pbI.x < 0 || pbI.x >= W || pbI.y < 0 || pbI.y >= H) continue;

        cv::line(frame, paI, pbI, bone.color, Config::LINE_THICKNESS,
                 cv::LINE_AA);
    }

    // 2. 绘制关键点
    for (int k = 0; k < 17; ++k) {
        float conf = det.kptScores[k];
        if (conf < Config::KPT_CONF_MIN_DRAW) continue;

        cv::Point pt(static_cast<int>(det.kpts[k].x),
                     static_cast<int>(det.kpts[k].y));
        if (pt.x < 0 || pt.x >= W || pt.y < 0 || pt.y >= H) continue;

        // 颜色按置信度分级
        cv::Scalar color = Config::COLOR_KPT_LOW;
        if (conf > 0.75f)      color = Config::COLOR_KPT_HIGH;
        else if (conf > 0.50f) color = Config::COLOR_KPT_MED;

        cv::circle(frame, pt, Config::KPT_RADIUS, color, -1, cv::LINE_AA);
    }
}

// ============================================================================
// drawJointAngles — 在关节旁标注角度值
// ============================================================================
static void drawJointAngles(cv::Mat& frame, const Detection& det,
                             const JointAngles& angles) {
    auto drawAngle = [&](int, int idxB, int, float angle,
                          cv::Scalar color) {
        if (det.kptScores[idxB] < Config::KPT_CONF_MIN_DRAW) return;
        cv::Point pt(static_cast<int>(det.kpts[idxB].x + 20),
                     static_cast<int>(det.kpts[idxB].y - 10));
        char buf[32];
        snprintf(buf, sizeof(buf), "%.0f°", angle);
        cv::putText(frame, buf, pt, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv::LINE_AA);
    };

    drawAngle((int)Kpt::LEFT_HIP,  (int)Kpt::LEFT_KNEE, (int)Kpt::LEFT_ANKLE,
              angles.leftKnee, Config::COLOR_LEFT);
    drawAngle((int)Kpt::RIGHT_HIP, (int)Kpt::RIGHT_KNEE,(int)Kpt::RIGHT_ANKLE,
              angles.rightKnee, Config::COLOR_RIGHT);
    drawAngle((int)Kpt::LEFT_SHOULDER, (int)Kpt::LEFT_ELBOW, (int)Kpt::LEFT_WRIST,
              angles.leftElbow, Config::COLOR_LEFT);
    drawAngle((int)Kpt::RIGHT_SHOULDER,(int)Kpt::RIGHT_ELBOW,(int)Kpt::RIGHT_WRIST,
              angles.rightElbow, Config::COLOR_RIGHT);
}

// ============================================================================
// drawHUD — 屏幕抬头信息
// ============================================================================
static void drawHUD(cv::Mat& frame, int frameCount, float dtwScore,
                     const JointAngles& angles, int detCount,
                     const std::string& actionName) {
    int y = 20;
    int dy = 22;
    cv::Scalar col = Config::COLOR_HUD;
    double fontScale = 0.55;

    auto putLine = [&](const std::string& text) {
        cv::putText(frame, text, cv::Point(10, y),
                    Config::HUD_FONT, fontScale, col, 1, cv::LINE_AA);
        y += dy;
    };

    char buf[128];

    snprintf(buf, sizeof(buf), "FPS: %d  |  Detections: %d  |  Action: %s",
             frameCount, detCount, actionName.c_str());
    putLine(buf);

    snprintf(buf, sizeof(buf), "DTW Score: %.2f", dtwScore);
    putLine(buf);

    snprintf(buf, sizeof(buf),
             "Knee: L=%.0f° R=%.0f°  |  Elbow: L=%.0f° R=%.0f°  |  Hip: L=%.0f° R=%.0f°",
             angles.leftKnee, angles.rightKnee,
             angles.leftElbow, angles.rightElbow,
             angles.leftHip, angles.rightHip);
    putLine(buf);
}

// ============================================================================
// processFrame — 核心处理管线
// ============================================================================
cv::Mat CVPipeline::processFrame() {
    // ---- 1. 采集帧 ----
    pImpl_->cap >> pImpl_->rawFrame;
    if (pImpl_->rawFrame.empty()) return {};

    cv::Mat frame = pImpl_->rawFrame.clone();
    int origW = frame.cols;
    int origH = frame.rows;

    pImpl_->detections.clear();

#ifdef RK3588_DEPLOY
    if (pImpl_->npuReady) {
        // ---- 2a. 预处理 ----
        preprocess(frame, pImpl_->letterboxBuffer, pImpl_->rgbBuffer,
                   pImpl_->scaleX, pImpl_->scaleY, pImpl_->padX, pImpl_->padY);

        // ---- 2b. 设置输入 ----
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index    = 0;
        inputs[0].type     = RKNN_TENSOR_UINT8;
        inputs[0].size     = Config::MODEL_INPUT_W * Config::MODEL_INPUT_H * 3;
        inputs[0].fmt      = RKNN_TENSOR_NHWC;  // RGB buffer is HWC
        inputs[0].buf      = pImpl_->rgbBuffer.data;
        inputs[0].pass_through = 0;

        int ret = rknn_inputs_set(pImpl_->rkCtx, 1, inputs);
        if (ret < 0) {
            std::cerr << "[CV] rknn_inputs_set 失败 ret=" << ret << "\n";
            return frame;
        }

        // ---- 2c. 推理 ----
        ret = rknn_run(pImpl_->rkCtx, nullptr);
        if (ret < 0) {
            std::cerr << "[CV] rknn_run 失败 ret=" << ret << "\n";
            return frame;
        }

        // ---- 2d. 获取输出 ----
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        outputs[0].is_prealloc = 0;
        ret = rknn_outputs_get(pImpl_->rkCtx, 1, outputs, nullptr);
        if (ret < 0) {
            std::cerr << "[CV] rknn_outputs_get 失败 ret=" << ret << "\n";
            return frame;
        }

        // ---- 2e. 后处理 (解码) ----
        float* outputData = (float*)outputs[0].buf;

        // 自动检测是否需要 sigmoid
        // 若存在远超出 [0,1] 范围的值 → 需要 sigmoid
        bool needSigmoid = false;
        int sampleCount = std::min(100, (int)(outputs[0].size / sizeof(float)));
        for (int i = 0; i < sampleCount; ++i) {
            float v = outputData[i];
            if (v < -3.0f || v > 3.0f) {
                needSigmoid = true;
                break;
            }
        }
        if (pImpl_->verbose && pImpl_->frameCount == 0) {
            std::cout << "[CV] 输出值检测: "
                      << (needSigmoid ? "需要 sigmoid" : "已预先激活")
                      << "\n";
        }

        decodeYOLOv8Pose(outputData, origW, origH,
                         pImpl_->scaleX, pImpl_->scaleY,
                         pImpl_->padX, pImpl_->padY,
                         Config::OBJ_THRESHOLD, pImpl_->detections,
                         needSigmoid);

        rknn_outputs_release(pImpl_->rkCtx, 1, outputs);

        // NMS
        nms(pImpl_->detections, Config::NMS_THRESHOLD, Config::MAX_DETECTIONS);
    }
#else
    // ---- 桌面模拟模式: 生成假的关键点用于 UI 调试 ----
    {
        // 简单的硬编码关键点模拟一个站立姿态
        Detection sim;
        sim.objScore = 0.9f;
        float cx = origW * 0.5f;
        float top = origH * 0.1f;
        float mid = origH * 0.5f;
        float bot = origH * 0.9f;
        float shx = origW * 0.4f; // shoulder x offset
        float hix = origW * 0.44f; // hip x offset

        sim.kpts[(int)Kpt::NOSE]           = {cx, top + 20};
        sim.kpts[(int)Kpt::LEFT_EYE]       = {cx - 10, top + 10};
        sim.kpts[(int)Kpt::RIGHT_EYE]      = {cx + 10, top + 10};
        sim.kpts[(int)Kpt::LEFT_EAR]       = {cx - 25, top + 15};
        sim.kpts[(int)Kpt::RIGHT_EAR]      = {cx + 25, top + 15};
        sim.kpts[(int)Kpt::LEFT_SHOULDER]  = {cx - shx, top + 70};
        sim.kpts[(int)Kpt::RIGHT_SHOULDER] = {cx + shx, top + 70};
        sim.kpts[(int)Kpt::LEFT_ELBOW]     = {cx - 100, mid - 10};
        sim.kpts[(int)Kpt::RIGHT_ELBOW]    = {cx + 100, mid - 10};
        sim.kpts[(int)Kpt::LEFT_WRIST]     = {cx - 150, mid + 30};
        sim.kpts[(int)Kpt::RIGHT_WRIST]    = {cx + 150, mid + 30};
        sim.kpts[(int)Kpt::LEFT_HIP]       = {cx - hix, mid + 30};
        sim.kpts[(int)Kpt::RIGHT_HIP]      = {cx + hix, mid + 30};
        sim.kpts[(int)Kpt::LEFT_KNEE]      = {cx - 50, bot - 40};
        sim.kpts[(int)Kpt::RIGHT_KNEE]     = {cx + 50, bot - 40};
        sim.kpts[(int)Kpt::LEFT_ANKLE]     = {cx - 40, bot};
        sim.kpts[(int)Kpt::RIGHT_ANKLE]    = {cx + 40, bot};

        for (int k = 0; k < 17; ++k) sim.kptScores[k] = 0.95f;

        pImpl_->detections.push_back(sim);
    }
#endif

    // ---- 3. 骨骼绘制 ----
    for (const auto& det : pImpl_->detections) {
        drawSkeleton(frame, det);
    }

    // ---- 4. 计算关节角度 ----
    if (!pImpl_->detections.empty()) {
        const auto& det = pImpl_->detections[0];

        auto& a = pImpl_->angles;
        a.leftKnee  = calcAngle(det.kpts[(int)Kpt::LEFT_HIP],
                                det.kpts[(int)Kpt::LEFT_KNEE],
                                det.kpts[(int)Kpt::LEFT_ANKLE]);
        a.rightKnee = calcAngle(det.kpts[(int)Kpt::RIGHT_HIP],
                                det.kpts[(int)Kpt::RIGHT_KNEE],
                                det.kpts[(int)Kpt::RIGHT_ANKLE]);
        a.leftElbow = calcAngle(det.kpts[(int)Kpt::LEFT_SHOULDER],
                                det.kpts[(int)Kpt::LEFT_ELBOW],
                                det.kpts[(int)Kpt::LEFT_WRIST]);
        a.rightElbow = calcAngle(det.kpts[(int)Kpt::RIGHT_SHOULDER],
                                 det.kpts[(int)Kpt::RIGHT_ELBOW],
                                 det.kpts[(int)Kpt::RIGHT_WRIST]);
        a.leftHip   = calcAngle(det.kpts[(int)Kpt::LEFT_SHOULDER],
                                det.kpts[(int)Kpt::LEFT_HIP],
                                det.kpts[(int)Kpt::LEFT_KNEE]);
        a.rightHip  = calcAngle(det.kpts[(int)Kpt::RIGHT_SHOULDER],
                                det.kpts[(int)Kpt::RIGHT_HIP],
                                det.kpts[(int)Kpt::RIGHT_KNEE]);

        // 更新膝角历史
        float avgKnee = (a.leftKnee + a.rightKnee) * 0.5f;
        pImpl_->kneeAngleHistory.push_back(avgKnee);
        if ((int)pImpl_->kneeAngleHistory.size() > Config::ANGLE_HISTORY_MAX) {
            pImpl_->kneeAngleHistory.pop_front();
        }

        // 标注角度
        drawJointAngles(frame, det, pImpl_->angles);
    }

    // ---- 5. HUD ----
    std::string actionLabel = "m01";
    drawHUD(frame, pImpl_->frameCount, pImpl_->currentDTWScore,
            pImpl_->angles, (int)pImpl_->detections.size(), actionLabel);

    pImpl_->frameCount++;
    pImpl_->displayFrame = frame;

    return frame;
}

// ============================================================================
// 结果查询
// ============================================================================
const Detection* CVPipeline::getDetection(int index) const {
    if (index < 0 || index >= (int)pImpl_->detections.size()) return nullptr;
    return &pImpl_->detections[index];
}

int CVPipeline::getDetectionCount() const {
    return (int)pImpl_->detections.size();
}

const JointAngles& CVPipeline::getJointAngles() const {
    return pImpl_->angles;
}

const std::deque<float>& CVPipeline::getKneeAngleHistory() const {
    return pImpl_->kneeAngleHistory;
}

float CVPipeline::getCurrentDTWScore() const {
    return pImpl_->currentDTWScore;
}

// ============================================================================
// calcAngle — 三点求夹角
// ============================================================================
float CVPipeline::calcAngle(const cv::Point2f& a,
                             const cv::Point2f& b,
                             const cv::Point2f& c) {
    cv::Point2f ba = a - b;
    cv::Point2f bc = c - b;
    float dot = ba.x * bc.x + ba.y * bc.y;
    float mag = std::sqrt(ba.x * ba.x + ba.y * ba.y) *
                std::sqrt(bc.x * bc.x + bc.y * bc.y);
    if (mag < 1e-6f) return 0.0f;
    float rad = std::acos(clampf(dot / mag, -1.0f, 1.0f));
    return rad * 180.0f / static_cast<float>(M_PI);
}

// ============================================================================
// computeDTW — DTW 动态时间规整
// ============================================================================
DTWResult CVPipeline::computeDTW(const std::string& templateJson) const {
    DTWResult result;

    // 读取模板
    std::ifstream ifs(templateJson);
    if (!ifs.is_open()) {
        std::cerr << "[CV] 无法打开模板: " << templateJson << "\n";
        return result;
    }
    json j;
    ifs >> j;
    std::vector<float> golden = j["angle_sequence"].get<std::vector<float>>();

    // 获取用户膝角历史
    const auto& hist = pImpl_->kneeAngleHistory;
    std::vector<float> user(hist.begin(), hist.end());

    if (golden.empty() || user.empty()) return result;

    int n = (int)user.size();
    int m = (int)golden.size();

    // DTW 矩阵
    std::vector<std::vector<float>> dtw(n + 1,
        std::vector<float>(m + 1, 1e9f));
    dtw[0][0] = 0.0f;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(user[i - 1] - golden[j - 1]);
            dtw[i][j] = cost + std::min({dtw[i - 1][j],
                                         dtw[i][j - 1],
                                         dtw[i - 1][j - 1]});
        }
    }

    result.rawDistance   = dtw[n][m];
    result.matchedLength = n + m;

    // 归一化到 [0, 1]
    float normalized = result.rawDistance / (float)(n + m);
    result.score = 1.0f / (1.0f + normalized / Config::ANGLE_NORM_RANGE);
    result.valid = true;

    // 更新当前评分
    pImpl_->currentDTWScore = result.score;

    return result;
}

// ============================================================================
// renderReferencePose — 渲染标准参考姿态
// ============================================================================
cv::Mat CVPipeline::renderReferencePose(int panelW, int panelH,
                                         const std::string& templateJson,
                                         const std::string& actionLabel) {
    cv::Mat canvas(panelH, panelW, CV_8UC3, cv::Scalar(30, 30, 30));

    // 画标题
    cv::putText(canvas, "Reference : " + actionLabel,
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.55, Config::COLOR_HUD, 1, cv::LINE_AA);

    // 读取模板膝角序列, 取中位数作为当前显示角度
    float kneeAngle = 90.0f; // 默认深蹲底部角度
    std::vector<float> goldenAngles;
    if (!templateJson.empty()) {
        std::ifstream ifs(templateJson);
        if (ifs.is_open()) {
            json j;
            ifs >> j;
            goldenAngles = j["angle_sequence"].get<std::vector<float>>();
            if (!goldenAngles.empty()) {
                kneeAngle = goldenAngles[goldenAngles.size() / 2];
            }
        }
    }

    // 基于膝角构建简化参考骨架
    // 将膝关节角度转换为 2D 关键点
    float cx = panelW * 0.5f;
    float baseY = panelH * 0.15f;
    float torsoLen = panelH * 0.25f;
    float thighLen = panelH * 0.20f;
    float shinLen  = panelH * 0.20f;

    // 参考关键点
    cv::Point2f refKpts[17];

    // 头部
    refKpts[0] = {cx, baseY};                    // nose
    refKpts[1] = {cx - 8, baseY - 8};           // left eye
    refKpts[2] = {cx + 8, baseY - 8};           // right eye
    refKpts[3] = {cx - 18, baseY - 3};          // left ear
    refKpts[4] = {cx + 18, baseY - 3};          // right ear

    float shY = baseY + 40;
    float shOff = 35;
    refKpts[5] = {cx - shOff, shY};             // left shoulder
    refKpts[6] = {cx + shOff, shY};             // right shoulder

    float hiY = shY + torsoLen;
    float hiOff = 30;
    refKpts[11] = {cx - hiOff, hiY};            // left hip
    refKpts[12] = {cx + hiOff, hiY};            // right hip

    // 膝角 → 膝关节位置
    float rad = kneeAngle * M_PI / 180.0f;
    float kneeX = cx - std::sin(rad) * thighLen * 0.3f;
    float kneeY = hiY + std::cos(rad) * thighLen * 0.5f + thighLen * 0.3f;
    refKpts[13] = {kneeX - 5, kneeY};           // left knee
    refKpts[14] = {kneeX + 5 + 60, kneeY};      // right knee

    float ankY = kneeY + shinLen * 0.7f;
    refKpts[15] = {cx - 25, ankY};              // left ankle
    refKpts[16] = {cx + 25, ankY};              // right ankle

    // 手臂
    float elY = shY + 60;
    refKpts[7] = {cx - 80, elY};          // left elbow
    refKpts[8] = {cx + 80, elY};          // right elbow
    refKpts[9] = {cx - 110, elY + 40};    // left wrist
    refKpts[10] = {cx + 110, elY + 40};   // right wrist

    // 画参考骨骼
    for (const auto& bone : BONES) {
        cv::Point pa(static_cast<int>(refKpts[bone.a].x),
                     static_cast<int>(refKpts[bone.a].y));
        cv::Point pb(static_cast<int>(refKpts[bone.b].x),
                     static_cast<int>(refKpts[bone.b].y));
        cv::line(canvas, pa, pb, bone.color, Config::LINE_THICKNESS,
                 cv::LINE_AA);
    }
    for (int k = 0; k < 17; ++k) {
        cv::Point pt(static_cast<int>(refKpts[k].x),
                     static_cast<int>(refKpts[k].y));
        cv::circle(canvas, pt, Config::KPT_RADIUS,
                   Config::COLOR_KPT_HIGH, -1, cv::LINE_AA);
    }

    // 标注膝角
    char angleBuf[32];
    snprintf(angleBuf, sizeof(angleBuf), "%.0f°", kneeAngle);
    cv::putText(canvas, angleBuf,
                cv::Point(static_cast<int>(refKpts[13].x - 30),
                          static_cast<int>(refKpts[13].y - 15)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, Config::COLOR_KPT_HIGH, 1,
                cv::LINE_AA);

    // 分隔线
    cv::line(canvas, cv::Point(panelW - 1, 0), cv::Point(panelW - 1, panelH),
             cv::Scalar(80, 80, 80), 1);

    return canvas;
}
