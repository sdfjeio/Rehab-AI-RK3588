#pragma once

// ============================================================================
// cv_pipeline.h — 视觉管线: NPU 推理 → 关键点解码 → 骨骼绘制 → DTW 评分
// ============================================================================

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// ---------------------------------------------------------------------------
// COCO 17 关键点枚举
// ---------------------------------------------------------------------------
enum class Kpt : int {
    NOSE = 0,
    LEFT_EYE, RIGHT_EYE,
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    COUNT = 17
};

// ---------------------------------------------------------------------------
// 单次检测结果
// ---------------------------------------------------------------------------
struct Detection {
    cv::Rect2f bbox;                       // 边界框 (原始图像坐标)
    cv::Point2f kpts[17];                  // 关键点 (原始图像坐标, 0~W-1, 0~H-1)
    float      kptScores[17];              // 各关键点置信度
    float      objScore;                   // 目标置信度
};

// ---------------------------------------------------------------------------
// 关节角度 (度)
// ---------------------------------------------------------------------------
struct JointAngles {
    float leftKnee  = 0.0f;   // 左膝
    float rightKnee = 0.0f;   // 右膝
    float leftElbow = 0.0f;   // 左肘
    float rightElbow = 0.0f;  // 右肘
    float leftHip   = 0.0f;   // 左髋
    float rightHip  = 0.0f;   // 右髋
};

// ---------------------------------------------------------------------------
// DTW 评分结果
// ---------------------------------------------------------------------------
struct DTWResult {
    float score           = 0.0f;   // 归一化评分 0~1
    float rawDistance     = 0.0f;   // 原始 DTW 距离
    int   matchedLength   = 0;      // 匹配使用的帧数
    bool  valid           = false;
};

// ---------------------------------------------------------------------------
// 标准参考姿态 (用于侧边对比)
// ---------------------------------------------------------------------------
struct ReferencePose {
    std::string actionName;
    cv::Point2f kpts[17];              // 参考关键点 (画布坐标)
    std::vector<float> kneeAngles;     // 膝角序列 (来自 golden template)
};

// ============================================================================
// CVPipeline — 视觉管线主类
// ============================================================================
class CVPipeline {
public:
    explicit CVPipeline(const std::string& rknnModelPath);
    ~CVPipeline();

    // ---- 生命周期 ----
    bool init(int cameraIndex);
    bool isReady() const;

    // ---- 核心处理 (每帧) ----
    // 抓取一帧, NPU 推理, 后处理, 骨骼绘制, 关节角度计算
    // 返回: 已绘制骨骼的 BGR 帧 (可直接 imshow)
    cv::Mat processFrame();

    // ---- 获取结果 ----
    const Detection*        getDetection(int index = 0) const;
    int                     getDetectionCount() const;
    const JointAngles&      getJointAngles() const;
    const std::deque<float>& getKneeAngleHistory() const;

    // ---- DTW 评分 ----
    DTWResult  computeDTW(const std::string& templateJson) const;
    float      getCurrentDTWScore() const;

    // ---- 静态工具 ----
    static float calcAngle(const cv::Point2f& a,
                           const cv::Point2f& b,
                           const cv::Point2f& c);
    static cv::Mat renderReferencePose(int panelW, int panelH,
                                       const std::string& templateJson,
                                       const std::string& actionLabel);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
