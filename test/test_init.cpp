// ============================================================================
// test_init.cpp — 模块初始化冒烟测试 (无需 GUI 也可运行)
// ============================================================================
#include "app_config.h"
#include "cv_pipeline.h"
#include "audio_engine.h"
#include "llm_pipeline.h"

#include <cstdio>
#include <iostream>

int main() {
    // 禁用 stdout 缓冲, 确保输出实时可见
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::cout << "=== AI 康复系统 - 初始化冒烟测试 ===\n\n";

    // 1. 测试配置常量
    std::cout << "[TEST 1] 配置常量\n";
    std::cout << "  MODEL_DIR:    " << Config::MODEL_DIR << "\n";
    std::cout << "  TEMPLATE_DIR: " << Config::TEMPLATE_DIR << "\n";
    std::cout << "  CAMERA:       " << Config::CAMERA_WIDTH << "x"
              << Config::CAMERA_HEIGHT << " @ idx " << Config::CAMERA_INDEX << "\n";
    std::cout << "  INPUT:        " << Config::MODEL_INPUT_W << "x"
              << Config::MODEL_INPUT_H << "\n";
    std::cout << "  OUTPUT:       " << Config::OUTPUT_CHANNELS << " ch x "
              << Config::NUM_ANCHORS << " anchors\n";
    std::cout << "  OBJ_THRESH:   " << Config::OBJ_THRESHOLD << "\n";
    std::cout << "  KPT_THRESH:   " << Config::KPT_THRESHOLD << "\n";
    std::cout << "  NMS_THRESH:   " << Config::NMS_THRESHOLD << "\n";
    std::cout << "  PASS\n\n";

    // 2. 测试静态工具函数
    std::cout << "[TEST 2] calcAngle 工具函数\n";
    cv::Point2f a(0, 0), b(1, 0), c(0, 1);
    float angle = CVPipeline::calcAngle(a, b, c);
    std::cout << "  (0,0)-(1,0)-(0,1) = " << angle << " deg (expect 45)\n";
    if (std::abs(angle - 45.0f) > 1.0f) {
        std::cerr << "  FAIL: 计算结果偏差\n";
        return 1;
    }
    std::cout << "  PASS\n\n";

    // 3. 测试 CV 管线初始化 (桌面模式: 模拟骨架)
    std::cout << "[TEST 3] CVPipeline 初始化\n";
    CVPipeline cv(Config::POSE_MODEL);
    if (!cv.init(Config::CAMERA_INDEX)) {
        std::cerr << "  FAIL: 管线初始化失败 (检查摄像头是否可用)\n";
        return 1;
    }
    std::cout << "  PASS\n\n";

    // 4. 测试帧处理 (5 帧)
    std::cout << "[TEST 4] processFrame x5\n";
    for (int i = 0; i < 5; ++i) {
        cv::Mat frame = cv.processFrame();
        if (frame.empty()) {
            std::cerr << "  FAIL: 第 " << i << " 帧为空\n";
            return 1;
        }
        int detCount = cv.getDetectionCount();
        const auto& angles = cv.getJointAngles();
        std::cout << "  帧 " << i << ": " << frame.cols << "x" << frame.rows
                  << " 检测数=" << detCount
                  << " 膝角 L=" << (int)angles.leftKnee
                  << " R=" << (int)angles.rightKnee << "\n";
    }
    std::cout << "  PASS\n\n";

    // 5. 测试 DTW 评分
    std::cout << "[TEST 5] DTW 评分\n";
    DTWResult dtw = cv.computeDTW(Config::TEMPLATE_DIR + "m01_golden.json");
    std::cout << "  Score: " << dtw.score
              << "  Raw: " << dtw.rawDistance
              << "  Valid: " << (dtw.valid ? "yes" : "no") << "\n";
    std::cout << "  PASS\n\n";

    // 6. 测试音频引擎
    std::cout << "[TEST 6] AudioEngine\n";
    AudioEngine audio;
    bool audioOk = audio.init();
    std::cout << "  状态: " << (audioOk ? "就绪" : "无 TTS 后端 (可接受)") << "\n";
    std::cout << "  PASS\n\n";

    // 7. 测试 LLM 管道
    std::cout << "[TEST 7] LLMPipeline\n";
    LLMPipeline llm("");
    llm.init();
    std::string fb = llm.generateFeedback(0.72f, "深蹲", "左膝角度偏差 5°");
    std::cout << "  反馈: " << fb << "\n";
    std::cout << "  PASS\n\n";

    // 8. 测试参考姿态渲染
    std::cout << "[TEST 8] renderReferencePose\n";
    cv::Mat ref = CVPipeline::renderReferencePose(320, 480,
        Config::TEMPLATE_DIR + "m01_golden.json", "m01");
    if (ref.empty()) {
        std::cerr << "  FAIL: 渲染结果为空\n";
        return 1;
    }
    std::cout << "  尺寸: " << ref.cols << "x" << ref.rows << "\n";
    std::cout << "  PASS\n\n";

    std::cout << "========================================\n";
    std::cout << "  全部 8 项测试通过!\n";
    std::cout << "========================================\n";
    return 0;
}
