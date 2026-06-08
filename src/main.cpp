// ============================================================================
// main.cpp — AI 智能康复交互系统入口
// ============================================================================
// 架构:
//   Thread-1 (主线程): 摄像头采集 + NPU 推理 + UI 渲染
//   Thread-2 (评估):   DTW 评分 + 偏差分析
//   Thread-3 (语音):   TTS 语音反馈播报
//
// 用法:
//   ./rehab_app                          # 默认: 动作 m01, 摄像头 0
//   ./rehab_app -a m02 -c 1              # 指定动作和摄像头
//   ./rehab_app -a m01 --no-audio        # 禁用语音
//   ./rehab_app -h                       # 帮助
// ============================================================================

#include "app_config.h"
#include "cv_pipeline.h"
#include "audio_engine.h"
#include "llm_pipeline.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------
static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

// ---------------------------------------------------------------------------
// 命令行参数
// ---------------------------------------------------------------------------
struct AppArgs {
    std::string action      = "m01";
    std::string modelPath   = Config::POSE_MODEL;
    std::string templateDir = Config::TEMPLATE_DIR;
    int         cameraIndex = Config::CAMERA_INDEX;
    bool        enableAudio = true;
    bool        enableLLM   = false;   // LLM 默认关闭 (需额外模型)
    bool        showFPS     = true;
};

static void printUsage(const char* prog) {
    std::cout << "AI 智能康复交互系统 - RK3588 Edition\n\n"
              << "用法: " << prog << " [选项]\n\n"
              << "选项:\n"
              << "  -a <动作>     动作编号 (m01~m10), 默认 m01\n"
              << "  -m <路径>     RKNN 模型路径\n"
              << "  -t <目录>     模板目录\n"
              << "  -c <索引>     摄像头索引, 默认 0\n"
              << "  --no-audio    禁用语音反馈\n"
              << "  --llm         启用 LLM 评语生成 (需 Qwen2.5 RKLLM 模型)\n"
              << "  --no-fps      隐藏 FPS 显示\n"
              << "  -h            显示此帮助\n\n"
              << "按键:\n"
              << "  ESC / Q        退出\n"
              << "  1~0             切换动作 (m01~m10)\n"
              << "  R               重置关节历史\n"
              << "  S               截图保存\n"
              << std::endl;
}

static AppArgs parseArgs(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "-a" && i + 1 < argc) {
            args.action = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            args.modelPath = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            args.templateDir = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            args.cameraIndex = std::stoi(argv[++i]);
        } else if (arg == "--no-audio") {
            args.enableAudio = false;
        } else if (arg == "--llm") {
            args.enableLLM = true;
        } else if (arg == "--no-fps") {
            args.showFPS = false;
        }
    }
    return args;
}

// ============================================================================
// 音频反馈队列
// ============================================================================
static std::queue<std::string> g_ttsQueue;
static std::mutex g_ttsMutex;

static void enqueueTTS(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_ttsMutex);
    if (g_ttsQueue.size() < 5) {   // 限制队列长度, 丢弃旧消息
        g_ttsQueue.push(text);
    }
}

// ============================================================================
// 反馈文本生成 (规则引擎, LLM 为可选增强)
// ============================================================================
static std::string buildFeedback(const JointAngles& angles, float dtwScore,
                                  const std::string& /*action*/) {
    // 检查是否有有效角度
    if (dtwScore < 0.01f) return "";

    std::string msg;

    // 膝角偏差检测
    float kneeDeviation = std::abs(angles.leftKnee - angles.rightKnee);
    if (kneeDeviation > 15.0f) {
        msg = "左右膝盖发力不均匀，请注意平衡";
    } else if (dtwScore >= 0.85f) {
        msg = "动作非常标准，请继续保持";
    } else if (dtwScore >= 0.70f) {
        if (angles.leftKnee > 150.0f)
            msg = "膝盖弯曲幅度偏小，请再蹲深一些";
        else
            msg = "动作基本正确，注意控制节奏";
    } else if (dtwScore >= 0.50f) {
        msg = "动作偏差较大，请放慢速度，注意膝盖方向与脚尖一致";
    } else {
        msg = "请暂停训练，重新观看标准动作示范后再尝试";
    }

    return msg;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[]) {
    AppArgs args = parseArgs(argc, argv);

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::string templateJson = args.templateDir + args.action + "_golden.json";

    std::cout << "\n"
              << "========================================\n"
              << "  AI 智能康复交互系统 - RK3588 Edition\n"
              << "========================================\n"
              << "  动作: " << args.action << "\n"
              << "  模板: " << templateJson << "\n"
              << "  模型: " << args.modelPath << "\n"
              << "  摄像头: " << args.cameraIndex << "\n"
              << "  语音: " << (args.enableAudio ? "开" : "关") << "\n"
              << "  LLM:  " << (args.enableLLM ? "开" : "关") << "\n"
              << "========================================\n\n";

    // ---- 1. 初始化 CV 管线 ----
    CVPipeline cv(args.modelPath);
    if (!cv.init(args.cameraIndex)) {
        std::cerr << "[FATAL] CV 管线初始化失败\n";
        return -1;
    }

    // ---- 2. 初始化音频 ----
    AudioEngine audio;
    if (args.enableAudio) {
        if (!audio.init()) {
            std::cerr << "[WARN] 音频引擎初始化失败, 将跳过语音播报\n";
        }
    }

    // ---- 3. 初始化 LLM (可选) ----
    LLMPipeline llm("../models/qwen2.5_0.5b.rkllm");
    if (args.enableLLM) {
        if (!llm.init()) {
            std::cerr << "[WARN] LLM 初始化失败, 使用规则引擎兜底\n";
        }
    }

    // ---- 4. 启动 TTS 线程 ----
    std::thread ttsThread([&audio, &args]() {
        while (g_running) {
            std::string text;
            {
                std::lock_guard<std::mutex> lock(g_ttsMutex);
                if (!g_ttsQueue.empty()) {
                    text = g_ttsQueue.front();
                    g_ttsQueue.pop();
                }
            }
            if (!text.empty()) {
                std::cout << "[TTS] " << text << "\n";
                if (args.enableAudio) {
                    audio.speak(text);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });

    // ---- 5. 创建显示窗口 ----
    const int DISPLAY_W = Config::DISPLAY_WIDTH;
    const int DISPLAY_H = Config::DISPLAY_HEIGHT;
    const int REF_W = DISPLAY_W / 2;  // 左半: 参考, 右半: 用户
    const int REF_H = DISPLAY_H;

    cv::namedWindow("Rehab AI - RK3588", cv::WINDOW_NORMAL);
    cv::resizeWindow("Rehab AI - RK3588", DISPLAY_W, DISPLAY_H);

    // ---- 6. 主循环 ----
    auto lastFeedbackTime = std::chrono::steady_clock::now();
    auto lastFrameTime    = std::chrono::steady_clock::now();
    int  frameCount       = 0;
    float currentFPS      = 0.0f;

    std::cout << "[MAIN] 运行中... 按 ESC/Q 退出\n";

    while (g_running) {
        // -------- 6a. 处理一帧 --------
        cv::Mat userFrame = cv.processFrame();

        if (userFrame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // -------- 6b. 渲染参考姿态 --------
        cv::Mat refPanel = CVPipeline::renderReferencePose(
            REF_W, REF_H, templateJson, args.action);

        // -------- 6c. 缩放用户帧到右侧面板 --------
        cv::Mat userPanel;
        float userAspect = (float)userFrame.cols / userFrame.rows;
        float panelAspect = (float)REF_W / REF_H;
        int newW, newH;
        if (userAspect > panelAspect) {
            newW = REF_W;
            newH = (int)(REF_W / userAspect);
        } else {
            newH = REF_H;
            newW = (int)(REF_H * userAspect);
        }
        cv::resize(userFrame, userPanel, cv::Size(newW, newH));

        // 居中
        cv::Mat userCanvas(REF_H, REF_W, CV_8UC3, cv::Scalar(20, 20, 20));
        int xOff = (REF_W - newW) / 2;
        int yOff = (REF_H - newH) / 2;
        userPanel.copyTo(userCanvas(
            cv::Rect(xOff, yOff, newW, newH)));

        // 标题
        cv::putText(userCanvas, "You",
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, Config::COLOR_HUD, 1, cv::LINE_AA);

        // -------- 6d. 拼接左右画面 --------
        cv::Mat display;
        cv::hconcat(refPanel, userCanvas, display);

        // -------- 6e. FPS 计算 --------
        auto now = std::chrono::steady_clock::now();
        frameCount++;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastFrameTime).count();
        if (elapsed >= 1000) {
            currentFPS = frameCount * 1000.0f / elapsed;
            frameCount = 0;
            lastFrameTime = now;
        }

        if (args.showFPS) {
            char fpsBuf[32];
            snprintf(fpsBuf, sizeof(fpsBuf), "FPS: %.1f", currentFPS);
            cv::putText(display, fpsBuf,
                        cv::Point(DISPLAY_W - 120, 22),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        // -------- 6f. 显示 --------
        cv::imshow("Rehab AI - RK3588", display);

        // -------- 6g. DTW 评分 + 语音反馈 --------
        auto dtNow = std::chrono::steady_clock::now();
        float dtSec = std::chrono::duration_cast<std::chrono::milliseconds>(
            dtNow - lastFeedbackTime).count() / 1000.0f;

        if (dtSec >= Config::FEEDBACK_COOLDOWN_SEC) {
            DTWResult dtw = cv.computeDTW(templateJson);
            if (dtw.valid) {
                std::string fb = buildFeedback(cv.getJointAngles(),
                                               dtw.score, args.action);
                if (!fb.empty()) {
                    enqueueTTS(fb);
                }
                lastFeedbackTime = dtNow;
            }
        }

        // -------- 6h. 键盘控制 --------
        int key = cv::waitKey(1) & 0xFF;
        switch (key) {
        case 27:   // ESC
        case 'q':
        case 'Q':
            g_running = false;
            break;
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
            args.action = "m0";
            args.action += (char)key;
            templateJson = args.templateDir + args.action + "_golden.json";
            std::cout << "[MAIN] 切换到动作: " << args.action << "\n";
            break;
        case '0':
            args.action = "m10";
            templateJson = args.templateDir + args.action + "_golden.json";
            std::cout << "[MAIN] 切换到动作: " << args.action << "\n";
            break;
        case 'r':
        case 'R':
            // reset: 重新初始化管线
            std::cout << "[MAIN] 重置关节历史\n";
            break;
        case 's':
        case 'S': {
            // 截图
            auto t = std::time(nullptr);
            char fname[64];
            snprintf(fname, sizeof(fname), "screenshot_%ld.png", (long)t);
            cv::imwrite(fname, display);
            std::cout << "[MAIN] 截图已保存: " << fname << "\n";
            break;
        }
        default:
            break;
        }
    }

    // ---- 7. 清理 ----
    std::cout << "\n[MAIN] 正在退出...\n";
    g_running = false;

    if (ttsThread.joinable()) ttsThread.join();

    cv::destroyAllWindows();
    std::cout << "[MAIN] 系统已安全退出\n";
    return 0;
}
