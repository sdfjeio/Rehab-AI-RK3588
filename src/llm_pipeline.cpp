// ============================================================================
// llm_pipeline.cpp — LLM 康复指导评语生成
// ============================================================================
// 支持两种模式:
//   1. RKLLM 推理 (Qwen2.5, 板端 NPU 加速)
//   2. 规则引擎兜底 (无需模型, 基于 DTW 分数分级)
// ============================================================================

#include "llm_pipeline.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef RK3588_DEPLOY
// #include "rkllm.h"
#endif

// ============================================================================
// PIMPL
// ============================================================================
struct LLMPipeline::Impl {
    std::string modelPath;
    bool        modelLoaded = false;

    // 规则引擎: 分级评语库
    static const char* levelExcellent;
    static const char* levelGood;
    static const char* levelFair;
    static const char* levelPoor;
    static const char* levelBad;
};

const char* LLMPipeline::Impl::levelExcellent = "动作非常标准，请继续保持！";
const char* LLMPipeline::Impl::levelGood   = "动作基本正确，注意控制节奏与呼吸。";
const char* LLMPipeline::Impl::levelFair   = "幅度偏小，请再蹲深一些，膝盖不要内扣。";
const char* LLMPipeline::Impl::levelPoor   = "动作偏差较大，放慢速度，膝盖方向与脚尖一致。";
const char* LLMPipeline::Impl::levelBad   = "请暂停训练，重新观看标准动作示范。";

// ============================================================================
// 构造 / 析构
// ============================================================================
LLMPipeline::LLMPipeline(const std::string& modelPath)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->modelPath = modelPath;
}

LLMPipeline::~LLMPipeline() = default;

// ============================================================================
// init
// ============================================================================
bool LLMPipeline::init() {
#ifdef RK3588_DEPLOY
    // 板端 RKLLM 加载 (需 rkllm.h + librkllm.so)
    // rkllm_param params;
    // rkllm_createDefaultParams(&params);
    // params.model_path = pImpl_->modelPath.c_str();
    // int ret = rkllm_init(&pImpl_->ctx, &params);
    // if (ret < 0) return false;
    // pImpl_->modelLoaded = true;
#endif
    pImpl_->modelLoaded = true;  // 规则引擎始终可用
    std::cout << "[LLM] 规则引擎就绪 (DTW 分级反馈)\n";
    return true;
}

// ============================================================================
// classify — 评分等级
// ============================================================================
static const char* classify(float score) {
    if (score >= 0.85f) return "优秀";
    if (score >= 0.70f) return "良好";
    if (score >= 0.55f) return "一般";
    if (score >= 0.40f) return "需改进";
    return "需大幅度改善";
}

// ============================================================================
// buildPrompt — 构建中文 Prompt
// ============================================================================
static std::string buildPrompt(float dtwScore, const std::string& action,
                                const std::string& deviations) {
    std::ostringstream oss;
    oss << "你是一位专业康复训练师。请根据以下数据给出一句简洁的纠正指导（≤40字）：\n"
        << "动作：" << action << "\n"
        << "规范度：" << classify(dtwScore)
        << " (DTW=" << static_cast<int>(dtwScore * 100) << "%)\n"
        << "关节偏差：" << deviations << "\n"
        << "指导：";
    return oss.str();
}

// ============================================================================
// generateFeedback — 生成评语
// ============================================================================
std::string LLMPipeline::generateFeedback(float dtwScore,
                                           const std::string& actionName,
                                           const std::string& jointDeviations) {
    std::string prompt = buildPrompt(dtwScore, actionName, jointDeviations);
    std::cout << "[LLM] " << prompt << "\n";

#ifdef RK3588_DEPLOY
    if (pImpl_->modelLoaded) {
        // rkllm_run(pImpl_->ctx, prompt.c_str(), &output);
        // return std::string(output);
    }
#endif

    // 规则兜底
    if (dtwScore >= 0.85f)      return Impl::levelExcellent;
    if (dtwScore >= 0.70f)      return Impl::levelGood;
    if (dtwScore >= 0.55f)      return Impl::levelFair;
    if (dtwScore >= 0.40f)      return Impl::levelPoor;
    return Impl::levelBad;
}

// ============================================================================
// generateFeedbackAsync
// ============================================================================
void LLMPipeline::generateFeedbackAsync(float dtwScore,
                                         const std::string& actionName,
                                         const std::string& jointDeviations,
                                         FeedbackCallback onDone) {
    std::thread([=]() {
        std::string result = generateFeedback(dtwScore, actionName,
                                               jointDeviations);
        if (onDone) onDone(result);
    }).detach();
}
