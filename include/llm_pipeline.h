#pragma once

#include <functional>
#include <memory>
#include <string>

// ==========================================
// LLM 管线: Qwen2.5 RKLLM → 语音指导文本
// ==========================================
class LLMPipeline {
public:
    LLMPipeline(const std::string& modelPath);
    ~LLMPipeline();

    // 加载 RKLLM 模型
    bool init();

    // 根据 DTW 评分 + 动作名 + 偏差信息 生成评语
    // dtwScore:   当前动作匹配度 (0.0~1.0, 越高越好)
    // actionName: 动作名 (如 "深蹲")
    // jointDeviations: 各关节偏差文本 (如 "左膝屈曲不足")
    std::string generateFeedback(
        float dtwScore,
        const std::string& actionName,
        const std::string& jointDeviations
    );

    // 异步生成 (回调模式, 避免阻塞主线程)
    using FeedbackCallback = std::function<void(const std::string&)>;
    void generateFeedbackAsync(
        float dtwScore,
        const std::string& actionName,
        const std::string& jointDeviations,
        FeedbackCallback onDone
    );

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
