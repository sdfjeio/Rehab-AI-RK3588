#pragma once

#include <memory>
#include <string>

// ==========================================
// 语音引擎: TTS 文本转语音播放
// ==========================================
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    // 初始化 (检测可用 TTS 后端: espeak / festival / gTTS)
    bool init();

    // 同步播放 (阻塞)
    void speak(const std::string& text);

    // 异步播放 (非阻塞, 独立线程)
    void speakAsync(const std::string& text);

    // 是否正在播放
    bool isSpeaking() const;

    // 停止当前播放
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
