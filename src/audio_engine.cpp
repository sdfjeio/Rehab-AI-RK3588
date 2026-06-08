// ============================================================================
// audio_engine.cpp — TTS 语音引擎实现
// ============================================================================
// 支持多种 TTS 后端, 按优先级探测:
//   1. espeak-ng (推荐, 中文支持好)
//   2. espeak
//   3. festival
//   4. gtts-cli (需联网)
// ============================================================================

#include "audio_engine.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>

// ============================================================================
// PIMPL 内部实现
// ============================================================================
struct AudioEngine::Impl {
    bool        ready     = false;
    std::string backend;         // "espeak-ng" / "espeak" / "festival" / "gtts-cli"
    std::string voiceParam;      // 语音参数 (如 "-v zh" 或 "-v mandarin")
    bool        asyncMode = false;
};

// ============================================================================
// 构造 / 析构
// ============================================================================
AudioEngine::AudioEngine()
    : pImpl_(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() {
    stop();
}

// ============================================================================
// detectBackend — 检测可用 TTS 后端
// ============================================================================
static bool checkCommand(const char* cmd) {
    char buf[128];
#ifdef _WIN32
    // Windows: use 'where' command, suppress output
    snprintf(buf, sizeof(buf), "where %s > nul 2>&1", cmd);
#else
    snprintf(buf, sizeof(buf), "which %s > /dev/null 2>&1", cmd);
#endif
    return std::system(buf) == 0;
}

bool AudioEngine::init() {
    if (checkCommand("espeak-ng")) {
        pImpl_->backend    = "espeak-ng";
        pImpl_->voiceParam = "-v zh";
        pImpl_->ready      = true;
    } else if (checkCommand("espeak")) {
        pImpl_->backend    = "espeak";
        pImpl_->voiceParam = "-v zh";
        pImpl_->ready      = true;
    } else if (checkCommand("festival")) {
        pImpl_->backend = "festival";
        pImpl_->ready   = true;
    } else if (checkCommand("gtts-cli")) {
        pImpl_->backend = "gtts-cli";
        pImpl_->ready   = true;
    }

    if (pImpl_->ready) {
        std::cout << "[TTS] 后端就绪: " << pImpl_->backend << "\n";
    } else {
        std::cerr << "[TTS] 未检测到 TTS 后端\n"
                  << "      板端安装: sudo apt install espeak-ng\n";
    }

    return pImpl_->ready;
}

// ============================================================================
// escapeText — 转义特殊字符防止 shell 注入
// ============================================================================
static std::string escapeText(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 4);
    for (char c : text) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

// ============================================================================
// speak — 同步播放
// ============================================================================
void AudioEngine::speak(const std::string& text) {
    if (!pImpl_->ready || text.empty()) return;

    std::string safe = escapeText(text);
    char cmd[1024];

    if (pImpl_->backend == "espeak-ng" || pImpl_->backend == "espeak") {
        snprintf(cmd, sizeof(cmd), "echo \"%s\" | %s %s",
                 safe.c_str(), pImpl_->backend.c_str(),
                 pImpl_->voiceParam.c_str());
    } else if (pImpl_->backend == "festival") {
        snprintf(cmd, sizeof(cmd), "echo \"%s\" | festival --tts",
                 safe.c_str());
    } else if (pImpl_->backend == "gtts-cli") {
        snprintf(cmd, sizeof(cmd),
                 "gtts-cli \"%s\" -l zh-cn -o /tmp/tts.mp3 && "
                 "ffplay -nodisp -autoexit /tmp/tts.mp3 2>/dev/null",
                 safe.c_str());
    } else {
        return;
    }

    std::system(cmd);
}

// ============================================================================
// speakAsync — 异步播放
// ============================================================================
void AudioEngine::speakAsync(const std::string& text) {
    if (!pImpl_->ready || text.empty()) return;

    // 防止 TTS 进程堆积
    stop();

    std::string safe = escapeText(text);
    char cmd[1024];

    if (pImpl_->backend == "espeak-ng" || pImpl_->backend == "espeak") {
        snprintf(cmd, sizeof(cmd), "echo \"%s\" | %s %s &",
                 safe.c_str(), pImpl_->backend.c_str(),
                 pImpl_->voiceParam.c_str());
    } else if (pImpl_->backend == "festival") {
        snprintf(cmd, sizeof(cmd), "echo \"%s\" | festival --tts &",
                 safe.c_str());
    } else {
        return;
    }

    std::system(cmd);
}

// ============================================================================
// isSpeaking
// ============================================================================
bool AudioEngine::isSpeaking() const {
    // espeak 默认同步, speak() 返回即播完
    return false;
}

// ============================================================================
// stop
// ============================================================================
void AudioEngine::stop() {
    // 终止所有可能的 TTS 进程
    std::system("pkill -9 espeak 2>/dev/null");
    std::system("pkill -9 espeak-ng 2>/dev/null");
    std::system("pkill -9 festival 2>/dev/null");
    std::system("pkill -9 ffplay 2>/dev/null");
}
