cmake_minimum_required(VERSION 3.14)
project(Rehab_AI_RK3588 VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# 构建模式
# ============================================================================
# 桌面模式: cmake -B build && cmake --build build
# 板端模式: cmake -B build -DRK3588=ON -DCMAKE_TOOLCHAIN_FILE=...
# ============================================================================
option(RK3588 "Build for RK3588 NPU (links librknnrt.so)" OFF)

# ============================================================================
# 交叉编译工具链 (板端部署时设置)
# ============================================================================
if(RK3588)
    add_definitions(-DRK3588_DEPLOY)

    # 取消注释并按实际 SDK 路径修改:
    # set(CMAKE_C_COMPILER  aarch64-linux-gnu-gcc)
    # set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    # set(CMAKE_SYSROOT /path/to/rk3588/sysroot)
    # set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    # set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    # set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    # set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

# ============================================================================
# 第三方头文件
# ============================================================================
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

# ============================================================================
# OpenCV (必需)
# ============================================================================
find_package(OpenCV REQUIRED)

# ============================================================================
# RKNN / RKLLM SDK (板端)
# ============================================================================
if(RK3588)
    # 按实际 SDK 路径修改:
    # set(RKNN_API_DIR ${CMAKE_SOURCE_DIR}/sdk/rknn/include)
    # set(RKNN_RT_LIB  ${CMAKE_SOURCE_DIR}/sdk/rknn/lib/librknnrt.so)
    # set(RKLLM_API_DIR ${CMAKE_SOURCE_DIR}/sdk/rkllm/include)
    # set(RKLLM_LIB     ${CMAKE_SOURCE_DIR}/sdk/rkllm/lib/librkllm.so)

    # 如 SDK 已安装到系统路径:
    # set(RKNN_RT_LIB  /usr/lib/aarch64-linux-gnu/librknnrt.so)
    # set(RKLLM_LIB    /usr/lib/aarch64-linux-gnu/librkllm.so)
endif()

# ============================================================================
# 源文件
# ============================================================================
set(SOURCES
    src/main.cpp
    src/cv_pipeline.cpp
    src/audio_engine.cpp
    src/llm_pipeline.cpp
)

# ============================================================================
# 可执行文件
# ============================================================================
add_executable(rehab_app ${SOURCES})

# ============================================================================
# 头文件搜索路径
# ============================================================================
target_include_directories(rehab_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${THIRDPARTY_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
if(RK3588)
    # target_include_directories(rehab_app PRIVATE ${RKNN_API_DIR})
    # target_include_directories(rehab_app PRIVATE ${RKLLM_API_DIR})
endif()

# ============================================================================
# 链接库
# ============================================================================
target_link_libraries(rehab_app PRIVATE
    ${OpenCV_LIBS}
    pthread
)
if(RK3588)
    # target_link_libraries(rehab_app PRIVATE ${RKNN_RT_LIB})
    # target_link_libraries(rehab_app PRIVATE ${RKLLM_LIB})
endif()

# ============================================================================
# 编译选项
# ============================================================================
target_compile_options(rehab_app PRIVATE
    -Wall -Wextra
    -O3 -DNDEBUG
)

# 板端 CPU 优化 (RK3588: 4×A76 + 4×A55)
if(RK3588)
    target_compile_options(rehab_app PRIVATE
        -march=armv8.2-a
        -mtune=cortex-a76.cortex-a55
    )
endif()

# ============================================================================
# 测试程序 (桌面模式冒烟测试)
# ============================================================================
add_executable(test_init
    test/test_init.cpp
    src/cv_pipeline.cpp
    src/audio_engine.cpp
    src/llm_pipeline.cpp
)

target_include_directories(test_init PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${THIRDPARTY_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_init PRIVATE
    ${OpenCV_LIBS}
    pthread
)

target_compile_options(test_init PRIVATE
    -Wall -Wextra
    -O2
)

# ============================================================================
# 安装规则 (板端部署)
# ============================================================================
if(RK3588)
    install(TARGETS rehab_app DESTINATION /home/user/rehab/bin)
    install(DIRECTORY models/  DESTINATION /home/user/rehab/models
            FILES_MATCHING PATTERN "*.rknn")
    install(DIRECTORY templates/ DESTINATION /home/user/rehab/templates
            FILES_MATCHING PATTERN "*.json")
endif()

# ============================================================================
# 调试信息
# ============================================================================
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "RK3588 mode: ${RK3588}")
message(STATUS "OpenCV found: ${OpenCV_FOUND}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
