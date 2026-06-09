# Makefile — AI Rehab System native build for RK3588 (Ubuntu 22.04 arm64)
#   $ sudo apt install build-essential librga-dev libdrm-dev librknnrt-dev
#   $ make          # build
#   $ make clean    # remove build artifacts

CXX := g++
CC  := gcc

# Include paths: project headers, vendor headers, system headers
INCLUDES := -Iinclude -Ivendor -I/usr/include/libdrm -I/usr/include/rga
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -Wall $(INCLUDES)
CFLAGS   := -std=c11   -O3 -DNDEBUG -Wall $(INCLUDES)
LDFLAGS  := -lrknnrt -lrga -ldrm -lpthread -lstdc++ -lm

CXX_SRCS := src/main_board.cc src/postprocess.cc src/yolov8-pose.cc src/ui.cpp
C_SRCS   := src/image_utils.c src/file_utils.c
OBJS     := $(CXX_SRCS:.cc=.o) $(C_SRCS:.c=.o)
TARGET   := rehab_app

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
