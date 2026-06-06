# Makefile — AI Rehab System cross-compilation for RK3588
# Cross-compiler: aarch64-none-linux-gnu-gcc 10.3

CROSS_PREFIX := C:/msys64/home/bin/aarch64-none-linux-gnu
CC           := $(CROSS_PREFIX)-gcc
CXX          := $(CROSS_PREFIX)-g++
SYSROOT      := C:/msys64/home/aarch64-none-linux-gnu/libc

INCLUDES     := -Iinclude -I$(SYSROOT)/usr/include -I$(SYSROOT)/usr/include/drm
CXXFLAGS     := -std=c++17 -O3 -DNDEBUG -Wall -Wextra $(INCLUDES)
CFLAGS       := -std=c11 -O3 -DNDEBUG -Wall -Wextra $(INCLUDES)
LDFLAGS      := -L$(SYSROOT)/usr/lib -L$(SYSROOT)/lib \
                -lrknnrt -lrga -ldrm -lpthread -lstdc++ -lm

# Source files
CXX_SRCS := src/main_board.cc src/postprocess.cc src/yolov8-pose.cc
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
