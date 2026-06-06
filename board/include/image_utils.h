// Minimal image_utils.h stub — types and functions for image handling
#ifndef RK_IMAGE_UTILS_H_
#define RK_IMAGE_UTILS_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_NV12,
    IMAGE_FORMAT_NV21,
} image_format_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    int width;
    int height;
    image_format_t format;
    int size;
    unsigned char *virt_addr;
    int fd;
} image_buffer_t;

typedef struct {
    float scale;
    int x_pad;
    int y_pad;
} letterbox_t;

int get_image_size(image_buffer_t *img);
int convert_image_with_letterbox(image_buffer_t *src, image_buffer_t *dst,
                                 letterbox_t *letterbox, int bg_color);

#ifdef __cplusplus
}
#endif

#endif // RK_IMAGE_UTILS_H_
