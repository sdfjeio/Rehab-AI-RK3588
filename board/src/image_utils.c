// image_utils.c — minimal implementation for Rockchip SDK compatibility
#include "image_utils.h"
#include <string.h>
#include <math.h>

int get_image_size(image_buffer_t *img) {
    int bpp;
    switch (img->format) {
        case IMAGE_FORMAT_GRAY8:   bpp = 1; break;
        case IMAGE_FORMAT_RGB888:  bpp = 3; break;
        case IMAGE_FORMAT_RGBA8888: bpp = 4; break;
        case IMAGE_FORMAT_NV12:
        case IMAGE_FORMAT_NV21:    bpp = 1; break;
        default:                   bpp = 3; break;
    }
    return img->width * img->height * bpp;
}

int convert_image_with_letterbox(image_buffer_t *src, image_buffer_t *dst,
                                 letterbox_t *letterbox, int bg_color) {
    float scale = fminf((float)dst->width / src->width,
                         (float)dst->height / src->height);
    int new_w = (int)(src->width * scale);
    int new_h = (int)(src->height * scale);
    int x_pad = (dst->width - new_w) / 2;
    int y_pad = (dst->height - new_h) / 2;

    // Fill with background color
    unsigned char bg_r = (bg_color >> 16) & 0xFF;
    unsigned char bg_g = (bg_color >> 8) & 0xFF;
    unsigned char bg_b = bg_color & 0xFF;
    for (int i = 0; i < (int)dst->size; i += 3) {
        dst->virt_addr[i]     = bg_r;
        dst->virt_addr[i + 1] = bg_g;
        dst->virt_addr[i + 2] = bg_b;
    }

    // Nearest-neighbor resize
    int src_bpp = (src->format == IMAGE_FORMAT_RGB888) ? 3 : 3;
    for (int y = 0; y < new_h; y++) {
        int src_y = y * src->height / new_h;
        for (int x = 0; x < new_w; x++) {
            int src_x = x * src->width / new_w;
            int dst_idx = ((y + y_pad) * dst->width + (x + x_pad)) * src_bpp;
            int src_idx = (src_y * src->width + src_x) * src_bpp;
            for (int c = 0; c < src_bpp; c++) {
                dst->virt_addr[dst_idx + c] = src->virt_addr[src_idx + c];
            }
        }
    }

    letterbox->scale  = scale;
    letterbox->x_pad  = x_pad;
    letterbox->y_pad  = y_pad;
    return 0;
}
