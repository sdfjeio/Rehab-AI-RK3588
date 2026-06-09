// Minimal DRM test with 32bpp (XRGB8888)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm.h>
#include <drm_mode.h>
#include "xf86drm.h"
#include "xf86drmMode.h"

int main() {
    int drm_fd = drmOpen(NULL, NULL);
    if (drm_fd < 0) {
        drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    }
    if (drm_fd < 0) { printf("FAIL: drmOpen\n"); return -1; }
    printf("DRM opened, fd=%d\n", drm_fd);

    uint32_t crtc_id = 0, conn_id = 0;
    drmModeModeInfo mode;
    int found = 0;
    drmModeRes *res = drmModeGetResources(drm_fd);
    if (!res) { printf("FAIL: resources\n"); return -1; }

    for (int i = 0; i < res->count_connectors && !found; i++) {
        drmModeConnector *conn = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            conn_id = conn->connector_id;
            if (conn->encoder_id) {
                drmModeEncoder *enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
                if (enc) { crtc_id = enc->crtc_id; drmModeFreeEncoder(enc); }
            }
            if (!crtc_id && res->count_crtcs > 0) crtc_id = res->crtcs[0];
            mode = conn->modes[0];
            printf("mode: %dx%d\n", mode.hdisplay, mode.vdisplay);
            found = 1;
        }
        drmModeFreeConnector(conn);
    }
    drmModeFreeResources(res);
    if (!found) { printf("FAIL: no display\n"); return -1; }

    int w = mode.hdisplay, h = mode.vdisplay;

    struct drm_mode_create_dumb dumb = {0};
    dumb.width  = w;
    dumb.height = h;
    dumb.bpp    = 32;
    if (ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &dumb) < 0) {
        printf("FAIL: create dumb buffer\n"); return -1;
    }
    printf("Buffer: %dx%d pitch=%d size=%llu\n", w, h, dumb.pitch, (unsigned long long)dumb.size);

    uint32_t fb_id;
    drmModeAddFB(drm_fd, w, h, 24, 32, dumb.pitch, dumb.handle, &fb_id);
    printf("FB id=%u\n", fb_id);

    struct drm_mode_map_dumb map = { .handle = dumb.handle };
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map);
    uint8_t *buf = (uint8_t *)mmap(0, dumb.size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, drm_fd, map.offset);

    // Fill with solid RED (XRGB little-endian: B=0, G=0, R=255, X=0)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t *p = buf + (y * dumb.pitch + x * 4);
            p[0] = 0; p[1] = 0; p[2] = 255; p[3] = 0;
        }
    }
    printf("Filled buffer with RED\n");

    int ret = drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);
    printf("drmModeSetCrtc RED ret=%d\n", ret);
    printf("Displaying RED for 5 seconds...\n");
    sleep(5);

    // Switch to GREEN
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t *p = buf + (y * dumb.pitch + x * 4);
            p[0] = 0; p[1] = 255; p[2] = 0; p[3] = 0;
        }
    }
    ret = drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);
    printf("Switched to GREEN, ret=%d. Wait 5s...\n", ret);
    sleep(5);

    // Switch to WHITE
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t *p = buf + (y * dumb.pitch + x * 4);
            p[0] = 255; p[1] = 255; p[2] = 255; p[3] = 0;
        }
    }
    ret = drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);
    printf("Switched to WHITE, ret=%d. Wait 5s...\n", ret);
    sleep(5);

    // Cleanup
    munmap(buf, dumb.size);
    drmModeRmFB(drm_fd, fb_id);
    struct drm_mode_destroy_dumb dd = { .handle = dumb.handle };
    ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
    drmClose(drm_fd);
    printf("Done.\n");
    return 0;
}
