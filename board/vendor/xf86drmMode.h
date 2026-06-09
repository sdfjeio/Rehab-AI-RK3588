// Minimal xf86drmMode.h stub — provides DRM mode-setting API prototypes
// Real implementation is in the target's libdrm.so
#ifndef _XF86DRMMODE_H_
#define _XF86DRMMODE_H_

#include <stdint.h>
#include <drm.h>
#include <drm_mode.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _drmModeRes {
    int count_fbs;
    uint32_t *fbs;
    int count_crtcs;
    uint32_t *crtcs;
    int count_connectors;
    uint32_t *connectors;
    int count_encoders;
    uint32_t *encoders;
    uint32_t min_width, max_width;
    uint32_t min_height, max_height;
} drmModeRes, *drmModeResPtr;

typedef struct _drmModeModeInfo {
    uint32_t clock;
    uint16_t hdisplay, hsync_start, hsync_end, htotal, hskew;
    uint16_t vdisplay, vsync_start, vsync_end, vtotal, vscan;
    uint32_t vrefresh;
    uint32_t flags;
    uint32_t type;
    char     name[32];
} drmModeModeInfo, *drmModeModeInfoPtr;

typedef struct _drmModeConnector {
    uint32_t connector_id;
    uint32_t encoder_id;
    uint32_t connector_type;
    uint32_t connector_type_id;
    uint32_t connection;
    uint32_t mmWidth, mmHeight;
    uint32_t subpixel;
    int count_modes;
    drmModeModeInfoPtr modes;
    int count_props;
    uint32_t *props;
    uint64_t *prop_values;
    int count_encoders;
    uint32_t *encoders;
} drmModeConnector, *drmModeConnectorPtr;

typedef struct _drmModeEncoder {
    uint32_t encoder_id;
    uint32_t encoder_type;
    uint32_t crtc_id;
    uint32_t possible_crtcs;
    uint32_t possible_clones;
} drmModeEncoder, *drmModeEncoderPtr;

#define DRM_MODE_CONNECTED        1
#define DRM_MODE_DISCONNECTED     2
#define DRM_MODE_UNKNOWNCONNECTION 3

extern drmModeResPtr      drmModeGetResources(int fd);
extern drmModeConnectorPtr drmModeGetConnector(int fd, uint32_t connectorId);
extern void                drmModeFreeConnector(drmModeConnectorPtr ptr);
extern drmModeEncoderPtr   drmModeGetEncoder(int fd, uint32_t encoder_id);
extern void                drmModeFreeEncoder(drmModeEncoderPtr ptr);
extern void                drmModeFreeResources(drmModeResPtr ptr);
extern int                 drmModeAddFB(int fd, uint32_t width, uint32_t height,
                                        uint8_t depth, uint8_t bpp,
                                        uint32_t pitch, uint32_t bo_handle,
                                        uint32_t *buf_id);
extern int                 drmModeRmFB(int fd, uint32_t bufferId);
extern int                 drmModeSetCrtc(int fd, uint32_t crtcId, uint32_t bufferId,
                                         uint32_t x, uint32_t y, uint32_t *connectors,
                                         int count, drmModeModeInfoPtr mode);

#ifdef __cplusplus
}
#endif

#endif
