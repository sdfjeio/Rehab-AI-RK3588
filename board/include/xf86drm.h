// Minimal xf86drm.h stub — provides DRM userspace API prototypes
// Real implementation is in the target's libdrm.so
#ifndef _XF86DRM_H_
#define _XF86DRM_H_

#include <stdint.h>
#include <drm.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int drmOpen(const char *name, const char *busid);
extern int drmClose(int fd);
extern int drmPrimeHandleToFD(int fd, uint32_t handle, uint32_t flags, int *prime_fd);
extern int drmIoctl(int fd, unsigned long request, void *arg);

#ifdef __cplusplus
}
#endif

#endif
