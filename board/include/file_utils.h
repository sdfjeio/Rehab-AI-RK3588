// Minimal file_utils.h stub — file reading utilities
#ifndef RK_FILE_UTILS_H_
#define RK_FILE_UTILS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int read_data_from_file(const char *path, char **out_data, size_t *out_size);
int write_data_to_file(const char *path, const char *data, size_t size);

#ifdef __cplusplus
}
#endif

#endif // RK_FILE_UTILS_H_
