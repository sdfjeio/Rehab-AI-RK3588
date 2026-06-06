// file_utils.c — minimal file I/O utilities
#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>

int read_data_from_file(const char *path, char **out_data, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return -1; }
    *out_data = (char *)malloc(sz);
    if (!*out_data) { fclose(fp); return -1; }
    size_t n = fread(*out_data, 1, sz, fp);
    fclose(fp);
    *out_size = n;
    return 0;
}

int write_data_to_file(const char *path, const char *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t n = fwrite(data, 1, size, fp);
    fclose(fp);
    return (n == size) ? 0 : -1;
}
