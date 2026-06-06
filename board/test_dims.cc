#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rknn_api.h"

int main() {
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (char*)"./model/yolov8s-pose-int8.rknn", 0, 0, NULL);
    if (ret < 0) { printf("rknn_init fail\n"); return -1; }

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("in=%d out=%d\n", io_num.n_input, io_num.n_output);

    for (int i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        printf("INPUT[%d]: n_dims=%d fmt=%d type=%d qnt=%d zp=%d scale=%f\n",
               i, attr.n_dims, attr.fmt, attr.type, attr.qnt_type, attr.zp, attr.scale);
        printf("  dims: [%d", attr.dims[0]);
        for (int d = 1; d < attr.n_dims; d++) printf(", %d", attr.dims[d]);
        printf("]\n");
    }

    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("OUTPUT[%d]: n_dims=%d fmt=%d type=%d qnt=%d zp=%d scale=%f\n",
               i, attr.n_dims, attr.fmt, attr.type, attr.qnt_type, attr.zp, attr.scale);
        printf("  dims: [%d", attr.dims[0]);
        for (int d = 1; d < attr.n_dims; d++) printf(", %d", attr.dims[d]);
        printf("]\n");
    }

    rknn_destroy(ctx);
    return 0;
}
