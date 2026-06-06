#include <stdio.h>
#include <string.h>
#include "rknn_api.h"

int main() {
    printf("Testing rknn_init...\n");
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (char*)"./model/yolov8s-pose-int8.rknn", 0, 0, NULL);
    printf("rknn_init ret=%d ctx=%p\n", ret, (void*)ctx);
    if (ret >= 0) {
        rknn_input_output_num io_num;
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        printf("rknn_query IN_OUT_NUM ret=%d in=%d out=%d\n", ret, io_num.n_input, io_num.n_output);
        rknn_destroy(ctx);
    }
    return 0;
}
