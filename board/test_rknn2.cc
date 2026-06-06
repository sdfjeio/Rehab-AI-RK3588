#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "rknn_api.h"

int main() {
    fprintf(stderr, "[TEST] rknn_init...\n");
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (char*)"./model/yolov8s-pose-int8.rknn", 0, 0, NULL);
    fprintf(stderr, "[TEST] rknn_init ret=%d ctx=%p\n", ret, (void*)ctx);
    if (ret < 0) return -1;

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    fprintf(stderr, "[TEST] IN_OUT_NUM ret=%d in=%d out=%d\n", ret, io_num.n_input, io_num.n_output);

    fprintf(stderr, "[TEST] querying input attrs...\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        fprintf(stderr, "[TEST] input[%d] ret=%d name=%s dims=%d fmt=%d type=%d\n",
                i, ret, input_attrs[i].name, input_attrs[i].n_dims, input_attrs[i].fmt, input_attrs[i].type);
    }

    fprintf(stderr, "[TEST] querying output attrs...\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        fprintf(stderr, "[TEST] output[%d] ret=%d name=%s dims=%d fmt=%d type=%d\n",
                i, ret, output_attrs[i].name, output_attrs[i].n_dims, output_attrs[i].fmt, output_attrs[i].type);
    }

    fprintf(stderr, "[TEST] sleeping 3 seconds to see if background threads crash...\n");
    sleep(3);
    fprintf(stderr, "[TEST] survived sleep. destroy ctx...\n");
    rknn_destroy(ctx);
    fprintf(stderr, "[TEST] done.\n");
    return 0;
}
