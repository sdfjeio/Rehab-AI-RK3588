// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov8-pose.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

#include <sys/time.h>

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolov8_pose_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov8_pose_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}


int inference_yolov8_pose_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        goto out;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        goto out;
    }
    // Set Input Data
    // For both INT8 and FP16: pass raw uint8 RGB. The NPU driver handles
    // uint8→int8 or uint8→float16 conversion internally, including any
    // mean/std normalization embedded during model conversion.
    inputs[0].type       = RKNN_TENSOR_UINT8;
    inputs[0].fmt        = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].size       = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf        = dst_img.virt_addr;

    // Debug: dump letterboxed input before NPU (always UINT8 now)
    {
        static int inp_dbg = 0;
        if (inp_dbg < 2) {
            unsigned char *dbg = (unsigned char *)inputs[0].buf;
            int mw = app_ctx->model_width;
            int mh = app_ctx->model_height;
            float sc = fminf((float)mw / (float)img->width, (float)mh / (float)img->height);
            int new_w = (int)(img->width * sc);
            int new_h = (int)(img->height * sc);
            int x_pad = (mw - new_w) / 2;
            int y_pad = (mh - new_h) / 2;
            fprintf(stderr, "[DEBUG input %d] model=%dx%d src=%dx%d scale=%.3f pad=[%d,%d]\n",
                    inp_dbg, mw, mh, img->width, img->height, sc, x_pad, y_pad);
            int img_start = (y_pad * mw + x_pad) * 3;
            fprintf(stderr, "[DEBUG input %d] top-pad@0: %02x %02x %02x  img@%d: %02x %02x %02x\n",
                    inp_dbg, dbg[0], dbg[1], dbg[2], img_start,
                    dbg[img_start], dbg[img_start+1], dbg[img_start+2]);
            int r_min=255,g_min=255,b_min=255,r_max=0,g_max=0,b_max=0;
            float r_sum=0,g_sum=0,b_sum=0;
            int ns = (new_w * new_h < 4096) ? new_w * new_h : 4096;
            for (int row = 0; row < ns / new_w && row < new_h; row++) {
                for (int col = 0; col < new_w && (row * new_w + col) < ns; col++) {
                    int off = ((row + y_pad) * mw + (col + x_pad)) * 3;
                    int r=dbg[off+0], gv=dbg[off+1], b=dbg[off+2];
                    if(r<r_min)r_min=r; if(r>r_max)r_max=r; r_sum+=r;
                    if(gv<g_min)g_min=gv; if(gv>g_max)g_max=gv; g_sum+=gv;
                    if(b<b_min)b_min=b; if(b>b_max)b_max=b; b_sum+=b;
                }
            }
            fprintf(stderr, "[DEBUG input %d] img RGB stats: R[%d,%d,%.0f] G[%d,%d,%.0f] B[%d,%d,%.0f]\n",
                    inp_dbg, r_min,r_max,r_sum/ns, g_min,g_max,g_sum/ns, b_min,b_max,b_sum/ns);
            inp_dbg++;
        }
    }

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    printf("rknn_run\n");
    int start_us,end_us;
    start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    end_us = getCurrentTimeUs() - start_us;
    printf("rknn_run time=%.2fms, FPS = %.2f\n",end_us / 1000.f, 
            1000.f * 1000.f / end_us);

    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    // Post Process
    start_us = getCurrentTimeUs();
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    end_us = getCurrentTimeUs() - start_us;
    printf("post_process time=%.2fms, FPS = %.2f\n",end_us / 1000.f, 
            1000.f * 1000.f / end_us);
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}