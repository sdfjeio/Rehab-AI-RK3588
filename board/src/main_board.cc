/*
 * main_board.cc — AI Rehabilitation System for RK3588
 *
 * Hardware pipeline:
 *   V4L2 Camera (NV12) → RGA (NV12→RGB888) → DRM Buffer → NPU Inference
 *   → CPU Skeleton + HUD Drawing → DRM Display
 *
 * Rehab features:
 *   - Color-coded COCO skeleton (left=cyan, right=magenta, center=green)
 *   - Joint angle computation (knee, elbow, hip, shoulder)
 *   - Real-time HUD overlay (FPS, angles, DTW score)
 *   - Keypoint confidence visualization
 *   - DTW action quality scoring vs golden templates
 *   - TTS audio feedback via espeak-ng
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <linux/videodev2.h>
#include <drm.h>
#include <drm_mode.h>
#include "xf86drm.h"
#include "xf86drmMode.h"
#include "im2d.h"
#include "RgaApi.h"
#include "rknn_api.h"
#include "yolov8-pose.h"
#include "ui.h"

/* =========================================================================
 * Hardware Parameters
 * ========================================================================= */
#define CAM_DEV         "/dev/video11"
#define CAM_WIDTH       640
#define CAM_HEIGHT      480
#define CAM_FORMAT      V4L2_PIX_FMT_NV12
#define BUFFER_COUNT    4
#define MODEL_BUF_WIDTH  640
#define MODEL_BUF_HEIGHT 480
#define DISP_FMT        RK_FORMAT_RGB_888
#define MODEL_PATH      "./model/yolov8s-pose.rknn"

/* =========================================================================
 * Rehab Parameters
 * ========================================================================= */
#define MAX_DTW_FRAMES      300
#define NUM_ACTIONS         10
#define MAX_FEEDBACK_LEN    256
#define ANGLE_SMOOTH_WINDOW 3

/* =========================================================================
 * COCO Skeleton — 17 keypoints, 19 bones
 * ========================================================================= */
static const int SKELETON[19][2] = {
    {15,13}, {13,11}, {16,14}, {14,12}, {11,12}, {5,11}, {6,12}, {5,6},
    {5,7},  {6,8},  {7,9},  {8,10},  {1,2}, {0,1}, {0,2}, {1,3},
    {2,4},  {3,5},  {4,6}
};

// Which side each bone belongs to: 0=center, 1=left, 2=right
static const int BONE_SIDE[19] = {
    2,2, 1,1, 0, 1,2, 0,
    1,2, 1,2, 0,0,0, 1,2, 1,2
};

/* =========================================================================
 * Colors (RGB)
 * ========================================================================= */
static const Color COLOR_LEFT    = {255, 128,   0};  // cyan-ish
static const Color COLOR_RIGHT   = {255,   0, 255};  // magenta
static const Color COLOR_CENTER  = {  0, 255,   0};  // green
static const Color COLOR_FACE    = {  0, 255, 255};  // yellow-cyan
static const Color COLOR_HUD     = {  0, 255,   0};  // green HUD
static const Color COLOR_ALERT   = {  0,   0, 255};  // red alert
static const Color COLOR_KPT_HI  = {  0, 255,   0};  // high confidence
static const Color COLOR_KPT_MED = {  0, 200, 255};  // medium confidence
static const Color COLOR_KPT_LO  = {  0,   0, 255};  // low confidence
static const Color COLOR_BG      = {  0,   0,   0};  // background

/* =========================================================================
 * 5x7 Bitmap Font — compact lookup for printable ASCII
 * ========================================================================= */
static const unsigned char glyph_space[5]    = {0x00,0x00,0x00,0x00,0x00};
static const unsigned char glyph_excl[5]     = {0x00,0x00,0x5F,0x00,0x00};
static const unsigned char glyph_dash[5]     = {0x00,0x00,0x06,0x00,0x00};
static const unsigned char glyph_dot[5]      = {0x00,0x00,0x60,0x00,0x00};
static const unsigned char glyph_slash[5]    = {0x20,0x10,0x08,0x04,0x02};
static const unsigned char glyph_0[5]        = {0x3E,0x51,0x49,0x45,0x3E};
static const unsigned char glyph_1[5]        = {0x00,0x42,0x7F,0x40,0x00};
static const unsigned char glyph_2[5]        = {0x42,0x61,0x51,0x49,0x46};
static const unsigned char glyph_3[5]        = {0x21,0x41,0x45,0x4B,0x31};
static const unsigned char glyph_4[5]        = {0x18,0x14,0x12,0x7F,0x10};
static const unsigned char glyph_5[5]        = {0x27,0x45,0x45,0x45,0x39};
static const unsigned char glyph_6[5]        = {0x3C,0x4A,0x49,0x49,0x30};
static const unsigned char glyph_7[5]        = {0x01,0x71,0x09,0x05,0x03};
static const unsigned char glyph_8[5]        = {0x36,0x49,0x49,0x49,0x36};
static const unsigned char glyph_9[5]        = {0x06,0x49,0x49,0x29,0x1E};
static const unsigned char glyph_colon[5]    = {0x00,0x36,0x36,0x00,0x00};
static const unsigned char glyph_A[5]        = {0x7E,0x11,0x11,0x11,0x7E};
static const unsigned char glyph_D[5]        = {0x7F,0x41,0x41,0x22,0x1C};
static const unsigned char glyph_E[5]        = {0x7F,0x49,0x49,0x49,0x41};
static const unsigned char glyph_F[5]        = {0x7F,0x09,0x09,0x09,0x01};
static const unsigned char glyph_H[5]        = {0x7F,0x08,0x08,0x08,0x7F};
static const unsigned char glyph_K[5]        = {0x7F,0x08,0x14,0x22,0x41};
static const unsigned char glyph_L[5]        = {0x7F,0x40,0x40,0x40,0x40};
static const unsigned char glyph_M[5]        = {0x7F,0x02,0x0C,0x02,0x7F};
static const unsigned char glyph_N[5]        = {0x7F,0x04,0x08,0x10,0x7F};
static const unsigned char glyph_P[5]        = {0x7F,0x09,0x09,0x09,0x06};
static const unsigned char glyph_R[5]        = {0x7F,0x09,0x19,0x29,0x46};
static const unsigned char glyph_S[5]        = {0x46,0x49,0x49,0x49,0x31};
static const unsigned char glyph_T[5]        = {0x01,0x01,0x7F,0x01,0x01};
static const unsigned char glyph_U[5]        = {0x3F,0x40,0x40,0x40,0x3F};
static const unsigned char glyph_W[5]        = {0x3F,0x40,0x38,0x40,0x3F};
static const unsigned char glyph_a[5]        = {0x20,0x54,0x54,0x54,0x78};
static const unsigned char glyph_c[5]        = {0x38,0x44,0x44,0x44,0x20};
static const unsigned char glyph_d[5]        = {0x38,0x44,0x44,0x48,0x7F};
static const unsigned char glyph_e[5]        = {0x38,0x54,0x54,0x54,0x18};
static const unsigned char glyph_g[5]        = {0x08,0x54,0x54,0x54,0x3C};
static const unsigned char glyph_i[5]        = {0x00,0x44,0x7D,0x40,0x00};
static const unsigned char glyph_k[5]        = {0x00,0x44,0x7D,0x40,0x3F};
static const unsigned char glyph_l[5]        = {0x00,0x41,0x7F,0x40,0x00};
static const unsigned char glyph_m[5]        = {0x7C,0x04,0x18,0x04,0x78};
static const unsigned char glyph_n[5]        = {0x7C,0x08,0x04,0x04,0x78};
static const unsigned char glyph_o[5]        = {0x38,0x44,0x44,0x44,0x38};
static const unsigned char glyph_p[5]        = {0x7C,0x14,0x14,0x14,0x08};
static const unsigned char glyph_r[5]        = {0x7C,0x08,0x04,0x04,0x08};
static const unsigned char glyph_s[5]        = {0x48,0x54,0x54,0x54,0x20};
static const unsigned char glyph_t[5]        = {0x04,0x3F,0x44,0x40,0x20};
static const unsigned char glyph_u[5]        = {0x3C,0x40,0x40,0x20,0x7C};
static const unsigned char glyph_w[5]        = {0x1C,0x20,0x18,0x20,0x1C};
static const unsigned char glyph_x[5]        = {0x44,0x28,0x10,0x28,0x44};

static const unsigned char glyph_B[5]        = {0x1F,0x11,0x1E,0x11,0x1F};
static const unsigned char glyph_C[5]        = {0x0E,0x11,0x10,0x11,0x0E};
static const unsigned char glyph_G[5]        = {0x0E,0x11,0x13,0x11,0x0E};
static const unsigned char glyph_I[5]        = {0x1F,0x04,0x04,0x04,0x1F};
static const unsigned char glyph_J[5]        = {0x0F,0x02,0x02,0x12,0x0C};
static const unsigned char glyph_O[5]        = {0x0E,0x11,0x11,0x11,0x0E};
static const unsigned char glyph_Q[5]        = {0x0E,0x11,0x15,0x12,0x0D};
static const unsigned char glyph_V[5]        = {0x11,0x11,0x11,0x0A,0x04};
static const unsigned char glyph_X[5]        = {0x11,0x0A,0x04,0x0A,0x11};
static const unsigned char glyph_Y[5]        = {0x11,0x0A,0x04,0x04,0x04};
static const unsigned char glyph_Z[5]        = {0x1F,0x02,0x04,0x08,0x1F};
static const unsigned char glyph_b[5]        = {0x10,0x10,0x1E,0x11,0x1E};
static const unsigned char glyph_f[5]        = {0x0E,0x08,0x1C,0x08,0x08};
static const unsigned char glyph_h[5]        = {0x10,0x10,0x1E,0x11,0x11};
static const unsigned char glyph_j[5]        = {0x04,0x00,0x06,0x04,0x14};
static const unsigned char glyph_q[5]        = {0x0E,0x11,0x11,0x1E,0x01};
static const unsigned char glyph_v[5]        = {0x11,0x11,0x0A,0x0A,0x04};
static const unsigned char glyph_y[5]        = {0x11,0x11,0x0A,0x04,0x08};
static const unsigned char glyph_z[5]        = {0x1F,0x02,0x04,0x08,0x1F};
static const unsigned char glyph_quote[5]    = {0x00,0x00,0x00,0x00,0x00}; // ' reserved
static const unsigned char *get_glyph(char ch) {
    switch (ch) {
        case ' ': return glyph_space; case '!': return glyph_excl;
        case '-': return glyph_dash;  case '.': return glyph_dot;
        case '/': return glyph_slash; case '0': return glyph_0;
        case '1': return glyph_1;     case '2': return glyph_2;
        case '3': return glyph_3;     case '4': return glyph_4;
        case '5': return glyph_5;     case '6': return glyph_6;
        case '7': return glyph_7;     case '8': return glyph_8;
        case '9': return glyph_9;     case ':': return glyph_colon;
        case 'A': return glyph_A;     case 'D': return glyph_D;
        case 'E': return glyph_E;     case 'F': return glyph_F;
        case 'H': return glyph_H;     case 'K': return glyph_K;
        case 'L': return glyph_L;     case 'M': return glyph_M;
        case 'N': return glyph_N;     case 'P': return glyph_P;
        case 'R': return glyph_R;     case 'S': return glyph_S;
        case 'T': return glyph_T;     case 'U': return glyph_U;
        case 'W': return glyph_W;     case 'a': return glyph_a;
        case 'c': return glyph_c;     case 'd': return glyph_d;
        case 'e': return glyph_e;     case 'g': return glyph_g;
        case 'i': return glyph_i;     case 'k': return glyph_k;
        case 'l': return glyph_l;     case 'm': return glyph_m;
        case 'n': return glyph_n;     case 'o': return glyph_o;
        case 'p': return glyph_p;     case 'r': return glyph_r;
        case 's': return glyph_s;     case 't': return glyph_t;
        case 'u': return glyph_u;     case 'w': return glyph_w;
        case 'x': return glyph_x;     case 'y': return glyph_y;
        case 'z': return glyph_z;     case 'B': return glyph_B;
        case 'C': return glyph_C;     case 'G': return glyph_G;
        case 'I': return glyph_I;     case 'J': return glyph_J;
        case 'O': return glyph_O;     case 'Q': return glyph_Q;
        case 'V': return glyph_V;     case 'X': return glyph_X;
        case 'Y': return glyph_Y;     case 'Z': return glyph_Z;
        case 'b': return glyph_b;     case 'f': return glyph_f;
        case 'h': return glyph_h;     case 'j': return glyph_j;
        case 'q': return glyph_q;     case 'v': return glyph_v;
        default:  return glyph_space;
    }
}

/* =========================================================================
 * Drawing Primitives (CPU, on DRM RGB888 buffer)
 * ========================================================================= */
uint8_t *g_buf = NULL;
int g_w = 0, g_h = 0, g_pitch = 0;

void set_pixel(int x, int y, const Color *c) {
    if (x < 0 || x >= g_w || y < 0 || y >= g_h) return;
    uint8_t *p = g_buf + (y * g_pitch + x * 3);
    p[2] = c->r; p[1] = c->g; p[0] = c->b;
}

void draw_line(int x0, int y0, int x1, int y1, const Color *c) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    for (;;) {
        set_pixel(x0, y0, c);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_rect(int left, int top, int w, int h, const Color *c) {
    for (int x = left; x < left + w && x < g_w; x++) {
        set_pixel(x, top, c);
        if (top + h - 1 >= 0 && top + h - 1 < g_h) set_pixel(x, top + h - 1, c);
    }
    for (int y = top; y < top + h && y < g_h; y++) {
        set_pixel(left, y, c);
        if (left + w - 1 >= 0 && left + w - 1 < g_w) set_pixel(left + w - 1, y, c);
    }
}

void draw_circle_filled(int cx, int cy, int r, const Color *c) {
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            if (x*x + y*y <= r*r) set_pixel(cx + x, cy + y, c);
        }
    }
}

void draw_char(int x, int y, char ch, const Color *c, int scale) {
    if (ch < 32 || ch > 127) ch = '?';
    const unsigned char *glyph = get_glyph(ch);
    for (int row = 0; row < 5; row++) {
        uint8_t line = glyph[row];
        for (int col = 0; col < 5; col++) {
            if (line & (1 << (4 - col))) {
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        set_pixel(x + col * scale + dx, y + row * scale + dy, c);
                    }
                }
            }
        }
    }
}

void draw_text(int x, int y, const char *text, const Color *c, int scale) {
    for (int i = 0; text[i]; i++) {
        draw_char(x + i * 6 * scale, y, text[i], c, scale);
    }
}

/* =========================================================================
 * Joint Angle Computation
 * ========================================================================= */
static float calc_angle(float x1, float y1, float x2, float y2,
                        float x3, float y3) {
    float v1x = x1 - x2, v1y = y1 - y2;
    float v2x = x3 - x2, v2y = y3 - y2;
    float dot = v1x * v2x + v1y * v2y;
    float cross = v1x * v2y - v1y * v2x;
    float angle = atan2f(cross, dot) * 180.0f / (float)M_PI;
    if (angle < 0) angle = -angle;
    return angle;
}

typedef struct {
    float left_knee, right_knee;
    float left_elbow, right_elbow;
    float left_hip, right_hip;
    float left_shoulder, right_shoulder;
} JointAngles;

static JointAngles compute_angles(const float kpts[17][3]) {
    JointAngles a = {0};
    // Knee: hip-knee-ankle  (left: 11-13-15, right: 12-14-16)
    a.left_knee  = calc_angle(kpts[11][0], kpts[11][1],
                               kpts[13][0], kpts[13][1],
                               kpts[15][0], kpts[15][1]);
    a.right_knee = calc_angle(kpts[12][0], kpts[12][1],
                               kpts[14][0], kpts[14][1],
                               kpts[16][0], kpts[16][1]);
    // Elbow: shoulder-elbow-wrist (left: 5-7-9, right: 6-8-10)
    a.left_elbow  = calc_angle(kpts[5][0], kpts[5][1],
                                kpts[7][0], kpts[7][1],
                                kpts[9][0], kpts[9][1]);
    a.right_elbow = calc_angle(kpts[6][0], kpts[6][1],
                                kpts[8][0], kpts[8][1],
                                kpts[10][0], kpts[10][1]);
    // Hip: shoulder-hip-knee (left: 5-11-13, right: 6-12-14)
    a.left_hip  = calc_angle(kpts[5][0], kpts[5][1],
                              kpts[11][0], kpts[11][1],
                              kpts[13][0], kpts[13][1]);
    a.right_hip = calc_angle(kpts[6][0], kpts[6][1],
                              kpts[12][0], kpts[12][1],
                              kpts[14][0], kpts[14][1]);
    // Shoulder: nose-shoulder-elbow (left: 0-5-7, right: 0-6-8)
    a.left_shoulder  = calc_angle(kpts[0][0], kpts[0][1],
                                   kpts[5][0], kpts[5][1],
                                   kpts[7][0], kpts[7][1]);
    a.right_shoulder = calc_angle(kpts[0][0], kpts[0][1],
                                   kpts[6][0], kpts[6][1],
                                   kpts[8][0], kpts[8][1]);
    return a;
}

/* =========================================================================
 * DTW — Dynamic Time Warping
 * ========================================================================= */
typedef struct {
    const float *angles;  // array of [left_knee, right_knee]
    int num_frames;
} GoldenTemplate;

// Squat golden template: knee angle progression (standing → squat → standing)
// Frames: 0=standing(170°) → 10=quarter(135°) → 20=parallel(90°) → 30=full(60°) → 40=parallel → 50=standing
#define SQUAT_TEMPLATE_FRAMES 51
static const float squat_template[SQUAT_TEMPLATE_FRAMES * 2] = {
    170,170, 168,168, 165,165, 160,160, 155,155,
    148,148, 140,140, 132,132, 123,123, 115,115,
    108,108, 102,102,  97, 97,  92, 92,  88, 88,
     84, 84,  80, 80,  76, 76,  72, 72,  68, 68,
     65, 65,  62, 62,  60, 60,  62, 62,  65, 65,
     68, 68,  72, 72,  76, 76,  80, 80,  85, 85,
     90, 90,  96, 96, 103,103, 110,110, 118,118,
    126,126, 135,135, 143,143, 150,150, 157,157,
    162,162, 166,166, 169,169, 170,170,
    // Note: extra row to make 51 entries
};

static GoldenTemplate g_template;

static float dtw_distance(const float *seq1, int len1,
                          const float *seq2, int len2) {
    // Simplified DTW using single feature (average of left+right knee)
    // Allocate on heap to avoid stack overflow
    float *prev = (float *)calloc(len2 + 1, sizeof(float));
    float *curr = (float *)calloc(len2 + 1, sizeof(float));
    if (!prev || !curr) { free(prev); free(curr); return 1e9f; }

    for (int j = 0; j <= len2; j++) prev[j] = 1e9f;
    prev[0] = 0;

    for (int i = 1; i <= len1; i++) {
        curr[0] = 1e9f;
        for (int j = 1; j <= len2; j++) {
            float cost = fabsf(seq1[i - 1] - seq2[j - 1]);
            float min_prev = fminf(prev[j], fminf(curr[j - 1], prev[j - 1]));
            curr[j] = cost + min_prev;
        }
        float *tmp = prev; prev = curr; curr = tmp;
    }

    float result = prev[len2];
    free(prev); free(curr);
    return result;
}

static float compute_dtw_score(const float *user_seq, int user_len,
                               const GoldenTemplate *tmpl) {
    if (user_len < 3) return 0.0f;

    // Compare using left knee angle
    float *tmpl_left = (float *)malloc(tmpl->num_frames * sizeof(float));
    float *user_left = (float *)malloc(user_len * sizeof(float));
    if (!tmpl_left || !user_left) {
        free(tmpl_left); free(user_left);
        return 0.0f;
    }

    for (int i = 0; i < tmpl->num_frames; i++)
        tmpl_left[i] = tmpl->angles[i * 2];  // left knee
    for (int i = 0; i < user_len; i++)
        user_left[i] = user_seq[i * 2];      // left knee

    float dist = dtw_distance(user_left, user_len,
                              tmpl_left, tmpl->num_frames);
    free(tmpl_left); free(user_left);

    // Normalize: 0=bad, 100=perfect
    float max_dist = 30.0f * tmpl->num_frames;
    float score = 100.0f * (1.0f - fminf(dist / max_dist, 1.0f));
    return score;
}

/* =========================================================================
 * Feedback Generation
 * ========================================================================= */
static void build_feedback(const JointAngles *a, float dtw_score,
                           char *out, int out_len) {
    const char *quality;
    if (dtw_score >= 85)      quality = "Excellent!";
    else if (dtw_score >= 70) quality = "Good";
    else if (dtw_score >= 50) quality = "Keep trying";
    else                      quality = "Watch form";

    int knee_avg = (int)((a->left_knee + a->right_knee) / 2.0f);
    char hint[128] = "";
    if (knee_avg > 150)       snprintf(hint, sizeof(hint), "Bend knees more");
    else if (knee_avg < 50)  snprintf(hint, sizeof(hint), "Don't go too low");
    else if (dtw_score >= 70) snprintf(hint, sizeof(hint), "Good depth!");

    snprintf(out, out_len, "%s | Knee: %d deg | %s | Score: %.0f",
             quality, knee_avg, hint, dtw_score);
}

/* =========================================================================
 * TTS Audio Feedback
 * ========================================================================= */
static void tts_speak(const char *text) {
    if (!text || !text[0]) return;
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "echo \"%s\" | espeak-ng -v zh 2>/dev/null &", text);
    system(cmd);
}

static int g_tts_enabled = 0;  // Disabled by default (no espeak on clean Buildroot)

/* =========================================================================
 * Main
 * ========================================================================= */
int main(int argc, char **argv) {
    const char *model_path = MODEL_PATH;
    if (argc > 1) model_path = argv[1];

    printf("=== AI Rehab System for RK3588 ===\n");
    printf("Model: %s\n", model_path);

    // --- Init golden template ---
    g_template.angles     = squat_template;
    g_template.num_frames = SQUAT_TEMPLATE_FRAMES;

    // --- 1. Init post process ---
    fprintf(stderr, "[DEBUG] step 1: init_post_process\n");
    init_post_process();
    fprintf(stderr, "[DEBUG] step 1 done\n");

    // --- 2. Load RKNN model ---
    fprintf(stderr, "[DEBUG] step 2: init_yolov8_pose_model\n");
    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    int ret = init_yolov8_pose_model(model_path, &app_ctx);
    if (ret < 0) {
        fprintf(stderr, "FAIL: Model init failed!\n");
        return -1;
    }
    fprintf(stderr, "[DEBUG] step 2 done\n");

    // --- 3. Open DRM display ---
    fprintf(stderr, "[DEBUG] step 3: drmOpen\n");
    int drm_fd = drmOpen("rockchip", NULL);
    if (drm_fd < 0) {
        fprintf(stderr, "FAIL: drmOpen (try /dev/dri/card0)\n");
        return -1;
    }
    fprintf(stderr, "[DEBUG] step 3 done, drm_fd=%d\n", drm_fd);

    uint32_t crtc_id = 0, conn_id = 0;
    drmModeModeInfo mode;
    int found = 0;
    drmModeRes *res = drmModeGetResources(drm_fd);
    fprintf(stderr, "[DEBUG] drmModeGetResources: %p\n", (void*)res);
    if (!res) { fprintf(stderr, "FAIL: drmModeGetResources\n"); return -1; }
    for (int i = 0; i < res->count_connectors && !found; i++) {
        drmModeConnector *conn = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            conn_id = conn->connector_id;
            drmModeEncoder *enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
            if (enc) {
                crtc_id = enc->crtc_id;
                drmModeFreeEncoder(enc);
            } else {
                crtc_id = res->crtcs[i >= res->count_crtcs ? 0 : i];
            }
            mode = conn->modes[0];
            found = 1;
        }
        drmModeFreeConnector(conn);
    }
    drmModeFreeResources(res);

    if (!found) {
        printf("FAIL: No connected DRM display\n");
        return -1;
    }
    int disp_w = mode.hdisplay;
    int disp_h = mode.vdisplay;
    printf("Display: %dx%d\n", disp_w, disp_h);

    // --- 4. Create two DRM dumb buffers ---
    // Buffer A: model/draw buffer (640x480 RGB888)
    struct drm_mode_create_dumb model_dumb = {0};
    model_dumb.width  = MODEL_BUF_WIDTH;
    model_dumb.height = MODEL_BUF_HEIGHT;
    model_dumb.bpp    = 24;
    ret = ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &model_dumb);
    if (ret < 0) { printf("FAIL: DRM model buffer\n"); return -1; }
    uint32_t model_handle = model_dumb.handle;
    uint32_t model_pitch  = model_dumb.pitch;
    uint32_t model_fb_id;
    drmModeAddFB(drm_fd, MODEL_BUF_WIDTH, MODEL_BUF_HEIGHT, 24, 24,
                 model_pitch, model_handle, &model_fb_id);

    struct drm_mode_map_dumb model_map = { .handle = model_handle };
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &model_map);
    uint8_t *model_buf = (uint8_t *)mmap(0, model_dumb.size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, drm_fd, model_map.offset);
    memset(model_buf, 0, model_dumb.size);

    int model_fd = -1;
    drmPrimeHandleToFD(drm_fd, model_handle, DRM_CLOEXEC, &model_fd);

    // Buffer B: display buffer (screen-native resolution)
    struct drm_mode_create_dumb disp_dumb = {0};
    disp_dumb.width  = disp_w;
    disp_dumb.height = disp_h;
    disp_dumb.bpp    = 24;
    ret = ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &disp_dumb);
    if (ret < 0) { printf("FAIL: DRM display buffer\n"); return -1; }
    uint32_t disp_handle = disp_dumb.handle;
    uint32_t disp_pitch  = disp_dumb.pitch;
    uint32_t disp_fb_id;
    drmModeAddFB(drm_fd, disp_w, disp_h, 24, 24, disp_pitch,
                 disp_handle, &disp_fb_id);

    struct drm_mode_map_dumb disp_map = { .handle = disp_handle };
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &disp_map);
    uint8_t *disp_buf = (uint8_t *)mmap(0, disp_dumb.size, PROT_READ | PROT_WRITE,
                             MAP_SHARED, drm_fd, disp_map.offset);
    memset(disp_buf, 0, disp_dumb.size);

    int disp_fd = -1;
    drmPrimeHandleToFD(drm_fd, disp_handle, DRM_CLOEXEC, &disp_fd);

    // Set global drawing context to model buffer (coordinates in 640x480)
    g_buf   = model_buf;
    g_w     = MODEL_BUF_WIDTH;
    g_h     = MODEL_BUF_HEIGHT;
    g_pitch = model_pitch;

    fprintf(stderr, "[DEBUG] step 4: RGA init\n");
    c_RkRgaInit();
    fprintf(stderr, "[DEBUG] step 4 done\n");

    // --- 6. Init V4L2 camera (multiplanar API) ---
    fprintf(stderr, "[DEBUG] step 5: opening camera\n");
    int cam_fd = open(CAM_DEV, O_RDWR);
    fprintf(stderr, "[DEBUG] step 5 done, cam_fd=%d\n", cam_fd);
    if (cam_fd < 0) { printf("FAIL: Camera open\n"); return -1; }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width       = CAM_WIDTH;
    fmt.fmt.pix_mp.height      = CAM_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = CAM_FORMAT;
    fmt.fmt.pix_mp.field       = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes  = 1;
    ioctl(cam_fd, VIDIOC_S_FMT, &fmt);

    struct v4l2_requestbuffers req = {0};
    req.count  = BUFFER_COUNT;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(cam_fd, VIDIOC_REQBUFS, &req);

    struct v4l2_plane mplane;
    int cam_dma_fds[BUFFER_COUNT];
    uint8_t *cam_mmap[BUFFER_COUNT];
    for (int i = 0; i < BUFFER_COUNT; i++) {
        memset(&mplane, 0, sizeof(mplane));
        struct v4l2_buffer buf = {0};
        buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.index    = i;
        buf.m.planes = &mplane;
        buf.length   = 1;
        ioctl(cam_fd, VIDIOC_QUERYBUF, &buf);
        cam_mmap[i] = (uint8_t *)mmap(NULL, mplane.length, PROT_READ, MAP_SHARED,
                                      cam_fd, mplane.m.mem_offset);
        struct v4l2_exportbuffer exp = {0};
        exp.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        exp.index = i;
        exp.plane = 0;
        ioctl(cam_fd, VIDIOC_EXPBUF, &exp);
        cam_dma_fds[i] = exp.fd;
        ioctl(cam_fd, VIDIOC_QBUF, &buf);
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(cam_fd, VIDIOC_STREAMON, &type);
    printf("Camera streaming...\n");

    // --- 7. Prepare RGA conversion params ---
    // Step 1: NV12→RGB888 cvtcolor into model buffer (640x480)
    rga_buffer_t src_rga = {0};
    src_rga.width   = CAM_WIDTH;
    src_rga.height  = CAM_HEIGHT;
    src_rga.wstride = CAM_WIDTH;
    src_rga.hstride = CAM_HEIGHT;
    src_rga.format  = RK_FORMAT_YCbCr_420_SP;

    rga_buffer_t model_rga = {0};
    model_rga.fd      = model_fd;
    model_rga.width   = MODEL_BUF_WIDTH;
    model_rga.height  = MODEL_BUF_HEIGHT;
    model_rga.wstride = MODEL_BUF_WIDTH;
    model_rga.hstride = MODEL_BUF_HEIGHT;
    model_rga.format  = DISP_FMT;

    // Step 2: RGB888→RGB888 resize from model buffer to display buffer
    rga_buffer_t disp_rga = {0};
    disp_rga.fd      = disp_fd;
    disp_rga.width   = disp_w;
    disp_rga.height  = disp_h;
    disp_rga.wstride = disp_w;
    disp_rga.hstride = disp_h;
    disp_rga.format  = DISP_FMT;

    // --- 8. UI init ---
    UIContext ui_ctx;
    ui_init(&ui_ctx);
    ui_set_screen(&ui_ctx, SCREEN_WELCOME);

    // --- 9. Rehab state ---
    float angle_history[MAX_DTW_FRAMES * 2];
    int   angle_count = 0;
    int   frame_count = 0;
    struct timeval last_time, now;
    gettimeofday(&last_time, NULL);

    printf("\n>>> Rehab system running. Tap screen to operate. <<<\n\n");

    // --- 10. Main loop ---
    while (1) {
        // 10a. Poll touch input
        ui_poll_touch(&ui_ctx);

        // 10b. Handle button taps → screen transitions
        if (ui_ctx.tapped_idx >= 0) {
            int t = ui_ctx.tapped_idx;
            switch (ui_ctx.screen) {
                case SCREEN_WELCOME:
                    if (t == 0) { // START
                        ui_set_screen(&ui_ctx, SCREEN_EXERCISE_SELECT);
                    }
                    if (t == 1) { // EXIT
                        printf("Exit requested.\n");
                        goto cleanup;
                    }
                    break;
                case SCREEN_EXERCISE_SELECT:
                    if (t >= 0 && t < 4) {
                        ui_ctx.exercise = (ExerciseType)t;
                        angle_count = 0;
                        ui_ctx.training_paused = 0;
                        training_btns[0].label = "PAUSE";
                        ui_set_screen(&ui_ctx, SCREEN_TRAINING);
                    }
                    if (t == 4) ui_set_screen(&ui_ctx, SCREEN_WELCOME);
                    break;
                case SCREEN_TRAINING:
                    if (t == 0) { // PAUSE/RESUME toggle
                        ui_ctx.training_paused = !ui_ctx.training_paused;
                        training_btns[0].label = ui_ctx.training_paused ? "RESUME" : "PAUSE";
                    }
                    if (t == 1) { // STOP → results
                        float score = 0;
                        if (angle_count > 10)
                            score = compute_dtw_score(angle_history, angle_count, &g_template);
                        ui_ctx.final_score    = score;
                        ui_ctx.avg_knee_angle  = (angle_count > 0) ?
                            (angle_history[(angle_count-1)*2] + angle_history[(angle_count-1)*2+1]) / 2.0f : 0;
                        ui_ctx.avg_elbow_angle = 0;
                        ui_ctx.total_frames    = frame_count;
                        ui_set_screen(&ui_ctx, SCREEN_RESULTS);
                    }
                    break;
                case SCREEN_RESULTS:
                    if (t == 0) { // REPEAT
                        angle_count = 0;
                        ui_ctx.training_paused = 0;
                        training_btns[0].label = "PAUSE";
                        ui_set_screen(&ui_ctx, SCREEN_TRAINING);
                    }
                    if (t == 1) ui_set_screen(&ui_ctx, SCREEN_WELCOME);
                    break;
            }
        }

        // 10c. Dequeue camera frame
        struct v4l2_plane dq_plane = {0};
        struct v4l2_buffer cam_buf = {0};
        cam_buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        cam_buf.memory   = V4L2_MEMORY_MMAP;
        cam_buf.m.planes = &dq_plane;
        cam_buf.length   = 1;
        if (ioctl(cam_fd, VIDIOC_DQBUF, &cam_buf) < 0) continue;

        int idx = cam_buf.index;
        src_rga.fd = cam_dma_fds[idx];

        // 10d. RGA: NV12 → RGB888 into model buffer
        ret = imcvtcolor(src_rga, model_rga, RK_FORMAT_YCbCr_420_SP, DISP_FMT);
        if (ret != IM_STATUS_SUCCESS) {
            ioctl(cam_fd, VIDIOC_QBUF, &cam_buf);
            continue;
        }
        imsync(0);

        // 10e. NPU inference + skeleton HUD (only when training and not paused)
        JointAngles angles = {0};

        if (ui_ctx.screen == SCREEN_TRAINING && !ui_ctx.training_paused) {
            image_buffer_t src_img;
            src_img.width     = MODEL_BUF_WIDTH;
            src_img.height    = MODEL_BUF_HEIGHT;
            src_img.format    = IMAGE_FORMAT_RGB888;
            src_img.size      = MODEL_BUF_WIDTH * MODEL_BUF_HEIGHT * 3;
            src_img.virt_addr = model_buf;
            src_img.fd        = model_fd;

            object_detect_result_list od_results;
            memset(&od_results, 0, sizeof(od_results));
            inference_yolov8_pose_model(&app_ctx, &src_img, &od_results);

            if (frame_count < 3) fprintf(stderr, "[DEBUG frame %d] detections=%d\n", frame_count, od_results.count);

            if (od_results.count > 0) {
                object_detect_result *det = &od_results.results[0];

                draw_rect(det->box.left, det->box.top,
                          det->box.right - det->box.left,
                          det->box.bottom - det->box.top, &COLOR_HUD);

                for (int b = 0; b < 19; b++) {
                    int k1 = SKELETON[b][0], k2 = SKELETON[b][1];
                    if (k1 >= 17 || k2 >= 17) continue;
                    float c1 = det->keypoints[k1][2];
                    float c2 = det->keypoints[k2][2];
                    if (c1 < 0.3f || c2 < 0.3f) continue;
                    const Color *bc;
                    switch (BONE_SIDE[b]) {
                        case 1:  bc = &COLOR_LEFT; break;
                        case 2:  bc = &COLOR_RIGHT; break;
                        default: bc = &COLOR_CENTER; break;
                    }
                    draw_line((int)det->keypoints[k1][0], (int)det->keypoints[k1][1],
                              (int)det->keypoints[k2][0], (int)det->keypoints[k2][1], bc);
                }

                for (int k = 0; k < 17; k++) {
                    float cf = det->keypoints[k][2];
                    if (cf < 0.3f) continue;
                    int cx = (int)det->keypoints[k][0], cy = (int)det->keypoints[k][1];
                    const Color *kc;
                    if (cf > 0.7f)       kc = &COLOR_KPT_HI;
                    else if (cf > 0.4f)  kc = &COLOR_KPT_MED;
                    else                  kc = &COLOR_KPT_LO;
                    draw_circle_filled(cx, cy, (cf > 0.7f) ? 4 : 3, kc);
                }

                angles = compute_angles(det->keypoints);
                if (angle_count < MAX_DTW_FRAMES) {
                    angle_history[angle_count * 2]     = angles.left_knee;
                    angle_history[angle_count * 2 + 1] = angles.right_knee;
                    angle_count++;
                }

                char at[32];
                snprintf(at, sizeof(at), "%.0f", angles.left_knee);
                draw_text((int)det->keypoints[13][0] + 8, (int)det->keypoints[13][1] - 8, at, &COLOR_LEFT, 1);
                snprintf(at, sizeof(at), "%.0f", angles.right_knee);
                draw_text((int)det->keypoints[14][0] + 8, (int)det->keypoints[14][1] - 8, at, &COLOR_RIGHT, 1);
            }

            // HUD
            gettimeofday(&now, NULL);
            float elapsed = (now.tv_sec - last_time.tv_sec) + (now.tv_usec - last_time.tv_usec) / 1000000.0f;
            float fps = (elapsed > 0) ? 1.0f / elapsed : 0;
            last_time = now;

            char hud[128];
            snprintf(hud, sizeof(hud), "FPS:%.1f", fps);
            draw_text(4, 4, hud, &COLOR_HUD, 1);

            char panel[64];
            snprintf(panel, sizeof(panel), "LK:%.0f RK:%.0f", angles.left_knee, angles.right_knee);
            draw_text(MODEL_BUF_WIDTH - 160, 4, panel, &COLOR_HUD, 1);
            snprintf(panel, sizeof(panel), "LE:%.0f RE:%.0f", angles.left_elbow, angles.right_elbow);
            draw_text(MODEL_BUF_WIDTH - 160, 14, panel, &COLOR_HUD, 1);
            snprintf(panel, sizeof(panel), "LH:%.0f RH:%.0f", angles.left_hip, angles.right_hip);
            draw_text(MODEL_BUF_WIDTH - 160, 24, panel, &COLOR_HUD, 1);

            if (angle_count > 10) {
                float sc = compute_dtw_score(angle_history, angle_count, &g_template);
                const Color *sc_c = (sc >= 70) ? &COLOR_CENTER :
                                    (sc >= 40) ? &COLOR_LEFT : &COLOR_ALERT;
                snprintf(hud, sizeof(hud), "SCORE: %.0f", sc);
                draw_text(MODEL_BUF_WIDTH / 2 - 30, MODEL_BUF_HEIGHT - 20, hud, sc_c, 1);
            }
            draw_text(4, MODEL_BUF_HEIGHT - 20, "Squat", &COLOR_HUD, 1);
        }

        // 10f. Draw training overlay buttons (only during training)
        if (ui_ctx.screen == SCREEN_TRAINING) {
            // Semi-transparent bottom bar background
            Color bar_bg = {20, 20, 40};
            draw_filled_rect(0, MODEL_BUF_HEIGHT - 42, MODEL_BUF_WIDTH, 42, &bar_bg);
            for (int i = 0; i < ui_ctx.button_count; i++) {
                draw_button_widget(&ui_ctx.buttons[i], i == ui_ctx.hover_idx);
            }
        }

        // 10g. Render UI for non-training screens
        if (ui_ctx.screen != SCREEN_TRAINING) {
            static int first_ui = 1;
            if (first_ui) {
                fprintf(stderr, "[DEBUG] ui_render screen=%d model_buf=%p g_w=%d g_h=%d pitch=%d\n",
                        ui_ctx.screen, (void*)model_buf, g_w, g_h, g_pitch);
                first_ui = 0;
            }
            ui_render(&ui_ctx);
        }

        // 10h. RGA resize: model buffer → display buffer
        ret = imresize(model_rga, disp_rga);
        if (ret != IM_STATUS_SUCCESS && frame_count < 3)
            fprintf(stderr, "[DEBUG frame %d] imresize ret=%d\n", frame_count, ret);
        imsync(0);

        // 10i. DRM display
        ret = drmModeSetCrtc(drm_fd, crtc_id, disp_fb_id, 0, 0, &conn_id, 1, &mode);
        if (frame_count < 3) fprintf(stderr, "[DEBUG frame %d] drmModeSetCrtc ret=%d\n", frame_count, ret);

        // 10j. Return camera buffer
        ioctl(cam_fd, VIDIOC_QBUF, &cam_buf);
        frame_count++;
    }

cleanup:

    // Cleanup (unreachable)
    release_yolov8_pose_model(&app_ctx);
    munmap(model_buf, model_dumb.size);
    munmap(disp_buf, disp_dumb.size);
    close(model_fd);
    close(disp_fd);
    close(cam_fd);
    drmModeRmFB(drm_fd, model_fb_id);
    drmModeRmFB(drm_fd, disp_fb_id);
    drmClose(drm_fd);
    c_RkRgaDeInit();
    return 0;
}
