// ui.cpp — Touch UI module for patient self-service
#include "ui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>

static int g_touch_fd = -1;

// Per-frame touch accumulator (display-space, 1024x600)
static int raw_x = 0, raw_y = 0;
static int raw_tracking = -1;
static int raw_touch = 0;

/* =========================================================================
 * Shared drawing helpers
 * ========================================================================= */

void draw_filled_rect(int x, int y, int w, int h, const Color *c) {
    for (int dy = 0; dy < h; dy++) {
        for (int dx = 0; dx < w; dx++) {
            set_pixel(x + dx, y + dy, c);
        }
    }
}

void draw_button_widget(const Button *btn, int is_hover) {
    // Fill
    Color fill = {40, 80, 160};
    if (is_hover)   fill = (Color){70, 120, 210};
    if (btn->is_pressed) fill = (Color){120, 160, 240};
    draw_filled_rect(btn->x, btn->y, btn->w, btn->h, &fill);

    // Border
    Color border = {180, 180, 200};
    draw_rect(btn->x, btn->y, btn->w, btn->h, &border);

    // Label (centered, scale 2)
    if (btn->label) {
        int len = (int)strlen(btn->label);
        int char_w = 6 * 2;   // scale=2
        int text_w = len * char_w;
        int text_h = 7 * 2;
        int tx = btn->x + (btn->w - text_w) / 2;
        int ty = btn->y + (btn->h - text_h) / 2;
        Color white = {200, 200, 200};
        draw_text(tx, ty, btn->label, &white, 2);
    }
}

/* =========================================================================
 * Touch input — non-blocking evdev read from /dev/input/event1
 * ========================================================================= */

int ui_init(UIContext *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->screen    = SCREEN_WELCOME;
    ctx->hover_idx = -1;

    g_touch_fd = open("/dev/input/event1", O_RDONLY | O_NONBLOCK);
    if (g_touch_fd < 0) {
        fprintf(stderr, "[UI] WARN: cannot open touch device\n");
    } else {
        fprintf(stderr, "[UI] touch device opened, fd=%d\n", g_touch_fd);
    }
    ctx->tapped_idx = -1;
    return (g_touch_fd >= 0) ? 0 : -1;
}

void ui_poll_touch(UIContext *ctx) {
    if (g_touch_fd < 0) { ctx->touch_down = 0; return; }

    struct input_event ev;
    int was_down = raw_tracking >= 0;

    static int last_hover = -1;  // Remember hover across SYN_REPORT frames

    // Drain all pending events
    while (read(g_touch_fd, &ev, sizeof(ev)) == sizeof(ev)) {
        if (ev.type == EV_ABS) {
            switch (ev.code) {
                case ABS_MT_POSITION_X: raw_x = ev.value; break;
                case ABS_MT_POSITION_Y: raw_y = ev.value; break;
                case ABS_MT_TRACKING_ID: raw_tracking = ev.value; break;
            }
        } else if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
            raw_touch = ev.value;
        } else if (ev.type == EV_SYN && ev.code == SYN_REPORT) {
            int is_down = raw_tracking >= 0;
            ctx->touch_x = raw_x;
            ctx->touch_y = raw_y;
            ctx->touch_down = is_down;
            ctx->touch_changed = (was_down && !is_down);
            ctx->tapped_idx = -1;  // Reset each frame

            // On finger lift, report which button was tapped
            if (was_down && !is_down && last_hover >= 0) {
                ctx->tapped_idx = last_hover;
            }
            was_down = is_down;

            // Hit-test hover
            ctx->hover_idx = -1;
            if (is_down && ctx->button_count > 0 && ctx->buttons) {
                int mx = TOUCH_TO_MODEL_X(raw_x);
                int my = TOUCH_TO_MODEL_Y(raw_y);
                for (int i = 0; i < ctx->button_count; i++) {
                    Button *b = &ctx->buttons[i];
                    if (mx >= b->x && mx < b->x + b->w &&
                        my >= b->y && my < b->y + b->h) {
                        ctx->hover_idx = i;
                        b->is_pressed = 1;
                    } else {
                        b->is_pressed = 0;
                    }
                }
            } else {
                for (int i = 0; i < ctx->button_count; i++)
                    ctx->buttons[i].is_pressed = 0;
            }
            if (ctx->hover_idx < 0) last_hover = -1;
            else if (is_down) last_hover = ctx->hover_idx;
        }
    }
}

/* =========================================================================
 * Screen rendering
 * ========================================================================= */

// ---- Per-screen button definitions -----------------------------------
// Buttons are defined in model-buffer space (640x480)
static Button welcome_btns[2];
static Button exercise_btns[5];   // 4 exercises + back
Button training_btns[2];   // pause + stop (non-static: accessed from main)
static Button results_btns[2];    // retry + menu

static void setup_welcome(UIContext *ctx) {
    welcome_btns[0] = (Button){180, 280, 280, 50, "START", 0};
    welcome_btns[1] = (Button){180, 350, 280, 50, "EXIT", 0};
    ctx->buttons     = welcome_btns;
    ctx->button_count = 2;
}

static void setup_exercise_select(UIContext *ctx) {
    // 4 exercise buttons in 2×2 grid
    exercise_btns[0] = (Button){ 80, 180, 220, 60, "SQUAT", 0};
    exercise_btns[1] = (Button){340, 180, 220, 60, "ARM RAISE", 0};
    exercise_btns[2] = (Button){ 80, 270, 220, 60, "SIDE RAISE", 0};
    exercise_btns[3] = (Button){340, 270, 220, 60, "FRONT RAISE", 0};
    exercise_btns[4] = (Button){220, 380, 200, 40, "BACK", 0};
    ctx->buttons     = exercise_btns;
    ctx->button_count = 5;
}

static void setup_training(UIContext *ctx) {
    training_btns[0] = (Button){420, 440, 100, 35, "PAUSE", 0};
    training_btns[1] = (Button){530, 440, 100, 35, "STOP", 0};
    ctx->buttons     = training_btns;
    ctx->button_count = 2;
}

static void setup_results(UIContext *ctx) {
    results_btns[0] = (Button){120, 380, 180, 45, "REPEAT", 0};
    results_btns[1] = (Button){340, 380, 180, 45, "MENU", 0};
    ctx->buttons     = results_btns;
    ctx->button_count = 2;
}

void ui_set_screen(UIContext *ctx, Screen screen) {
    ctx->screen = screen;
    ctx->hover_idx = -1;
    switch (screen) {
        case SCREEN_WELCOME:         setup_welcome(ctx); break;
        case SCREEN_EXERCISE_SELECT: setup_exercise_select(ctx); break;
        case SCREEN_TRAINING:        setup_training(ctx); break;
        case SCREEN_RESULTS:         setup_results(ctx); break;
    }
}

// ---- Screen renderers ------------------------------------------------

static void render_welcome(UIContext *ctx) {
    Color c;
    // Title
    c = (Color){0, 200, 100};
    draw_text(120, 120, "RK3588 AI Rehab", &c, 3);

    // Subtitle
    c = (Color){150, 150, 150};
    draw_text(160, 180, "v1.0 — RKNPU2", &c, 1);

    // Footer
    draw_text(180, 440, "Touch START to begin", &c, 1);
}

static void render_exercise_select(UIContext *ctx) {
    Color c = {0, 200, 100};
    draw_text(140, 100, "Select Exercise", &c, 2);

    c = (Color){150, 150, 150};
    draw_text(180, 440, "Touch an exercise to start", &c, 1);
}

static void render_results(UIContext *ctx) {
    Color c = {0, 200, 100};
    draw_text(200, 80, "RESULTS", &c, 2);

    char buf[64];
    c = (Color){255, 255, 255};
    snprintf(buf, sizeof(buf), "Score: %.0f / 100", ctx->final_score);
    draw_text(200, 160, buf, &c, 2);

    // Rating
    const char *rating;
    Color rc;
    if (ctx->final_score >= 85)      { rating = "EXCELLENT!"; rc = (Color){0, 255, 0}; }
    else if (ctx->final_score >= 70) { rating = "GOOD";       rc = (Color){0, 200, 200}; }
    else                             { rating = "KEEP TRYING"; rc = (Color){255, 200, 0}; }
    draw_text(200, 200, rating, &rc, 2);

    snprintf(buf, sizeof(buf), "Avg Knee: %.0f deg", ctx->avg_knee_angle);
    draw_text(200, 260, buf, &c, 1);

    snprintf(buf, sizeof(buf), "Frames: %d", ctx->total_frames);
    draw_text(200, 280, buf, &c, 1);
}

void ui_render(UIContext *ctx) {
    // Clear to mid-bright background for visibility
    Color bg = {30, 40, 80};
    draw_filled_rect(0, 0, g_w, g_h, &bg);

    // Draw title bar
    Color bar = {10, 10, 30};
    draw_filled_rect(0, 0, g_w, 40, &bar);

    // Screen title text
    Color white = {200, 200, 200};
    switch (ctx->screen) {
        case SCREEN_WELCOME:         draw_text(10, 10, "RK Rehab System", &white, 1); break;
        case SCREEN_EXERCISE_SELECT: draw_text(10, 10, "Select Exercise", &white, 1); break;
        case SCREEN_RESULTS:         draw_text(10, 10, "Training Results", &white, 1); break;
        default: break;
    }

    // Screen-specific content
    switch (ctx->screen) {
        case SCREEN_WELCOME:         render_welcome(ctx); break;
        case SCREEN_EXERCISE_SELECT: render_exercise_select(ctx); break;
        case SCREEN_RESULTS:         render_results(ctx); break;
        default: break;
    }

    // Draw all buttons
    for (int i = 0; i < ctx->button_count; i++) {
        draw_button_widget(&ctx->buttons[i], i == ctx->hover_idx);
    }
}
