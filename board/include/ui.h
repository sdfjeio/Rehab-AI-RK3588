#ifndef UI_H_
#define UI_H_

#include <stdint.h>

/* ---- Shared color type ----------------------------------------------- */
typedef struct { uint8_t r, g, b; } Color;

/* ---- Exercise types -------------------------------------------------- */
typedef enum {
    EXERCISE_SQUAT = 0,
    EXERCISE_ARM_RAISE,
    EXERCISE_LATERAL_RAISE,
    EXERCISE_FRONT_RAISE,
    EXERCISE_COUNT
} ExerciseType;

/* ---- Screen state machine -------------------------------------------- */
typedef enum {
    SCREEN_WELCOME = 0,
    SCREEN_EXERCISE_SELECT,
    SCREEN_TRAINING,
    SCREEN_RESULTS
} Screen;

/* ---- Button ---------------------------------------------------------- */
typedef struct {
    int x, y, w, h;
    const char *label;
    int is_pressed;
} Button;

/* ---- UIContext — all UI state ---------------------------------------- */
typedef struct {
    Screen        screen;
    ExerciseType  exercise;

    // Touch state (display-space raw values, 1024x600 range)
    int touch_x, touch_y;
    int touch_down;
    int touch_changed;

    // Buttons (dynamically set per screen)
    Button       *buttons;
    int           button_count;
    int           hover_idx;
    int           tapped_idx;   // Button index tapped this frame, -1 = none

    // Training mode flags
    int           training_paused;

    // Results data
    float         final_score;
    float         avg_knee_angle;
    float         avg_elbow_angle;
    int           total_frames;
} UIContext;

/* ---- Touch coords scaling: display(1024x600) → model(640x480) -------- */
#define TOUCH_TO_MODEL_X(tx) ((int)((tx) * 640 / 1024))
#define TOUCH_TO_MODEL_Y(ty) ((int)((ty) * 480 / 600))

/* ---- Drawing globals (defined in main_board.cc) ---------------------- */
extern uint8_t *g_buf;
extern int g_w, g_h, g_pitch;

/* ---- Shared drawing primitives --------------------------------------- */
void set_pixel(int x, int y, const Color *c);
void draw_line(int x0, int y0, int x1, int y1, const Color *c);
void draw_rect(int left, int top, int w, int h, const Color *c);
void draw_circle_filled(int cx, int cy, int r, const Color *c);
void draw_text(int x, int y, const char *text, const Color *c, int scale);

/* ---- UI drawing helpers ---------------------------------------------- */
void draw_filled_rect(int x, int y, int w, int h, const Color *c);
void draw_button_widget(const Button *btn, int is_hover);

/* ---- Exposed for label toggling during training ---------------------- */
extern Button training_btns[2];

/* ---- UI API ---------------------------------------------------------- */
int  ui_init(UIContext *ctx);
void ui_poll_touch(UIContext *ctx);
void ui_render(UIContext *ctx);
void ui_set_screen(UIContext *ctx, Screen screen);

#endif // UI_H_
