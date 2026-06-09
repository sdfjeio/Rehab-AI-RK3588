#!/bin/bash
# install.sh — Install AI Rehab launcher on Ubuntu 22.04 RK3588 board
# Run as root:  sudo ./scripts/install.sh

set -e

APP_DIR="/data/rehab"
LAUNCHER="/usr/local/bin/rehab-launch"
DESKTOP="/usr/share/applications/rehab.desktop"
ICON="/usr/share/icons/hicolor/48x48/apps/rehab.png"

echo "=== AI Rehab Installer ==="

# 1. Install launcher script
echo "[1/4] Installing launcher script..."
cp scripts/launch.sh "$LAUNCHER"
chmod +x "$LAUNCHER"
echo "  -> $LAUNCHER"

# 2. Install desktop entry
echo "[2/4] Installing desktop entry..."
cp assets/rehab.desktop "$DESKTOP"
echo "  -> $DESKTOP"

# 3. Install icon (use generic app icon if none provided)
echo "[3/4] Installing icon..."
if [ -f assets/rehab.png ]; then
    cp assets/rehab.png "$ICON"
else
    # Fallback: copy a system icon
    if [ -f /usr/share/icons/hicolor/48x48/apps/gnome-logo-text-dark.png ]; then
        cp /usr/share/icons/hicolor/48x48/apps/gnome-logo-text-dark.png "$ICON"
    fi
fi
gtk-update-icon-cache /usr/share/icons/hicolor/ 2>/dev/null || true
echo "  -> $ICON"

# 4. Verify app binary exists
echo "[4/4] Checking app binary..."
if [ -x "$APP_DIR/rehab_app" ]; then
    echo "  -> $APP_DIR/rehab_app OK"
else
    echo "  -> WARNING: $APP_DIR/rehab_app not found or not executable"
    echo "     Build it first: cd $APP_DIR && make"
fi

echo ""
echo "=== Installation complete ==="
echo "You can now find 'AI 康复训练' in the GNOME app menu."
echo "Or run directly: $LAUNCHER"
