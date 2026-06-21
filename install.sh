#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# OpenXenos — cross-platform auto-start installer
#   Linux  → systemd user service
#   macOS  → launchd user agent
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_HOME="$HOME"
UV_BIN="$(which uv 2>/dev/null || echo "$USER_HOME/.local/bin/uv")"
SERVICE_SRC="$SCRIPT_DIR/openxenos.service"

# ── Detect OS ────────────────────────────────────────────────
case "$(uname -s)" in
    Linux)  OS=linux ;;
    Darwin) OS=macos ;;
    *)
        echo "✗ Unsupported OS: $(uname -s)"
        echo "  OpenXenos auto-start supports Linux (systemd) and macOS (launchd)."
        echo "  You can still run it manually: uv run openxenos"
        exit 1
        ;;
esac

echo "╔══════════════════════════════════════╗"
echo "║   OpenXenos — Auto-start Installer  ║"
echo "║   OS: $OS                              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Install functions (defined before use) ──────────────────

install_linux() {
    echo ""
    echo "▸ Installing systemd user service..."

    if ! command -v systemctl &>/dev/null; then
        echo "  ✗ systemctl not found — is systemd available?"
        exit 1
    fi

    SYSTEMD_DIR="$USER_HOME/.config/systemd/user"
    SERVICE_DST="$SYSTEMD_DIR/openxenos.service"
    mkdir -p "$SYSTEMD_DIR"

    sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" "$SERVICE_SRC" > "$SERVICE_DST"

    echo "  ✓ Service written to $SERVICE_DST"

    systemctl --user daemon-reload
    systemctl --user enable --now openxenos

    echo "  ✓ Service enabled and started"
}

install_macos() {
    echo ""
    echo "▸ Installing launchd user agent..."

    PLIST_DST="$USER_HOME/Library/LaunchAgents/com.openxenos.server.plist"
    PLIST_DST_DIR="$(dirname "$PLIST_DST")"
    mkdir -p "$PLIST_DST_DIR"

    LOG_DIR="$USER_HOME/Library/Logs"
    mkdir -p "$LOG_DIR"

    cat > "$PLIST_DST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.openxenos.server</string>

    <key>ProgramArguments</key>
    <array>
        <string>$UV_BIN</string>
        <string>run</string>
        <string>openxenos</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$USER_HOME/.local/bin</string>
        <key>HOME</key>
        <string>$USER_HOME</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/openxenos.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/openxenos.err</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
PLIST

    echo "  ✓ plist written to $PLIST_DST"

    # Unload if already running, then load
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    launchctl load "$PLIST_DST"

    echo "  ✓ Agent loaded"
}

# ── Step 1: Check prerequisites ─────────────────────────────
echo "▸ Checking prerequisites..."

if ! command -v uv &>/dev/null && [ ! -x "$UV_BIN" ]; then
    echo "  ✗ uv not found. Install it first:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✓ uv: $UV_BIN"

# ── Step 2: Create .env if missing ──────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo ""
    echo "▸ Creating .env from template..."
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "  ✓ Created $SCRIPT_DIR/.env"
    echo "  ⚠  Edit it now to set your ANTHROPIC_AUTH_TOKEN:"
    echo "     nano $SCRIPT_DIR/.env"
    echo ""
    read -rp "  Press Enter after editing (or Ctrl-C to abort)... " _
fi
echo "  ✓ .env present"

# ── Step 3: Install the service ─────────────────────────────
if [ "$OS" = "linux" ]; then
    install_linux
elif [ "$OS" = "macos" ]; then
    install_macos
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Installation complete!             ║"
echo "╠══════════════════════════════════════╣"
echo "║                                      ║"
echo "║   Port:     2222                     ║"

if [ "$OS" = "linux" ]; then
    echo "║   Service:  systemd (user)           ║"
    echo "║                                      ║"
    echo "║   Check status:                      ║"
    echo "║     systemctl --user status openxenos ║"
    echo "║                                      ║"
    echo "║   Tail logs:                         ║"
    echo "║     journalctl --user -u openxenos -f ║"
elif [ "$OS" = "macos" ]; then
    echo "║   Service:  launchd (user agent)     ║"
    echo "║                                      ║"
    echo "║   Check status:                      ║"
    echo "║     launchctl list | grep openxenos  ║"
    echo "║                                      ║"
    echo "║   Tail logs:                         ║"
    echo "║     tail -f ~/Library/Logs/openxenos.log ║"
fi

echo "║                                      ║"
echo "║   Claude Code config:                ║"
echo "║     base_url: http://localhost:2222/v1║"
echo "║     api_key:  anything               ║"
echo "╚══════════════════════════════════════╝"
