#!/usr/bin/env bash
# install.sh - Install system dependencies and setup Python environment
set -e

echo "=== Hand Gesture Control - Installer ==="

# Detect package manager
if command -v apt &>/dev/null; then
    PKG_MGR="apt"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
elif command -v pacman &>/dev/null; then
    PKG_MGR="pacman"
else
    echo "Error: Unsupported package manager. Install dependencies manually."
    exit 1
fi

echo "Detected package manager: $PKG_MGR"

# Install system dependencies
case "$PKG_MGR" in
    apt)
        echo "Installing system packages via apt..."
        sudo apt update
        sudo apt install -y v4l-utils python3-dev python3-venv libevdev2
        ;;
    dnf)
        echo "Installing system packages via dnf..."
        sudo dnf install -y v4l-utils python3-devel python3-virtualenv libevdev
        ;;
    pacman)
        echo "Installing system packages via pacman..."
        sudo pacman -S --noconfirm v4l-utils python python-evdev
        ;;
esac

# Setup Python virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment and installing Python packages..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd user directory if it doesn't exist
mkdir -p ~/.config/systemd/user

echo ""
echo "=== Installation Complete ==="
echo "Next steps:"
echo "  1. Activate the virtual environment: source $VENV_DIR/bin/activate"
echo "  2. Test with: python -m src.main --debug"
echo "  3. Install as service: cp gesture-control.service ~/.config/systemd/user/ && systemctl --user enable gesture-control.service"
