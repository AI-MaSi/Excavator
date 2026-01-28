#!/bin/bash
# MASI Robot Setup Script
# Configures I2C, installs dependencies, sets up OLED service

set -e

echo "========================================"
echo "  MASI Robot Setup"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="/boot/firmware/config.txt"
USER_HOME=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)
USERNAME="${SUDO_USER:-$USER}"

# --- I2C Configuration ---
echo ""
echo "[1/5] Configuring I2C buses..."

# Backup config
cp "$CONFIG" "$CONFIG.bak.$(date +%Y%m%d_%H%M%S)"

# Enable hardware I2C at 1MHz
if ! grep -q "^dtparam=i2c_arm=on" "$CONFIG"; then
    # Add after the optional hardware interfaces comment if exists, otherwise at start of [all]
    if grep -q "optional hardware interfaces" "$CONFIG"; then
        sed -i '/optional hardware interfaces/a dtparam=i2c_arm=on' "$CONFIG"
    else
        sed -i '1i dtparam=i2c_arm=on' "$CONFIG"
    fi
fi

# Set baudrate to 1MHz
if grep -q "^dtparam=i2c_arm_baudrate=" "$CONFIG"; then
    sed -i 's/^dtparam=i2c_arm_baudrate=.*/dtparam=i2c_arm_baudrate=1000000/' "$CONFIG"
else
    sed -i '/dtparam=i2c_arm=on/a dtparam=i2c_arm_baudrate=1000000' "$CONFIG"
fi
echo "  OK: Hardware I2C (bus 1) at 1MHz"

# Remove existing virtual I2C configs
sed -i '/dtoverlay=i2c-gpio,bus=3/d' "$CONFIG"
sed -i '/dtoverlay=i2c-gpio,bus=4/d' "$CONFIG"
sed -i '/# Virtual I2C buses/d' "$CONFIG"
sed -i '/# Bus 3:/d' "$CONFIG"
sed -i '/# Bus 4:/d' "$CONFIG"

# Add virtual I2C buses
cat >> "$CONFIG" << 'EOF'

# Virtual I2C buses (bit-banged)
# Bus 3: GPIO 16/19 ~200kHz (ADC)
dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=16,i2c_gpio_scl=19,i2c_gpio_delay_us=1
# Bus 4: GPIO 20/21 ~100kHz (OLED)
dtoverlay=i2c-gpio,bus=4,i2c_gpio_sda=20,i2c_gpio_scl=21
EOF

echo "  OK: Virtual I2C bus 3 (GPIO 18/19, 200kHz) - ADC"
echo "  OK: Virtual I2C bus 4 (GPIO 20/21, 100kHz) - OLED"

# --- System packages ---
echo ""
echo "[2/5] Installing system packages..."
apt-get update -qq
apt-get install -y python3-pip python3-pil python3-smbus i2c-tools git > /dev/null 2>&1
echo "  OK: System packages installed"

# --- Python packages ---
echo ""
echo "[3/5] Installing Python packages..."

pip3 install --break-system-packages --quiet \
    numpy \
    pandas \
    numba \
    pyyaml \
    pyserial \
    adafruit-blinka \
    adafruit-circuitpython-ssd1306 \
    2>/dev/null

echo "  OK: Python packages installed"
echo "      numpy, pandas, numba, pyyaml, pyserial"
echo "      adafruit-blinka, ssd1306"

# --- OLED systemd service ---
echo ""
echo "[4/5] Setting up OLED service..."

cat > /etc/systemd/system/oled.service << EOF
[Unit]
Description=OLED Display Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 ${SCRIPT_DIR}/oled/oled.py
WorkingDirectory=${SCRIPT_DIR}/oled
Restart=always
RestartSec=5
Nice=10
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable oled.service > /dev/null 2>&1
echo "  OK: OLED service created and enabled"

# --- Shell aliases and sudoers ---
echo ""
echo "[5/5] Setting up shell aliases..."

BASHRC="$USER_HOME/.bashrc"

# OLED aliases
if ! grep -q "oled-stop" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << 'EOF'

# OLED control
alias oled-stop='sudo systemctl stop oled.service'
alias oled-start='sudo systemctl start oled.service'
alias oled-status='systemctl status oled.service'
EOF
    echo "  OK: OLED aliases added"
else
    echo "  OK: OLED aliases already present"
fi

# Passwordless sudo for OLED control
cat > /etc/sudoers.d/oled << EOF
${USERNAME} ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop oled.service
${USERNAME} ALL=(ALL) NOPASSWD: /usr/bin/systemctl start oled.service
EOF
chmod 440 /etc/sudoers.d/oled

# --- Summary ---
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "I2C Buses:"
echo "  Bus 1: Hardware (GPIO 2/3)   - 1MHz    - IMU/Sensors"
echo "  Bus 3: Virtual  (GPIO 18/19) - 200kHz  - ADC"
echo "  Bus 4: Virtual  (GPIO 20/21) - 100kHz  - OLED"
echo ""
echo "OLED Wiring (corner pins 38-40):"
echo "  Pin 38 (GPIO 20) -> SDA"
echo "  Pin 40 (GPIO 21) -> SCL"
echo "  Pin 39 (GND)     -> GND"
echo "  Pin 17 (3.3V)    -> VCC"
echo ""
echo "Commands:"
echo "  oled-start / oled-stop / oled-status"
echo ""
echo ">>> REBOOT REQUIRED: sudo reboot <<<"
