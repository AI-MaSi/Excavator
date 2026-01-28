#!/bin/bash
# OLED Display Setup Script
# Sets up I2C buses and OLED systemd service

set -e

echo "=== OLED Display Setup ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup.sh"
    exit 1
fi

CONFIG="/boot/firmware/config.txt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- I2C Configuration ---
echo ""
echo "[1/4] Configuring I2C buses..."

# Backup config
cp "$CONFIG" "$CONFIG.bak.$(date +%Y%m%d_%H%M%S)"

# Enable hardware I2C at 1MHz if not already
if ! grep -q "^dtparam=i2c_arm=on" "$CONFIG"; then
    sed -i '/\[all\]/i dtparam=i2c_arm=on' "$CONFIG"
    echo "  Added: Hardware I2C enabled"
fi

if ! grep -q "^dtparam=i2c_arm_baudrate=1000000" "$CONFIG"; then
    sed -i '/dtparam=i2c_arm=on/a dtparam=i2c_arm_baudrate=1000000' "$CONFIG"
    echo "  Added: Hardware I2C baudrate 1MHz"
else
    echo "  OK: Hardware I2C already at 1MHz"
fi

# Remove any existing virtual I2C configs
sed -i '/dtoverlay=i2c-gpio,bus=3/d' "$CONFIG"
sed -i '/dtoverlay=i2c-gpio,bus=4/d' "$CONFIG"
sed -i '/# Virtual I2C buses/d' "$CONFIG"
sed -i '/# Bus 3:/d' "$CONFIG"
sed -i '/# Bus 4:/d' "$CONFIG"

# Add virtual I2C buses at end of file
cat >> "$CONFIG" << 'EOF'

# Virtual I2C buses (bit-banged)
# Bus 3: GPIO 16/19 ~200kHz (ADC)
dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=16,i2c_gpio_scl=19,i2c_gpio_delay_us=1
# Bus 4: GPIO 20/21 ~100kHz (OLED)
dtoverlay=i2c-gpio,bus=4,i2c_gpio_sda=20,i2c_gpio_scl=21
EOF

echo "  Added: Virtual I2C bus 3 (GPIO 16/19, 200kHz)"
echo "  Added: Virtual I2C bus 4 (GPIO 20/21, 100kHz)"

# --- Install packages ---
echo ""
echo "[2/4] Installing packages..."
apt-get install -y python3-pil python3-smbus i2c-tools > /dev/null 2>&1 || true
pip3 install adafruit-circuitpython-ssd1306 adafruit-blinka adafruit-extended-bus --break-system-packages --quiet 2>/dev/null || true
echo "  OK: Packages installed"

# --- Systemd service ---
echo ""
echo "[3/4] Creating systemd service..."

cat > /etc/systemd/system/oled.service << EOF
[Unit]
Description=OLED Display Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 ${SCRIPT_DIR}/oled.py
WorkingDirectory=${SCRIPT_DIR}
Restart=always
RestartSec=5
Nice=10
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable oled.service
echo "  OK: Service created and enabled"

# --- Shell aliases ---
echo ""
echo "[4/4] Setting up aliases..."

ALIAS_BLOCK='
# OLED control aliases
alias oled-stop="sudo systemctl stop oled.service"
alias oled-start="sudo systemctl start oled.service"
alias oled-status="systemctl status oled.service"'

# Add to user's bashrc if not present
USER_HOME=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)
BASHRC="$USER_HOME/.bashrc"

if ! grep -q "oled-stop" "$BASHRC" 2>/dev/null; then
    echo "$ALIAS_BLOCK" >> "$BASHRC"
    echo "  OK: Aliases added to $BASHRC"
else
    echo "  OK: Aliases already in $BASHRC"
fi

# --- Sudoers for passwordless control ---
cat > /etc/sudoers.d/oled << EOF
${SUDO_USER:-$USER} ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop oled.service
${SUDO_USER:-$USER} ALL=(ALL) NOPASSWD: /usr/bin/systemctl start oled.service
EOF
chmod 440 /etc/sudoers.d/oled
echo "  OK: Passwordless sudo configured"

# --- Done ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "I2C Bus Layout:"
echo "  Bus 1: Hardware (GPIO 2/3)  - 1MHz"
echo "  Bus 3: Virtual  (GPIO 16/19) - 200kHz (ADC)"
echo "  Bus 4: Virtual  (GPIO 20/21) - 100kHz (OLED)"
echo ""
echo "OLED Wiring (directly adjacent to each other at corner of board!):"
echo "  Pin 38 (GPIO 20) -> SDA"
echo "  Pin 40 (GPIO 21) -> SCL"
echo "  Pin 39 (GND)     -> GND"
echo "  Pin 17 (3.3V)    -> VCC"
echo ""
echo "Next steps:"
echo "  1. Reboot: sudo reboot"
echo "  2. After reboot, verify: i2cdetect -y 4"
echo "  3. Control: oled-start, oled-stop, oled-status"
