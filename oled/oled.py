import socket
import struct
import fcntl
import os
from adafruit_extended_bus import ExtendedI2C
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
from time import sleep, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, 'Montserrat-VariableFont_wght.ttf')
UPDATE_INTERVAL = 2  # seconds between updates

# OLED setup on virtual I2C bus 4 (GPIO 20=SDA, GPIO 21=SCL)
# frequency=None suppresses warning; bus speed is set by device tree, not Python
i2c = ExtendedI2C(4, frequency=None)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3D, reset=None)

# Pre-load fonts once
FONT_SMALL = ImageFont.truetype(FONT_PATH, 12)
FONT_LARGE = ImageFont.truetype(FONT_PATH, 19)

# Reusable image buffer
IMAGE = Image.new("1", (128, 64))
DRAW = ImageDraw.Draw(IMAGE)


def get_default_interface():
    """Get default route interface from /proc/net/route (no subprocess)."""
    try:
        with open('/proc/net/route', 'r') as f:
            for line in f.readlines()[1:]:  # skip header
                parts = line.strip().split()
                if parts[1] == '00000000':  # default route
                    return parts[0]
    except (IOError, IndexError):
        pass
    return None


def get_ip_address(interface):
    """Get IP using ioctl (no subprocess)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            sock.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', interface.encode()[:15])
        )[20:24])
        sock.close()
        return ip
    except (IOError, OSError):
        return None


def get_ssid(interface):
    """Read SSID from /proc/net/wireless or sysfs (no subprocess)."""
    try:
        # Try wireless extension via ioctl
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        buf = struct.pack('256s', interface.encode()[:15])
        result = fcntl.ioctl(sock.fileno(), 0x8B1B, buf)  # SIOCGIWESSID request
        sock.close()
        ssid = result[16:48].decode('utf-8').rstrip('\x00')
        return ssid if ssid else None
    except (IOError, OSError):
        pass
    return None


def get_signal_strength(interface):
    """Read signal from /proc/net/wireless (no subprocess)."""
    try:
        with open('/proc/net/wireless', 'r') as f:
            for line in f.readlines()[2:]:  # skip 2 header lines
                if interface in line:
                    parts = line.split()
                    # Column 3 is link quality, column 4 is signal level
                    signal = float(parts[3].rstrip('.'))
                    # Convert to dBm-like value if it's 0-100 range
                    if signal > 0:
                        return int(-100 + signal * 0.5 + 20)
    except (IOError, IndexError, ValueError):
        pass
    return None


def get_cpu_temp():
    """Read CPU temp (already efficient)."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read()) // 1000
    except IOError:
        return None


def draw_signal_bars(rssi, x, y):
    """Draw WiFi signal strength bars."""
    if rssi is None:
        return
    bars = 3 if rssi > -65 else 2 if rssi > -75 else 1 if rssi > -85 else 0
    for i in range(bars):
        y_pos = y - i * 5
        DRAW.rectangle((x, y_pos, x + 5, y_pos + 3), fill=255)


def update_display(interface, ssid, network_name, ip, rssi, cpu_temp):
    """Render display with all info on one screen."""
    DRAW.rectangle((0, 0, 127, 63), fill=0)  # clear

    # Top line: network type + signal bars
    is_wifi = interface and 'wlan' in interface
    label = f"SSID: {ssid}" if is_wifi and ssid else f"Net: {network_name}"
    DRAW.text((2, 0), label[:18], font=FONT_SMALL, fill=255)  # truncate long names

    if is_wifi and rssi:
        draw_signal_bars(rssi, 118, 10)

    DRAW.line((0, 14, 127, 14), fill=255)

    # IP address (large)
    ip_str = ip or "No IP"
    ip_w = DRAW.textlength(ip_str, font=FONT_LARGE)
    DRAW.text(((128 - ip_w) // 2, 18), ip_str, font=FONT_LARGE, fill=255)

    # CPU temp at bottom
    if cpu_temp is not None:
        temp_str = f"CPU: {cpu_temp}C"
        temp_w = DRAW.textlength(temp_str, font=FONT_SMALL)
        DRAW.text(((128 - temp_w) // 2, 50), temp_str, font=FONT_SMALL, fill=255)

    oled.image(IMAGE)
    oled.show()


# Initial clear
oled.fill(0)
oled.show()

# State for change detection
prev_state = None

while True:
    interface = get_default_interface()
    ip = get_ip_address(interface) if interface else None
    is_wifi = interface and 'wlan' in interface
    ssid = get_ssid(interface) if is_wifi else None
    network_name = ssid or ("WiFi" if is_wifi else ("Wired" if interface else "None"))
    rssi = get_signal_strength(interface) if is_wifi else None
    cpu_temp = get_cpu_temp()

    state = (ssid, network_name, ip, rssi, cpu_temp)
    if state != prev_state:
        update_display(interface, ssid, network_name, ip, rssi, cpu_temp)
        prev_state = state

    sleep(UPDATE_INTERVAL)
