import subprocess
import board
import digitalio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
from time import sleep

# OLED setup
RESET_PIN = digitalio.DigitalInOut(board.D4)
i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3D, reset=RESET_PIN)


def clear_display():
    oled.fill(0)
    oled.show()


def get_active_interface():
    try:
        cmd = "ip route get 1.1.1.1 | awk '{print $5}' | head -n 1"
        interface = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return interface
    except subprocess.CalledProcessError:
        return None


def get_ip_address(interface):
    try:
        cmd = f"ip addr show {interface} | grep 'inet ' | awk '{{print $2}}' | cut -d'/' -f1"
        IP = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return IP
    except subprocess.CalledProcessError:
        return "No IP Found"


def get_ssid(interface):
    try:
        cmd = f"iwgetid -r {interface}"
        SSID = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return SSID
    except subprocess.CalledProcessError:
        return "Not Connected"


# new feature, get RSSI
def get_rssi(interface):
    try:
        # Use iwconfig to get signal level
        cmd = f"iwconfig {interface} | grep 'Signal level' | awk '{{print $4}}' | cut -d'=' -f2"
        rssi = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return rssi
    except subprocess.CalledProcessError:
        return "RSSI Unavailable"



def update_display(interface, network_name, IP, rssi=None):
    # Create blank image for drawing.
    width = oled.width
    height = oled.height
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)

    font_path = '/home/pi/GitHub/Excavator/misc/show_ip/Montserrat-VariableFont_wght.ttf'
    font_size = 12
    font_size_ip = 19
    font_savonia = ImageFont.truetype(font_path, font_size)
    font_savonia_ip = ImageFont.truetype(font_path, font_size_ip)

    network_label = "SSID: " if "wlan" in interface else "Network:"

    draw.text((0, 8), f"{network_label}", font=font_savonia, fill=255)
    draw.text((1, 25), f"{network_name}", font=font_savonia, fill=255)
    draw.text((1, 43), f"{IP}", font=font_savonia_ip, fill=255)

    if rssi:
        draw.text((80, 8), f"RSSI: {rssi}", font=font_savonia, fill=255)

    oled.image(image)
    oled.show()


clear_display()

previous_network_name, previous_IP, previous_rssi = None, None, None
while True:
    interface = get_active_interface()
    IP = get_ip_address(interface) if interface else "NONE"
    network_name = get_ssid(interface) if "wlan" in interface else "Wired" if interface else "NONE"
    rssi = get_rssi(interface) if "wlan" in interface else None

    if network_name != previous_network_name or IP != previous_IP or (rssi and previous_rssi != rssi):
        update_display(interface, network_name, IP, rssi)
        previous_network_name, previous_IP, previous_rssi = network_name, IP, rssi

    sleep(30)
