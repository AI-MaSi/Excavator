# This code prints the SSID and IP-address so you can connect to the excavator
# Adafruit SSD1306 128x64 screen used
# set to run every time excavator boots up. After service "network.target"

import subprocess
import board
import digitalio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

# OLED setup
RESET_PIN = digitalio.DigitalInOut(board.D4)
i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3D, reset=RESET_PIN)

# Clear display
oled.fill(0)
oled.show()

# Get IP address
cmd = "hostname -I | cut -d' ' -f1"
IP = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()

# Get connected SSID
cmd = "iwgetid -r"
SSID = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()

# Create blank image for drawing
width = oled.width
height = oled.height
image = Image.new("1", (width, height))

# Get drawing object to draw on image
draw = ImageDraw.Draw(image)
font_path = '/home/pi/GitHub/Excavator/misc/show_ip/Montserrat-VariableFont_wght.ttf'
font_size = 11
font_size_ip = 19

font = ImageFont.load_default()
font_savonia = ImageFont.truetype(font_path, font_size)
font_savonia_ip = ImageFont.truetype(font_path, font_size_ip)

# Draw SSID and IP on the image
draw.text((0, 10), "SSID: " + SSID, font=font_savonia, fill=255)
draw.text((1, 35), "IP: " + IP, font=font_savonia_ip, fill=255)

# Display image
oled.image(image)
oled.show()
