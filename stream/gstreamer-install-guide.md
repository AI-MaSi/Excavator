# GStreamer Installation Guide

## On Excavator (Linux)

1. Update system packages:
```bash
sudo apt-get update
sudo apt-get upgrade
```

2. Install GStreamer and dependencies:
```bash
sudo apt-get install libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-bad1.0-dev \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-tools \
     gstreamer1.0-gl \
     gstreamer1.0-gtk3
sudo apt-get install gstreamer1.0-qt5
```

## On Receiving PC (Windows 10)

1. Download GStreamer:
   - URL: https://gstreamer.freedesktop.org/data/pkg/windows/1.24.10/msvc/gstreamer-1.0-msvc-x86_64-1.24.10.msi
   - Install the downloaded package. Select "Complete" installation!

2. Add to PATH:
   - Press Windows key + R
   - Type `sysdm.cpl` and press Enter
   - Go to "Advanced" tab
   - Click "Environment Variables"
   - Under "System Variables", select "Path"
   - Click "Edit" → "New"
   - Add: `C:\gstreamer\1.0\msvc_x86_64\bin`
   - Click "OK" on all windows
   - Restart Command Prompt/PowerShell

## Usage (With VLC using RTP, low latency UDP connection) 
(assuming camera is `/dev/video0` and receiver IP is set)

On excavator:
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! rtph264pay mtu=1200 ! udpsink host=<receiver_IP> port=8081
```

On receiving PC:
```bash
gst-launch-1.0 udpsrc port=8081 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
```
