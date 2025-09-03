# Elderly Guardian — Edge Fall Detection

Real-time fall detection system running YOLOv8 Pose on RK3588 SBC with Metis M.2 accelerator, monitoring video feed and triggering smart home alerts when falls are detected.

## Overview

This system watches a Logitech C920 video feed using computer vision to detect falls in real-time, automatically triggering a TP-Link Kasa smart plug for downstream alerts and automations.

## Hardware Requirements

- **SBC/SoC**: RK3588 (e.g., Aetina/AIO-style dev box)
- **AI Accelerator**: Metis M.2 (Axelera) via Voyager SDK
- **Camera**: Logitech C920 (USB, 1280×720@30fps recommended)
- **Wi-Fi**: Edimax EW-7822UAC (AC1200 USB)
- **Smart Plug**: TP-Link Kasa (e.g., KP115) for alerts/automations

### Why This Hardware Combo?

- **RK3588**: Provides excellent I/O and video pipeline capabilities
- **Metis + Voyager**: Offloads heavy pose estimation to dedicated accelerator
- **C920**: Rock-solid UVC camera with MJPEG/H.264 support
- **Kasa**: Easy local automation without cloud dependencies

## Installation

### Directory Structure

```
/home/aetina/
├── voyager-sdk/                         # Voyager SDK (AXELERA_FRAMEWORK)
└── fall_detection_app/
    ├── fall_detection.py                # Main application
    ├── app-config.yaml                  # Configuration file
    ├── yolov8lpose-fall-detector.yaml   # Voyager pipeline/model config
    └── fall_demo.mp4                    # Output video (generated)
```

### Environment Setup

1. **Environment Variables** (add to `~/.bashrc`):

```bash
# Voyager SDK home
export AXELERA_FRAMEWORK=/home/aetina/voyager-sdk

# Python module discovery
export PYTHONPATH="$AXELERA_FRAMEWORK:$PYTHONPATH"

# Native libraries and GStreamer plugins
export LD_LIBRARY_PATH="$AXELERA_FRAMEWORK/lib:$LD_LIBRARY_PATH"
export GST_PLUGIN_PATH="$AXELERA_FRAMEWORK/gst:$GST_PLUGIN_PATH"
```

Reload your shell:
```bash
source ~/.bashrc
```

2. **System Dependencies**:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip v4l-utils \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav
```

3. **Python Environment**:

```bash
cd /home/aetina/fall_detection_app
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python pyyaml python-kasa
```

### Configuration Files

**app-config.yaml**:
```yaml
config:
  data_root: /home/aetina/voyager-sdk
  pipeline_path: /home/aetina/fall_detection_app/yolov8lpose-fall-detector.yaml
  source: "usb:/dev/video20:1280x720@30"
```

## Usage

### Basic Startup

```bash
cd /home/aetina/fall_detection_app
source venv/bin/activate

python3 fall_detection.py app-config.yaml \
  --floor-band-pct 0.22 --ankle-floor-band 0.20 --hips-floor-band 0.15 \
  --descent-window 1.0 --require-descent-px 55 --bottom-descent-px 30 \
  --prone-confirm-frames 6 \
  --prox-height-frac 0.55 --prox-area-frac 0.28 --retreat-area-drop-frac 0.10 \
  --min-area 90000 --max-area 250000 --min-conf 0.65 --min-kp-visible 8 \
  --fall-latch-seconds 7 --recovery-frames 10 --recovery-upright-deg 20 --recovery-ascend-px 50 \
  --kasa-ip 10.0.0.214 --kasa-on-seconds 5 --kasa-cooldown 30 --fall-frames 5 \
  --out /home/aetina/fall_demo.mp4 --log INFO
```

### Command Line Options

- `--display`: Opens preview window (requires GUI/X11)
- `--out`: Specify output video file path
- `--kasa-ip`: IP address of TP-Link Kasa smart plug
- `--log`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Smart Plug Setup

### Discover Kasa Device IP

```bash
source venv/bin/activate
kasa discover  # Shows all Kasa devices and their IPs on your network
```

**Note**: Consider setting a static DHCP lease in your router to prevent IP changes.

### Alert Integration

The system pulses the Kasa smart plug (ON → OFF) when falls are detected, which can trigger:
- Alexa routines
- Home automation systems
- Emergency notifications

## How It Works

### Processing Pipeline

1. **Video Source**: Voyager SDK captures from USB camera (`/dev/video20`)
2. **Preprocessing**: Frames letterboxed to 640×640, normalized for YOLOv8
3. **AI Inference**: YOLOv8 Pose runs on Metis accelerator via Voyager SDK
4. **Detection**: Returns bounding boxes and 17 keypoints per person
5. **Fall Analysis**: Custom heuristics analyze pose and motion patterns

### Fall Detection Logic

**Quality Filters**:
- Bounding box area: 90,000 - 250,000 pixels²
- Pose confidence ≥ 0.65
- Minimum 8 visible keypoints (confidence ≥ 0.35)

**Fall Detection Methods**:

**Method A - Downward Motion**:
- Detects rapid downward movement of core body and bounding box
- Checks if person ends up horizontal near floor level
- Analyzes descent within 1.0 second window

**Method B - Floor-Prone Detection**:
- Identifies sustained horizontal position at floor level
- Catches slow or soft falls that Method A might miss

**False Positive Prevention**:
- **Proximity Guard**: Suppresses alerts when person is very close to camera
- **Retreat Guard**: Ignores rapid area shrinkage (walking away)
- **Recovery Latch**: Maintains alert state until clear recovery is detected

## Troubleshooting

### Common Issues

**No pose detection**:
- Verify `pipeline_path` in configuration
- Ensure Voyager SDK can write to `voyager-sdk/weights` directory
- YOLOv8 weights download automatically on first run

**GStreamer errors**:
```bash
export AXELERA_FRAMEWORK=/home/aetina/voyager-sdk
export PYTHONPATH="$AXELERA_FRAMEWORK:$PYTHONPATH"
export LD_LIBRARY_PATH="$AXELERA_FRAMEWORK/lib:$LD_LIBRARY_PATH"
export GST_PLUGIN_PATH="$AXELERA_FRAMEWORK/gst:$GST_PLUGIN_PATH"
```

**Display window won't open**:
- Run headless by omitting `--display` flag
- Ensure `--out` parameter is set for video recording

**Kasa IP changes**:
- Use `kasa discover` to find current IP
- Set DHCP reservation in router settings

### Performance Tuning

**Reduce false positives near camera**:
```bash
--prox-height-frac 0.60 --prox-area-frac 0.32
```

**Improve soft fall detection**:
```bash
--prone-confirm-frames 4 --floor-band-pct 0.25
```

**Reduce USB bandwidth (if needed)**:
```yaml
source: "usb:/dev/video20:640x480@30"
```

## Auto-Start Service

Create systemd service for automatic startup:

**`/etc/systemd/system/elderly-guardian.service`**:
```ini
[Unit]
Description=Elderly Guardian Fall Detection
After=network-online.target

[Service]
User=aetina
WorkingDirectory=/home/aetina/fall_detection_app
Environment=AXELERA_FRAMEWORK=/home/aetina/voyager-sdk
Environment=PYTHONPATH=/home/aetina/voyager-sdk
Environment=LD_LIBRARY_PATH=/home/aetina/voyager-sdk/lib
Environment=GST_PLUGIN_PATH=/home/aetina/voyager-sdk/gst
ExecStart=/home/aetina/fall_detection_app/venv/bin/python3 \
  /home/aetina/fall_detection_app/fall_detection.py \
  /home/aetina/fall_detection_app/app-config.yaml \
  --out /home/aetina/fall_demo.mp4 --log INFO \
  --kasa-ip 10.0.0.214 --kasa-on-seconds 5 --kasa-cooldown 30 \
  --floor-band-pct 0.22 --ankle-floor-band 0.20 --hips-floor-band 0.15 \
  --descent-window 1.0 --require-descent-px 55 --bottom-descent-px 30 \
  --prone-confirm-frames 6 --fall-latch-seconds 7
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable elderly-guardian
sudo systemctl start elderly-guardian
```

## Demo Scenarios

For testing and demonstration, the system handles:

1. **Normal Activity**: Person walking through room (no alerts)
2. **Sitting**: Person sits on couch/chair (suppressed by floor logic)
3. **Actual Fall**: Person falls to floor (triggers red "FALL" alert and Kasa pulse)
4. **Recovery**: Person stands up (alert clears after recovery criteria met)

## Future Enhancements

- **Room Calibration**: Save per-site floor ROI for improved accuracy
- **Activity Recognition**: Add "lying on couch/bed/chair" detection
- **Integration Options**: MQTT/Webhook support alongside Kasa alerts
- **Multi-Camera Support**: Expand to multiple camera feeds
