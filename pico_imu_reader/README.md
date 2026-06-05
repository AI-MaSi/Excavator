# pico_imu_reader

Firmware for **Seeed XIAO RP2040** that reads up to four ISM330DHCX IMUs over two I2C buses and streams fused orientation data as binary frames over USB CDC. Used as the IMU front-end for the kaivuriprokkis excavator project.

---

## Hardware assembly

### Board

**Seeed XIAO RP2040** — pin assignments below are fixed and match the physical wiring. Do not change them without rewiring.

### I2C bus wiring

| Bus  | SDA | SCL | IMU addresses |
|------|-----|-----|---------------|
| I2C0 | P28 | P29 | 0x6A, 0x6B    |
| I2C1 | P6  | P7  | 0x6A, 0x6B    |

Up to two IMUs per bus — set each sensor's `SDO` pin low for 0x6A or high for 0x6B. The firmware probes all four slots at runtime; only detected IMUs are streamed.

Power the IMU modules from the **3V3 and GND pins on one side of the board**; I2C0 (P28/P29) and I2C1 (P6/P7) are on the opposite side.

### IMU

**ISM330DHCX** (detected via `WHO_AM_I == 0x6B`). Accelerometer range: ±2 g. Gyro range: 250 dps (default). Internal ODR: 416 Hz; output rate to host: 200 Hz.

---

## Building

Requires the **Raspberry Pi Pico VS Code extension** or a manual CMake toolchain with `arm-none-eabi-gcc`.

```bash
# Clone with submodule
git submodule update --init --recursive

# Configure (VS Code extension does this automatically)
cmake -B build -DPICO_BOARD=seeed_xiao_rp2040

# Build all targets
cmake --build build
```

Three firmware targets are produced:

| Target         | Description                                                  |
|----------------|--------------------------------------------------------------|
| `main`         | Full stack: AHRS fusion, startup calibration, USB CDC stream |
| `simple_stream`| Minimal stream without AHRS — raw IMU values only            |
| `i2c_scan_demo`| Bus scan utility; prints detected devices over USB serial    |

Flash the desired `.uf2` from `build/` by holding BOOT and connecting USB, then copying the file to the mass-storage drive.

---

## Startup calibration (`main` firmware)

On power-up the board runs a **30-second stationary calibration window**:

- Streams `CAL_WAIT` to the host during the window.
- Collects gyro and accelerometer statistics for each IMU.
- Acceptance thresholds (hardcoded in `main.c`):
  - Gyro stddev < 0.5 dps per axis
  - Accel stddev < 0.05 g per axis
  - Accel norm 0.75 – 1.25 g
- If passed: applies measured gyro mean as bias offset and begins streaming.
- If failed: sends `ERR_CAL` — power-cycle and keep the machine still during the window.

Leave the machine **completely stationary** for the first 30 seconds after power-on.

---

## Binary protocol (USB CDC)

All frames share a common 3-byte header: `0xAA 0xBB <version>`.

| Frame            | `version` | Description                                                       |
|------------------|-----------|-------------------------------------------------------------------|
| Descriptor       | `0x02`    | Sent at stream start — sensor count, sample rate, bus/addr per sensor |
| Data             | `0x01`    | One frame per sample: sequence, timestamp, per-sensor floats       |
| Control/error    | `0x02`    | `CAL_WAIT`, `ERR_CAL`, `ERR_IMU`, `ERR_I2C` status codes         |
| Calibration report | `0x02`  | Per-IMU stats sent after calibration window                       |

Each data frame contains 7 floats per sensor: `qw qx qy qz pitch roll yaw` (gravity-based pitch/roll).

Pitch and roll use the gravity-vector method, not `2*atan2(y, w)`. The host-side reader must match this convention.

---

## Verifying hardware with `i2c_scan_demo`

Flash `i2c_scan_demo.uf2`, then open a serial terminal (115200 baud). The firmware scans both buses and prints detected addresses. Use this to confirm all IMUs are physically present before flashing the main firmware.

---

## Settings

AHRS settings are hardcoded in `src/imu_reader/settings/settings.c`:

```c
.sampleRate          = 200,     // Hz
.gyroRangeDps        = 250.0f,
.ahrsGain            = 4.5f,
.ahrsAccelRejection  = 20.0f,
.ahrsRecoveryPeriodS = 0.5f,
```

Edit and rebuild to change them. There is no runtime configuration interface.
