# Hardware Wiring

Physical connections between the STM32F407ZG and the robot peripherals. Pin
assignments are the source of truth in `stm32_firmware/Core/Inc/pin_config.h`;
this document records the cable-level wiring and the gotchas that cost time
during bring-up. Update it whenever a cable or adapter changes.

## Debug / Flash — DAPLink (CMSIS-DAP)

SWD five-wire to the DAPLink. Required for flashing with pyOCD and for the
non-intrusive memory reads used by `MDK-ARM/debug_tools/`.

| DAPLink | STM32 | Note |
|---------|-------|------|
| SWCLK | SWCLK (PA14) | board pin usually labeled `CLK` |
| SWDIO | SWDIO (PA13) | board pin usually labeled `DIO` |
| 3V3 | 3V3 | only if the board is not self-powered |
| GND | GND | mandatory, common ground |
| RST/NRST | NRST | recommended; lets pyOCD reset the target |

Gotcha: the board's SWD header may be labeled `GND 3V3 RST DIO CLK`. `RXD/TXD`
next to it are the UART pair, not SWD — do not wire DAPLink UART pins to SWD.
While the target runs at full speed (blocking I2C + watchdog) the initial SWD
handshake is flaky; connect-under-reset (assert NRST during attach) is required.

## IMU — BMI088 (I2C)

| STM32 | BMI088 | Note |
|-------|--------|------|
| PB8 (I2C1_SCL) | SCL | AF4, open-drain, 400 kHz |
| PB9 (I2C1_SDA) | SDA | AF4, open-drain |
| PB1 (EXTI1) | GYRO INT3 | 1 kHz data-ready interrupt; drives the whole scheduler |
| 3V3 / GND | VCC / GND | |

The accelerometer and gyroscope are two dies at I2C addresses 0x18 and 0x68.
`ACC_PWR_CTRL` (reg 0x7D) must be written 0x04 (not 0x03) or the accelerometer
stays inactive and its data registers read zero — see `bmi088.c`. The PB1 DRDY
interrupt increments `g_system_ticks`; if it does not fire, the main loop body
never runs and no sensor is read.

## Wheel Motors — DDSM315 (RS485)

| STM32 | DDSM315 bus | Note |
|-------|-------------|------|
| PA2 (USART2_TX) | RS485 TX | half-duplex RS485, 115200 baud |
| PA3 (USART2_RX) | RS485 RX | self-echo is discarded before each read |
| GND | GND | |

Left motor ID = 1, right motor ID = 2. Torque command is current-mode; the
firmware maps body-frame torque to per-motor current. Direction signs
`WHEEL_DIR_L/R` must be verified on the bench (calibration step 1).

## Hip Servos — ST3215 (via Waveshare Bus Servo Adapter A)

The ST3215 is a single-wire half-duplex TTL bus servo. It connects to the
STM32 through a **Waveshare Bus Servo Adapter (A)**, which converts the
single-wire bus into a 2-wire UART. The firmware therefore uses USART3 in
**full-duplex** mode (TX + RX on separate pins).

| STM32 | Adapter board | Note |
|-------|---------------|------|
| PB10 (USART3_TX) | TXD | **same-name** (TX→TXD), per Waveshare wiki |
| PB11 (USART3_RX) | RXD | **same-name** (RX→RXD) |
| GND | GND | mandatory common ground |

Servo IDs: LF=1, RF=2, LB=3, RB=4. Bus speed 1 Mbps, 8N1.

Requirements / gotchas:
- The adapter's **jumper must be in position A** (UART mode); position B is for
  USB and will not pass UART traffic.
- Wire same-name (MCU TX → board TXD, MCU RX → board RXD). The adapter crosses
  the pair internally; crossing again on the MCU side breaks communication.
- Servo power is independent of the STM32 3V3; share GND only.
- ST3215 ships with ID = 0; each servo must be re-addressed to 1/2/3/4 with the
  vendor tool before the firmware can query it.
- The firmware was originally `HAL_HalfDuplex_Init` on PB10 only (single-wire).
  Driving the single-wire bus directly works for native ST3215 without an
  adapter, but with this adapter board a 2-wire full-duplex UART is required.

## Pi5 Bridge — UART

| STM32 | Pi5 | Note |
|-------|-----|------|
| PC6 (USART6_TX) | Pi RX | AF8, 921600 baud |
| PC7 (USART6_RX) | Pi TX | RX uses DMA + IDLE line detection |
| GND | GND | |

Frames are `A5 | version | type | length | seq:u16 | timestamp:u32 | payload |
crc8 | 5A`. The Pi must send a HELLO frame whose 16-byte payload equals
`KUAFU_MODEL_HASH` before heartbeat/action traffic is accepted. See
`docs/contracts/interface.md`.

## Power / Common Ground

Every peripheral supply shares GND with the STM32. Servo and motor power must
not be drawn from the STM32 3V3 rail. Verify battery sag under load as part of
calibration (required measurement 7).
