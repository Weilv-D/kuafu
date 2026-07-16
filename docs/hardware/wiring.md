# STM32 Hardware Wiring

`stm32_firmware/Core/Inc/pin_config.h` is the firmware pin source of truth. This
document records the cable-level configuration that passed powered bring-up on
2026-07-16.

## DAPLink

| DAPLink | STM32F407ZG | Note |
|---|---|---|
| SWCLK | PA14 / SWCLK | Debug clock |
| SWDIO | PA13 / SWDIO | Debug data |
| NRST | NRST | Required for reliable connect-under-reset |
| GND | GND | Mandatory common ground |
| 3V3 sense | 3V3 | Voltage reference; do not back-power a powered board |

Probe ID is `LU_2022_8888`; target name is `stm32f407zgtx`.

## BMI088

| STM32F407ZG | BMI088 | Configuration |
|---|---|---|
| PB8 | SCL | I2C1, 400 kHz |
| PB9 | SDA | I2C1 |
| PB1 | gyro INT3 | 1 kHz data-ready |
| 3V3 | VCC | Sensor supply |
| GND | GND | Common ground |

The accelerometer address is `0x18`; the gyroscope address is `0x68`.

## DDSM315 Wheel Bus

The installed RS485 board is an auto-direction module whose TTL labels describe
electrical direction. The TTL pair must therefore be crossed:

| STM32F407ZG | RS485 module | Configuration |
|---|---|---|
| PA2 / USART2_TX | RX | 115200 baud, 8N1 |
| PA3 / USART2_RX | TX | Continuous receive |
| GND | GND | Mandatory common ground |

The differential pair that passed bring-up is:

| RS485 module | DDSM315 bus |
|---|---|
| A | B |
| B | A |

Left motor ID is 1 and right motor ID is 2. Both motors share the differential
pair. Change an ID only with one motor connected to the bus.

## ST3215 Servo Bus

The Waveshare Bus Servo Adapter A converts the single-wire servo bus to a
two-wire UART. Its labels are used directly:

| STM32F407ZG | Adapter A | Configuration |
|---|---|---|
| PB10 / USART3_TX | TXD | 1 Mbps, 8N1 |
| PB11 / USART3_RX | RXD | Full-duplex UART side |
| GND | GND | Mandatory common ground |

The adapter jumper is in position A. Servo IDs and firmware order are
`[1,2,3,4]=[A_l,A_r,B_l,B_r]`.

## Raspberry Pi 5 Link

| STM32F407ZG | Raspberry Pi 5 | Configuration |
|---|---|---|
| PC6 / USART6_TX | RX | 921600 baud |
| PC7 / USART6_RX | TX | Circular DMA reception |
| GND | GND | Mandatory common ground |

The Pi must send a compatible model-hash `HELLO`, a fresh heartbeat, and an
explicit mode request before either wheel can be enabled.

## Power

Servo and wheel power are independent of the STM32 3V3 rail. All supplies share
one ground reference. Battery voltage measurement is not wired; firmware reports
`battery_mv=0` as unavailable and performs no battery-voltage fault check.
