#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

#include "stm32f4xx.h"
#include "kuafu_generated.h"

/* ========================================================================== */
/*                           System Configuration                             */
/* ========================================================================== */
#define SYS_CLOCK_HZ            168000000U
#define CONTROL_LOOP_FREQ       1000U           /* IMU & Fusion loop at 1 kHz */
#define MOTOR_LOOP_FREQ         250U            /* DDSM Motor loop at 250 Hz  */
#define SERVO_LOOP_FREQ         50U             /* ST3215 Servo loop at 50 Hz */

/* ========================================================================== */
/*                           Peripheral Pin Mapping                           */
/* ========================================================================== */

/* USART1: CH340 USB-to-TTL Program / Debug Port */
#define DEBUG_USART             USART1
#define DEBUG_USART_CLK_EN()    __HAL_RCC_USART1_CLK_ENABLE()
#define DEBUG_GPIO_CLK_EN()     __HAL_RCC_GPIOA_CLK_ENABLE()
#define DEBUG_TX_PIN            GPIO_PIN_9
#define DEBUG_TX_PORT           GPIOA
#define DEBUG_RX_PIN            GPIO_PIN_10
#define DEBUG_RX_PORT           GPIOA
#define DEBUG_USART_AF          GPIO_AF7_USART1

/* USART2: DDSM315 RS485 Motor Bus Port */
#define DDSM_USART              USART2
#define DDSM_USART_CLK_EN()     __HAL_RCC_USART2_CLK_ENABLE()
#define DDSM_GPIO_CLK_EN()      __HAL_RCC_GPIOA_CLK_ENABLE()
#define DDSM_TX_PIN             GPIO_PIN_2
#define DDSM_TX_PORT            GPIOA
#define DDSM_RX_PIN             GPIO_PIN_3
#define DDSM_RX_PORT            GPIOA
#define DDSM_USART_AF           GPIO_AF7_USART2
#define DDSM_LEFT_ID            1
#define DDSM_RIGHT_ID           2

/* Bench-only DDSM315 ID assignment. Set to 1..253, connect exactly one motor
 * to the RS485 bus, power-cycle that motor, flash once, then restore to 0.
 * The firmware sends the official ID-set frame five times while all wheel
 * actuation remains disabled. */
#define DDSM_ID_CALIBRATION_TARGET 0

/* USART3: ST3215 Servo Port (full-duplex via Waveshare Bus Servo Adapter A).
 * The ST3215 single-wire half-duplex bus is converted to a 2-wire UART by the
 * adapter board; wire STM32 TX(PB10)->adapter TXD, RX(PB11)->adapter RXD
 * (same-name, per Waveshare wiki), GND common. Adapter jumper must be at A. */
#define SERVO_USART             USART3
#define SERVO_USART_CLK_EN()    __HAL_RCC_USART3_CLK_ENABLE()
#define SERVO_GPIO_CLK_EN()     __HAL_RCC_GPIOB_CLK_ENABLE()
#define SERVO_TX_PIN            GPIO_PIN_10
#define SERVO_TX_PORT           GPIOB
#define SERVO_RX_PIN            GPIO_PIN_11
#define SERVO_RX_PORT           GPIOB
#define SERVO_USART_AF          GPIO_AF7_USART3
#define SERVO_LF_ID             1
#define SERVO_RF_ID             2
#define SERVO_LB_ID             3
#define SERVO_RB_ID             4

/* USART6: Raspberry Pi 5 Bridge Port */
#define PI_USART                USART6
#define PI_USART_CLK_EN()       __HAL_RCC_USART6_CLK_ENABLE()
#define PI_GPIO_CLK_EN()        __HAL_RCC_GPIOC_CLK_ENABLE()
#define PI_TX_PIN               GPIO_PIN_6
#define PI_TX_PORT              GPIOC
#define PI_RX_PIN               GPIO_PIN_7
#define PI_RX_PORT              GPIOC
#define PI_USART_AF             GPIO_AF8_USART6

/* I2C1: BMI088 IMU Interface */
#define IMU_I2C                 I2C1
#define IMU_I2C_CLK_EN()        __HAL_RCC_I2C1_CLK_ENABLE()
#define IMU_GPIO_CLK_EN()       __HAL_RCC_GPIOB_CLK_ENABLE()
#define IMU_SCL_PIN             GPIO_PIN_8
#define IMU_SCL_PORT            GPIOB
#define IMU_SDA_PIN             GPIO_PIN_9
#define IMU_SDA_PORT            GPIOB
#define IMU_I2C_AF              GPIO_AF4_I2C1

/* EXTI1: Gyro Data Ready Interrupt */
#define IMU_INT_PIN             GPIO_PIN_1
#define IMU_INT_PORT            GPIOB
#define IMU_INT_GPIO_CLK_EN()   __HAL_RCC_GPIOB_CLK_ENABLE()
#define IMU_INT_EXTI_IRQn       EXTI1_IRQn

/* ========================================================================== */
/*                         LQR & Kinematics Constants                         */
/* ========================================================================== */

/* LQR Gain K Vector: [e_x, theta, e_x_dot, theta_dot] */
#define LQR_K0                  KUAFU_LQR_K0
#define LQR_K1                  KUAFU_LQR_K1
#define LQR_K2                  KUAFU_LQR_K2
#define LQR_K3                  KUAFU_LQR_K3
#define LQI_KI                  KUAFU_LQI_KI

/* DDSM315 Current to Torque scaling factor */
/* tau = 0.75 * I_amps. Command maps raw [-32767, 32767] to [-8.0, 8.0] Amps.
 * Hence: I_raw = tau * (32767.0f / 6.0f) ~= 5461.17f * tau */
#define DDSM_TORQUE_TO_RAW      5461.17f
#define DDSM_RAW_TO_TORQUE      (1.0f / DDSM_TORQUE_TO_RAW)

/* Wheel geometry & rated torque (mirrors kuafu_physics.py) */
#define WHEEL_RADIUS_M          R_WHEEL_M

/* Base-layer heading/rate tracking (mirrors kuafu_physics.py) */
#define YAW_KP                  KUAFU_YAW_KP
#define YAW_KD                  KUAFU_YAW_KD

/* Wheel rotation direction mapping (body frame: +torque/+velocity = forward).
 * Left/right hub motors may be mirror mounted; set the affected side to -1.0f.
 * Applied to both torque commands and velocity feedback so the LQR and the Pi
 * telemetry share one consistent body frame. Verify on bench (defaults: none). */
#define WHEEL_DIR_L             (+1.0f)
#define WHEEL_DIR_R             (+1.0f)

/* 5-Bar Linkage Kinematics parameters (meters) */
#define KIN_LEG_A               (A_LEN_MM * 0.001f)
#define KIN_LEG_B               (B_LEN_MM * 0.001f)
#define KIN_LEG_C               (-(AX_MM) * 0.001f)
#define KIN_MIN_LEG_D0          (D0_MIN_MM * 0.001f)
#define KIN_MAX_LEG_D0          (D0_MAX_MM * 0.001f)

/* ST3215 Servo mapping */
#define SERVO_CENTER_TICKS      2048
#define SERVO_TICKS_PER_RAD     (4096.0f / (2.0f * 3.14159265f))

/* Bench-only mode for measuring the per-servo dwell zero. While enabled the
 * firmware keeps wheel actuation inactive, disables all ST3215 torque, and
 * only polls servo feedback. Set back to 0 after SERVO_CENTER_INIT is filled. */
#define SERVO_ZERO_CALIBRATION_MODE 0

/* --------------------------------------------------------------------------
 * Servo interface contract (firmware is the source of truth):
 *   Pi sends sim-frame joint angles (dwell = 0), order [LF, RF, LB, RB] =
 *   [hip_A_l, hip_A_r, hip_B_l, hip_B_r]. Firmware maps each servo as:
 *     tick = SERVO_DIR[i] * q[i] * SERVO_TICKS_PER_RAD + SERVO_CENTER[i]
 *   - SERVO_DIR: maps the shared joint sign to each servo's raw tick sign.
 *     For extension (qA<0, qB>0), the expected raw tick changes are
 *     [LF decreases, RF increases, LB increases, RB decreases]. Do not use
 *     viewing-dependent CW/CCW.
 *   - SERVO_CENTER: per-servo mechanical zero tick (dwell posture).
 *   See docs/hardware/calibration.md for ordering and the physical test.
 * ------------------------------------------------------------------------ */
#define SERVO_DIR_INIT          { +1, -1, +1, -1 }   /* [LF, RF, LB, RB] */
#define SERVO_CENTER_INIT       { 275, 1097, 2809, 1023 } /* [LF, RF, LB, RB], 2026-07-16 */

/* ========================================================================== */
/*                     IMU Attitude Axis Mapping (calibratable)               */
/* ========================================================================== */
/* Map the balance-relevant tilt/rate to the physical IMU mounting.
 * Verify on bench: tilt the body forward/back and confirm ATT_PITCH() rises
 * monotonically with the correct sign. Adjust source/sign/index if mounted
 * on a different axis or orientation. */
#define ATT_PITCH(mahony)        (+1.0f * (mahony)->pitch)  /* body pitch (rad) */
#define ATT_PITCH_RATE_IDX       1                          /* gyro[] index (Y) */
#define ATT_PITCH_RATE_SIGN      (+1.0f)                    /* pitch-rate sign  */

/* ========================================================================== */
/*                            Safety Thresholds                               */
/* ========================================================================== */
#define SAFETY_MAX_PITCH_RAD    0.785398f       /* 45 degrees */
#define SAFETY_MAX_PITCH_RATE_RAD_S 8.0f
#define SAFETY_MAX_TEMP_C       65.0f
#define SAFETY_OVERTEMP_DEBOUNCE_MS 100U
#define SAFETY_HEARTBEAT_MS     200U            /* Pi heartbeat timeout */
#define SAFETY_ACTION_MS        80U             /* RL action freshness timeout */
#define SAFETY_IMU_MAX_AGE_MS   20U
#define SAFETY_WHEEL_MAX_AGE_MS 50U
#define SAFETY_SERVO_MAX_AGE_MS 250U
#define SAFETY_FRESHNESS_DEBOUNCE_TICKS 3U  /* 250 Hz control loop; 3 ticks = 12 ms */
#define SAFETY_MODE_TRANSITION_GRACE_MS 100U

#endif /* PIN_CONFIG_H */
