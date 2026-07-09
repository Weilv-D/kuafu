#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

#include "stm32f4xx.h"

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

/* USART3: ST3215 Half-Duplex TTL Bus Servo Port */
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
#define LQR_K0                  -4.47f
#define LQR_K1                  -61.18f
#define LQR_K2                  -5.82f
#define LQR_K3                  -4.02f

/* DDSM315 Current to Torque scaling factor */
/* tau = 0.75 * I_amps. Command maps raw [-32767, 32767] to [-8.0, 8.0] Amps.
 * Hence: I_raw = tau * (32767.0f / 6.0f) ~= 5461.17f * tau */
#define DDSM_TORQUE_TO_RAW      5461.17f
#define DDSM_RAW_TO_TORQUE      (1.0f / DDSM_TORQUE_TO_RAW)
#define DDSM_MAX_TORQUE_NM      1.1f

/* 5-Bar Linkage Kinematics parameters (meters) */
#define KIN_LEG_A               0.093f          /* Upper leg crank length 'a' (93mm) */
#define KIN_LEG_B               0.149f          /* Lower leg rod length 'b' (149mm)  */
#define KIN_LEG_C               0.026f          /* Hips spacing offset 'c' (26mm)    */
#define KIN_MIN_LEG_D0          0.058f          /* 58mm minimum virtual leg length    */
#define KIN_MAX_LEG_D0          0.207f          /* 207mm maximum virtual leg length   */

/* ST3215 Servo mapping */
#define SERVO_CENTER_TICKS      2048
#define SERVO_TICKS_PER_RAD     (4096.0f / (2.0f * 3.14159265f))

/* ========================================================================== */
/*                            Safety Thresholds                               */
/* ========================================================================== */
#define SAFETY_MAX_PITCH_RAD    0.785398f       /* 45 degrees */
#define SAFETY_MAX_TEMP_C       65.0f
#define SAFETY_HEARTBEAT_MS     200U            /* Pi heartbeat timeout */

#endif /* PIN_CONFIG_H */
