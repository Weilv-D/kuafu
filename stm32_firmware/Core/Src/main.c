#include "stm32f4xx_hal.h"
#include "pin_config.h"
#include "crc8.h"
#include "pi_link.h"
#include "bmi088.h"
#include "ddsm315.h"
#include "st3215.h"
#include "mahony.h"
#include "kinematics.h"
#include "lqr_controller.h"
#include "safety_state.h"

/* Peripheral Handles */
I2C_HandleTypeDef hi2c1;
UART_HandleTypeDef huart1; /* CH340 Debug/Program */
UART_HandleTypeDef huart2; /* DDSM RS485 */
UART_HandleTypeDef huart3; /* ST3215 Half-Duplex Servo */
UART_HandleTypeDef huart6; /* Pi 5 Bridge */
DMA_HandleTypeDef hdma_usart6_rx;

/* Real-Time Telemetry and State variables */
volatile uint32_t g_system_ticks = 0;
BMI088_t g_imu;
MahonyFilter_t g_mahony;
LQRController_t g_lqr;

DDSM_State_t g_ddsm_left;
DDSM_State_t g_ddsm_right;
ST3215_State_t g_servos[4];

/* Per-servo direction (mirror) and mechanical zero (calibratable, see pin_config.h) */
static const int8_t  g_servo_dir[4]    = SERVO_DIR_INIT;    /* [LF, RF, LB, RB] */
static const int16_t g_servo_center[4] = SERVO_CENTER_INIT; /* [LF, RF, LB, RB] */

/* Wheel torque commands computed once per 250 Hz cycle (slot 0), sent in slots 0/1 */
static float g_ctrl_tau_l = 0.0f;
static float g_ctrl_tau_r = 0.0f;

/* Max consecutive servo failures before a fatal FAULT lockdown */
#define SERVO_FAIL_LIMIT         3

/* DMA Buffer for Pi Bridge (USART6 RX) */
#define PI_RX_BUF_SIZE           128
uint8_t g_pi_rx_buf[PI_RX_BUF_SIZE];

/* Function Prototypes */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART3_HalfDuplex_Init(void);
static void MX_USART6_UART_Init(void);
static void MX_DMA_Init(void);
static void System_Initial_Setup(void);
static void Pi_Command_Snapshot(Pi_Command_Heartbeat_t *hb, Pi_Command_Action_t *act);
static int16_t Servo_Angle_To_Ticks(float angle_rad, int idx);
static float Servo_Feedback_Angle(int idx);
void Error_Handler(void);

int main(void) {
    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* Configure the system clock to 168 MHz */
    SystemClock_Config();

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    MX_USART2_UART_Init();
    MX_USART3_HalfDuplex_Init();
    MX_USART6_UART_Init();

    /* Initialize drivers, CRC tables and control states */
    crc8_init();
    pi_link_init();
    safety_state_init();
    lqr_init(&g_lqr);
    mahony_init(&g_mahony, 2.0f, 0.005f); /* Kp = 2.0, Ki = 0.005 */

    g_ddsm_left.id = DDSM_LEFT_ID;
    g_ddsm_right.id = DDSM_RIGHT_ID;
    for (int i = 0; i < 4; i++) {
        g_servos[i].id = i + 1; /* IDs: 1, 2, 3, 4 */
        g_servos[i].is_online = 1;
        g_servos[i].consecutive_failures = 0;
    }

    /* Start Pi Link bridge USART6 reception via DMA */
    __HAL_UART_ENABLE_IT(&huart6, UART_IT_IDLE);
    HAL_UART_Receive_DMA(&huart6, g_pi_rx_buf, PI_RX_BUF_SIZE);

    /* Run initial sensor detection and calibration */
    System_Initial_Setup();

    uint32_t last_tick = 0;
    uint32_t last_servo_loop_time = 0;
    uint8_t active_servo_query_idx = 0;
    uint32_t temp_refresh_counter = 0;

    /* Main background scheduler loop */
    while (1) {
        /* Soft real-time scheduler aligned to system ticks (1ms resolution) */
        if (g_system_ticks != last_tick) {
            /* Use the actual elapsed time; blocking bus I/O can skip ticks */
            uint32_t dticks = g_system_ticks - last_tick;
            last_tick = g_system_ticks;
            float fusion_dt = (float)dticks * 0.001f;

            /* 1. Read IMU sensors (safe to do here in the main loop background) */
            bmi088_read_accel(&g_imu);
            bmi088_read_gyro(&g_imu);

            /* Refresh chip temperature at low rate (~10 Hz) for safety/telemetry */
            if (++temp_refresh_counter >= 100) {
                temp_refresh_counter = 0;
                bmi088_read_temp(&g_imu);
            }

            float gx = g_imu.gyro[0];
            float gy = g_imu.gyro[1];
            float gz = g_imu.gyro[2];

            /* 2. Run sensor fusion update */
            if (g_safety_state.is_gyro_calibrated) {
                gx -= g_safety_state.gyro_calib_offset[0];
                gy -= g_safety_state.gyro_calib_offset[1];
                gz -= g_safety_state.gyro_calib_offset[2];

                mahony_update(&g_mahony,
                              g_imu.accel[0], g_imu.accel[1], g_imu.accel[2],
                              gx, gy, gz, fusion_dt);
            } else {
                safety_state_gyro_calib_update(gx, gy, gz);
            }

            /* Balance-relevant tilt & rate mapped to the physical IMU mounting.
             * Pitch rate uses the bias-corrected gyro (offset is 0 until calibrated). */
            float body_pitch = ATT_PITCH(&g_mahony);
            float body_pitch_rate = ATT_PITCH_RATE_SIGN *
                (g_imu.gyro[ATT_PITCH_RATE_IDX] - g_safety_state.gyro_calib_offset[ATT_PITCH_RATE_IDX]);

            /* 3. Run the Slot scheduler (250Hz DDSM Motor & Pi Link) */
            uint8_t slot = last_tick % 4;

            /* Safety state machine update (at 250 Hz, inside slot scheduler) */
            if (slot == 3) {
                float max_temp = g_imu.temperature;
                for (int i = 0; i < 4; i++) {
                    if (g_servos[i].temperature_c > max_temp) {
                        max_temp = g_servos[i].temperature_c;
                    }
                }
                safety_state_update(body_pitch, body_pitch_rate, max_temp, 0.004f);
            }

            /* --- Slot 0: Compute LQR (once/cycle), command & poll Left DDSM --- */
            if (slot == 0) {
                if (g_safety_state.current_mode != STATE_FAULT && g_safety_state.current_mode != STATE_INIT) {
                    /* Snapshot Pi commands atomically (updated in USART6 ISR) */
                    Pi_Command_Heartbeat_t hb;
                    Pi_Command_Action_t act;
                    Pi_Command_Snapshot(&hb, &act);
                    (void)hb;

                    /* Average wheel velocity in body frame -> forward speed ẋ.
                     * yaw rate (gz) drives the conditional damping term. */
                    float wheel_vel_avg =
                        (WHEEL_DIR_L * g_ddsm_left.velocity_rads +
                         WHEEL_DIR_R * g_ddsm_right.velocity_rads) * 0.5f;
                    float yaw_rate = gz;

                    /* Run LQR once per 250 Hz cycle; cache both wheel commands. */
                    lqr_update(&g_lqr,
                               body_pitch,
                               body_pitch_rate,
                               wheel_vel_avg,
                               yaw_rate,
                               act.delta_torque_l,
                               act.delta_torque_r,
                               &g_ctrl_tau_l,
                               &g_ctrl_tau_r);

                    ddsm_set_torque(&huart2, DDSM_LEFT_ID, WHEEL_DIR_L * g_ctrl_tau_l);
                } else {
                    /* Safe stop wheels if in FAULT or INIT */
                    g_ctrl_tau_l = 0.0f;
                    g_ctrl_tau_r = 0.0f;
                    ddsm_set_torque(&huart2, DDSM_LEFT_ID, 0.0f);
                }

                /* Discard RS485 self-echo, then block-read Left feedback (10 bytes) */
                __HAL_UART_CLEAR_OREFLAG(&huart2);
                __HAL_UART_FLUSH_DRREGISTER(&huart2);
                uint8_t rx_buf[10];
                if (HAL_UART_Receive(&huart2, rx_buf, 10, 2) == HAL_OK) {
                    ddsm_parse_feedback(rx_buf, &g_ddsm_left);
                }
            }

            /* --- Slot 1: Command & Poll Right DDSM Motor (uses cached torque) --- */
            else if (slot == 1) {
                if (g_safety_state.current_mode != STATE_FAULT && g_safety_state.current_mode != STATE_INIT) {
                    ddsm_set_torque(&huart2, DDSM_RIGHT_ID, WHEEL_DIR_R * g_ctrl_tau_r);
                } else {
                    ddsm_set_torque(&huart2, DDSM_RIGHT_ID, 0.0f);
                }

                /* Discard RS485 self-echo, then block-read Right feedback (10 bytes) */
                __HAL_UART_CLEAR_OREFLAG(&huart2);
                __HAL_UART_FLUSH_DRREGISTER(&huart2);
                uint8_t rx_buf[10];
                if (HAL_UART_Receive(&huart2, rx_buf, 10, 2) == HAL_OK) {
                    ddsm_parse_feedback(rx_buf, &g_ddsm_right);
                }
            }

            /* --- Slot 2: Queue telemetry data to Raspberry Pi 5 --- */
            else if (slot == 2) {
                pi_link_send_imu(&huart6, 
                                 g_mahony.roll, 
                                 g_mahony.pitch, 
                                 g_mahony.yaw, 
                                 g_imu.gyro[0], 
                                 g_imu.gyro[1], 
                                 g_imu.gyro[2]);

                /* Report joint feedback in the shared sim/body frame so it is
                 * symmetric with the command contract (mirror + zero applied). */
                float servo_pos[4], servo_vel[4], servo_cur[4];
                for (int i = 0; i < 4; i++) {
                    servo_pos[i] = Servo_Feedback_Angle(i);
                    servo_vel[i] = (float)g_servo_dir[i] * g_servos[i].velocity_rads;
                    servo_cur[i] = g_servos[i].current_a;
                }

                /* Single-turn wheel angle (raw); velocity/torque mapped to body frame */
                float wheel_l_pos = g_ddsm_left.position_rad;
                float wheel_r_pos = g_ddsm_right.position_rad;

                pi_link_send_joints(&huart6,
                                    wheel_l_pos, WHEEL_DIR_L * g_ddsm_left.velocity_rads, WHEEL_DIR_L * g_ddsm_left.torque,
                                    wheel_r_pos, WHEEL_DIR_R * g_ddsm_right.velocity_rads, WHEEL_DIR_R * g_ddsm_right.torque,
                                    servo_pos, servo_vel, servo_cur);
            }

            /* --- Slot 3: Diagnostic packages & main controller logic --- */
            else if (slot == 3) {
                uint16_t dummy_battery_mv = 18500; /* Simulated 5S battery (18.5V) */
                pi_link_send_diag(&huart6, dummy_battery_mv, (uint8_t)g_imu.temperature, g_safety_state.error_mask);

                if (g_safety_state.current_mode == STATE_FAULT) {
                    pi_link_send_fault(&huart6, g_safety_state.active_fault);
                }
            }
        }

        /* --- 50 Hz Background Loop: ST3215 Servo Control --- */
        uint32_t current_time = HAL_GetTick();
        if (current_time - last_servo_loop_time >= (1000 / SERVO_LOOP_FREQ)) {
            last_servo_loop_time = current_time;

            uint8_t ids[4] = {SERVO_LF_ID, SERVO_RF_ID, SERVO_LB_ID, SERVO_RB_ID};

            if (g_safety_state.current_mode == STATE_ACTIVE) {
                /* Target hip angles received directly from Pi RL agent.
                 * Order matches sim: target_q = [LF, RF, LB, RB]. */
                Pi_Command_Heartbeat_t hb;
                Pi_Command_Action_t act;
                Pi_Command_Snapshot(&hb, &act);
                (void)hb;

                int16_t pos_ticks[4];
                uint16_t speed_ticks[4] = {2000, 2000, 2000, 2000};
                uint8_t accels[4] = {50, 50, 50, 50};

                for (int i = 0; i < 4; i++) {
                    pos_ticks[i] = Servo_Angle_To_Ticks(act.target_q[i], i);
                }

                st3215_sync_write_pos(&huart3, ids, 4, pos_ticks, speed_ticks, accels);
            }
            else if (g_safety_state.current_mode == STATE_STAND || g_safety_state.current_mode == STATE_CLIMB) {
                /* Standing/Climbing virtual height mode */
                Pi_Command_Heartbeat_t hb;
                Pi_Command_Action_t act;
                Pi_Command_Snapshot(&hb, &act);
                (void)act;

                float q_hip_A = 0.0f; /* front chain (LF, RF) */
                float q_hip_B = 0.0f; /* back chain  (LB, RB) */

                /* Compute inverse kinematics mapping for target height */
                if (kinematics_solve_ik(hb.target_leg_d0, &q_hip_A, &q_hip_B) == 0) {
                    int16_t pos_ticks[4];
                    uint16_t speed_ticks[4] = {1500, 1500, 1500, 1500};
                    uint8_t accels[4] = {30, 30, 30, 30};

                    pos_ticks[0] = Servo_Angle_To_Ticks(q_hip_A, 0); /* LF (A chain) */
                    pos_ticks[1] = Servo_Angle_To_Ticks(q_hip_A, 1); /* RF (A chain) */
                    pos_ticks[2] = Servo_Angle_To_Ticks(q_hip_B, 2); /* LB (B chain) */
                    pos_ticks[3] = Servo_Angle_To_Ticks(q_hip_B, 3); /* RB (B chain) */

                    st3215_sync_write_pos(&huart3, ids, 4, pos_ticks, speed_ticks, accels);
                }
            }
            else if (g_safety_state.current_mode == STATE_FAULT) {
                /* Lockdown: Disable servo torque to allow gravity lock */
                for (int i = 0; i < 4; i++) {
                    st3215_set_torque_enable(&huart3, ids[i], 0);
                }
            }

            /* Sequentially query feedback from one online servo to prevent blocking.
             * A servo that stays unreachable triggers a fatal FAULT lockdown. */
            ST3215_State_t *qs = &g_servos[active_servo_query_idx];
            if (qs->is_online) {
                if (st3215_read_state(&huart3, ids[active_servo_query_idx], qs) == 0) {
                    qs->consecutive_failures = 0;
                } else if (g_safety_state.current_mode == STATE_INIT) {
                    /* Ignore transient boot races before calibration; keep retrying */
                    qs->consecutive_failures = 0;
                } else {
                    qs->consecutive_failures++;
                    if (qs->consecutive_failures >= SERVO_FAIL_LIMIT) {
                        qs->is_online = 0;
                        safety_state_trigger_fault(FAULT_SERVO); /* fatal lockdown */
                    }
                }
            }
            active_servo_query_idx = (active_servo_query_idx + 1) % 4;
        }
    }
}

/* System Setup and Gyro calibration */
static void System_Initial_Setup(void) {
    HAL_Delay(500); /* Wait for supply voltage stabilization */

    /* 1. Initialize BMI088 IMU over I2C */
    while (bmi088_init(&g_imu, &hi2c1) != 0) {
        /* Blink an LED or print error if IMU is missing */
        HAL_Delay(100);
    }

    /* 2. Configure DDSM315 motors to Current Control Mode and Enable */
    ddsm_set_mode(&huart2, DDSM_LEFT_ID, DDSM_MODE_CURRENT);
    HAL_Delay(10);
    ddsm_set_enable(&huart2, DDSM_LEFT_ID, 1);
    HAL_Delay(10);

    ddsm_set_mode(&huart2, DDSM_RIGHT_ID, DDSM_MODE_CURRENT);
    HAL_Delay(10);
    ddsm_set_enable(&huart2, DDSM_RIGHT_ID, 1);
    HAL_Delay(10);

    /* 3. Enable ST3215 servos torque */
    for (int i = 0; i < 4; i++) {
        st3215_set_torque_enable(&huart3, i + 1, 1);
        HAL_Delay(5);
    }

    /* Enable EXTI Line 1 (PB1) Interrupt for Gyro DRDY sync */
    HAL_NVIC_SetPriority(IMU_INT_EXTI_IRQn, 0, 0); /* Highest priority */
    HAL_NVIC_EnableIRQ(IMU_INT_EXTI_IRQn);
}

/**
 * @brief EXTI1 callback: triggered at 1kHz by Gyro DRDY pin (PB1).
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    if (GPIO_Pin == IMU_INT_PIN) {
        g_system_ticks++;
    }
}

/**
 * @brief Atomically snapshots the Pi command state (updated in the USART6 ISR)
 *        to avoid torn multi-byte reads in the main loop.
 */
static void Pi_Command_Snapshot(Pi_Command_Heartbeat_t *hb, Pi_Command_Action_t *act) {
    HAL_NVIC_DisableIRQ(USART6_IRQn);
    *hb = g_pi_cmd_heartbeat;
    *act = g_pi_cmd_action;
    HAL_NVIC_EnableIRQ(USART6_IRQn);
}

/**
 * @brief Maps a sim-frame joint angle (dwell = 0) to a servo tick command,
 *        applying the per-servo mirror direction and mechanical zero.
 */
static int16_t Servo_Angle_To_Ticks(float angle_rad, int idx) {
    return (int16_t)((float)g_servo_dir[idx] * angle_rad * SERVO_TICKS_PER_RAD) + g_servo_center[idx];
}

/**
 * @brief Converts a decoded servo position (driver frame, relative to
 *        SERVO_CENTER_TICKS) into the sim/command frame, applying the per-servo
 *        mechanical zero and mirror direction. Inverse of Servo_Angle_To_Ticks.
 */
static float Servo_Feedback_Angle(int idx) {
    float center_offset = (float)(SERVO_CENTER_TICKS - g_servo_center[idx]) / SERVO_TICKS_PER_RAD;
    return (float)g_servo_dir[idx] * (g_servos[idx].position_rad + center_offset);
}

/**
 * @brief System Clock Configuration to 168MHz (from 8MHz HSE)
 */
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 8;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2; /* SYSCLK = 168MHz */
    RCC_OscInitStruct.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4; /* APB1 = 42MHz */
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2; /* APB2 = 84MHz */
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* Configure EXTI1 Interrupt Pin (PB1) */
    GPIO_InitStruct.Pin = IMU_INT_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING; /* Gyro INT3 goes high when data ready */
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(IMU_INT_PORT, &GPIO_InitStruct);
}

static void MX_DMA_Init(void) {
    __HAL_RCC_DMA2_CLK_ENABLE();

    /* DMA2_Stream1_Channel5 for USART6_RX */
    HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
}

static void MX_I2C1_Init(void) {
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 400000; /* 400 kHz Fast Mode */
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    
    IMU_GPIO_CLK_EN();
    IMU_I2C_CLK_EN();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = IMU_SCL_PIN | IMU_SDA_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD; /* Open Drain */
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = IMU_I2C_AF;
    HAL_GPIO_Init(IMU_SCL_PORT, &GPIO_InitStruct);

    if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_USART1_UART_Init(void) {
    huart1.Instance = DEBUG_USART;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;

    DEBUG_GPIO_CLK_EN();
    DEBUG_USART_CLK_EN();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = DEBUG_TX_PIN | DEBUG_RX_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = DEBUG_USART_AF;
    HAL_GPIO_Init(DEBUG_TX_PORT, &GPIO_InitStruct);

    if (HAL_UART_Init(&huart1) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_USART2_UART_Init(void) {
    huart2.Instance = DDSM_USART;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;

    DDSM_GPIO_CLK_EN();
    DDSM_USART_CLK_EN();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = DDSM_TX_PIN | DDSM_RX_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = DDSM_USART_AF;
    HAL_GPIO_Init(DDSM_TX_PORT, &GPIO_InitStruct);

    if (HAL_UART_Init(&huart2) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_USART3_HalfDuplex_Init(void) {
    huart3.Instance = SERVO_USART;
    huart3.Init.BaudRate = 1000000;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;

    SERVO_GPIO_CLK_EN();
    SERVO_USART_CLK_EN();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = SERVO_TX_PIN; /* In Half-Duplex, only the TX pin is used */
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD; /* Open-Drain for bidirectional bus line */
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = SERVO_USART_AF;
    HAL_GPIO_Init(SERVO_TX_PORT, &GPIO_InitStruct);

    if (HAL_HalfDuplex_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_USART6_UART_Init(void) {
    huart6.Instance = PI_USART;
    huart6.Init.BaudRate = 921600;
    huart6.Init.WordLength = UART_WORDLENGTH_8B;
    huart6.Init.StopBits = UART_STOPBITS_1;
    huart6.Init.Parity = UART_PARITY_NONE;
    huart6.Init.Mode = UART_MODE_TX_RX;
    huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart6.Init.OverSampling = UART_OVERSAMPLING_16;

    PI_GPIO_CLK_EN();
    PI_USART_CLK_EN();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = PI_TX_PIN | PI_RX_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = PI_USART_AF;
    HAL_GPIO_Init(PI_TX_PORT, &GPIO_InitStruct);

    if (HAL_UART_Init(&huart6) != HAL_OK) {
        Error_Handler();
    }

    /* Associate DMA handle to USART6 RX */
    hdma_usart6_rx.Instance = DMA2_Stream1;
    hdma_usart6_rx.Init.Channel = DMA_CHANNEL_5;
    hdma_usart6_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart6_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart6_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart6_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart6_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart6_rx.Init.Mode = DMA_CIRCULAR;
    hdma_usart6_rx.Init.Priority = DMA_PRIORITY_HIGH;
    hdma_usart6_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_usart6_rx) != HAL_OK) {
        Error_Handler();
    }

    __HAL_LINKDMA(&huart6, hdmarx, hdma_usart6_rx);

    /* Enable USART6 IRQ */
    HAL_NVIC_SetPriority(USART6_IRQn, 1, 1);
    HAL_NVIC_EnableIRQ(USART6_IRQn);
}

/**
 * @brief USART6 global interrupt handler. Processes IDLE line detections.
 */
void USART6_IRQHandler(void) {
    if (__HAL_UART_GET_FLAG(&huart6, UART_FLAG_IDLE) != RESET) {
        __HAL_UART_CLEAR_IDLEFLAG(&huart6);

        /* Determine received size */
        uint16_t counter = __HAL_DMA_GET_COUNTER(&hdma_usart6_rx);
        uint16_t rx_len = PI_RX_BUF_SIZE - counter;

        if (rx_len > 0) {
            pi_link_parse_packet(g_pi_rx_buf, rx_len);
        }

        /* Reset DMA reception */
        HAL_UART_DMAStop(&huart6);
        HAL_UART_Receive_DMA(&huart6, g_pi_rx_buf, PI_RX_BUF_SIZE);
    }
    HAL_UART_IRQHandler(&huart6);
}

/**
 * @brief DMA2 Stream1 (USART6 RX) global interrupt handler.
 */
void DMA2_Stream1_IRQHandler(void) {
    HAL_DMA_IRQHandler(&hdma_usart6_rx);
}

/**
 * @brief EXTI1 (PB1 Pin) global interrupt handler.
 */
void EXTI1_IRQHandler(void) {
    HAL_GPIO_EXTI_IRQHandler(IMU_INT_PIN);
}

/**
 * @brief Fatal error handler: stop actuators intent, lock down and slow-blink LED2.
 */
void Error_Handler(void) {
    __disable_irq();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_13; /* LED2 (PC13, active low) */
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        for (volatile uint32_t i = 0; i < 4000000; i++) {
            __NOP();
        }
    }
}
