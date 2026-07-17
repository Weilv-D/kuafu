#include "stm32f4xx_hal.h"
#include "pin_config.h"
#include "crc8.h"
#include "pi_link.h"
#include "pi_transport.h"
#include "bmi088.h"
#include "ddsm315.h"
#include "st3215.h"
#include "mahony.h"
#include "kinematics.h"
#include "lqr_controller.h"
#include "safety_state.h"
#include "servo_mapping.h"
#include "startup_manager.h"
#include "firmware_runtime.h"

/* Peripheral Handles */
I2C_HandleTypeDef hi2c1;
UART_HandleTypeDef huart1; /* CH340 Debug/Program */
UART_HandleTypeDef huart2; /* DDSM RS485 */
UART_HandleTypeDef huart3; /* ST3215 Half-Duplex Servo */
UART_HandleTypeDef huart6; /* Pi 5 Bridge */
IWDG_HandleTypeDef hiwdg;
DMA_HandleTypeDef hdma_usart6_rx;

/* Real-Time Telemetry and State variables */
volatile uint32_t g_system_ticks = 0;
BMI088_t g_imu;
MahonyFilter_t g_mahony;
LQRController_t g_lqr;

DDSM_State_t g_ddsm_left;
DDSM_State_t g_ddsm_right;
ST3215_State_t g_servos[4];
StartupManager_t g_startup_manager;
uint8_t g_actuator_discovery_step;
uint8_t g_actuator_configured;
static DDSM_Bus_t g_ddsm_bus;
static ST3215_Bus_t g_st3215_bus;

/* Wheel torque commands computed and sent together at each 250 Hz deadline. */
static float g_ctrl_tau_l = 0.0f;
static float g_ctrl_tau_r = 0.0f;
static volatile float g_body_gyro[3] = {0.0f, 0.0f, 0.0f};

/* Max consecutive servo failures before a fatal FAULT lockdown */
#define SERVO_FAIL_LIMIT         3

/* DMA Buffer for Pi Bridge (USART6 RX) */
#define PI_RX_BUF_SIZE           256
uint8_t g_pi_rx_buf[PI_RX_BUF_SIZE];
static PiTransport_t g_pi_transport;
static volatile uint8_t g_pi_poll_requested = 0U;
static uint8_t g_reset_cause = 0U;

/* Function Prototypes */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USART6_UART_Init(void);
static void MX_DMA_Init(void);
static void MX_IWDG_Init(void);
static void Pi_Command_Snapshot(Pi_Command_Heartbeat_t *hb, Pi_Command_Action_t *act);
static void Actuator_Feedback_Snapshot(DDSM_State_t *left,
                                       DDSM_State_t *right,
                                       ST3215_State_t servos[4]);
static uint8_t Read_Reset_Cause(void);
static uint16_t Device_Age_Ms(const DeviceHealth_t *health, uint32_t now_ms);
static uint16_t Device_Error_Count(const DeviceHealth_t *health);
static uint16_t Device_Timeout_Count(const DeviceHealth_t *health);
static uint16_t Device_Checksum_Count(const DeviceHealth_t *health);
static uint16_t Device_Protocol_Count(const DeviceHealth_t *health);
void Error_Handler(void);

int main(void) {
    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();
    g_reset_cause = Read_Reset_Cause();

    /* Configure the system clock to 168 MHz */
    SystemClock_Config();

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    MX_USART2_UART_Init();
    MX_USART3_UART_Init();
    MX_USART6_UART_Init();
    MX_IWDG_Init();

    /* Initialize drivers, CRC tables and control states */
    crc8_init();
    pi_link_init();
    pi_transport_init(&g_pi_transport, g_pi_rx_buf, PI_RX_BUF_SIZE);
    safety_state_init();
    lqr_init(&g_lqr);
    mahony_init(&g_mahony, 2.0f, 0.005f); /* Kp = 2.0, Ki = 0.005 */

    g_ddsm_left.id = DDSM_LEFT_ID;
    g_ddsm_right.id = DDSM_RIGHT_ID;
    device_health_init(&g_ddsm_left.health);
    device_health_init(&g_ddsm_right.health);
#if DDSM_ID_CALIBRATION_TARGET > 0
    {
        uint8_t id_packet[DDSM_FRAME_SIZE];
        uint8_t repeat;
        ddsm_build_set_id(id_packet, DDSM_ID_CALIBRATION_TARGET);
        safety_state_trigger_fault(FAULT_WHEEL_LEFT | FAULT_WHEEL_RIGHT);
        HAL_Delay(250U);
        for (repeat = 0U; repeat < 5U; ++repeat) {
            (void)HAL_UART_Transmit(&huart2, id_packet, DDSM_FRAME_SIZE, 20U);
            HAL_Delay(4U);
        }
    }
#endif
    ddsm_bus_init(&g_ddsm_bus, &huart2);
    for (int i = 0; i < 4; i++) {
        g_servos[i].id = i + 1; /* IDs: 1, 2, 3, 4 */
        device_health_init(&g_servos[i].health);
        g_servos[i].health.online = 1U; /* optimistic discovery; first valid read timestamps it */
    }
    st3215_bus_init(&g_st3215_bus, &huart3);

    /* Start Pi Link bridge USART6 reception via DMA */
    __HAL_UART_ENABLE_IT(&huart6, UART_IT_IDLE);
    HAL_UART_Receive_DMA(&huart6, g_pi_rx_buf, PI_RX_BUF_SIZE);

    /* Arm gyro data-ready before starting the non-blocking device sequence. */
    HAL_NVIC_SetPriority(IMU_INT_EXTI_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(IMU_INT_EXTI_IRQn);

#if SERVO_ZERO_CALIBRATION_MODE
    /* A calibration image must never leave INIT and start commanding the
     * placeholder center ticks. FAULT keeps wheel commands at zero while the
     * background servo feedback poll remains active. */
    safety_state_trigger_fault(FAULT_SERVO);
#endif

    uint32_t last_tick = 0;
    uint32_t next_wheel_tx_ms = 0U;
    uint8_t next_wheel_is_right = 0U;
    uint8_t active_servo_query_idx = 0;
    uint32_t last_servo_query_ms = 0U;
    uint8_t fault_servo_disable_idx = 0U;
    uint32_t temp_refresh_counter = 0;
    uint8_t health_telemetry_divider = 0U;
    uint8_t bmi_init_in_progress = 0U;
    uint8_t servo_enable_step = 0U;
    uint8_t servos_enabled = 0U;
    uint8_t wheel_enable_mask = 0U;
    uint32_t next_actuator_enable_ms = 0U;
    uint32_t last_pi_poll_ms = 0U;
    FirmwareRuntime_t firmware_runtime;
    uint8_t control_deadline_pending = 0U;
    uint8_t servo_deadline_pending = 0U;

    startup_manager_init(&g_startup_manager, HAL_GetTick());
    firmware_runtime_init(&firmware_runtime, HAL_GetTick());

    /* Main background scheduler loop */
    while (1) {
        uint32_t startup_now = HAL_GetTick();
        StartupInputs_t startup_inputs;
        StartupOutputs_t startup_outputs;
        FirmwareRuntimeInputs_t runtime_inputs;
        FirmwareRuntimeOutputs_t runtime_outputs;
        DDSM_State_t left_feedback;
        DDSM_State_t right_feedback;
        ST3215_State_t servo_feedback[4];
        Pi_Command_Heartbeat_t runtime_heartbeat;
        Pi_Command_Action_t runtime_action;
        uint8_t wheel_authorized;
        uint8_t startup_servos_online = 1U;

        if (g_pi_poll_requested || startup_now != last_pi_poll_ms) {
            g_pi_poll_requested = 0U;
            last_pi_poll_ms = startup_now;
            (void)pi_transport_poll(&g_pi_transport,
                                    (uint16_t)__HAL_DMA_GET_COUNTER(&hdma_usart6_rx));
        }

        ddsm_bus_step(&g_ddsm_bus, startup_now);
        st3215_bus_step(&g_st3215_bus, startup_now);
        Actuator_Feedback_Snapshot(&left_feedback, &right_feedback, servo_feedback);
        Pi_Command_Snapshot(&runtime_heartbeat, &runtime_action);
        (void)runtime_action;
        wheel_authorized = (uint8_t)(g_startup_manager.phase == STARTUP_READY &&
                                     g_safety_state.current_mode != STATE_FAULT &&
                                     pi_link_is_compatible() && pi_link_heartbeat_fresh() &&
                                     runtime_heartbeat.mode_request >= (uint8_t)STATE_STAND &&
                                     runtime_heartbeat.mode_request <= (uint8_t)STATE_CLIMB);

        runtime_inputs.now_ms = startup_now;
        runtime_inputs.mode = g_safety_state.current_mode;
        runtime_inputs.link_compatible = pi_link_is_compatible();
        runtime_inputs.heartbeat_fresh = pi_link_heartbeat_fresh();
        runtime_inputs.action_fresh = pi_link_action_fresh();
        runtime_inputs.wheel_authorized = wheel_authorized;
        runtime_inputs.wheel_bus_idle = ddsm_bus_is_idle(&g_ddsm_bus);
        runtime_inputs.servo_bus_idle = st3215_bus_is_idle(&g_st3215_bus);
        runtime_outputs = firmware_runtime_step(&firmware_runtime, &runtime_inputs);
        if (runtime_outputs.control_due) control_deadline_pending = 1U;
        if (runtime_outputs.servo_due) servo_deadline_pending = 1U;

        if (bmi_init_in_progress) {
            int bmi_result = bmi088_init_step(&g_imu, startup_now);
            if (bmi_result != 0) {
                bmi_init_in_progress = 0U;
            }
        }

        for (int i = 0; i < 4; ++i) {
            if (!device_health_is_fresh(&servo_feedback[i].health,
                                        startup_now,
                                        SAFETY_SERVO_MAX_AGE_MS)) {
                startup_servos_online = 0U;
            }
        }
        startup_inputs.now_ms = startup_now;
        startup_inputs.imu_initialized = g_imu.initialized;
        startup_inputs.gyro_calibrated = g_safety_state.is_gyro_calibrated;
        startup_inputs.wheel_l_online = device_health_is_fresh(&left_feedback.health,
                                                               startup_now,
                                                               SAFETY_WHEEL_MAX_AGE_MS);
        startup_inputs.wheel_r_online = device_health_is_fresh(&right_feedback.health,
                                                               startup_now,
                                                               SAFETY_WHEEL_MAX_AGE_MS);
        startup_inputs.servos_online = startup_servos_online;
        startup_inputs.actuator_configured = g_actuator_configured;
        startup_outputs = startup_manager_step(&g_startup_manager, &startup_inputs);

        if (startup_outputs.request_imu_init && !g_imu.initialized && !bmi_init_in_progress) {
            bmi088_begin_init(&g_imu, &hi2c1, startup_now);
            bmi_init_in_progress = 1U;
        }

        if (g_startup_manager.phase == STARTUP_ACTUATOR_DISCOVERY &&
            !g_actuator_configured) {
            int discovery_result = -1;
            if (g_actuator_discovery_step == 0U) {
                discovery_result = ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_left,
                                                       DDSM_MODE_CURRENT, startup_now);
            } else if (g_actuator_discovery_step == 1U) {
                /* Explicit zero-torque clears any stale torque retained across
                 * power cycles so the motor does not spin before the Pi arms. */
                discovery_result = ddsm_bus_queue_torque(&g_ddsm_bus, &g_ddsm_left,
                                                         0.0f, startup_now);
            } else if (g_actuator_discovery_step == 2U) {
                discovery_result = ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_left,
                                                         0U, startup_now);
            } else if (g_actuator_discovery_step == 3U) {
                discovery_result = ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_right,
                                                       DDSM_MODE_CURRENT, startup_now);
            } else if (g_actuator_discovery_step == 4U) {
                discovery_result = ddsm_bus_queue_torque(&g_ddsm_bus, &g_ddsm_right,
                                                         0.0f, startup_now);
            } else if (g_actuator_discovery_step == 5U) {
                discovery_result = ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_right,
                                                         0U, startup_now);
            } else if (g_actuator_discovery_step == 6U) {
                discovery_result = st3215_bus_queue_torque(
                    &g_st3215_bus, ST3215_BROADCAST_ID, 0U);
            }
            if (g_actuator_discovery_step < 7U && discovery_result == 0) {
                ++g_actuator_discovery_step;
            }
            g_actuator_configured = (uint8_t)(g_actuator_discovery_step >= 7U);
        }

        if (startup_outputs.enable_actuators && !servos_enabled &&
            (int32_t)(startup_now - next_actuator_enable_ms) >= 0) {
            int enable_result = -1;
            if (servo_enable_step < 4U) {
#if SERVO_ZERO_CALIBRATION_MODE
                enable_result = st3215_bus_queue_torque(
                    &g_st3215_bus, (uint8_t)(servo_enable_step + 1U), 0U);
#else
                enable_result = st3215_bus_queue_torque(
                    &g_st3215_bus, (uint8_t)(servo_enable_step + 1U), 1U);
#endif
            }
            if (servo_enable_step < 4U && enable_result == 0) {
                ++servo_enable_step;
            }
            next_actuator_enable_ms = startup_now + 5U;
            servos_enabled = (uint8_t)(servo_enable_step >= 4U);
        }

        /* Wheel power is a separately armed safety domain.  Discovery and
         * zero-current polling run while disabled; enabling requires an
         * authenticated, fresh Pi mode request and is revoked on link loss. */
        if (wheel_authorized && wheel_enable_mask != 0x03U && ddsm_bus_is_idle(&g_ddsm_bus)) {
            if ((wheel_enable_mask & 0x01U) == 0U) {
                if (ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_left, 1U, startup_now) == 0) {
                    wheel_enable_mask |= 0x01U;
                }
            } else if (ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_right, 1U, startup_now) == 0) {
                wheel_enable_mask |= 0x02U;
            }
        } else if (!wheel_authorized && wheel_enable_mask != 0U && ddsm_bus_is_idle(&g_ddsm_bus)) {
            if ((wheel_enable_mask & 0x01U) != 0U) {
                if (ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_left, 0U, startup_now) == 0) {
                    wheel_enable_mask &= (uint8_t)~0x01U;
                }
            } else if (ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_right, 0U, startup_now) == 0) {
                wheel_enable_mask &= (uint8_t)~0x02U;
            }
        }

        /* The DDSM315 bus permits at most one request/response transaction per
         * 4 ms.  Dispatch exactly one motor at each bus deadline and alternate
         * sides after every successful submission.  This keeps both feedback
         * ages bounded even when one motor times out or adapter echo shifts a
         * frame, and avoids two independent deadlines becoming phase-locked. */
        if (g_actuator_configured && ddsm_bus_is_idle(&g_ddsm_bus) &&
            (int32_t)(startup_now - next_wheel_tx_ms) >= 0) {
            int wheel_result;
            DDSM_State_t *wheel = next_wheel_is_right ? &g_ddsm_right : &g_ddsm_left;
            float torque = next_wheel_is_right
                         ? WHEEL_DIR_R * g_ctrl_tau_r
                         : WHEEL_DIR_L * g_ctrl_tau_l;
            if (runtime_outputs.wheel_intent_allowed) {
                wheel_result = ddsm_bus_queue_torque(&g_ddsm_bus, wheel,
                                                     torque, startup_now);
            } else {
                wheel_result = ddsm_bus_queue_query(&g_ddsm_bus, wheel, startup_now);
            }
            if (wheel_result == 0) {
                next_wheel_is_right ^= 1U;
                next_wheel_tx_ms = startup_now + 4U;
            }
        }

        if (startup_outputs.fault_requested) {
            safety_state_trigger_fault(FAULT_INIT);
        }

        /* Reaching this point proves the scheduler is alive even before DRDY. */
        HAL_IWDG_Refresh(&hiwdg);

        /* Soft real-time scheduler aligned to system ticks (1ms resolution) */
        if (g_system_ticks != last_tick) {
            /* Use the actual elapsed time; blocking bus I/O can skip ticks */
            uint32_t dticks = g_system_ticks - last_tick;
            last_tick = g_system_ticks;
            float fusion_dt = (float)dticks * 0.001f;

            /* DRDY can arrive while the BMI088 register sequence is still in
             * progress.  Do not read or calibrate from an uninitialized IMU. */
            if (!g_imu.initialized) {
                continue;
            }

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
                safety_state_gyro_calib_update(gx, gy, gz, HAL_GetTick());
            }
            g_body_gyro[0] = gx;
            g_body_gyro[1] = gy;
            g_body_gyro[2] = gz;

            /* Balance-relevant tilt & rate mapped to the physical IMU mounting.
             * Pitch rate uses the bias-corrected gyro (offset is 0 until calibrated). */
            float body_pitch = ATT_PITCH(&g_mahony);
            float body_pitch_rate = ATT_PITCH_RATE_SIGN *
                (g_imu.gyro[ATT_PITCH_RATE_IDX] - g_safety_state.gyro_calib_offset[ATT_PITCH_RATE_IDX]);

            /* 3. Run the 250 Hz motor deadline and lower-rate telemetry slots. */
            uint8_t slot = last_tick % 4;
            uint8_t control_due = control_deadline_pending;
            if (control_due) control_deadline_pending = 0U;

            /* Safety state machine update (at 250 Hz, inside slot scheduler) */
            if (control_due && g_startup_manager.phase >= STARTUP_ACTUATOR_DISCOVERY) {
                float max_temp = g_imu.temperature;
                for (int i = 0; i < 4; i++) {
                    if (servo_feedback[i].temperature_c > max_temp) {
                        max_temp = servo_feedback[i].temperature_c;
                    }
                }
                SafetyInputs_t safety_inputs;
                SafetyDecision_t safety_decision;
                uint32_t safety_now = HAL_GetTick();
                uint8_t servos_fresh = 1U;
                for (int i = 0; i < 4; ++i) {
                    if (!device_health_is_fresh(&servo_feedback[i].health,
                                                safety_now,
                                                SAFETY_SERVO_MAX_AGE_MS)) {
                        servos_fresh = 0U;
                    }
                }
                safety_inputs.now_ms = safety_now;
                safety_inputs.pitch_rad = body_pitch;
                safety_inputs.pitch_rate_rads = body_pitch_rate;
                safety_inputs.max_temp_c = max_temp;
                safety_inputs.gyro_calibrated = g_safety_state.is_gyro_calibrated;
                safety_inputs.startup_ready = servos_enabled;
                safety_inputs.imu_fresh = device_health_is_fresh(&g_imu.health,
                                                                 safety_now,
                                                                 SAFETY_IMU_MAX_AGE_MS);
                safety_inputs.wheel_l_fresh = device_health_is_fresh(&left_feedback.health,
                                                                     safety_now,
                                                                     SAFETY_WHEEL_MAX_AGE_MS);
                safety_inputs.wheel_r_fresh = device_health_is_fresh(&right_feedback.health,
                                                                     safety_now,
                                                                     SAFETY_WHEEL_MAX_AGE_MS);
                safety_inputs.servos_fresh = servos_fresh;
                safety_inputs.link_compatible = pi_link_is_compatible();
                safety_inputs.heartbeat_fresh = pi_link_heartbeat_fresh();
                safety_inputs.action_fresh = pi_link_action_fresh();
                safety_inputs.requested_mode = g_pi_cmd_heartbeat.mode_request;
                safety_decision = safety_state_update(&safety_inputs);
                if (safety_decision.enter_hold) {
                    pi_link_enter_hold();
                } else if (safety_decision.clear_action) {
                    pi_link_clear_action();
                }
            }

            /* --- Control deadline: compute and cache both wheel commands --- */
            if (control_due && g_startup_manager.phase >= STARTUP_ACTUATOR_DISCOVERY) {
                if (runtime_outputs.wheel_intent_allowed) {
                    /* Snapshot Pi commands atomically (updated in USART6 ISR) */
                    Pi_Command_Heartbeat_t hb;
                    Pi_Command_Action_t act;
                    Pi_Command_Snapshot(&hb, &act);
                    /* Average wheel velocity in body frame -> forward speed ẋ.
                     * All command and residual fields now use the shared P0 contract. */
                    float wheel_vel_l = WHEEL_DIR_L * left_feedback.velocity_rads;
                    float wheel_vel_r = WHEEL_DIR_R * right_feedback.velocity_rads;
                    float yaw_rate = gz;
                    if (!pi_link_is_compatible() || !pi_link_heartbeat_fresh()) {
                        /* Enter a fresh local hold reference immediately; do not
                         * jerk-limit a stale command back to zero from its old
                         * moving reference. */
                        lqr_reset(&g_lqr, g_lqr.x_est, g_mahony.yaw);
                    }

                    /* Run LQR once per 250 Hz cycle; cache both wheel commands. */
                    lqr_update(&g_lqr,
                               body_pitch,
                                body_pitch_rate,
                                wheel_vel_l,
                                wheel_vel_r,
                                g_mahony.yaw,
                                yaw_rate,
                                hb.target_velocity,
                                hb.target_yaw_rate,
                                runtime_outputs.residual_allowed
                                    ? act.delta_torque_common : 0.0f,
                                runtime_outputs.residual_allowed
                                    ? act.delta_torque_yaw : 0.0f,
                                &g_ctrl_tau_l,
                                &g_ctrl_tau_r);

                } else {
                    /* Safe stop wheels if in FAULT or INIT */
                    g_ctrl_tau_l = 0.0f;
                    g_ctrl_tau_r = 0.0f;
                }
            }

            /* --- Slot 2: Queue telemetry data to Raspberry Pi 5 --- */
            if (slot == 2) {
                pi_link_send_imu(&huart6, 
                                 g_mahony.roll, 
                                 g_mahony.pitch, 
                                 g_mahony.yaw, 
                                  g_body_gyro[0],
                                  g_body_gyro[1],
                                  g_body_gyro[2]);

                /* Report joint feedback in the shared sim/body frame so it is
                 * symmetric with the command contract (mirror + zero applied). */
                float servo_pos[4], servo_vel[4], servo_cur[4];
                for (int i = 0; i < 4; i++) {
                    servo_pos[i] = servo_tick_to_angle(servo_feedback[i].position_tick, (uint8_t)i);
                    servo_vel[i] = (float)servo_direction((uint8_t)i) * servo_feedback[i].velocity_rads;
                    servo_cur[i] = servo_feedback[i].current_a;
                }

                /* Single-turn wheel angle (raw); velocity/torque mapped to body frame */
                float wheel_l_pos = left_feedback.position_rad;
                float wheel_r_pos = right_feedback.position_rad;

                pi_link_send_joints(&huart6,
                                    wheel_l_pos, WHEEL_DIR_L * left_feedback.velocity_rads, WHEEL_DIR_L * left_feedback.torque,
                                    wheel_r_pos, WHEEL_DIR_R * right_feedback.velocity_rads, WHEEL_DIR_R * right_feedback.torque,
                                    servo_pos, servo_vel, servo_cur);
            }

            /* --- Slot 3: Diagnostic packages & main controller logic --- */
            else if (slot == 3) {
                /* Battery sensing is not populated on this hardware.  Zero is
                 * the protocol sentinel for unavailable, not an undervoltage. */
                pi_link_send_diag(&huart6, 0U, (uint8_t)g_imu.temperature,
                                  safety_state_legacy_fault_mask());

                if (++health_telemetry_divider >= 25U) {
                    Pi_HealthTelemetry_t health;
                    int i;
                    uint32_t health_now = HAL_GetTick();
                    health_telemetry_divider = 0U;
                    health.fault_mask = g_safety_state.fault_mask;
                    health.mode = (uint8_t)g_safety_state.current_mode;
                    health.reset_cause = g_reset_cause;
                    health.imu_age_ms = Device_Age_Ms(&g_imu.health, health_now);
                    health.wheel_l_age_ms = Device_Age_Ms(&left_feedback.health, health_now);
                    health.wheel_r_age_ms = Device_Age_Ms(&right_feedback.health, health_now);
                    health.imu_errors = Device_Error_Count(&g_imu.health);
                    health.wheel_l_errors = Device_Error_Count(&left_feedback.health);
                    health.wheel_r_errors = Device_Error_Count(&right_feedback.health);
                    health.wheel_l_timeout_errors = Device_Timeout_Count(&left_feedback.health);
                    health.wheel_l_checksum_errors = Device_Checksum_Count(&left_feedback.health);
                    health.wheel_l_protocol_errors = Device_Protocol_Count(&left_feedback.health);
                    health.wheel_r_timeout_errors = Device_Timeout_Count(&right_feedback.health);
                    health.wheel_r_checksum_errors = Device_Checksum_Count(&right_feedback.health);
                    health.wheel_r_protocol_errors = Device_Protocol_Count(&right_feedback.health);
                    for (i = 0; i < 4; ++i) {
                        health.servo_age_ms[i] = Device_Age_Ms(&servo_feedback[i].health,
                                                               health_now);
                        health.servo_errors[i] = Device_Error_Count(&servo_feedback[i].health);
                    }
                    (void)pi_link_send_health(&huart6, &health);
                }

                if (g_safety_state.current_mode == STATE_FAULT) {
                    pi_link_send_fault(&huart6, safety_state_legacy_fault_mask());
                }
            }
        }

        /* --- 50 Hz Background Loop: ST3215 Servo Control --- */
        uint32_t current_time = HAL_GetTick();
        if (g_actuator_configured && servo_deadline_pending) {
            servo_deadline_pending = 0U;

            uint8_t ids[4] = {SERVO_LF_ID, SERVO_RF_ID, SERVO_LB_ID, SERVO_RB_ID};

#if SERVO_ZERO_CALIBRATION_MODE
            /* Position commands are intentionally suppressed. System setup
             * has already disabled torque; feedback polling below still runs. */
            (void)ids;
#else
            if (g_safety_state.current_mode == STATE_ACTIVE &&
                runtime_outputs.servo_intent_allowed) {
                /* Pi supplies bounded workspace residuals.  Project them through
                 * the same dwell-relative (Qx,D0) five-bar IK as the simulator. */
                Pi_Command_Heartbeat_t hb;
                Pi_Command_Action_t act;
                Pi_Command_Snapshot(&hb, &act);
                int16_t pos_ticks[4];
                uint16_t speed_ticks[4] = {2000, 2000, 2000, 2000};
                uint8_t accels[4] = {50, 50, 50, 50};
                float action_scale = runtime_outputs.residual_allowed ? 1.0f : 0.0f;
                float d0_max = (fabsf(hb.target_velocity) > D0_GATE_V_THRESH ||
                                fabsf(hb.target_yaw_rate) > D0_GATE_W_THRESH)
                                   ? D0_GATE_MAX_HIGH * 0.001f : KIN_MAX_LEG_D0;
                float roll_term_mm = -(KUAFU_ROLL_KP * g_mahony.roll +
                                       KUAFU_ROLL_KD * g_body_gyro[0]);
                float d0_l = hb.target_leg_d0 + 0.001f * roll_term_mm / 2.0f +
                             0.001f * D0_RESIDUAL_SCALE_MM * action_scale * act.d0_l;
                float d0_r = hb.target_leg_d0 - 0.001f * roll_term_mm / 2.0f +
                             0.001f * D0_RESIDUAL_SCALE_MM * action_scale * act.d0_r;
                if (d0_l < KIN_MIN_LEG_D0) d0_l = KIN_MIN_LEG_D0;
                if (d0_l > d0_max) d0_l = d0_max;
                if (d0_r < KIN_MIN_LEG_D0) d0_r = KIN_MIN_LEG_D0;
                if (d0_r > d0_max) d0_r = d0_max;
                float qA_l, qB_l, qA_r, qB_r;
                if (kinematics_solve_ik_xy(QX_RESIDUAL_SCALE_MM * 0.001f * action_scale * act.qx_l, d0_l, &qA_l, &qB_l) == 0 &&
                    kinematics_solve_ik_xy(QX_RESIDUAL_SCALE_MM * 0.001f * action_scale * act.qx_r, d0_r, &qA_r, &qB_r) == 0) {
                    pos_ticks[0] = servo_angle_to_tick(qA_l, 0);
                    pos_ticks[1] = servo_angle_to_tick(qA_r, 1);
                    pos_ticks[2] = servo_angle_to_tick(qB_l, 2);
                    pos_ticks[3] = servo_angle_to_tick(qB_r, 3);
                    (void)st3215_bus_queue_sync_write(&g_st3215_bus, ids, 4U,
                                                      pos_ticks, speed_ticks, accels);
                }
            }
            else if ((g_safety_state.current_mode == STATE_STAND ||
                      g_safety_state.current_mode == STATE_CLIMB) &&
                     runtime_outputs.servo_intent_allowed) {
                /* Standing/Climbing virtual height mode */
                Pi_Command_Heartbeat_t hb;
                Pi_Command_Action_t act;
                Pi_Command_Snapshot(&hb, &act);
                (void)act;

                float q_hip_A = 0.0f; /* A chain, pivot x=-c (LF, RF) */
                float q_hip_B = 0.0f; /* B chain, pivot x=+c (LB, RB) */

                /* Compute inverse kinematics mapping for target height */
                if (kinematics_solve_ik(hb.target_leg_d0, &q_hip_A, &q_hip_B) == 0) {
                    int16_t pos_ticks[4];
                    uint16_t speed_ticks[4] = {1500, 1500, 1500, 1500};
                    uint8_t accels[4] = {30, 30, 30, 30};

                    pos_ticks[0] = servo_angle_to_tick(q_hip_A, 0); /* LF (A chain) */
                    pos_ticks[1] = servo_angle_to_tick(q_hip_A, 1); /* RF (A chain) */
                    pos_ticks[2] = servo_angle_to_tick(q_hip_B, 2); /* LB (B chain) */
                    pos_ticks[3] = servo_angle_to_tick(q_hip_B, 3); /* RB (B chain) */

                    (void)st3215_bus_queue_sync_write(&g_st3215_bus, ids, 4U,
                                                      pos_ticks, speed_ticks, accels);
                }
            }
            else if (g_safety_state.current_mode == STATE_FAULT) {
                /* Lockdown: disable servo torque once to allow gravity lock.
                 * Sending the disable every 50 Hz cycle floods the full-duplex
                 * bus with echo bytes that desync subsequent read queries, so a
                 * one-shot flag is used instead of repeated transmission. */
                if (fault_servo_disable_idx < 4U &&
                    st3215_bus_queue_torque(&g_st3215_bus,
                                            ids[fault_servo_disable_idx], 0U) == 0) {
                    ++fault_servo_disable_idx;
                }
            }
#endif
        }

        /* Poll every servo round-robin, including offline devices, so a valid
         * frame can restore health after line noise or a temporary disconnect. */
        current_time = HAL_GetTick();
        if (g_startup_manager.phase >= STARTUP_ACTUATOR_DISCOVERY &&
            current_time - last_servo_query_ms >= 5U &&
            st3215_bus_queue_read(&g_st3215_bus,
                                  &g_servos[active_servo_query_idx],
                                  g_safety_state.current_mode == STATE_INIT
                                      ? 0U : SERVO_FAIL_LIMIT,
                                  current_time) == 0) {
            last_servo_query_ms = current_time;
            active_servo_query_idx = (active_servo_query_idx + 1) % 4;
        }
    }
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

static void MX_IWDG_Init(void) {
    hiwdg.Instance = IWDG;
    hiwdg.Init.Prescaler = IWDG_PRESCALER_64;
    hiwdg.Init.Reload = 4095;
    if (HAL_IWDG_Init(&hiwdg) != HAL_OK) {
        Error_Handler();
    }
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
    HAL_NVIC_SetPriority(USART2_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
}

static void Actuator_Feedback_Snapshot(DDSM_State_t *left,
                                       DDSM_State_t *right,
                                       ST3215_State_t servos[4]) {
    int i;
    HAL_NVIC_DisableIRQ(USART2_IRQn);
    HAL_NVIC_DisableIRQ(USART3_IRQn);
    *left = g_ddsm_left;
    *right = g_ddsm_right;
    for (i = 0; i < 4; ++i) servos[i] = g_servos[i];
    HAL_NVIC_EnableIRQ(USART3_IRQn);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
}

static uint8_t Read_Reset_Cause(void) {
    uint8_t cause = 0U;
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_PINRST)) cause |= (1U << 0);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_PORRST)) cause |= (1U << 1);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_SFTRST)) cause |= (1U << 2);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_IWDGRST)) cause |= (1U << 3);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_WWDGRST)) cause |= (1U << 4);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_LPWRRST)) cause |= (1U << 5);
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_BORRST)) cause |= (1U << 6);
    __HAL_RCC_CLEAR_RESET_FLAGS();
    return cause;
}

static uint16_t Device_Age_Ms(const DeviceHealth_t *health, uint32_t now_ms) {
    uint32_t age;
    if (health == NULL || health->last_valid_ms == 0U) return UINT16_MAX;
    age = (uint32_t)(now_ms - health->last_valid_ms);
    return age > UINT16_MAX ? UINT16_MAX : (uint16_t)age;
}

static uint16_t Device_Error_Count(const DeviceHealth_t *health) {
    uint32_t total;
    if (health == NULL) return UINT16_MAX;
    total = (uint32_t)health->timeout_count + health->checksum_count + health->protocol_count;
    return total > UINT16_MAX ? UINT16_MAX : (uint16_t)total;
}

static uint16_t Device_Timeout_Count(const DeviceHealth_t *health) {
    return (health == NULL) ? UINT16_MAX : health->timeout_count;
}

static uint16_t Device_Checksum_Count(const DeviceHealth_t *health) {
    return (health == NULL) ? UINT16_MAX : health->checksum_count;
}

static uint16_t Device_Protocol_Count(const DeviceHealth_t *health) {
    return (health == NULL) ? UINT16_MAX : health->protocol_count;
}

static void MX_USART3_UART_Init(void) {
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

    /* Full-duplex: separate TX (PB10) and RX (PB11) lines. The ST3215 bus servos
     * attach through a Waveshare Bus Servo Adapter (A), which converts the
     * single-wire half-duplex servo bus into a 2-wire UART (TXD/RXD). Both pins
     * are AF push-pull; the adapter board drives its own lines. */
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = SERVO_TX_PIN | SERVO_RX_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = SERVO_USART_AF;
    HAL_GPIO_Init(SERVO_TX_PORT, &GPIO_InitStruct);

    if (HAL_UART_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }
    HAL_NVIC_SetPriority(USART3_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(USART3_IRQn);
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
    /* The Pi RX stream is consumed incrementally from a DMA ring. */
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
        g_pi_poll_requested = 1U;
    }
    HAL_UART_IRQHandler(&huart6);
}

void USART2_IRQHandler(void) {
    HAL_UART_IRQHandler(&huart2);
}

void USART3_IRQHandler(void) {
    HAL_UART_IRQHandler(&huart3);
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ddsm_bus_on_tx_complete(&g_ddsm_bus);
    } else if (huart == &huart3) {
        st3215_bus_on_tx_complete(&g_st3215_bus);
    } else if (huart == &huart6) {
        pi_link_on_tx_complete(huart);
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ddsm_bus_on_rx_byte(&g_ddsm_bus, HAL_GetTick());
    } else if (huart == &huart3) {
        st3215_bus_on_rx_byte(&g_st3215_bus, HAL_GetTick());
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ddsm_bus_on_uart_error(&g_ddsm_bus, huart);
    } else if (huart == &huart3) {
        st3215_bus_on_uart_error(&g_st3215_bus);
    } else if (huart == &huart6) {
        pi_link_on_tx_error(huart);
    }
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
