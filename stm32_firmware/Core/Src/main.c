#include "stm32f4xx_hal.h"
#include "pin_config.h"
#include <math.h>
#include <string.h>
#include "crc8.h"
#include "pi_link.h"
#include "pi_transport.h"
#include "bmi088.h"
#include "ddsm315.h"
#include "st3215.h"
#include "mahony.h"
#include "kinematics.h"
#include "balance_trace.h"
#include "safety_state.h"
#include "servo_mapping.h"
#include "startup_manager.h"
#include "firmware_runtime.h"
#include "control_section.h"
#include "dma_rx_ring.h"

/* Peripheral Handles */
I2C_HandleTypeDef hi2c1;
UART_HandleTypeDef huart1; /* CH340 Debug/Program */
UART_HandleTypeDef huart2; /* DDSM RS485 */
UART_HandleTypeDef huart3; /* ST3215 Half-Duplex Servo */
UART_HandleTypeDef huart6; /* Pi 5 Bridge */
IWDG_HandleTypeDef hiwdg;
DMA_HandleTypeDef hdma_usart6_rx;
DMA_HandleTypeDef hdma_usart3_rx;

/* ST3215 servo bus RX ring: 1 Mbaud is far beyond single-byte HAL IRQ
 * servicing; a circular DMA plus main-loop consumption removes the overrun
 * storm entirely (see st3215.c note).  The DmaRxRing wrapper reconstructs an
 * absolute producer cursor from NDTR so a main-loop stall that lets the DMA
 * lap the ring is counted in overrun_count instead of silently corrupting
 * in-flight frames. */
#define ST3215_RX_BUF_SIZE 64U
static uint8_t g_st3215_rx_buf[ST3215_RX_BUF_SIZE];
static DmaRxRing_t g_st3215_ring;

/* Real-Time Telemetry and State variables */
volatile uint32_t g_system_ticks = 0;
BMI088_t g_imu;
MahonyFilter_t g_mahony;

/* One-pole low-pass on pitch rate to suppress gyro noise amplified by K3.
 * α = 0.2 at 1 kHz → τ ≈ 4.5 ms → fc ≈ 35 Hz, well above balance bandwidth. */
static float g_pitch_rate_filt = 0.0f;
#define PITCH_RATE_FILTER_ALPHA 0.2f

/* Latest fused attitude, shared between the DRDY-driven fusion block and the
 * ms-deadline-driven safety block (which must keep running even if DRDY stops). */
static float g_body_pitch = 0.0f;
static float g_body_pitch_rate = 0.0f;

DDSM_State_t g_ddsm_left;
DDSM_State_t g_ddsm_right;
ST3215_State_t g_servos[4];
StartupManager_t g_startup_manager;
uint8_t g_actuator_discovery_step;
uint8_t g_actuator_configured;
/* Per-wheel mask (bit0=left, bit1=right) recording that a current-loop mode
 * frame has been (re)sent while the motor is enabled.  Mode switching while
 * the motor is disabled is unreliable on the DDSM315, so once the wheels are
 * armed we re-assert current mode on each wheel exactly once. */
static uint8_t g_wheel_mode_sent = 0U;
static DDSM_Bus_t g_ddsm_bus;
static ST3215_Bus_t g_st3215_bus;

/* Wheel torque commands computed and sent together at each 250 Hz deadline. */
static float g_ctrl_tau_l = 0.0f;
static float g_ctrl_tau_r = 0.0f;
static volatile float g_body_gyro[3] = {0.0f, 0.0f, 0.0f};

/* Final wheel-output authorization.  Updated once per 250 Hz control
 * deadline as the conjunction of three independent verdicts:
 *   - firmware_runtime intent (mode operational + startup authorized),
 *   - the actuator supervisor (verified enables, fresh leg support, safe
 *     posture — see actuator_supervisor.c),
 *   - current-loop mode re-assertion completed on both wheels.
 * Held between deadlines so every later consumer in the same control period
 * (LQR computation, bus dispatch) sees the same verdict; defaults to closed
 * until the first deadline opens it.  SWD-visible for bring-up forensics. */
static uint8_t g_wheel_output_gate = 0U;

/* The 250 Hz control section (safety -> supervisor verdict -> gate -> LQR ->
 * trace).  Extracted into control_section.c so the composition that
 * authorises wheel torque is host-testable; the scheduler samples inputs and
 * applies outputs.  SWD-visible for bring-up forensics. */
static ControlSection_t g_control_section;

/* Body-frame torque actually accepted by the DDSM queue at dispatch time and
 * the ms it was queued; the balance trace reports these as "sent" (distinct
 * from the LQR target) with an age-based validity bit. */
static float g_sent_tau_l = 0.0f;
static float g_sent_tau_r = 0.0f;
static uint32_t g_sent_tau_ms = 0U;

/* Set by the fusion block: attitude/rate are trustworthy for control only
 * after a FULL update or a bounded GYRO_ONLY window (see mahony.c). */
static uint8_t g_imu_control_valid = 0U;

/* ms of the last successfully queued STAND/ACTIVE leg-hold sync-write; the
 * supervisor's "leg target transmitted" input must reflect a real position
 * command on the wire, not merely a torque-enable frame. */
static uint32_t g_leg_hold_tx_ms = 0U;
#define LEG_HOLD_MAX_AGE_MS 100U

/* Consecutive read failures before the round-robin poll marks a servo
 * offline (offline_after).  Freshness fault latching is the safety layer's
 * job (SAFETY_SERVO_MAX_AGE_MS + debounce); this only drives device_health
 * online state. */
#define SERVO_FAIL_LIMIT         3

/* DMA Buffer for Pi Bridge (USART6 RX) */
#define PI_RX_BUF_SIZE           256
uint8_t g_pi_rx_buf[PI_RX_BUF_SIZE];
static PiTransport_t g_pi_transport;
static volatile uint8_t g_pi_poll_requested = 0U;
static uint8_t g_reset_cause = 0U;

/* Main-loop stall forensics: the IMU data-ready handler (EXTI1, 1 kHz,
 * highest priority) watches the loop heartbeat; on a >100 ms stall it
 * snapshots UART status/error evidence to the RAM-top dump area so an
 * interrupt storm starving the loop is visible after the IWDG reset. */
volatile uint32_t g_loop_heartbeat_ms = 0U;
volatile uint32_t g_uart2_err_cnt = 0U;
volatile uint32_t g_uart3_err_cnt = 0U;
volatile uint32_t g_uart6_err_cnt = 0U;
volatile uint32_t g_uart2_rx_cnt = 0U;
volatile uint32_t g_uart3_rx_cnt = 0U;
/* Set by the UART error ISR when an RX error (ORE/NE/FE) fires: the HAL clears
 * CR3.DMAR and aborts the circular RX DMA on overrun, so without a re-arm the
 * bus stays deaf until reboot (observed on USART3: one boot-time ORE killed
 * all servo feedback).  The main loop performs the actual re-arm because the
 * DMA abort completes asynchronously (Receive_DMA returns BUSY inside the ISR). */
volatile uint8_t g_uart3_rx_rearm = 0U;
volatile uint8_t g_uart6_rx_rearm = 0U;

#define STALL_DUMP_BASE  0x2001FFC0U  /* 12 words; ends just below FAULT_DUMP @0x2001FFF0 */
#define STALL_DUMP_MAGIC 0x57A11EDU

void main_loop_stall_check(void) {
    volatile uint32_t *dump = (volatile uint32_t *)STALL_DUMP_BASE;
    uint32_t now = HAL_GetTick();
    if (dump[0] == STALL_DUMP_MAGIC) return;             /* keep first stall */
    if ((uint32_t)(now - g_loop_heartbeat_ms) <= 100U) return;
    dump[1] = g_loop_heartbeat_ms;
    dump[2] = g_system_ticks;
    dump[3] = huart2.Instance->SR;
    dump[4] = huart3.Instance->SR;
    dump[5] = huart6.Instance->SR;
    dump[6] = g_uart2_err_cnt;
    dump[7] = g_uart3_err_cnt;
    dump[8] = g_uart6_err_cnt;
    dump[9] = g_uart2_rx_cnt;
    dump[10] = g_uart3_rx_cnt;
    dump[11] = hdma_usart6_rx.Instance->NDTR;
    dump[0] = STALL_DUMP_MAGIC;
}

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

    /* Keep the HAL tick alive under UART interrupt storms: HAL timeouts and
     * delay loops depend on it, and a starved tick turns a bounded 2 ms I2C
     * timeout into an unbounded wait. */
    NVIC_SetPriority(SysTick_IRQn, 0U);

    /* The ST3215 RX ring must exist BEFORE MX_USART3_UART_Init starts its
     * circular DMA stream: the DMA transfer-complete callback notes every
     * completed lap through dma_rx_ring_note_lap, and a lap counted between
     * the DMA start and a later zeroing dma_rx_ring_init would be erased,
     * skewing the absolute producer cursor and re-feeding overwritten bytes
     * to the parser as spurious checksum failures. */
    dma_rx_ring_init(&g_st3215_ring, g_st3215_rx_buf, ST3215_RX_BUF_SIZE);

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    MX_USART2_UART_Init();
    MX_USART3_UART_Init();  /* starts circular RX DMA into g_st3215_rx_buf */
    MX_USART6_UART_Init();
    MX_IWDG_Init();

    /* Initialize drivers, CRC tables and control states */
    crc8_init();
    pi_link_init();
    pi_transport_init(&g_pi_transport, g_pi_rx_buf, PI_RX_BUF_SIZE);
    safety_state_init();
    control_section_init(&g_control_section, HAL_GetTick());
    balance_trace_init();
    balance_trace_set_metadata("kuafu-stm32", KUAFU_MODEL_HASH);
    mahony_init(&g_mahony, 10.0f, 0.002f); /* Kp high for fast balance tracking */

    g_ddsm_left.id = DDSM_LEFT_ID;
    g_ddsm_right.id = DDSM_RIGHT_ID;
    device_health_init(&g_ddsm_left.health);
    device_health_init(&g_ddsm_right.health);
#if DDSM_ID_CALIBRATION_TARGET > 0
    {
        uint8_t id_packet[DDSM_FRAME_SIZE];
        uint8_t repeat;
        ddsm_build_set_id(id_packet, DDSM_ID_CALIBRATION_TARGET);
        /* FAULT_INIT (serious) is the latch: wheel-freshness faults are
         * TRANSIENT and would auto-recover to STAND once the freshly
         * re-IDed motor answers polls — re-enabling wheel authorization on
         * a bench image that must never actuate.  The transient bits stay
         * in the mask for diagnostic telemetry. */
        safety_state_trigger_fault(FAULT_INIT | FAULT_WHEEL_LEFT | FAULT_WHEEL_RIGHT);
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
        /* Freshness requires a real first valid frame: an optimistic online
         * flag here would make a servo that never replied look "fresh" for
         * the first SAFETY_SERVO_MAX_AGE_MS after boot, which — given the
         * ~800 ms power-wait + IMU + discovery sequence — could carry INIT
         * into STAND and briefly open the wheel gate on legs whose feedback
         * has never been seen.  The round-robin poll below restores online
         * state on the first valid frame. */
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
     * placeholder center ticks.  FAULT_INIT (serious) is the latch: a bare
     * FAULT_SERVO is TRANSIENT and auto-recovers to STAND as soon as the
     * polled servos report fresh — re-arming the enable sequencer and wheel
     * authorization on an image that must stay torque-free.  The transient
     * bit stays in the mask for diagnostic telemetry; the wheel commands
     * stay at zero and the background feedback poll keeps running. */
    safety_state_trigger_fault(FAULT_INIT | FAULT_SERVO);
#endif

    uint32_t last_tick = 0;
    uint32_t next_wheel_tx_ms = 0U;
    uint8_t next_wheel_is_right = 0U;
    uint8_t active_servo_query_idx = 0;
    uint32_t last_servo_query_ms = 0U;
    uint8_t fault_servo_disable_idx = 0U;
    uint32_t temp_refresh_counter = 0;
    uint8_t health_telemetry_divider = 0U;
    uint32_t last_diag_tx_ms = 0U; /* wall-clock slot for diag/health/fault frames */
    uint8_t bmi_init_in_progress = 0U;
    uint8_t servo_enable_step = 0U;
    uint8_t servos_enabled = 0U;
    uint8_t wheel_enable_mask = 0U;
    uint32_t next_actuator_enable_ms = 0U;
    uint32_t last_pi_poll_ms = 0U;
    FirmwareRuntime_t firmware_runtime;
    uint8_t servo_deadline_pending = 0U;

    startup_manager_init(&g_startup_manager, HAL_GetTick());
    firmware_runtime_init(&firmware_runtime, HAL_GetTick());

    /* Main background scheduler loop */
    while (1) {
        uint32_t startup_now = HAL_GetTick();
        g_loop_heartbeat_ms = startup_now;
        StartupInputs_t startup_inputs;
        StartupOutputs_t startup_outputs;
        FirmwareRuntimeInputs_t runtime_inputs;
        FirmwareRuntimeOutputs_t runtime_outputs;
        DDSM_State_t left_feedback;
        DDSM_State_t right_feedback;
        ST3215_State_t servo_feedback[4];
        uint8_t wheel_authorized;
        uint8_t startup_servos_online = 1U;

        if (g_uart6_rx_rearm) {
            /* Synchronous abort first: HAL clears CR3.DMAR and disables the RX
             * DMA on an ORE/NE/FE, and that teardown completes ASYNCHRONOUSLY.
             * Re-arming without waiting lets the late abort completion wipe the
             * fresh Receive_DMA state (observed on USART3: bus permanently deaf
             * with ORE latched, EIE=0 and no further error callbacks). */
            (void)HAL_UART_AbortReceive(&huart6);
            __HAL_UART_CLEAR_OREFLAG(&huart6);
            if (HAL_UART_Receive_DMA(&huart6, g_pi_rx_buf, PI_RX_BUF_SIZE) == HAL_OK) {
                /* The restarted stream begins at the ring base; the transport
                 * must not re-parse pre-restart content as new data. */
                pi_transport_reset(&g_pi_transport);
                g_uart6_rx_rearm = 0U;
            }
        } else if ((hdma_usart6_rx.Instance->CR & DMA_SxCR_EN) == 0U) {
            /* Deaf-DMA watchdog: the abort-after-rearm race can end with the
             * stream disabled and no error callback left to fire (EIE cleared),
             * so re-detection must not rely on the error path alone. */
            g_uart6_rx_rearm = 1U;
        }
        if (g_uart3_rx_rearm) {
            (void)HAL_UART_AbortReceive(&huart3);
            __HAL_UART_CLEAR_OREFLAG(&huart3);
            if (HAL_UART_Receive_DMA(&huart3, g_st3215_rx_buf, ST3215_RX_BUF_SIZE) == HAL_OK) {
                /* Rebase, not re-init: the synchronous abort completes the
                 * old stream, so a lap racing the re-arm is real and must
                 * stay counted (see dma_rx_ring_rebase). */
                dma_rx_ring_rebase(&g_st3215_ring);
                g_uart3_rx_rearm = 0U;
            }
        } else if ((hdma_usart3_rx.Instance->CR & DMA_SxCR_EN) == 0U) {
            g_uart3_rx_rearm = 1U;
        }
        if (g_pi_poll_requested || startup_now != last_pi_poll_ms) {
            g_pi_poll_requested = 0U;
            last_pi_poll_ms = startup_now;
            /* Ordering contract (pi_transport.h): lap count first, NDTR after. */
            uint32_t pi_laps = g_pi_transport.lap_count;
            uint16_t pi_ndtr = (uint16_t)__HAL_DMA_GET_COUNTER(&hdma_usart6_rx);
            (void)pi_transport_poll(&g_pi_transport, pi_laps, pi_ndtr);
        }

        ddsm_bus_step(&g_ddsm_bus, startup_now);
        /* Drain the ST3215 DMA ring into the bus parser (main-loop context).
         * The absolute producer cursor is reconstructed as lap_count * size
         * + write index (ordering contract in dma_rx_ring.h: lap count read
         * first, NDTR after); a genuine consumer lag past one full ring is
         * billed to overrun_count, and consume() returns only the still-
         * valid oldest bytes. */
        {
            uint32_t laps = g_st3215_ring.lap_count;
            uint16_t ndtr = (uint16_t)__HAL_DMA_GET_COUNTER(&hdma_usart3_rx);
            uint16_t write_idx = (ndtr == 0U) ? ST3215_RX_BUF_SIZE
                                              : (uint16_t)(ST3215_RX_BUF_SIZE - ndtr);
            uint8_t chunk[ST3215_RX_BUF_SIZE];
            uint16_t count;
            dma_rx_ring_update_producer(&g_st3215_ring, laps, write_idx);
            while ((count = dma_rx_ring_consume(&g_st3215_ring,
                                                chunk, ST3215_RX_BUF_SIZE)) > 0U) {
                for (uint16_t i = 0U; i < count; ++i) {
                    ++g_uart3_rx_cnt;
                    st3215_bus_rx_byte_from_dma(&g_st3215_bus, chunk[i], startup_now);
                }
            }
        }
        st3215_bus_step(&g_st3215_bus, startup_now);
        Actuator_Feedback_Snapshot(&left_feedback, &right_feedback, servo_feedback);
        /* Wheel torque authorization for the LQR self-balance loop.
         * Self-balancing is a baseline capability that must work standalone
         * (no Pi), so it is gated only on startup completion + an operational
         * mode + no fault.  INIT is deliberately excluded: the documented
         * state contract keeps the wheel power domain locked (read-only
         * queries) until the safety machine reaches STAND.  Pi link freshness
         * only gates the ACTIVE-mode motion commands, not the balance loop
         * itself. */
        wheel_authorized = (uint8_t)(g_startup_manager.phase == STARTUP_READY &&
                                     g_safety_state.current_mode != STATE_INIT &&
                                     g_safety_state.current_mode != STATE_FAULT);

        runtime_inputs.now_ms = startup_now;
        runtime_inputs.mode = g_safety_state.current_mode;
        runtime_inputs.link_compatible = pi_link_is_compatible();
        runtime_inputs.heartbeat_fresh = pi_link_heartbeat_fresh();
        runtime_inputs.action_fresh = pi_link_action_fresh();
        runtime_inputs.wheel_authorized = wheel_authorized;
        runtime_inputs.wheel_bus_idle = ddsm_bus_is_idle(&g_ddsm_bus);
        runtime_inputs.servo_bus_idle = st3215_bus_is_idle(&g_st3215_bus);
        runtime_outputs = firmware_runtime_step(&firmware_runtime, &runtime_inputs);
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
        /* Per-channel IMU validity: a dead accelerometer must not be masked by
         * a live gyro (or vice versa) through the legacy aggregate health. */
        startup_inputs.accel_valid = bmi088_accel_healthy(&g_imu, startup_now,
                                                          SAFETY_IMU_MAX_AGE_MS);
        startup_inputs.gyro_valid = bmi088_gyro_healthy(&g_imu, startup_now,
                                                        SAFETY_IMU_MAX_AGE_MS);
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
                /* Bring the motor out of its power-up state first (Protocol-3
                 * mode switching is only honoured once the motor is enabled). */
                discovery_result = ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_left,
                                                         0U, startup_now);
            } else if (g_actuator_discovery_step == 1U) {
                /* Select the current (torque) loop.  Protocol-3 form: byte[9]=mode
                 * with no CRC.  Re-asserted while enabled after arming too. */
                discovery_result = ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_left,
                                                       DDSM_MODE_CURRENT, startup_now);
            } else if (g_actuator_discovery_step == 2U) {
                /* Explicit zero-torque clears any stale torque retained across
                 * power cycles so the motor does not spin before the Pi arms. */
                discovery_result = ddsm_bus_queue_torque(&g_ddsm_bus, &g_ddsm_left,
                                                         0.0f, startup_now);
            } else if (g_actuator_discovery_step == 3U) {
                discovery_result = ddsm_bus_queue_enable(&g_ddsm_bus, &g_ddsm_right,
                                                         0U, startup_now);
            } else if (g_actuator_discovery_step == 4U) {
                discovery_result = ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_right,
                                                       DDSM_MODE_CURRENT, startup_now);
            } else if (g_actuator_discovery_step == 5U) {
                discovery_result = ddsm_bus_queue_torque(&g_ddsm_bus, &g_ddsm_right,
                                                         0.0f, startup_now);
            } else if (g_actuator_discovery_step == 6U) {
                discovery_result = st3215_bus_queue_torque(
                    &g_st3215_bus, ST3215_BROADCAST_ID, 0U);
            }
            if (g_actuator_discovery_step < 7U && discovery_result == 0) {
                ++g_actuator_discovery_step;
            }
            g_actuator_configured = (uint8_t)(g_actuator_discovery_step >= 7U);
        }

        /* Mode gate: the enable sequencer must never run while the machine
         * is in FAULT.  The fault-entry edge resets servos_enabled so the
         * software stops claiming "verified", and the 50 Hz FAULT branch
         * physically disables servo torque; a mode-blind sequencer would
         * immediately re-send the enable frames, fight the disable one-shot,
         * and re-set servos_enabled to 1 while the joints are physically
         * torque-free.  A transient fault (wheel/servo freshness) then
         * recovers with that stale flag, the supervisor clears its re-issue
         * stage on the racing 0->1 observation, and the wheel gate re-opens
         * on limp legs.  Re-issue is for RECOVERY only: the sequencer re-runs
         * once the safety machine has left FAULT.  INIT stays allowed because
         * at boot the sequencer must complete before INIT can exit to STAND
         * (startup_ready == servos_enabled feeds the transition). */
        if (startup_outputs.enable_actuators && !servos_enabled &&
            g_safety_state.current_mode != STATE_FAULT &&
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
     * zero-current polling run while disabled.  Arming is authorized by the
     * standalone balance conditions only — startup READY, an operational
     * (non-INIT) mode, and no latched fault — because self-balancing is a
     * baseline capability that must not depend on the Pi link; Pi link
     * freshness gates ACTIVE-mode commands (residuals, velocity), never the
     * balance loop itself.  Authorization is revoked whenever any condition
     * drops, which queues the physical disable frames below. */
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

        /* Once both wheels are armed, re-assert the current-loop mode on any
         * wheel that has not yet received it while enabled.  Mode switching is
         * only reliably honoured by the DDSM315 after the motor is enabled. */
        if (g_actuator_configured && wheel_enable_mask == 0x03U &&
            g_wheel_mode_sent != 0x03U && ddsm_bus_is_idle(&g_ddsm_bus)) {
            if ((g_wheel_mode_sent & 0x01U) == 0U) {
                if (ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_left,
                                        DDSM_MODE_CURRENT, startup_now) == 0) {
                    g_wheel_mode_sent |= 0x01U;
                }
            } else if ((g_wheel_mode_sent & 0x02U) == 0U) {
                if (ddsm_bus_queue_mode(&g_ddsm_bus, &g_ddsm_right,
                                        DDSM_MODE_CURRENT, startup_now) == 0) {
                    g_wheel_mode_sent |= 0x02U;
                }
            }
        }

        if (startup_outputs.fault_requested) {
            safety_state_trigger_fault(FAULT_INIT);
        }

        /* Reaching this point proves the scheduler is alive even before DRDY. */
        HAL_IWDG_Refresh(&hiwdg);

        /* ================================================================
         * 250 Hz control section (HAL_GetTick deadline; deliberately
         * independent of the gyro data-ready interrupt).  The whole
         * sequence -- safety state machine, actuator-supervisor verdict,
         * wheel output gate, LQR computation, balance trace -- runs inside
         * control_section_step (host-testable); the scheduler samples its
         * inputs from the loop-local snapshots and applies its outputs for
         * the dispatch layer.  If the IMU dies, imu_fresh goes stale and
         * the debounced freshness fault latches FAULT here regardless of
         * DRDY.
         * ================================================================ */
        if (runtime_outputs.control_due &&
            g_startup_manager.phase >= STARTUP_ACTUATOR_DISCOVERY) {
            ControlSectionInputs_t cs_in;
            ControlSectionOutputs_t cs_out;
            uint32_t cs_now = HAL_GetTick();
            uint8_t servos_fresh = 1U;
            uint8_t hold_tx_recent = (uint8_t)(
                servos_enabled &&
                (uint32_t)(cs_now - g_leg_hold_tx_ms) <= LEG_HOLD_MAX_AGE_MS);
            float max_temp = g_imu.temperature;
            uint32_t sent_age;
            for (int i = 0; i < 4; ++i) {
                if (!device_health_is_fresh(&servo_feedback[i].health,
                                            cs_now, SAFETY_SERVO_MAX_AGE_MS)) {
                    servos_fresh = 0U;
                }
                if (servo_feedback[i].temperature_c > max_temp) {
                    max_temp = servo_feedback[i].temperature_c;
                }
            }
            memset(&cs_in, 0, sizeof(cs_in));
            cs_in.now_ms = cs_now;
            cs_in.runtime = runtime_outputs;
            cs_in.imu_control_valid = g_imu_control_valid;
            cs_in.pitch_rad = g_body_pitch;
            cs_in.pitch_rate_rad_s = g_body_pitch_rate;
            cs_in.wheel_vel_l_rads = WHEEL_DIR_L * left_feedback.velocity_rads;
            cs_in.wheel_vel_r_rads = WHEEL_DIR_R * right_feedback.velocity_rads;
            cs_in.yaw_rad = g_mahony.yaw;
            cs_in.yaw_rate_rads = g_body_gyro[2];
            cs_in.max_temp_c = max_temp;
            cs_in.imu_fresh = device_health_is_fresh(&g_imu.health, cs_now,
                                                     SAFETY_IMU_MAX_AGE_MS);
            cs_in.wheel_l_fresh = device_health_is_fresh(&left_feedback.health,
                                                         cs_now,
                                                         SAFETY_WHEEL_MAX_AGE_MS);
            cs_in.wheel_r_fresh = device_health_is_fresh(&right_feedback.health,
                                                         cs_now,
                                                         SAFETY_WHEEL_MAX_AGE_MS);
            cs_in.servos_fresh = servos_fresh;
            cs_in.requested_mode = g_pi_cmd_heartbeat.mode_request;
            cs_in.startup_ready = servos_enabled;
            cs_in.actuator_configured = g_actuator_configured;
            cs_in.wheel_authorized = wheel_authorized;
            cs_in.wheel_bus_idle = runtime_inputs.wheel_bus_idle;
            cs_in.servo_bus_idle = runtime_inputs.servo_bus_idle;
            cs_in.link_compatible = pi_link_is_compatible();
            cs_in.heartbeat_fresh = pi_link_heartbeat_fresh();
            cs_in.action_fresh = pi_link_action_fresh();
            cs_in.leg_hold_tx_recent = hold_tx_recent;
            cs_in.leg_feedback_fresh = startup_servos_online;
            cs_in.leg_posture_safe = (uint8_t)(fabsf(g_body_pitch) <= PITCH_FADE_END_RAD);
            cs_in.servo_enable_verified = servos_enabled;
            cs_in.wheel_enable_verified = (uint8_t)(wheel_enable_mask == 0x03U);
            cs_in.wheel_mode_complete = (uint8_t)(g_wheel_mode_sent == 0x03U);
            Pi_Command_Snapshot(&cs_in.heartbeat, &cs_in.action);
            cs_in.startup_phase = (uint32_t)g_startup_manager.phase;
            cs_in.imu_age_ms = Device_Age_Ms(&g_imu.health, cs_now);
            cs_in.wheel_left_age_ms = Device_Age_Ms(&left_feedback.health, cs_now);
            cs_in.wheel_right_age_ms = Device_Age_Ms(&right_feedback.health, cs_now);
            sent_age = cs_now - g_sent_tau_ms;
            cs_in.wheel_sent_age_ms =
                (uint16_t)(sent_age > UINT16_MAX ? UINT16_MAX : sent_age);
            cs_in.feedback_torque_left_nm = WHEEL_DIR_L * left_feedback.torque;
            cs_in.feedback_torque_right_nm = WHEEL_DIR_R * right_feedback.torque;
            cs_in.sent_torque_left_nm = g_sent_tau_l;
            cs_in.sent_torque_right_nm = g_sent_tau_r;

            cs_out = control_section_step(&g_control_section, &cs_in);
            g_wheel_output_gate = cs_out.wheel_output_gate;
            g_ctrl_tau_l = cs_out.torque_left_nm;
            g_ctrl_tau_r = cs_out.torque_right_nm;

            if (cs_out.fault_entry_edge) {
                /* FAULT entry resets the one-shot enable bookkeeping.  The
                 * FAULT paths physically disable servo torque and queue
                 * wheel disables, so the software must not keep claiming
                 * "enabled/verified": the 50 Hz disable sequence becomes
                 * runnable again (fault_servo_disable_idx), the servo
                 * re-enable path re-runs on recovery (servo_enable_step/
                 * servos_enabled), and wheel current-loop mode is
                 * re-asserted after re-arming (g_wheel_mode_sent).
                 * wheel_enable_mask clears itself through the disable queue
                 * when authorization drops. */
                servo_enable_step = 0U;
                servos_enabled = 0U;
                fault_servo_disable_idx = 0U;
                g_wheel_mode_sent = 0U;
            }
        }
        /* Wheel bus dispatch at the DDSM315 bus limit.
         * At 115200 baud a 10-byte frame takes ~0.87 ms TX + ~0.87 ms RX;
         * the 4 ms spacing matches the original bus scheduling and gives each
         * wheel a 125 Hz update rate.  Removing it caused RS485 overruns.
         * Positioned after the 250 Hz control section so a frame on the wire
         * never predates the verdict and torque computed in the same period;
         * g_wheel_output_gate is the persistent authorization.  Closed states
         * keep polling so wheel feedback freshness survives every revocation:
         * torque command frames (0x64) elicit the same status reply as a
         * query, so the freshness channel is identical in every branch.
         *
         * Zero-torque, not a query, whenever the motor is still enabled: the
         * DDSM315 current loop HOLDS its last setpoint, so a closed gate on an
         * enabled wheel would otherwise leave the pre-closure LQR torque
         * actively driving the robot (supervisor HOLDING demotions keep the
         * mode operational, so the TILT fault backstop only arrives after the
         * robot has fallen).  Zero-torque pins the setpoint to coast.  Plain
         * queries are reserved for motors that are (or may be) disabled --
         * INIT/discovery and the disable-drain window -- where a command
         * frame would be pointless and the mode re-assertion owns re-arming. */
        if (g_actuator_configured && ddsm_bus_is_idle(&g_ddsm_bus) &&
            (int32_t)(startup_now - next_wheel_tx_ms) >= 0) {
            int wheel_result;
            uint8_t sent_torque_frame = 0U;
            DDSM_State_t *wheel = next_wheel_is_right ? &g_ddsm_right : &g_ddsm_left;
            float torque = next_wheel_is_right
                         ? WHEEL_DIR_R * g_ctrl_tau_r
                         : WHEEL_DIR_L * g_ctrl_tau_l;
            if (g_wheel_output_gate) {
                wheel_result = ddsm_bus_queue_torque(&g_ddsm_bus, wheel,
                                                     torque, startup_now);
                sent_torque_frame = (uint8_t)(wheel_result == 0);
            } else if (g_safety_state.current_mode == STATE_FAULT ||
                       wheel_enable_mask != 0U) {
                wheel_result = ddsm_bus_queue_torque(&g_ddsm_bus, wheel,
                                                     0.0f, startup_now);
            } else {
                wheel_result = ddsm_bus_queue_query(&g_ddsm_bus, wheel, startup_now);
            }
            if (wheel_result == 0) {
                next_wheel_is_right ^= 1U;
                next_wheel_tx_ms = startup_now + 4U;
                if (sent_torque_frame) {
                    /* Body-frame value of the accepted frame, for the trace's
                     * "sent" channels (distinct from the LQR target). */
                    if (next_wheel_is_right) {
                        g_sent_tau_l = g_ctrl_tau_l;
                    } else {
                        g_sent_tau_r = g_ctrl_tau_r;
                    }
                    g_sent_tau_ms = startup_now;
                }
            }
        }

        /* Soft real-time scheduler aligned to system ticks (1ms resolution) */
        if (g_system_ticks != last_tick) {
            /* Use the actual elapsed time; blocking bus I/O can skip ticks.
             * Clamp the fusion step: after a long main-loop stall (e.g. an
             * interrupt storm) dticks spans hundreds of ms, and integrating
             * the quaternion with dt~1 s would blow up the attitude. */
            uint32_t dticks = g_system_ticks - last_tick;
            last_tick = g_system_ticks;
            float fusion_dt = (float)dticks * 0.001f;
            if (fusion_dt > 0.02f) fusion_dt = 0.02f;

            /* DRDY can arrive while the BMI088 register sequence is still in
             * progress.  Do not read or calibrate from an uninitialized IMU.
             * A guarded block (not `continue`): a continue here would also skip
             * every scheduler stage after this tick block -- the 50 Hz leg
             * writer and the servo round-robin poll -- silently coupling their
             * execution to the IMU init state.  Those stages are independently
             * guarded (actuator_configured / startup phase) and must not depend
             * on where this block decides to bail out. */
            if (g_imu.initialized) {

                /* 1. Read IMU sensors (safe to do here in the main loop background).
                 * Return codes matter: a failed channel keeps its previous sample
                 * and must not be fused as if it were fresh. */
                (void)bmi088_read_accel(&g_imu);
                (void)bmi088_read_gyro(&g_imu);

                /* Refresh chip temperature at low rate (~10 Hz) for safety/telemetry */
                if (++temp_refresh_counter >= 100) {
                    temp_refresh_counter = 0;
                    bmi088_read_temp(&g_imu);
                }

                BMI088SampleValidity_t imu_validity;
                bmi088_get_sample_validity(&g_imu, HAL_GetTick(),
                                           BMI088_DEFAULT_MAX_AGE_MS, &imu_validity);
                uint8_t accel_ok = (uint8_t)(imu_validity.accel_valid &&
                                             imu_validity.accel_fresh);
                uint8_t gyro_ok = (uint8_t)(imu_validity.gyro_valid &&
                                            imu_validity.gyro_fresh);

                float gx = g_imu.gyro[0];
                float gy = g_imu.gyro[1];
                float gz = g_imu.gyro[2];

                /* 2. Run sensor fusion update with per-channel validity. */
                if (g_safety_state.is_gyro_calibrated) {
                    gx -= g_safety_state.gyro_calib_offset[0];
                    gy -= g_safety_state.gyro_calib_offset[1];
                    gz -= g_safety_state.gyro_calib_offset[2];
                }
                /* Always attempt the fusion update.  Mahony is accel-referenced, so
                 * running with a zero gyro offset at start is safe; the bias
                 * estimate is refined in the background (below) once the robot
                 * settles.  An invalid gyro is rejected inside the filter and
                 * revokes control validity instead of integrating garbage. */
                MahonyUpdateStatus_t fuse_status = mahony_update_validated(
                    &g_mahony,
                    g_imu.accel[0], g_imu.accel[1], g_imu.accel[2],
                    gx, gy, gz, fusion_dt, accel_ok, gyro_ok);
                g_imu_control_valid = (uint8_t)(
                    fuse_status == MAHONY_UPDATE_FULL ||
                    (fuse_status == MAHONY_UPDATE_GYRO_ONLY && g_mahony.control_valid));

                if (!g_imu_control_valid) {
                    /* Attitude no longer trustworthy: drop balance authority now.
                     * The ms-deadline safety block latches FAULT_IMU via freshness
                     * once the failure outlives its debounce window.  Telemetry
                     * slots below still run on the last published attitude. */
                    g_ctrl_tau_l = 0.0f;
                    g_ctrl_tau_r = 0.0f;
                }

                if (g_imu_control_valid) {
                    if (!g_safety_state.is_gyro_calibrated && gyro_ok) {
                        safety_state_gyro_calib_update(gx, gy, gz, HAL_GetTick());
                    }
                    g_body_gyro[0] = gx;
                    g_body_gyro[1] = gy;
                    g_body_gyro[2] = gz;

                    /* Balance-relevant tilt & rate mapped to the physical IMU
                     * mounting.  Pitch rate uses the bias-corrected gyro (offset is
                     * 0 until calibrated). */
                    float body_pitch = ATT_PITCH(&g_mahony);
                    float body_pitch_rate = ATT_PITCH_RATE_SIGN *
                        (g_imu.gyro[ATT_PITCH_RATE_IDX] - g_safety_state.gyro_calib_offset[ATT_PITCH_RATE_IDX]);
                    /* Low-pass filter pitch rate: the raw gyro Y has broadband noise
                     * that K3 amplifies into torque chatter.  A 35 Hz one-pole filter
                     * preserves the balance-relevant dynamics while cutting noise. */
                    g_pitch_rate_filt += PITCH_RATE_FILTER_ALPHA *
                        (body_pitch_rate - g_pitch_rate_filt);
                    body_pitch_rate = g_pitch_rate_filt;

                    /* Publish the latest attitude for the ms-deadline safety block. */
                    g_body_pitch = body_pitch;
                    g_body_pitch_rate = body_pitch_rate;
                }

                /* 3. State telemetry on the 1 kHz tick cadence.  Control and
                 * trace live in the 250 Hz control section above; this block
                 * only publishes fusion results.  Diagnostic telemetry lives
                 * on the wall clock below, so a stopped DRDY freezes state
                 * telemetry but never diagnostics, fault frames, or fault
                 * detection itself. */
                uint8_t slot = last_tick % 4;

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
            }
        }

        /* --- 250 Hz diagnostic telemetry on the wall clock (HAL tick) ---
         * Diag, health, and FAULT frames report fault state, per-device ages,
         * and error counters — none of which depend on the fusion path.  They
         * are therefore driven by the SysTick millisecond, not the DRDY tick
         * and not g_imu.initialized: when the IMU path dies (I2C failure or a
         * stopped data-ready), the 250 Hz control section still latches the
         * fault on the wall clock and these frames still tell the Pi — and
         * the operator — exactly what failed and when.  State telemetry
         * (IMU/joints, above) intentionally stays DRDY-gated: with the fusion
         * path down its values are frozen and worthless, and the age fields
         * here already carry that information. */
        {
            uint32_t diag_now = HAL_GetTick();
            if (diag_now != last_diag_tx_ms && (diag_now & 3U) == 3U) {
                last_diag_tx_ms = diag_now;
                /* Battery sensing is not populated on this hardware.  Zero is
                 * the protocol sentinel for unavailable, not an undervoltage.
                 * The BMI088 covers -40..85 degC; saturate before the uint8
                 * field because a negative float-to-unsigned cast is UB and
                 * would wrap (e.g. -5 degC -> 251) on a cold bench.  With the
                 * sensor path down this reports the last read value, which the
                 * health frame's imu_age_ms dates. */
                uint8_t diag_temp_c =
                    (g_imu.temperature < 0.0f) ? 0U :
                    ((g_imu.temperature > 255.0f) ? 255U
                                                  : (uint8_t)g_imu.temperature);
                pi_link_send_diag(&huart6, 0U, diag_temp_c,
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
            uint8_t ids[4] = {SERVO_LF_ID, SERVO_RF_ID, SERVO_LB_ID, SERVO_RB_ID};
            /* A servo deadline is "served" once a frame was actually queued,
             * the mode has nothing to transmit (INIT), or the request is
             * statically unsolvable (IK rejected its already-clamped inputs —
             * deterministic, so an immediate retry cannot help).  A BUSY servo
             * bus keeps the deadline pending so the write is retried on the
             * next scheduler pass instead of being dropped for a whole 20 ms
             * period.  Five dropped periods would expire LEG_HOLD_MAX_AGE_MS,
             * demote the actuator supervisor, and close the wheel gate for
             * ~100 ms on a balancing robot — with one degraded servo timing
             * out ~10 ms of every 24 ms poll cycle that happened often enough
             * to matter.  The wheel dispatch layer already had this retry
             * shape (next_wheel_tx_ms only advances on success). */
            uint8_t servo_deadline_served = 0U;

#if SERVO_ZERO_CALIBRATION_MODE
            /* Position commands are intentionally suppressed. System setup
             * has already disabled torque; feedback polling below still runs. */
            (void)ids;
            servo_deadline_served = 1U;
#else
            if (g_safety_state.current_mode == STATE_ACTIVE) {
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
                    if (st3215_bus_queue_sync_write(&g_st3215_bus, ids, 4U,
                                                    pos_ticks, speed_ticks, accels) == 0) {
                        g_leg_hold_tx_ms = current_time;
                        servo_deadline_served = 1U;
                    }
                } else {
                    servo_deadline_served = 1U; /* unsolvable request: skip */
                }
            }
            else if (g_safety_state.current_mode == STATE_STAND ||
                     g_safety_state.current_mode == STATE_CLIMB) {
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

                    if (st3215_bus_queue_sync_write(&g_st3215_bus, ids, 4U,
                                                    pos_ticks, speed_ticks, accels) == 0) {
                        g_leg_hold_tx_ms = current_time;
                        servo_deadline_served = 1U;
                    }
                } else {
                    servo_deadline_served = 1U; /* unsolvable request: skip */
                }
            }
            else if (g_safety_state.current_mode == STATE_FAULT) {
                /* Lockdown: disable servo torque once to allow gravity lock.
                 * Sending the disable every 50 Hz cycle floods the full-duplex
                 * bus with echo bytes that desync subsequent read queries, so a
                 * one-shot flag is used instead of repeated transmission.
                 * A busy bus leaves the deadline pending so the (single next)
                 * disable frame is retried on the following pass. */
                if (fault_servo_disable_idx >= 4U) {
                    servo_deadline_served = 1U; /* one-shot complete */
                } else if (st3215_bus_queue_torque(&g_st3215_bus,
                                                   ids[fault_servo_disable_idx],
                                                   0U) == 0) {
                    ++fault_servo_disable_idx;
                    servo_deadline_served = 1U;
                }
            }
            else {
                /* INIT: no leg command on the wire; the deadline is served. */
                servo_deadline_served = 1U;
            }
#endif
            if (servo_deadline_served) {
                servo_deadline_pending = 0U;
            }
        }

        /* Poll every servo round-robin, including offline devices, so a valid
         * frame can restore health after line noise or a temporary disconnect.
         * 6 ms spacing (24 ms full cycle) deliberately mismatches the 50 Hz
         * (20 ms) sync-write period: at 5 ms the cycles phase-locked and the
         * same servo always landed right after the write burst, eating every
         * delayed reply (observed: S1 timeout bursts, S2-S4 clean). */
        current_time = HAL_GetTick();
        if (g_startup_manager.phase >= STARTUP_ACTUATOR_DISCOVERY &&
            current_time - last_servo_query_ms >= 6U &&
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
        /* Highest-priority context: runs even when a UART interrupt storm
         * starves SysTick and the main loop, so stall forensics still work. */
        main_loop_stall_check();
    }
}

/**
 * @brief Snapshots the Pi command state.  Frames are parsed in main-loop
 *        context (pi_transport_poll); the short IRQ mask keeps the copy
 *        atomic against any future ISR-side producer at zero cost.
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
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* DMA2_Stream1_Channel5 for USART6_RX */
    HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);

    /* DMA1_Stream1_Channel4 for USART3_RX */
    HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA1_Stream1_IRQn);
}

static void MX_IWDG_Init(void) {
    hiwdg.Instance = IWDG;
    /* ~1.0 s window: LSI ~32 kHz / 16 = 2 kHz, reload 2000.  The previous
     * 64/4095 setting (~8.2 s) let a hung main loop hold the last wheel
     * torque far too long for a self-balancing robot. */
    hiwdg.Init.Prescaler = IWDG_PRESCALER_16;
    hiwdg.Init.Reload = 2000;
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

    /* DMA1_Stream1_Channel4 circular RX (main loop consumes the ring). */
    hdma_usart3_rx.Instance = DMA1_Stream1;
    hdma_usart3_rx.Init.Channel = DMA_CHANNEL_4;
    hdma_usart3_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart3_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart3_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart3_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart3_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart3_rx.Init.Mode = DMA_CIRCULAR;
    hdma_usart3_rx.Init.Priority = DMA_PRIORITY_HIGH;
    hdma_usart3_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_usart3_rx) != HAL_OK) {
        Error_Handler();
    }
    __HAL_LINKDMA(&huart3, hdmarx, hdma_usart3_rx);
    if (HAL_UART_Receive_DMA(&huart3, g_st3215_rx_buf, ST3215_RX_BUF_SIZE) != HAL_OK) {
        Error_Handler();
    }

    /* USART3 IRQ stays enabled for TX-complete (TX remains interrupt-driven). */
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
    /* Swallow NE/FE before HAL sees them: in DMA reception HAL treats ANY
     * flagged error as blocking and aborts the whole circular RX stream
     * (UART_DMAAbortOnError), so one noisy byte on this half-duplex bus cost
     * a ~1 ms blind window that corrupted every in-flight reply (observed:
     * ~25%% servo poll failures, freshness FAULT bursts).  A framing/noise
     * blip now costs at most one garbage byte; the parser resynchronizes on
     * the 0xFFFF header and the checksum drops a single frame.  Reading SR
     * then DR clears the flags; DR is normally empty here because the DMA
     * drains RXNE immediately, and stealing a byte in the rare race costs
     * exactly one frame as well.  ORE still goes through the HAL abort path
     * (g_uart3_rx_rearm in the main loop restores the stream). */
    uint32_t sr = huart3.Instance->SR;
    if ((sr & (USART_SR_NE | USART_SR_FE)) != 0U) {
        volatile uint32_t drained = huart3.Instance->DR;
        (void)drained;
    }
    HAL_UART_IRQHandler(&huart3);
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ddsm_bus_on_tx_complete_at(&g_ddsm_bus, HAL_GetTick());
    } else if (huart == &huart3) {
        st3215_bus_on_tx_complete(&g_st3215_bus);
    } else if (huart == &huart6) {
        pi_link_on_tx_complete(huart);
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ++g_uart2_rx_cnt;
        ddsm_bus_on_rx_byte(&g_ddsm_bus, HAL_GetTick());
    } else if (huart == &huart3) {
        /* One completed servo-ring lap: the truth source for the producer
         * reconstruction in the main-loop drain (see dma_rx_ring.h). */
        dma_rx_ring_note_lap(&g_st3215_ring);
    } else if (huart == &huart6) {
        pi_transport_note_lap(&g_pi_transport);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart2) {
        ++g_uart2_err_cnt;
        ddsm_bus_on_uart_error(&g_ddsm_bus, huart);
    } else if (huart == &huart3) {
        ++g_uart3_err_cnt;
        st3215_bus_on_uart_error(&g_st3215_bus);
        if ((huart->ErrorCode & (HAL_UART_ERROR_ORE | HAL_UART_ERROR_NE |
                                 HAL_UART_ERROR_FE)) != 0U) {
            g_uart3_rx_rearm = 1U;
        }
    } else if (huart == &huart6) {
        ++g_uart6_err_cnt;
        /* RX-side errors (overrun/noise/framing) must not drop queued TX
         * telemetry: circular DMA reception keeps running and the streaming
         * decoder resynchronizes on its own.  Only genuine TX failures fall
         * through to the TX queue recovery.  Overrun still makes the HAL
         * abort the RX DMA, so schedule a re-arm. */
        if ((huart->ErrorCode & (HAL_UART_ERROR_ORE | HAL_UART_ERROR_NE |
                                 HAL_UART_ERROR_FE)) == 0U) {
            pi_link_on_tx_error(huart);
        } else {
            g_uart6_rx_rearm = 1U;
        }
    }
}

/**
 * @brief DMA2 Stream1 (USART6 RX) global interrupt handler.
 */
void DMA2_Stream1_IRQHandler(void) {
    HAL_DMA_IRQHandler(&hdma_usart6_rx);
}

/**
 * @brief DMA1 Stream1 (USART3 RX) global interrupt handler.
 */
void DMA1_Stream1_IRQHandler(void) {
    HAL_DMA_IRQHandler(&hdma_usart3_rx);
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
