#ifndef CONTROL_SECTION_H
#define CONTROL_SECTION_H

#include <stdint.h>

#include "actuator_supervisor.h"
#include "balance_trace.h"
#include "firmware_runtime.h"
#include "lqr_controller.h"
#include "pi_link.h"
#include "safety_state.h"

/* The 250 Hz control section, extracted from the scheduler so the exact
 * composition that authorises wheel torque is host-testable:
 *
 *   safety state machine -> actuator-supervisor verdict -> wheel output
 *   gate -> LQR/LQI computation -> balance-trace sample
 *
 * One call consumes one wall-clock deadline.  The gate verdict combines the
 * runtime intent, the supervisor verdict, and completed current-loop
 * re-assertion, and is returned to the caller, which holds it until the next
 * deadline for the bus dispatch layer.  Everything the section needs is
 * passed in as sampled inputs; the shared robot state (g_safety_state, the
 * Pi command globals, the balance trace ring) is touched through the same
 * module APIs the firmware uses. */

typedef struct {
    uint32_t now_ms;

    /* Runtime verdicts for this deadline (firmware_runtime_step output). */
    FirmwareRuntimeOutputs_t runtime;

    /* Estimation: latest published attitude and body-frame rates. */
    uint8_t imu_control_valid;
    float pitch_rad;
    float pitch_rate_rad_s;
    float wheel_vel_l_rads;
    float wheel_vel_r_rads;
    float yaw_rad;
    float yaw_rate_rads;

    /* Safety state machine inputs. */
    float max_temp_c;
    uint8_t imu_fresh;
    uint8_t wheel_l_fresh;
    uint8_t wheel_r_fresh;
    uint8_t servos_fresh;
    uint8_t requested_mode;
    uint8_t startup_ready;

    /* Actuator-supervisor inputs (physical lifecycle bookkeeping). */
    uint8_t actuator_configured;
    uint8_t wheel_authorized;
    uint8_t wheel_bus_idle;
    uint8_t servo_bus_idle;
    uint8_t link_compatible;
    uint8_t heartbeat_fresh;
    uint8_t action_fresh;
    uint8_t leg_hold_tx_recent;
    uint8_t leg_feedback_fresh;
    uint8_t leg_posture_safe;
    uint8_t servo_enable_verified;
    uint8_t wheel_enable_verified;
    /* Current-loop re-assertion completed on both wheels. */
    uint8_t wheel_mode_complete;

    /* Sampled Pi commands (caller snapshots atomically). */
    Pi_Command_Heartbeat_t heartbeat;
    Pi_Command_Action_t action;

    /* Trace provenance the scheduler owns. */
    uint32_t startup_phase;
    uint16_t imu_age_ms;
    uint16_t wheel_left_age_ms;
    uint16_t wheel_right_age_ms;
    uint16_t wheel_sent_age_ms;   /* age of the last accepted torque frame */
    float feedback_torque_left_nm;
    float feedback_torque_right_nm;
    float sent_torque_left_nm;
    float sent_torque_right_nm;
} ControlSectionInputs_t;

typedef struct {
    /* Final wheel-torque authorization for this control period. */
    uint8_t wheel_output_gate;
    /* Body-frame wheel torque commands; zero whenever the gate is closed. */
    float torque_left_nm;
    float torque_right_nm;
    /* Pulses for exactly one deadline when the safety machine ENTERS FAULT;
     * the caller responds by resetting its physical enable bookkeeping. */
    uint8_t fault_entry_edge;
} ControlSectionOutputs_t;

typedef struct {
    uint8_t hold_ref_anchored;
    uint32_t prev_safety_mode;
    uint32_t prev_trace_mode;
    uint32_t last_lqr_ms;
    uint8_t have_last_lqr_ms;
    LQRController_t lqr;
    ActuatorSupervisor_t supervisor;
} ControlSection_t;

void control_section_init(ControlSection_t *section, uint32_t now_ms);

ControlSectionOutputs_t control_section_step(ControlSection_t *section,
                                             const ControlSectionInputs_t *inputs);

#endif
