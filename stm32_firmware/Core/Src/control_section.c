#include "control_section.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pin_config.h"

/* Validity bits of the trace sample (balance_trace.h keeps fields raw). */
#define TRACE_VALID_IMU      (1u << 0)
#define TRACE_VALID_WHEEL_L  (1u << 1)
#define TRACE_VALID_WHEEL_R  (1u << 2)
#define TRACE_VALID_TARGET   (1u << 3)
#define TRACE_VALID_SENT     (1u << 4)
#define TRACE_VALID_FEEDBACK (1u << 5)
#define TRACE_SENT_MAX_AGE_MS 10U

void control_section_init(ControlSection_t *section, uint32_t now_ms) {
    if (section == NULL) return;
    memset(section, 0, sizeof(*section));
    lqr_init(&section->lqr);
    actuator_supervisor_init(&section->supervisor);
    section->last_lqr_ms = now_ms;
}

ControlSectionOutputs_t control_section_step(ControlSection_t *section,
                                             const ControlSectionInputs_t *in) {
    ControlSectionOutputs_t out;
    SafetyInputs_t safety_inputs;
    SafetyDecision_t safety_decision;
    ActuatorSupervisorInputs_t actuator_inputs;
    ActuatorSupervisorOutputs_t actuator_outputs;
    uint8_t estimation_ok;
    uint32_t mode_now;
    uint32_t trace_now;
    BalanceTraceInput_t trace_in;

    memset(&out, 0, sizeof(out));
    if (section == NULL || in == NULL) return out;

    /* 1. Safety state machine: freshness/tilt/thermal faults and mode
     * transitions, at the wall-clock deadline (independent of DRDY). */
    memset(&safety_inputs, 0, sizeof(safety_inputs));
    safety_inputs.now_ms = in->now_ms;
    safety_inputs.pitch_rad = in->pitch_rad;
    safety_inputs.pitch_rate_rads = in->pitch_rate_rad_s;
    safety_inputs.max_temp_c = in->max_temp_c;
    safety_inputs.startup_ready = in->startup_ready;
    safety_inputs.imu_fresh = in->imu_fresh;
    safety_inputs.wheel_l_fresh = in->wheel_l_fresh;
    safety_inputs.wheel_r_fresh = in->wheel_r_fresh;
    safety_inputs.servos_fresh = in->servos_fresh;
    safety_inputs.link_compatible = in->link_compatible;
    safety_inputs.heartbeat_fresh = in->heartbeat_fresh;
    safety_inputs.action_fresh = in->action_fresh;
    safety_inputs.requested_mode = in->requested_mode;
    safety_decision = safety_state_update(&safety_inputs);
    if (safety_decision.enter_hold) {
        pi_link_enter_hold();
    } else if (safety_decision.clear_action) {
        pi_link_clear_action();
    }

    if (g_safety_state.current_mode == STATE_FAULT &&
        section->prev_safety_mode != (uint32_t)STATE_FAULT) {
        out.fault_entry_edge = 1U;
    }
    section->prev_safety_mode = (uint32_t)g_safety_state.current_mode;

    /* 2. Actuator-supervisor verdict over the physical enable lifecycle. */
    memset(&actuator_inputs, 0, sizeof(actuator_inputs));
    actuator_inputs.now_ms = in->now_ms;
    actuator_inputs.mode = g_safety_state.current_mode;
    actuator_inputs.fault_mask = g_safety_state.fault_mask;
    actuator_inputs.startup_ready = in->startup_ready;
    actuator_inputs.actuator_configured = in->actuator_configured;
    actuator_inputs.link_compatible = in->link_compatible;
    actuator_inputs.heartbeat_fresh = in->heartbeat_fresh;
    actuator_inputs.wheel_authorized = in->wheel_authorized;
    actuator_inputs.wheel_bus_idle = in->wheel_bus_idle;
    actuator_inputs.servo_bus_idle = in->servo_bus_idle;
    /* "Transmitted" means a real leg-position sync-write went out after the
     * enables, not just torque-enable frames. */
    actuator_inputs.leg_target_tx_complete = in->leg_hold_tx_recent;
    actuator_inputs.leg_feedback_fresh = in->leg_feedback_fresh;
    actuator_inputs.leg_posture_safe = in->leg_posture_safe;
    actuator_inputs.leg_target_held =
        (uint8_t)(in->leg_hold_tx_recent && in->leg_feedback_fresh);
    actuator_inputs.servo_enable_verified = in->servo_enable_verified;
    actuator_inputs.wheel_enable_verified = in->wheel_enable_verified;
    actuator_outputs = actuator_supervisor_step(&section->supervisor,
                                                &actuator_inputs, NULL);

    /* 3. Wheel output gate: runtime intent AND supervisor verdict AND
     * completed current-loop re-assertion.  Current-loop re-assertion must
     * complete before the first torque frame: a motor that reverted to its
     * default velocity loop across a disable/enable cycle would read the
     * identical torque frame as a speed command. */
    out.wheel_output_gate = (uint8_t)(
        in->runtime.wheel_intent_allowed &&
        actuator_outputs.wheel_output_allowed &&
        in->wheel_mode_complete);

    /* Estimation authority needs BOTH the filter's own validity verdict and
     * the freshness age.  g_imu_control_valid is refreshed only inside the
     * DRDY-driven fusion block: if the data-ready interrupt itself dies, that
     * flag freezes at its last value while the attitude is frozen with it.
     * Requiring imu_fresh here (already a safety input, 20 ms pair age) cuts
     * the stale-authority window from the freshness-fault debounce (~52 ms)
     * to the freshness bound itself. */
    estimation_ok = (uint8_t)(in->imu_control_valid && in->imu_fresh);

    /* 4. LQR/LQI computation under the gate. */
    if (out.wheel_output_gate && estimation_ok) {
        /* Velocity/yaw references are an ACTIVE-mode contract
         * (firmware_runtime.velocity_command_active): STAND is a position
         * hold, CLIMB is height-only, and the heartbeat fields are not
         * trusted to be zeroed by the sender. */
        float vx_cmd = in->runtime.velocity_command_active
                           ? in->heartbeat.target_velocity : 0.0f;
        float wz_cmd = in->runtime.velocity_command_active
                           ? in->heartbeat.target_yaw_rate : 0.0f;
        if (!in->link_compatible || !in->heartbeat_fresh) {
            /* Re-anchor the hold reference ONCE on link loss, not every
             * cycle: resetting x_ref to x_est continuously pins x_error at
             * zero and silently disables the K0 position loop and the LQI
             * integrator, so the robot cruises away instead of holding its
             * position. */
            if (!section->hold_ref_anchored) {
                lqr_reset(&section->lqr, section->lqr.x_est, in->yaw_rad);
                section->hold_ref_anchored = 1U;
            }
        } else {
            section->hold_ref_anchored = 0U;
        }

        /* Integrate with the MEASURED wall time since the previous LQR
         * pass: a skipped deadline must not be integrated as a nominal
         * 4 ms step (and vice versa). */
        {
            float elapsed_s = section->have_last_lqr_ms
                ? (float)(in->now_ms - section->last_lqr_ms) * 0.001f
                : BASE_DT;
            section->last_lqr_ms = in->now_ms;
            section->have_last_lqr_ms = 1U;

            (void)lqr_update_elapsed_dt(&section->lqr,
                                        elapsed_s,
                                        in->pitch_rad,
                                        in->pitch_rate_rad_s,
                                        in->wheel_vel_l_rads,
                                        in->wheel_vel_r_rads,
                                        in->yaw_rad,
                                        in->yaw_rate_rads,
                                        vx_cmd,
                                        wz_cmd,
                                        in->runtime.residual_allowed
                                            ? in->action.delta_torque_common : 0.0f,
                                        in->runtime.residual_allowed
                                            ? in->action.delta_torque_yaw : 0.0f,
                                        &out.torque_left_nm,
                                        &out.torque_right_nm);

            /* Pitch-gated torque fade: full authority inside the
             * recoverable band, linearly decaying to zero as the body tips
             * past PITCH_FADE_END so a fallen robot does not spin its
             * wheels at full torque. */
            {
                float ap = fabsf(in->pitch_rad);
                float scale;
                if (ap <= PITCH_FADE_START_RAD) {
                    scale = 1.0f;
                } else if (ap >= PITCH_FADE_END_RAD) {
                    scale = 0.0f;
                } else {
                    scale = (PITCH_FADE_END_RAD - ap) /
                            (PITCH_FADE_END_RAD - PITCH_FADE_START_RAD);
                }
                out.torque_left_nm *= scale;
                out.torque_right_nm *= scale;
            }
        }
    }

    /* 5. Balance-trace sample for this deadline.  Recording runs on the
     * wall clock, not the DRDY tick, so an IMU death (the fault class the
     * trace most needs to capture) still fills the bounded post-fault
     * window.  All torques are body-frame Nm; sent-vs-target distinguishes
     * what the controller asked for from what the bus actually accepted. */
    trace_now = in->now_ms;
    mode_now = (uint32_t)g_safety_state.current_mode;
    if (mode_now == (uint32_t)STATE_FAULT &&
        section->prev_trace_mode != (uint32_t)STATE_FAULT) {
        balance_trace_note_event(BALANCE_TRACE_EVENT_FAULT, trace_now);
    }
    section->prev_trace_mode = mode_now;

    memset(&trace_in, 0, sizeof(trace_in));
    trace_in.timestamp_ms = trace_now;
    trace_in.mode = mode_now;
    trace_in.fault_mask = (uint32_t)g_safety_state.fault_mask;
    trace_in.startup_state = in->startup_phase;
    trace_in.actuator_state = (uint32_t)section->supervisor.phase;
    trace_in.imu_age_ms = in->imu_age_ms;
    trace_in.wheel_left_age_ms = in->wheel_left_age_ms;
    trace_in.wheel_right_age_ms = in->wheel_right_age_ms;
    trace_in.pitch_rad = in->pitch_rad;
    trace_in.pitch_rate_rad_s = in->pitch_rate_rad_s;
    trace_in.wheel_left_rad_s = in->wheel_vel_l_rads;
    trace_in.wheel_right_rad_s = in->wheel_vel_r_rads;
    trace_in.target_torque_left_nm = out.torque_left_nm;
    trace_in.target_torque_right_nm = out.torque_right_nm;
    trace_in.sent_torque_left_nm = in->sent_torque_left_nm;
    trace_in.sent_torque_right_nm = in->sent_torque_right_nm;
    trace_in.feedback_torque_left_nm = in->feedback_torque_left_nm;
    trace_in.feedback_torque_right_nm = in->feedback_torque_right_nm;
    trace_in.validity =
        (estimation_ok ? TRACE_VALID_IMU : 0U) |
        (in->wheel_l_fresh ? TRACE_VALID_WHEEL_L : 0U) |
        (in->wheel_r_fresh ? TRACE_VALID_WHEEL_R : 0U) |
        ((out.wheel_output_gate && estimation_ok)
            ? TRACE_VALID_TARGET : 0U) |
        ((in->wheel_sent_age_ms <= TRACE_SENT_MAX_AGE_MS)
            ? TRACE_VALID_SENT : 0U) |
        ((in->wheel_l_fresh && in->wheel_r_fresh)
            ? TRACE_VALID_FEEDBACK : 0U);
    balance_trace_record(&trace_in);

    return out;
}
