#include "actuator_supervisor.h"
#include "test_support.h"

#include <string.h>

typedef struct {
    int hold_calls;
    int servo_calls;
    int wheel_calls;
    int zero_calls;
    int fail_wheel;
} ActuatorTestCtx;

static int queue_hold(void *opaque) {
    ((ActuatorTestCtx *)opaque)->hold_calls++;
    return ACTUATOR_OP_OK;
}
static int queue_servo(void *opaque, uint8_t enable) {
    (void)enable;
    ((ActuatorTestCtx *)opaque)->servo_calls++;
    return ACTUATOR_OP_OK;
}
static int queue_wheel(void *opaque, uint8_t enable) {
    ActuatorTestCtx *ctx = (ActuatorTestCtx *)opaque;
    ctx->wheel_calls++;
    return (enable && ctx->fail_wheel) ? ACTUATOR_OP_ERROR : ACTUATOR_OP_OK;
}
static int queue_zero(void *opaque) {
    ((ActuatorTestCtx *)opaque)->zero_calls++;
    return ACTUATOR_OP_OK;
}

static ActuatorSupervisorInputs_t healthy_inputs(void) {
    ActuatorSupervisorInputs_t in;
    memset(&in, 0, sizeof(in));
    in.mode = STATE_STAND;
    in.startup_ready = 1U;
    in.actuator_configured = 1U;
    in.link_compatible = 1U;
    in.heartbeat_fresh = 1U;
    in.wheel_authorized = 1U;
    in.wheel_bus_idle = 1U;
    in.servo_bus_idle = 1U;
    in.leg_target_tx_complete = 1U;
    in.leg_feedback_fresh = 1U;
    in.leg_posture_safe = 1U;
    in.leg_target_held = 1U;
    in.servo_enable_verified = 1U;
    in.wheel_enable_verified = 1U;
    return in;
}

void run_actuator_supervisor_tests(void) {
    ActuatorSupervisor_t supervisor;
    ActuatorSupervisorInputs_t in = healthy_inputs();
    ActuatorTestCtx ctx;
    ActuatorSupervisorOps_t ops;
    ActuatorSupervisorOutputs_t out;

    memset(&ctx, 0, sizeof(ctx));
    ops.queue_leg_hold = queue_hold;
    ops.queue_servo_enable = queue_servo;
    ops.queue_wheel_enable = queue_wheel;
    ops.queue_wheel_zero = queue_zero;
    ops.ctx = &ctx;
    actuator_supervisor_init(&supervisor);

    /* A recovery cannot skip the verified leg target/feedback stage. */
    in.leg_target_tx_complete = 0U;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(!out.wheel_output_allowed);
    TEST_TRUE(out.request_hold);
    in.leg_target_tx_complete = 1U;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(!out.wheel_output_allowed);
    TEST_TRUE(ctx.servo_calls > 0);
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.wheel_output_allowed);

    /* A transient fault clears authorization and zeroes before recovery. */
    in.fault_mask = FAULT_SERVO;
    in.mode = STATE_FAULT;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.clear_motion);
    TEST_TRUE(!out.wheel_output_allowed);
    TEST_TRUE(ctx.zero_calls > 0);
    in.fault_mask = FAULT_NONE;
    in.mode = STATE_STAND;
    in.wheel_enable_verified = 1U;
    in.leg_target_held = 0U; /* require a new post-fault feedback window */
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(!out.wheel_output_allowed);
    in.leg_target_held = 1U;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.wheel_output_allowed);

    /* Serious faults and TX failures remain latched despite healthy input. */
    in.fault_mask = FAULT_TILT;
    in.mode = STATE_FAULT;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_EQ_INT(ACTUATOR_SUPERVISOR_LATCHED, out.phase);
    in.fault_mask = FAULT_NONE;
    in.mode = STATE_STAND;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_EQ_INT(ACTUATOR_SUPERVISOR_LATCHED, out.phase);
    TEST_TRUE(!out.wheel_output_allowed);

    actuator_supervisor_init(&supervisor);
    memset(&ctx, 0, sizeof(ctx));
    ctx.fail_wheel = 1;
    in = healthy_inputs();
    in.wheel_enable_verified = 0U;
    /* Step 1: startup recovery re-issues the servo enable first and must not
     * release wheel output in the same step. */
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(ctx.servo_calls > 0);
    TEST_TRUE(!out.wheel_output_allowed);
    /* Step 2: the wheel enable attempt fails on TX and latches. */
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_EQ_INT(ACTUATOR_SUPERVISOR_LATCHED, out.phase);
    TEST_TRUE(!out.wheel_output_allowed);

    /* Every recovery re-issues the servo enable; repeats never shortcut. */
    actuator_supervisor_init(&supervisor);
    memset(&ctx, 0, sizeof(ctx));
    in = healthy_inputs();
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(!out.wheel_output_allowed);      /* reissue step */
    TEST_EQ_INT(1, ctx.servo_calls);
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.wheel_output_allowed);
    TEST_EQ_INT(1, ctx.servo_calls);           /* no re-issue while healthy */
    in.fault_mask = FAULT_SERVO;
    in.mode = STATE_FAULT;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.clear_motion);
    in.fault_mask = FAULT_NONE;
    in.mode = STATE_STAND;
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(!out.wheel_output_allowed);
    TEST_EQ_INT(2, ctx.servo_calls);           /* re-issued after the fault */
    out = actuator_supervisor_step(&supervisor, &in, &ops);
    TEST_TRUE(out.wheel_output_allowed);

    /* ops-less integration (verdict-only, as in the scheduler): no executor
     * is wired, so authorization must come from observing the external
     * sequencer.  A "verified" flag that never drops cannot shortcut the
     * re-issue stage; it must drop (physical enables do not survive a
     * fault/restart) and return on a later step. */
    actuator_supervisor_init(&supervisor);
    in = healthy_inputs();
    out = actuator_supervisor_step(&supervisor, &in, NULL);
    TEST_TRUE(!out.wheel_output_allowed);      /* still verified: no shortcut */
    out = actuator_supervisor_step(&supervisor, &in, NULL);
    TEST_TRUE(!out.wheel_output_allowed);      /* ...and no progress at all */
    in.servo_enable_verified = 0U;             /* sequencer re-enabling */
    out = actuator_supervisor_step(&supervisor, &in, NULL);
    TEST_TRUE(!out.wheel_output_allowed);
    in.servo_enable_verified = 1U;             /* enable completed again */
    out = actuator_supervisor_step(&supervisor, &in, NULL);
    TEST_TRUE(!out.wheel_output_allowed);      /* re-issue clears, READY next */
    out = actuator_supervisor_step(&supervisor, &in, NULL);
    TEST_TRUE(out.wheel_output_allowed);
}
