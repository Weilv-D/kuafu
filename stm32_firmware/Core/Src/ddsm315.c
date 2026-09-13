#include "ddsm315.h"
#include "crc8.h"
#include "pin_config.h"

#include <stddef.h>
#include <string.h>

#define RPM_TO_RADS (2.0f * 3.14159265f / 60.0f)
#define POS_TO_RAD  (2.0f * 3.14159265f / 32768.0f)
#define DDSM_OFFLINE_AFTER 3U

/* Nestable critical section.  These helpers execute in BOTH main and
 * interrupt context, so interrupts are restored to the saved PRIMASK rather
 * than unconditionally enabled — a bare __enable_irq() would unmask
 * interrupts in the middle of an ISR. */
static inline uint32_t bus_lock_irqs(void) {
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    return primask;
}

static inline void bus_unlock_irqs(uint32_t primask) {
    __set_PRIMASK(primask);
}

static uint8_t deadline_reached(uint32_t now_ms, uint32_t deadline_ms) {
    return (uint8_t)((int32_t)(now_ms - deadline_ms) >= 0);
}

static void finish_failure(DDSM_Bus_t *bus, DeviceFailure_t failure) {
    if (bus == NULL) return;
    if (bus->target != NULL) {
        /* Health counters are read-modify-write and this runs in both main
         * (deadline timeout) and ISR (parse failure) context against a
         * target whose ISR may simultaneously mark it valid. */
        uint32_t primask = bus_lock_irqs();
        device_health_mark_failure(&bus->target->health, failure, DDSM_OFFLINE_AFTER);
        bus_unlock_irqs(primask);
    }
    bus->error_count++;
    bus->status = failure == DEVICE_FAILURE_TIMEOUT ? DDSM_TX_STATUS_TIMEOUT
                                                     : DDSM_TX_STATUS_ERROR;
    bus->last_tx.status = bus->status;
    bus->target = NULL;
    bus->phase = DDSM_BUS_IDLE;
}

static int start_pending(DDSM_Bus_t *bus, uint32_t now_ms) {
    uint8_t i;
    if (bus == NULL || bus->phase != DDSM_BUS_IDLE || bus->pending_count == 0U ||
        bus->huart == NULL) return 0;
    /* The queue cursors are shared with the UART ISR (a completed transaction
     * pops the next pending frame from interrupt context).  Popping under a
     * short IRQ mask keeps the read-modify-write of pending_head/count atomic
     * against concurrent enqueues from the main loop. */
    {
        uint32_t primask = bus_lock_irqs();
        i = bus->pending_head;
        bus->pending_head = (uint8_t)((bus->pending_head + 1U) % DDSM_QUEUE_DEPTH);
        bus->pending_count--;
        bus_unlock_irqs(primask);
    }
    memcpy(bus->tx, bus->pending[i].packet, DDSM_FRAME_SIZE);
    bus->target = bus->pending[i].target;
    bus->expect_reply = bus->pending[i].expect_reply;
    bus->queued_mode = bus->pending[i].mode;
    bus->deadline_ms = now_ms + DDSM_TRANSACTION_TIMEOUT_MS;
    bus->rx_len = 0U;
    bus->phase = DDSM_BUS_TX;
    bus->status = DDSM_TX_STATUS_ACTIVE;
    bus->last_tx.id = bus->target != NULL ? bus->target->id : 0U;
    bus->last_tx.expect_reply = bus->expect_reply;
    bus->last_tx.queued_ms = bus->pending[i].queued_ms;
    bus->last_tx.status = DDSM_TX_STATUS_ACTIVE;
    if (HAL_UART_Transmit_IT(bus->huart, bus->tx, DDSM_FRAME_SIZE) != HAL_OK) {
        finish_failure(bus, DEVICE_FAILURE_PROTOCOL);
        return -1;
    }
    return 0;
}

static void finish_idle(DDSM_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL) return;
    bus->target = NULL;
    bus->phase = DDSM_BUS_IDLE;
    bus->last_tx.finished_ms = now_ms;
    if (bus->pending_count != 0U) (void)start_pending(bus, now_ms);
}

static void arm_rx(DDSM_Bus_t *bus) {
    if (bus != NULL && bus->huart != NULL) {
        (void)HAL_UART_Receive_IT(bus->huart, &bus->rx_byte, 1U);
    }
}

void ddsm_build_torque(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float torque_nm) {
    int16_t raw;
    if (torque_nm > DDSM_MAX_TORQUE_NM) torque_nm = DDSM_MAX_TORQUE_NM;
    if (torque_nm < -DDSM_MAX_TORQUE_NM) torque_nm = -DDSM_MAX_TORQUE_NM;
    raw = (int16_t)(torque_nm * DDSM_TORQUE_TO_RAW);
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x64U;
    packet[2] = (uint8_t)((uint16_t)raw >> 8);
    packet[3] = (uint8_t)raw;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_speed(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float rpm) {
    int16_t raw = (int16_t)(rpm * 10.0f);
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x64U;
    packet[2] = (uint8_t)((uint16_t)raw >> 8);
    packet[3] = (uint8_t)raw;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_enable(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t enable) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0xA0U;
    packet[2] = enable ? 0x08U : 0x09U;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_mode(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t mode) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0xA0U;
    /* DDSM315 UART protocol (Protocol 3, documented "no feedback, no CRC"):
     *     A0 frame: byte[2] is the sub-command, mode VALUE lives in byte[9].
     *     Reference frame:  01 A0 01 00 00 00 00 00 00 00   (byte[9]=mode)
     * The official DDSM example firmware (ddsm_ctrl.cpp) uses the same layout
     * (value in byte[9], no CRC) and it is accepted by the hardware.
     *
     * NOTE: the DDSM315 wiki is internally contradictory. Its prose says the
     * mode value goes in byte[9] with no CRC, but its own serial-capture
     * example shows `mode` in byte[2] with a trailing CRC (the DDSM210-style
     * framing). Field evidence on this robot (mode byte persistently reads
     * 0x02 while the wheels behave like a current loop) shows the wiki prose
     * is the correct interpretation for this firmware: byte[9]=mode, NO CRC.
     * A prior commit put the value in byte[2] with a CRC; that frame was
     * ignored by the motor, leaving it in its default velocity loop — the
     * real cause of the earlier "sprinting", NOT a frame-format bug. */
    packet[9] = mode;
}

void ddsm_build_query(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x74U;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_set_id(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = 0xAAU;
    packet[1] = 0x55U;
    packet[2] = 0x53U;
    packet[3] = id;
    packet[9] = crc8_calculate(packet, 9U);
}

int ddsm_parse_feedback(const uint8_t packet[DDSM_FRAME_SIZE], DDSM_State_t *state) {
    int16_t raw_current;
    int16_t raw_speed;
    uint16_t raw_position;
    if (packet == NULL || state == NULL) return -3;
    if (crc8_calculate(packet, 9U) != packet[9]) return -1;
    if (packet[0] != state->id) return -2;
    state->mode = packet[1];
    raw_current = (int16_t)(((uint16_t)packet[2] << 8) | packet[3]);
    raw_speed = (int16_t)(((uint16_t)packet[4] << 8) | packet[5]);
    raw_position = (uint16_t)(((uint16_t)packet[6] << 8) | packet[7]);
    state->torque = (float)raw_current * DDSM_RAW_TO_TORQUE;
    /* DDSM315 speed field is in 0.1 RPM (matches ddsm_build_speed's rpm*10
     * encoding). Forgetting the 0.1 factor made every feedback velocity 10x
     * too large, which destabilized the LQR hold -- a wheel at rest reported
     * ~33 rad/s once disturbed, locking the loop into a false steady-state
     * spin at OMEGA_NOLOAD. */
    state->velocity_rads = (float)raw_speed * 0.1f * RPM_TO_RADS;
    state->position_rad = (float)raw_position * POS_TO_RAD;
    state->error_code = packet[8];
    return 0;
}

void ddsm_bus_init(DDSM_Bus_t *bus, UART_HandleTypeDef *huart) {
    if (bus == NULL) return;
    memset(bus, 0, sizeof(*bus));
    bus->huart = huart;
    bus->phase = DDSM_BUS_IDLE;
    bus->status = DDSM_TX_STATUS_IDLE;
    arm_rx(bus);
}

uint8_t ddsm_bus_is_idle(const DDSM_Bus_t *bus) {
    return (uint8_t)(bus != NULL && bus->phase == DDSM_BUS_IDLE);
}

int ddsm_bus_submit(DDSM_Bus_t *bus,
                    DDSM_State_t *target,
                    const uint8_t packet[DDSM_FRAME_SIZE],
                    uint32_t now_ms) {
    uint8_t slot;
    uint8_t expect_reply;
    uint8_t mode;
    if (bus == NULL || target == NULL || packet == NULL || bus->huart == NULL) return -1;
    expect_reply = packet[1] == 0xA0U ? 0U : 1U;
    mode = (packet[1] == 0xA0U) ? packet[9] : 0U;
    if (bus->phase != DDSM_BUS_IDLE || bus->pending_count != 0U) {
        /* Enqueue under the same IRQ mask discipline as start_pending: the
         * completion ISR pops this queue from interrupt context, so the
         * full/advance/publish sequence must be atomic. */
        uint32_t primask = bus_lock_irqs();
        if (bus->pending_count >= DDSM_QUEUE_DEPTH) {
            bus_unlock_irqs(primask);
            bus->queue_overflow_count++;
            return -2;
        }
        slot = bus->pending_tail;
        memcpy(bus->pending[slot].packet, packet, DDSM_FRAME_SIZE);
        bus->pending[slot].target = target;
        bus->pending[slot].expect_reply = expect_reply;
        bus->pending[slot].mode = mode;
        bus->pending[slot].queued_ms = now_ms;
        bus->pending_tail = (uint8_t)((bus->pending_tail + 1U) % DDSM_QUEUE_DEPTH);
        bus->pending_count++;
        bus_unlock_irqs(primask);
        bus->status = DDSM_TX_STATUS_QUEUED;
        return 0;
    }
    memcpy(bus->tx, packet, DDSM_FRAME_SIZE);
    bus->target = target;
    bus->queued_mode = mode;
    bus->expect_reply = expect_reply;
    bus->deadline_ms = now_ms + DDSM_TRANSACTION_TIMEOUT_MS;
    bus->phase = DDSM_BUS_TX;
    bus->rx_len = 0U;
    bus->status = DDSM_TX_STATUS_ACTIVE;
    bus->last_tx.id = target->id;
    bus->last_tx.expect_reply = expect_reply;
    bus->last_tx.queued_ms = now_ms;
    bus->last_tx.status = DDSM_TX_STATUS_ACTIVE;
    if (HAL_UART_Transmit_IT(bus->huart, bus->tx, DDSM_FRAME_SIZE) != HAL_OK) {
        finish_failure(bus, DEVICE_FAILURE_PROTOCOL);
        return -3;
    }
    return 0;
}

int ddsm_bus_queue_torque(DDSM_Bus_t *bus, DDSM_State_t *target,
                          float torque_nm, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_torque(packet, target->id, torque_nm);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_enable(DDSM_Bus_t *bus, DDSM_State_t *target,
                          uint8_t enable, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_enable(packet, target->id, enable);
    /* The reply expectation is derived inside submit from the frame itself
     * (0xA0 control frames never elicit a reply), so the bus-level flag for
     * the ACTIVE transaction is never second-guessed here. */
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_mode(DDSM_Bus_t *bus, DDSM_State_t *target,
                        uint8_t mode, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_mode(packet, target->id, mode);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_query(DDSM_Bus_t *bus, DDSM_State_t *target,
                         uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_query(packet, target->id);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

void ddsm_bus_step(DDSM_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL) return;
    if (bus->phase == DDSM_BUS_IDLE) {
        (void)start_pending(bus, now_ms);
        return;
    }
    if (deadline_reached(now_ms, bus->deadline_ms)) {
        if (bus->expect_reply) {
            finish_failure(bus, DEVICE_FAILURE_TIMEOUT);
        } else {
            /* No-reply mode/enable is complete only as TX completion plus
             * quiet expiry; it is never reported as feedback/ACK. */
            bus->status = DDSM_TX_STATUS_TX_COMPLETE;
            bus->last_tx.status = DDSM_TX_STATUS_TX_COMPLETE;
            finish_idle(bus, now_ms);
        }
        bus->rx_len = 0U;
    }
}

void ddsm_bus_on_tx_complete_at(DDSM_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL || bus->phase != DDSM_BUS_TX) return;
    bus->tx_complete_ms = now_ms;
    bus->last_tx.tx_complete_ms = now_ms;
    bus->last_tx.status = DDSM_TX_STATUS_TX_COMPLETE;
    bus->status = DDSM_TX_STATUS_TX_COMPLETE;
    if (bus->expect_reply) bus->phase = DDSM_BUS_RX;
    else finish_idle(bus, now_ms);
}

void ddsm_bus_on_tx_complete(DDSM_Bus_t *bus) {
    ddsm_bus_on_tx_complete_at(bus, 0U);
}

void ddsm_bus_on_rx_byte(DDSM_Bus_t *bus, uint32_t now_ms) {
    int result;
    if (bus == NULL) return;
    if (bus->rx_len < DDSM_FRAME_SIZE) {
        bus->rx[bus->rx_len++] = bus->rx_byte;
    }
    if (bus->rx_len == DDSM_FRAME_SIZE) {
        if (memcmp(bus->rx, bus->tx, DDSM_FRAME_SIZE) == 0) {
            bus->rx_len = 0U;
        } else if ((bus->phase == DDSM_BUS_TX || bus->phase == DDSM_BUS_RX) &&
                   bus->target != NULL && bus->rx[0] == bus->target->id &&
                   crc8_calculate(bus->rx, 9U) == bus->rx[9]) {
            result = ddsm_parse_feedback(bus->rx, bus->target);
            if (result == 0) {
                {
                    uint32_t primask = bus_lock_irqs();
                    device_health_mark_valid(&bus->target->health, now_ms);
                    bus_unlock_irqs(primask);
                }
                bus->mode_feedback = bus->target->mode;
                bus->status = DDSM_TX_STATUS_FEEDBACK_VALID;
                bus->last_tx.status = DDSM_TX_STATUS_FEEDBACK_VALID;
                bus->last_tx.finished_ms = now_ms;
                finish_idle(bus, now_ms);
                bus->rx_len = 0U;
            } else {
                finish_failure(bus, result == -1 ? DEVICE_FAILURE_CHECKSUM
                                                 : DEVICE_FAILURE_PROTOCOL);
                bus->rx_len = 0U;
            }
        } else if ((bus->phase == DDSM_BUS_TX || bus->phase == DDSM_BUS_RX) &&
                   bus->target != NULL && bus->rx[0] == bus->target->id) {
            /* An auto-direction adapter can leave a partial echo directly in
             * front of the motor reply. A target-ID byte at the start of one
             * invalid window is therefore not a transaction boundary. Record
             * the bad candidate and keep sliding until a complete frame or
             * the bounded transaction deadline is reached. */
            {
                uint32_t primask = bus_lock_irqs();
                device_health_mark_failure(&bus->target->health,
                                           DEVICE_FAILURE_CHECKSUM,
                                           DDSM_OFFLINE_AFTER);
                bus_unlock_irqs(primask);
            }
            memmove(bus->rx, &bus->rx[1], DDSM_FRAME_SIZE - 1U);
            bus->rx_len = DDSM_FRAME_SIZE - 1U;
        } else {
            memmove(bus->rx, &bus->rx[1], DDSM_FRAME_SIZE - 1U);
            bus->rx_len = DDSM_FRAME_SIZE - 1U;
        }
    }
    arm_rx(bus);
}

void ddsm_bus_on_uart_error(DDSM_Bus_t *bus, UART_HandleTypeDef *huart) {
    if (bus == NULL || bus->huart == NULL || huart != bus->huart) {
        if (bus != NULL) {
            bus->rx_len = 0U;
            arm_rx(bus);
        }
        return;
    }
    if (bus->phase != DDSM_BUS_IDLE) {
        if (bus->target != NULL) {
            uint32_t primask = bus_lock_irqs();
            device_health_mark_uart_error(&bus->target->health, (uint32_t)huart->ErrorCode);
            bus_unlock_irqs(primask);
        }
    }
    bus->rx_len = 0U;
    bus->status = DDSM_TX_STATUS_ERROR;
    bus->last_tx.status = DDSM_TX_STATUS_ERROR;
    bus->error_count++;
    arm_rx(bus);
}

DDSM_TxStatus_t ddsm_bus_status(const DDSM_Bus_t *bus) {
    return bus == NULL ? DDSM_TX_STATUS_IDLE : bus->status;
}

uint8_t ddsm_bus_mode_feedback(const DDSM_Bus_t *bus) {
    return bus == NULL ? 0U : bus->mode_feedback;
}

uint32_t ddsm_bus_queue_depth(const DDSM_Bus_t *bus) {
    return bus == NULL ? 0U : bus->pending_count;
}

const DDSM_TxSnapshot_t *ddsm_bus_get_last_tx(const DDSM_Bus_t *bus) {
    return bus == NULL ? NULL : &bus->last_tx;
}
