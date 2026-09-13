#ifndef DDSM315_H
#define DDSM315_H

#include "stm32f4xx_hal.h"
#include "device_health.h"
#include <stdint.h>

#define DDSM_MODE_CURRENT        1U
#define DDSM_MODE_SPEED          2U
#define DDSM_MODE_POSITION       3U
#define DDSM_MODE_DISABLE        9U
#define DDSM_FRAME_SIZE         10U
/* High-load replies can exceed the nominal ~2 ms round trip; 8 ms caused
 * retry chains that also starved the other wheel. 12 ms keeps the worst-case
 * per-wheel service rate well above the 50 ms freshness budget. */
#define DDSM_TRANSACTION_TIMEOUT_MS 12U
#define DDSM_QUEUE_DEPTH 4U

typedef enum {
    DDSM_TX_STATUS_IDLE = 0,
    DDSM_TX_STATUS_QUEUED = 1,
    DDSM_TX_STATUS_ACTIVE = 2,
    DDSM_TX_STATUS_TX_COMPLETE = 3,
    DDSM_TX_STATUS_FEEDBACK_VALID = 4,
    DDSM_TX_STATUS_TIMEOUT = 5,
    DDSM_TX_STATUS_ERROR = 6
} DDSM_TxStatus_t;

typedef struct {
    uint8_t id;
    uint8_t mode;
    uint8_t expect_reply;
    uint8_t status;
    uint32_t queued_ms;
    uint32_t tx_complete_ms;
    uint32_t finished_ms;
} DDSM_TxSnapshot_t;

typedef struct {
    uint8_t id;
    uint8_t mode;
    float torque;
    float velocity_rads;
    float position_rad;
    uint8_t error_code;
    DeviceHealth_t health;
} DDSM_State_t;

typedef enum {
    DDSM_BUS_IDLE = 0,
    DDSM_BUS_TX = 1,
    DDSM_BUS_RX = 2
} DDSM_BusPhase_t;

/* Concurrency contract: submit/step/queue_* run in main-loop context;
 * on_tx_complete/on_rx_byte/on_uart_error run in the USART IRQ.  The queue
 * cursors (pending_head/tail/count) are the only fields both contexts
 * read-modify-write, and every such mutation is enclosed in a short IRQ mask
 * (see ddsm315.c).  Phase transitions out of IDLE happen exclusively in
 * start_pending, which is reached either from the ISR (transaction finished,
 * next frame pending) or from the main loop when the bus is IDLE — states
 * that cannot overlap — so phase itself needs no masking.  Diagnostic status
 * words are single writes from either context; a torn read costs at most one
 * stale diagnostic sample. */

typedef struct {
    UART_HandleTypeDef *huart;
    DDSM_State_t *target;
    uint8_t tx[DDSM_FRAME_SIZE];
    uint8_t rx[DDSM_FRAME_SIZE];
    uint32_t deadline_ms;
    DDSM_BusPhase_t phase;
    uint8_t rx_byte;
    uint8_t rx_len;
    /* Fire-and-forget commands (mode switch, enable) elicit no motor reply.
     * When 0, a transaction timeout is treated as a clean completion and does
     * NOT penalise the target's health -- otherwise the repeated 12 ms no-reply
     * windows stack into a false stale burst and latch a spurious wheel FAULT. */
    uint8_t expect_reply;
    uint8_t queued_mode;
    uint8_t mode_feedback;
    DDSM_TxStatus_t status;
    uint32_t tx_complete_ms;
    uint32_t error_count;
    uint32_t queue_overflow_count;
    DDSM_TxSnapshot_t last_tx;
    struct {
        DDSM_State_t *target;
        uint8_t packet[DDSM_FRAME_SIZE];
        uint8_t expect_reply;
        uint8_t mode;
        uint32_t queued_ms;
    } pending[DDSM_QUEUE_DEPTH];
    uint8_t pending_head;
    uint8_t pending_tail;
    uint8_t pending_count;
} DDSM_Bus_t;

void ddsm_build_torque(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float torque_nm);
void ddsm_build_speed(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float rpm);
void ddsm_build_enable(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t enable);
void ddsm_build_mode(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t mode);
void ddsm_build_query(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id);
void ddsm_build_set_id(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id);
int ddsm_parse_feedback(const uint8_t packet[DDSM_FRAME_SIZE], DDSM_State_t *state);

void ddsm_bus_init(DDSM_Bus_t *bus, UART_HandleTypeDef *huart);
uint8_t ddsm_bus_is_idle(const DDSM_Bus_t *bus);
int ddsm_bus_submit(DDSM_Bus_t *bus,
                    DDSM_State_t *target,
                    const uint8_t packet[DDSM_FRAME_SIZE],
                    uint32_t now_ms);
int ddsm_bus_queue_torque(DDSM_Bus_t *bus, DDSM_State_t *target,
                          float torque_nm, uint32_t now_ms);
int ddsm_bus_queue_enable(DDSM_Bus_t *bus, DDSM_State_t *target,
                          uint8_t enable, uint32_t now_ms);
int ddsm_bus_queue_mode(DDSM_Bus_t *bus, DDSM_State_t *target,
                        uint8_t mode, uint32_t now_ms);
int ddsm_bus_queue_query(DDSM_Bus_t *bus, DDSM_State_t *target,
                         uint32_t now_ms);
void ddsm_bus_step(DDSM_Bus_t *bus, uint32_t now_ms);
void ddsm_bus_on_tx_complete(DDSM_Bus_t *bus);
void ddsm_bus_on_tx_complete_at(DDSM_Bus_t *bus, uint32_t now_ms);
void ddsm_bus_on_rx_byte(DDSM_Bus_t *bus, uint32_t now_ms);
void ddsm_bus_on_uart_error(DDSM_Bus_t *bus, UART_HandleTypeDef *huart);
DDSM_TxStatus_t ddsm_bus_status(const DDSM_Bus_t *bus);
uint8_t ddsm_bus_mode_feedback(const DDSM_Bus_t *bus);
uint32_t ddsm_bus_queue_depth(const DDSM_Bus_t *bus);
const DDSM_TxSnapshot_t *ddsm_bus_get_last_tx(const DDSM_Bus_t *bus);

#endif
