#ifndef STM32F4XX_HAL_H
#define STM32F4XX_HAL_H

#include <stdint.h>

typedef struct {
    uint32_t unused;
} I2C_HandleTypeDef;

typedef struct {
    uint32_t unused;
    uint32_t ErrorCode;
} UART_HandleTypeDef;

typedef enum {
    HAL_OK = 0,
    HAL_ERROR = 1,
    HAL_BUSY = 2,
    HAL_TIMEOUT = 3
} HAL_StatusTypeDef;

#define I2C_MEMADD_SIZE_8BIT 1U
#define HAL_UART_ERROR_NONE   0x00000000U
#define HAL_UART_ERROR_PE     0x00000001U
#define HAL_UART_ERROR_NE     0x00000002U
#define HAL_UART_ERROR_FE     0x00000004U
#define HAL_UART_ERROR_ORE    0x00000008U
#define HAL_UART_ERROR_DMA    0x00000010U
#define __HAL_UART_CLEAR_OREFLAG(huart) ((void)(huart))
#define __HAL_UART_FLUSH_DRREGISTER(huart) ((void)(huart))
#define __disable_irq() ((void)0)
#define __enable_irq() ((void)0)

uint32_t HAL_GetTick(void);
HAL_StatusTypeDef HAL_I2C_Mem_Write(I2C_HandleTypeDef *hi2c,
                                    uint16_t dev_address,
                                    uint16_t mem_address,
                                    uint16_t mem_address_size,
                                    uint8_t *data,
                                    uint16_t size,
                                    uint32_t timeout);
HAL_StatusTypeDef HAL_I2C_Mem_Read(I2C_HandleTypeDef *hi2c,
                                   uint16_t dev_address,
                                   uint16_t mem_address,
                                   uint16_t mem_address_size,
                                   uint8_t *data,
                                   uint16_t size,
                                   uint32_t timeout);
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *huart,
                                      uint8_t *data,
                                      uint16_t size);
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart,
                                     uint8_t *data,
                                     uint16_t size);
HAL_StatusTypeDef HAL_UART_Abort(UART_HandleTypeDef *huart);
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart,
                                   uint8_t *data,
                                   uint16_t size,
                                   uint32_t timeout);

#endif
