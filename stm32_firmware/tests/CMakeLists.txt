cmake_minimum_required(VERSION 3.20)
project(kuafu_firmware_host_tests C)

enable_testing()

add_executable(firmware_host_tests
    test_main.c
    test_servo_mapping.c
    test_device_health.c
    test_support.c
    ../Core/Src/crc8.c
    ../Core/Src/kinematics.c
    ../Core/Src/servo_mapping.c
    ../Core/Src/device_health.c
)

target_include_directories(firmware_host_tests PRIVATE
    hal_stubs
    ../Core/Inc
)

if(MSVC)
    target_compile_options(firmware_host_tests PRIVATE /W4 /WX)
else()
    target_compile_options(firmware_host_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME firmware_host_tests COMMAND firmware_host_tests)
