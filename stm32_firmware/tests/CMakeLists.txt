cmake_minimum_required(VERSION 3.20)
project(kuafu_firmware_host_tests C)

enable_testing()

add_executable(firmware_host_tests
    test_main.c
    test_servo_mapping.c
    test_device_health.c
    test_safety_state.c
    test_startup_manager.c
    test_bmi088.c
    test_ddsm315.c
    test_st3215.c
    test_pi_link.c
    test_firmware_runtime.c
    test_control_section.c
    test_lqr.c
    test_balance_trace.c
    test_actuator_supervisor.c
    test_mahony.c
    test_dma_rx_ring.c
    test_pi_transport.c
    test_support.c
    ../Core/Src/crc8.c
    ../Core/Src/kinematics.c
    ../Core/Src/servo_mapping.c
    ../Core/Src/device_health.c
    ../Core/Src/safety_state.c
    ../Core/Src/startup_manager.c
    ../Core/Src/bmi088.c
    ../Core/Src/ddsm315.c
    ../Core/Src/st3215.c
    ../Core/Src/pi_link.c
    ../Core/Src/pi_transport.c
    ../Core/Src/firmware_runtime.c
    ../Core/Src/control_section.c
    ../Core/Src/lqr_controller.c
    ../Core/Src/balance_trace.c
    ../Core/Src/actuator_supervisor.c
    ../Core/Src/mahony.c
    ../Core/Src/dma_rx_ring.c
)

target_include_directories(firmware_host_tests PRIVATE
    hal_stubs
    ../Core/Inc
)

if(MSVC)
    # /utf-8: sources contain UTF-8 comments (em-dashes, CJK); without this
    # MSVC on a non-UTF-8 system code page raises C4819, which /WX promotes
    # to a build failure.
    target_compile_options(firmware_host_tests PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(firmware_host_tests PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(firmware_host_tests PRIVATE m)
endif()

add_test(NAME firmware_host_tests COMMAND firmware_host_tests)
