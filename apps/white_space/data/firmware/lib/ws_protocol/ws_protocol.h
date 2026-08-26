#pragma once
// Pure protocol logic shared by the firmware and the test suite.
//
// Everything here must stay free of Arduino/RP2040 headers and of global state: `pio test` does not
// compile firmware.cpp (test_build_src defaults to no), so this header is the seam that lets the
// tests reach real production code instead of a copy of it. PlatformIO's library dependency finder
// picks up lib/ for both `pio run` and `pio test`, so both get the same definitions.

#include <stdint.h>

// Modbus RTU CRC-16 (poly 0xA001, init 0xFFFF), transmitted low byte first.
// Used for the motor drive commands on Serial1.
inline uint16_t crc16_modbus(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (int j = 8; j != 0; j--) {
            if ((crc & 0x0001) != 0) {
                crc >>= 1;
                crc ^= 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}
