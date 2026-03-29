import serial
import time

# -----------------------------
# SERIAL SETUP
# -----------------------------
PORT = '/dev/serial0'   # Raspberry Pi UART
BAUDRATE = 9600

ser = serial.Serial(PORT, BAUDRATE, timeout=1)
time.sleep(2)  # wait for sensor

# -----------------------------
# MODBUS COMMANDS (NPK)
# -----------------------------
NITROGEN_CMD   = bytes([0x01, 0x03, 0x00, 0x1E, 0x00, 0x01, 0xE4, 0x0C])
PHOSPHORUS_CMD = bytes([0x01, 0x03, 0x00, 0x1F, 0x00, 0x01, 0xB5, 0xCC])
POTASSIUM_CMD  = bytes([0x01, 0x03, 0x00, 0x20, 0x00, 0x01, 0x85, 0xC0])

# -----------------------------
# FUNCTION: READ DATA
# -----------------------------
def read_npk(command, name):
    ser.reset_input_buffer()
    ser.write(command)
    time.sleep(0.1)

    response = ser.read(7)

    if len(response) == 7:
        print(f"{name} RAW:", response.hex())
        return response[4]
    else:
        print(f"{name} ERROR: No response")
        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        nitrogen = read_npk(NITROGEN_CMD, "Nitrogen")
        time.sleep(0.3)

        phosphorus = read_npk(PHOSPHORUS_CMD, "Phosphorus")
        time.sleep(0.3)

        potassium = read_npk(POTASSIUM_CMD, "Potassium")
        time.sleep(0.3)

        print("\n===== NPK VALUES =====")

        print("Nitrogen  :", nitrogen if nitrogen is not None else "Error", "mg/kg")
        print("Phosphorus:", phosphorus if phosphorus is not None else "Error", "mg/kg")
        print("Potassium :", potassium if potassium is not None else "Error", "mg/kg")

        print("======================\n")

        time.sleep(2)

except KeyboardInterrupt:
    print("Program stopped")
    ser.close()
