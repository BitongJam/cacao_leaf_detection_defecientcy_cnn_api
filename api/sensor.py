import serial
import time

# Initialize serial communication
ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

# MODBUS query to read 3 registers starting from 0x1E
npk_query = bytearray([0x01, 0x03, 0x00, 0x1e, 0x00, 0x03, 0x65, 0xcd])

def get_npk_values():
    ser.write(npk_query)  # Send MODBUS query
    time.sleep(0.15)  #  Wait for the response

    # Read the response (expected length is 11 bytes)
    response = ser.read(11)
    print("Raw Response:", response.hex())  # Debugging: Print raw hex response


    # Validate response format
    if len(response) == 11 and response[0] == 0x01 and response[1] == 0x03 and response[2] == 0x06:
        # Parse the NPK values
        nitrogen = (response[3] << 8) | response[4]  # Combine high and low bytes
        phosphorus = (response[5] << 8) | response[6]
        potassium = (response[7] << 8) | response[8]
        return nitrogen, phosphorus, potassium
    else:
        print("Invalid or incomplete response")
        return None, None, None

def main():
    while True:
        nitrogen, phosphorus, potassium = get_npk_values()
        if nitrogen is not None:
            print(f"Nitrogen (N): {nitrogen} mg/kg")
            print(f"Phosphorus (P): {phosphorus} mg/kg")
            print(f"Potassium (K): {potassium} mg/kg")
            print("-" * 30)
        else:
            print("Failed to read NPK values")
        time.sleep(5)  # Wait before sending the next query
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")