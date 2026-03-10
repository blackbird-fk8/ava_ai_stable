import time
import serial

PORT = "COM5"   # change this to your relay's COM port
BAUD = 9600

# LCUS-2 / CH340 relay commands
RELAY1_ON  = bytes.fromhex("A0 01 01 A2")
RELAY1_OFF = bytes.fromhex("A0 01 00 A1")
RELAY2_ON  = bytes.fromhex("A0 02 01 A3")
RELAY2_OFF = bytes.fromhex("A0 02 00 A2")

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Turning relay 2 ON")
ser.write(RELAY2_ON)
time.sleep(5)

print("Turning relay 2 OFF")
ser.write(RELAY2_OFF)

ser.close()
print("Done")

relay.close()