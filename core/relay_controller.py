import time
import serial


class RelayController:
    def __init__(self, port: str, baud: int = 9600, timeout: float = 1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.relay = None

    def connect(self):
        if self.relay is None or not self.relay.is_open:
            self.relay = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2)
            self.alarm_off()

    def close(self):
        if self.relay is not None:
            try:
                self.alarm_off()
            except Exception:
                pass
            try:
                self.relay.close()
            except Exception:
                pass
            self.relay = None

    def _write(self, data: bytes):
        if self.relay is None or not self.relay.is_open:
            raise RuntimeError("Relay is not connected.")
        self.relay.write(data)

    def strobe_on(self):
        self._write(b'\xA0\x01\x01\xA2')

    def strobe_off(self):
        self._write(b'\xA0\x01\x00\xA1')

    def horn_on(self):
        self._write(b'\xA0\x02\x01\xA3')

    def horn_off(self):
        self._write(b'\xA0\x02\x00\xA2')

    def alarm_on(self):
        self.strobe_on()
        self.horn_on()

    def alarm_off(self):
        try:
            self.strobe_off()
        except Exception:
            pass
        try:
            self.horn_off()
        except Exception:
            pass