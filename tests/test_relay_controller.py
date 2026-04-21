import unittest
from unittest.mock import patch

from core.relay_controller import (
    HORN_OFF_COMMAND,
    HORN_ON_COMMAND,
    STROBE_OFF_COMMAND,
    STROBE_ON_COMMAND,
    RelayController,
)


class FakeSerialPort:
    def __init__(self):
        self.is_open = True
        self.writes = []

    def write(self, data):
        self.writes.append(data)

    def close(self):
        self.is_open = False


class RelayControllerTests(unittest.TestCase):
    def test_alarm_on_skips_disabled_outputs(self):
        relay = RelayController("COM1", enable_strobe=False, enable_horn=False)
        relay.relay = FakeSerialPort()

        result = relay.alarm_on()

        self.assertTrue(result)
        self.assertEqual(relay.relay.writes, [])

    def test_strobe_and_horn_commands_are_sent_when_enabled(self):
        relay = RelayController("COM1", enable_strobe=True, enable_horn=True)
        relay.relay = FakeSerialPort()

        relay.strobe_on()
        relay.horn_on()

        self.assertEqual(relay.relay.writes, [STROBE_ON_COMMAND, HORN_ON_COMMAND])

    def test_alarm_off_sends_off_commands(self):
        relay = RelayController("COM1")
        relay.relay = FakeSerialPort()

        relay.alarm_off()

        self.assertEqual(relay.relay.writes, [STROBE_OFF_COMMAND, HORN_OFF_COMMAND])

    @patch("core.relay_controller.time.sleep", return_value=None)
    @patch("core.relay_controller.serial.Serial")
    @patch("core.relay_controller.SERIAL_INSTALLED", True)
    def test_connect_initializes_and_turns_alarm_off(self, serial_ctor, _sleep):
        fake_port = FakeSerialPort()
        serial_ctor.return_value = fake_port
        relay = RelayController("COM7")

        connected = relay.connect()

        self.assertTrue(connected)
        self.assertIs(relay.relay, fake_port)
        self.assertEqual(fake_port.writes, [STROBE_OFF_COMMAND, HORN_OFF_COMMAND])


if __name__ == "__main__":
    unittest.main()
