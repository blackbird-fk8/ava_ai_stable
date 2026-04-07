"""
Serial relay controller for SCARE AI alarm hardware.

Controls strobe lights and horn via serial commands to a relay module.
"""

import logging
import time
import serial

logger = logging.getLogger(__name__)


class RelayController:
    """Controls alarm hardware (strobe and horn) via serial relay module."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 1.0) -> None:
        """
        Initialize relay controller.

        Args:
            port: Serial port name (e.g., 'COM5')
            baud: Baud rate (default: 9600)
            timeout: Serial timeout in seconds (default: 1.0)
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.relay = None
        logger.info(f"RelayController initialized for {port} at {baud} baud")

    def connect(self) -> bool:
        """
        Connect to the relay module.

        Returns:
            True if connection successful, False otherwise
        """
        if self.relay is not None and self.relay.is_open:
            logger.debug("Already connected to relay")
            return True

        try:
            self.relay = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2)  # Wait for relay to initialize
            self.alarm_off()  # Ensure alarm is off on startup
            logger.info(f"Connected to relay on {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect to relay on {self.port}: {e}")
            self.relay = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to relay: {e}")
            return False

    def close(self) -> None:
        """Safely disconnect from relay and turn off alarm."""
        if self.relay is None:
            return

        try:
            self.alarm_off()
            logger.debug("Alarm turned off before closing")
        except Exception as e:
            logger.warning(f"Error turning off alarm during close: {e}")

        try:
            if self.relay.is_open:
                self.relay.close()
            logger.info("Relay connection closed")
        except Exception as e:
            logger.warning(f"Error closing relay connection: {e}")
        finally:
            self.relay = None

    def _write(self, data: bytes, description: str = "") -> bool:
        """
        Write data to relay.

        Args:
            data: Bytes to write
            description: Optional description of the command

        Returns:
            True if write successful, False otherwise
        """
        if self.relay is None or not self.relay.is_open:
            logger.error(f"Relay not connected. Cannot write {description}")
            return False

        try:
            self.relay.write(data)
            logger.debug(f"Relay command sent: {description}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to write to relay ({description}): {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error writing to relay ({description}): {e}")
            return False

    def strobe_on(self) -> bool:
        """Turn on strobe light. Returns True if successful."""
        return self._write(b'\xA0\x01\x01\xA2', "STROBE_ON")

    def strobe_off(self) -> bool:
        """Turn off strobe light. Returns True if successful."""
        return self._write(b'\xA0\x01\x00\xA1', "STROBE_OFF")

    def horn_on(self) -> bool:
        """Turn on horn. Returns True if successful."""
        return self._write(b'\xA0\x02\x01\xA3', "HORN_ON")

    def horn_off(self) -> bool:
        """Turn off horn. Returns True if successful."""
        return self._write(b'\xA0\x02\x00\xA2', "HORN_OFF")

    def alarm_on(self) -> bool:
        """Turn on full alarm (strobe + horn). Returns True if both successful."""
        strobe_ok = self.strobe_on()
        horn_ok = self.horn_on()
        return strobe_ok and horn_ok

    def alarm_off(self) -> bool:
        """Turn off full alarm (strobe + horn). Always attempts both, returns True if at least one succeeds."""
        strobe_ok = self.strobe_off()
        horn_ok = self.horn_off()
        return strobe_ok or horn_ok