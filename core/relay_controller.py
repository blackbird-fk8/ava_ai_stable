"""Serial relay controller for SCARE AI alarm hardware."""

import logging
import time
from typing import Optional

import serial

logger = logging.getLogger(__name__)

STROBE_ON_COMMAND = b"\xA0\x01\x01\xA2"
STROBE_OFF_COMMAND = b"\xA0\x01\x00\xA1"
HORN_ON_COMMAND = b"\xA0\x02\x01\xA3"
HORN_OFF_COMMAND = b"\xA0\x02\x00\xA2"


class RelayController:
    """Controls alarm hardware (strobe and horn) via serial relay module."""

    def __init__(
        self,
        port: str,
        baud: int = 9600,
        timeout: float = 1.0,
        enable_strobe: bool = True,
        enable_horn: bool = True,
    ) -> None:
        """
        Initialize relay controller.

        Args:
            port: Serial port name (e.g., 'COM5')
            baud: Baud rate (default: 9600)
            timeout: Serial timeout in seconds (default: 1.0)
            enable_strobe: Whether strobe commands should activate output 1
            enable_horn: Whether horn commands should activate output 2
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.enable_strobe = enable_strobe
        self.enable_horn = enable_horn
        self.relay: Optional[serial.Serial] = None
        logger.info(
            "RelayController initialized for %s at %s baud (strobe=%s, horn=%s)",
            port,
            baud,
            enable_strobe,
            enable_horn,
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
            self.relay = None
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

    @property
    def is_connected(self) -> bool:
        """Return True when a serial connection is open."""
        return self.relay is not None and self.relay.is_open

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
        if not self.enable_strobe:
            logger.debug("Skipping STROBE_ON because strobe output is disabled")
            return True
        return self._write(STROBE_ON_COMMAND, "STROBE_ON")

    def strobe_off(self) -> bool:
        """Turn off strobe light. Returns True if successful."""
        return self._write(STROBE_OFF_COMMAND, "STROBE_OFF")

    def horn_on(self) -> bool:
        """Turn on horn. Returns True if successful."""
        if not self.enable_horn:
            logger.debug("Skipping HORN_ON because horn output is disabled")
            return True
        return self._write(HORN_ON_COMMAND, "HORN_ON")

    def horn_off(self) -> bool:
        """Turn off horn. Returns True if successful."""
        return self._write(HORN_OFF_COMMAND, "HORN_OFF")

    def alarm_on(self) -> bool:
        """Turn on full alarm (strobe + horn). Returns True if both successful."""
        if not self.enable_strobe and not self.enable_horn:
            logger.debug("Skipping ALARM_ON because all outputs are disabled")
            return True

        strobe_ok = self.strobe_on()
        horn_ok = self.horn_on()
        return strobe_ok and horn_ok

    def alarm_off(self) -> bool:
        """Turn off full alarm (strobe + horn). Always attempts both, returns True if at least one succeeds."""
        strobe_ok = self.strobe_off()
        horn_ok = self.horn_off()
        return strobe_ok or horn_ok
