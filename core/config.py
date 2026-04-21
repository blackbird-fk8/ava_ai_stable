"""Shared application configuration for SCARE AI."""

from dataclasses import asdict, dataclass, fields
import json
import logging
from typing import Any, Dict


@dataclass
class AppConfig:
    active_mode: str = "AVA Alert"
    camera_index: int = 0
    frame_width: int = 320
    frame_height: int = 240
    face_match_threshold: float = 0.35
    animal_classifier_confidence: float = 0.60
    warning_duration: int = 10
    alarm_duration: int = 10
    known_cooldown: int = 3
    post_alarm_cooldown: int = 5
    frame_skip: int = 3
    person_confirm_frames: int = 2
    animal_confirm_frames: int = 2
    enable_strobe: bool = True
    enable_horn: bool = True
    enable_event_photos: bool = True
    relay_port: str = "COM5"
    relay_baud: int = 9600
    weed_conf_threshold: float = 0.15
    weed_frame_skip: int = 3
    weed_spray_cooldown: float = 3.0
    weed_spray_duration: float = 1.0
    weed_zone_x_min: float = 0.30
    weed_zone_x_max: float = 0.70
    weed_zone_y_min: float = 0.30
    weed_zone_y_max: float = 0.70
    food_conf_threshold: float = 0.55
    food_frame_skip: int = 3
    food_simulation_interval: float = 5.0
    food_infer_width: int = 224
    food_infer_height: int = 224


_VALID_CONFIG_KEYS = {field.name for field in fields(AppConfig)}


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    """Convert an AppConfig instance into a JSON-serializable dictionary."""
    return asdict(config)


def load_app_config(path: str, logger: logging.Logger | None = None) -> AppConfig:
    """Load config from disk, merging valid keys onto defaults."""
    cfg = AppConfig()
    if not path:
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        if logger is not None:
            logger.warning("Config file not found at %s. Using defaults.", path)
        return cfg
    except json.JSONDecodeError as e:
        if logger is not None:
            logger.warning("Invalid JSON in config file %s: %s. Using defaults.", path, e)
        return cfg
    except OSError as e:
        if logger is not None:
            logger.warning("Failed to read config file %s: %s. Using defaults.", path, e)
        return cfg

    if not isinstance(data, dict):
        if logger is not None:
            logger.warning("Config file %s did not contain a JSON object. Using defaults.", path)
        return cfg

    valid_data = {key: value for key, value in data.items() if key in _VALID_CONFIG_KEYS}
    if logger is not None and len(valid_data) != len(data):
        unknown_keys = sorted(set(data) - _VALID_CONFIG_KEYS)
        logger.debug("Ignoring unknown config keys from %s: %s", path, ", ".join(unknown_keys))

    return AppConfig(**{**config_to_dict(cfg), **valid_data})


def save_app_config(path: str, config: AppConfig) -> None:
    """Persist config to disk as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2)
