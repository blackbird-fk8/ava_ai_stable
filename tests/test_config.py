import json
import tempfile
import unittest
from pathlib import Path

from core.config import AppConfig, config_to_dict, load_app_config, save_app_config


class ConfigTests(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        cfg = load_app_config("does-not-exist.json")

        self.assertEqual(cfg, AppConfig())

    def test_partial_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"camera_index": 2, "enable_horn": False}), encoding="utf-8")

            cfg = load_app_config(str(path))

        self.assertEqual(cfg.camera_index, 2)
        self.assertFalse(cfg.enable_horn)
        self.assertEqual(cfg.frame_width, AppConfig().frame_width)

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"camera_index": 3, "mystery_value": 99}), encoding="utf-8")

            cfg = load_app_config(str(path))

        self.assertEqual(cfg.camera_index, 3)
        self.assertFalse(hasattr(cfg, "mystery_value"))

    def test_invalid_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text("{not-json", encoding="utf-8")

            cfg = load_app_config(str(path))

        self.assertEqual(cfg, AppConfig())

    def test_save_and_load_round_trip(self):
        cfg = AppConfig(camera_index=4, relay_port="COM12", weed_spray_duration=1.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_app_config(str(path), cfg)

            loaded = load_app_config(str(path))

        self.assertEqual(loaded, cfg)
        self.assertEqual(config_to_dict(loaded)["relay_port"], "COM12")


if __name__ == "__main__":
    unittest.main()
