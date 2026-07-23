import importlib
import os
import tempfile
import unittest

from config import Config


class SaraSmokeTests(unittest.TestCase):
    def test_config_validate(self):
        Config.validate(force=True)

    def test_intent_engine(self):
        from sara.core.intent.engine import detect_intent

        intent, match = detect_intent("open chrome")
        self.assertEqual(intent, "open_app")
        self.assertTrue(match)

    def test_calc_utils(self):
        from sara.orchestrator.calc_utils import _safe_calc, _parse_duration_to_seconds

        self.assertEqual(_safe_calc("12 * (3 + 4)"), "The answer is 84.")
        self.assertEqual(_parse_duration_to_seconds("5 minutes"), 300)

    def test_preferences_db(self):
        from sara.core.memory import PreferencesDB

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        try:
            db = PreferencesDB(db_path=tmp.name)
            self.assertTrue(db.set_preference("test_key", "test_value"))
            self.assertEqual(db.get_preference("test_key"), "test_value")
            self.assertTrue(db.delete_preference("test_key"))
            db.close()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_reminder_manager_shutdown(self):
        from sara.tools.reminders import ReminderManager

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        try:
            manager = ReminderManager(db_path=tmp.name)
            manager.start()
            manager.shutdown()
            self.assertFalse(manager._thread and manager._thread.is_alive())
            self.assertIsNone(manager._conn)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_audio_module_imports(self):
        importlib.import_module("sara.audio.aec")
        importlib.import_module("sara.audio.stt.engine")

    def test_tool_router_import(self):
        from sara.core.tool_router import TOOL_NAME_TO_INTENT, resolve_tool_call, build_fake_match

        self.assertIn("weather", TOOL_NAME_TO_INTENT)
        resolved = resolve_tool_call("what's the weather in Mumbai", "qwen2.5")
        self.assertEqual(resolved["name"], "weather")
        self.assertTrue(resolved["arguments"]["location"].lower().startswith("mumbai"))

        fake_match = build_fake_match(resolved["name"], resolved["arguments"])
        self.assertTrue(fake_match)
        self.assertEqual(fake_match.group(1), resolved["arguments"]["location"])

    def test_tts_engine_initialization(self):
        from sara.audio.tts import TextToSpeech

        tts = TextToSpeech()
        try:
            tts.speak("Hello from Sara!")
        finally:
            tts.shutdown()

    def test_vision_module_import(self):
        from sara.tools.vision import VisionAssistant

        assistant = VisionAssistant()
        self.assertTrue(hasattr(assistant, "capture_screenshot"))


if __name__ == "__main__":
    unittest.main()
