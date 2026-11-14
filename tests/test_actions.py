import json
import unittest
from pathlib import Path

from src.actions import ActionEngine


REPO_ROOT = Path(__file__).resolve().parents[1]


class ActionEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ActionEngine()

    def test_every_model_label_has_assistant_support(self):
        with (REPO_ROOT / "configs/label_mapping.json").open(encoding="utf-8") as file:
            labels = set(json.load(file)["label2id"])
        self.assertEqual(labels, self.engine.supported_intents())

    def test_every_known_intent_has_a_real_handler(self):
        self.engine._json = lambda *_args, **_kwargs: {}
        for intent in sorted(self.engine.supported_intents()):
            with self.subTest(intent=intent):
                result = self.engine.execute(intent, "test request")
                self.assertNotEqual(result.title, "Request understood")

    def test_calculator(self):
        result = self.engine.execute("qa_maths", "calculate (245 * 18) / 7")
        self.assertEqual(result.status, "completed")
        self.assertIn("630", result.message)

    def test_list_state_is_session_local_and_reversible(self):
        added = self.engine.execute("lists_createoradd", "add milk to my shopping list")
        queried = self.engine.execute("lists_query", "show my shopping list")
        removed = self.engine.execute("lists_remove", "remove milk from my shopping list")
        self.assertEqual(added.status, "completed")
        self.assertIn("milk", queried.message.lower())
        self.assertEqual(removed.status, "completed")
        self.assertEqual(self.engine.lists["default"], [])

    def test_alarm_state(self):
        set_result = self.engine.execute("alarm_set", "set an alarm for 7 AM tomorrow")
        query_result = self.engine.execute("alarm_query", "show my alarms")
        self.assertEqual(set_result.status, "completed")
        self.assertIn("7 AM tomorrow", query_result.message)

    def test_virtual_device_does_not_claim_physical_execution(self):
        result = self.engine.execute("iot_hue_lighton", "turn the bedroom lights on")
        self.assertEqual(result.status, "completed")
        self.assertIn("virtual demo state", result.message.lower())
        self.assertEqual(self.engine.devices["lights"], "on")

    def test_explicit_command_routing(self):
        self.assertEqual(self.engine.detect_explicit_intent("calculate 12 * 8"), "qa_maths")
        self.assertEqual(self.engine.detect_explicit_intent("weather in Mysuru tomorrow"), "weather_query")
        self.assertEqual(self.engine.detect_explicit_intent("turn the bedroom light off"), "iot_hue_lightoff")

    def test_low_confidence_clarification_is_actionable(self):
        result = self.engine.clarification([
            {"intent": "weather_query", "probability": 0.35},
            {"intent": "datetime_query", "probability": 0.22},
            {"intent": "general_quirky", "probability": 0.12},
        ])
        self.assertEqual(result.status, "needs_input")
        self.assertIn("weather query", result.message)


if __name__ == "__main__":
    unittest.main()
