import json
import unittest
from pathlib import Path

from src.actions import ActionEngine
from src.examples import INTENT_EXAMPLES


REPO_ROOT = Path(__file__).resolve().parents[1]


class ActionEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ActionEngine()

    def test_every_model_label_has_assistant_support(self):
        with (REPO_ROOT / "configs/label_mapping.json").open(encoding="utf-8") as file:
            labels = set(json.load(file)["label2id"])
        self.assertEqual(labels, self.engine.supported_intents())
        self.assertEqual(labels, set(INTENT_EXAMPLES))

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

    def test_intentional_typo_routing(self):
        self.assertEqual(self.engine.detect_explicit_intent("calclate 12 * 8"), "qa_maths")
        self.assertEqual(self.engine.detect_explicit_intent("whts the wethr in Bengaluru tmrw"), "weather_query")
        self.assertEqual(self.engine.detect_explicit_intent("hows the trffic to airport"), "transport_traffic")
        self.assertEqual(self.engine.detect_explicit_intent("turn the ligths on"), "iot_hue_lighton")

    def test_weather_location_and_time_extraction(self):
        self.assertEqual(self.engine._weather_location("weather in Bengaluru tomorrow"), "Bengaluru")
        self.assertEqual(self.engine._weather_location("naale bengaluru weather eniratte"), "Bengaluru")
        self.assertEqual(self.engine._weather_location("कल बेंगलुरु का मौसं कैसा रहेगा"), "Bengaluru")
        self.assertEqual(self.engine._weather_location("ನಾಳೆ ಬೆಂಗಳೂರಿನ ಹವಮನ ಹೇಗಿದೆ"), "Bengaluru")
        self.assertTrue(self.engine._asks_tomorrow("wethr in Bengaluru tmrw"))
        self.assertTrue(self.engine._asks_tomorrow("naale bengaluru weather eniratte"))
        self.assertTrue(self.engine._asks_tomorrow("कल बेंगलुरु का मौसम"))
        self.assertTrue(self.engine._asks_tomorrow("ನಾಳೆ ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ"))

    def test_weather_returns_an_answer_not_a_search_handoff(self):
        responses = iter([
            {"results": [{"name": "Bengaluru", "latitude": 12.97, "longitude": 77.59}]},
            {
                "current": {
                    "temperature_2m": 25, "apparent_temperature": 26,
                    "relative_humidity_2m": 60, "wind_speed_10m": 8,
                },
                "daily": {
                    "temperature_2m_min": [19, 20],
                    "temperature_2m_max": [27, 28],
                    "precipitation_probability_max": [20, 35],
                },
            },
        ])
        self.engine._json = lambda *_args, **_kwargs: next(responses)
        result = self.engine.execute("weather_query", "whts the wethr in Bengaluru tmrw")
        self.assertEqual(result.status, "completed")
        self.assertIn("Bengaluru tomorrow", result.message)

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
