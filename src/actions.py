"""Bounded action engine for the IntentShield research demonstration.

The classifier decides the intent. This module performs only reversible or
read-only actions. External services are optional and use public, key-free APIs.
"""

from __future__ import annotations

import ast
import html
import json
import math
import operator
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass
class ActionResult:
    status: str
    title: str
    message: str
    link: str | None = None
    link_label: str | None = None


class ActionEngine:
    """Small, transparent assistant layer for every trained intent.

    State-changing demo actions stay inside this Python session.  Requests that
    require a real account, purchase, post, or physical device are returned as
    drafts or explicit hand-offs; they are never reported as completed.
    """

    KNOWN_INTENTS = {
        "alarm_query", "alarm_set", "audio_volume_down", "audio_volume_mute",
        "audio_volume_other", "calendar_set", "cooking_query", "cooking_recipe",
        "datetime_convert", "datetime_query", "email_querycontact", "general_greet",
        "general_joke", "general_quirky", "iot_cleaning", "iot_coffee",
        "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
        "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
        "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness",
        "music_likeness", "music_query", "music_settings", "news_query",
        "play_audiobook", "play_game", "play_podcasts", "play_radio",
        "qa_currency", "qa_definition", "qa_factoid", "qa_maths",
        "recommendation_events", "recommendation_locations", "recommendation_movies",
        "social_post", "social_query", "takeaway_order", "takeaway_query",
        "transport_query", "transport_ticket", "transport_traffic", "weather_query",
    }

    def __init__(self):
        self.lists: dict[str, list[str]] = {"default": []}
        self.alarms: list[str] = []
        self.calendar: list[str] = []
        self.audio = {"volume": 50, "muted": False}
        self.devices = {
            "lights": "off", "brightness": 60, "colour": "warm white",
            "plug": "off", "cleaner": "idle", "coffee": "idle",
        }
        self.music_preferences = {"liked": [], "disliked": []}

    @classmethod
    def supported_intents(cls) -> set[str]:
        return set(cls.KNOWN_INTENTS)

    @staticmethod
    def detect_explicit_intent(text: str) -> str | None:
        """High-precision command cues used only as an assistant fallback.

        This does not alter the frozen research prediction or its reported
        confidence.  It prevents obvious commands such as "calculate 2+2"
        from becoming useless when the confidence firewall pauses the model.
        """
        lowered = text.lower().strip()
        rules = [
            (r"\b(calculate|solve)\b|^[\d\s()+*/%.^\-]+$", "qa_maths"),
            (r"\b(convert|exchange)\b.*\b[A-Z]{3}\b.*\b[A-Z]{3}\b", "qa_currency"),
            (r"\b(weather|temperature|forecast|rain)\b", "weather_query"),
            (r"\b(set|create|add)\b.*\balarm\b|\bwake me\b", "alarm_set"),
            (r"\b(show|list|what).*(alarm|alarms)\b", "alarm_query"),
            (r"\b(add|put)\b.*\b(list|shopping)\b", "lists_createoradd"),
            (r"\b(show|read|what).*(my list|shopping list)\b", "lists_query"),
            (r"\b(remove|delete)\b.*\b(list|shopping)\b", "lists_remove"),
            (r"\b(recipe|how (?:do|can) i cook|how to cook)\b", "cooking_recipe"),
            (r"\b(define|meaning of)\b", "qa_definition"),
            (r"\b(latest|headlines?)\b.*\b(news|ai|technology|sports)\b", "news_query"),
            (r"\b(traffic|congestion)\b", "transport_traffic"),
            (r"\b(time|date)\b.*\b(now|today|current)\b", "datetime_query"),
            (r"\b(turn|switch)\b.*\blight.*\bon\b", "iot_hue_lighton"),
            (r"\b(turn|switch)\b.*\blight.*\boff\b", "iot_hue_lightoff"),
            (r"\b(mute|silence)\b", "audio_volume_mute"),
            (r"\b(volume)\b.*\b(down|lower|decrease)\b", "audio_volume_down"),
        ]
        for pattern, intent in rules:
            if re.search(pattern, text if "[A-Z]" in pattern else lowered, re.I):
                return intent
        return None

    @staticmethod
    def clarification(predictions: list[dict]) -> ActionResult:
        names = [item["intent"].replace("_", " ") for item in predictions[:3]]
        options = ", ".join(names)
        return ActionResult(
            "needs_input", "I need a little more detail",
            f"I am not confident enough to act. Did you mean: {options}? Please add the item, place, time, or topic.",
        )

    @staticmethod
    def _json(url: str, timeout: int = 8):
        request = urllib.request.Request(url, headers={"User-Agent": "IntentShield/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    @staticmethod
    def _query_link(base: str, query: str):
        return base + urllib.parse.quote_plus(query.strip())

    def execute(self, intent: str, text: str) -> ActionResult:
        handlers = {
            "alarm_query": self._alarm_query,
            "alarm_set": self._alarm_set,
            "audio_volume_down": self._audio_down,
            "audio_volume_mute": self._audio_mute,
            "audio_volume_other": self._audio_other,
            "calendar_set": self._calendar_set,
            "cooking_query": self._cooking_query,
            "cooking_recipe": self._cooking_recipe,
            "qa_maths": self._math,
            "datetime_query": self._datetime,
            "datetime_convert": self._datetime_convert,
            "weather_query": self._weather,
            "qa_currency": self._currency,
            "qa_definition": self._knowledge,
            "qa_factoid": self._knowledge,
            "lists_createoradd": self._list_add,
            "lists_query": self._list_query,
            "lists_remove": self._list_remove,
            "news_query": self._news,
            "music_query": self._music,
            "music_likeness": self._music_like,
            "music_dislikeness": self._music_dislike,
            "music_settings": self._music_settings,
            "play_radio": self._music,
            "play_podcasts": self._music,
            "play_audiobook": self._music,
            "play_game": self._game,
            "recommendation_movies": self._movies,
            "recommendation_locations": self._places,
            "recommendation_events": self._events,
            "transport_traffic": self._traffic,
            "transport_query": self._traffic,
            "transport_ticket": self._travel,
            "takeaway_query": self._food,
            "takeaway_order": self._food,
            "social_post": self._social,
            "social_query": self._social,
            "email_querycontact": self._email,
            "general_greet": self._greet,
            "general_joke": self._joke,
            "general_quirky": self._quirky,
            "iot_cleaning": self._iot_cleaning,
            "iot_coffee": self._iot_coffee,
            "iot_hue_lightchange": self._iot_lightchange,
            "iot_hue_lightdim": self._iot_lightdim,
            "iot_hue_lightoff": self._iot_lightoff,
            "iot_hue_lighton": self._iot_lighton,
            "iot_hue_lightup": self._iot_lightup,
            "iot_wemo_off": self._iot_plugoff,
            "iot_wemo_on": self._iot_plugon,
        }
        if intent in handlers:
            return handlers[intent](text)
        return ActionResult(
            "understood",
            "Request understood",
            f"IntentShield routed this to '{intent.replace('_', ' ')}'. A production connector is required to complete the action.",
        )

    @staticmethod
    def _clean_after_prefix(text: str, pattern: str) -> str:
        return re.sub(pattern, "", text.strip(), flags=re.I).strip(" .?!")

    def _alarm_set(self, text):
        detail = self._clean_after_prefix(text, r"^(please\s+)?(set|create|add)\s+(an?\s+)?alarm\s*(for|at|to)?\s*")
        if not detail:
            return ActionResult("needs_input", "When should I set it?", "Example: set an alarm for 7 AM tomorrow")
        self.alarms.append(detail)
        return ActionResult("completed", "Alarm saved for this demo", f"Alarm: {detail}. It remains active only while this session is open.")

    def _alarm_query(self, _text):
        if not self.alarms:
            return ActionResult("completed", "No active demo alarms", "Set one by saying: set an alarm for 7 AM tomorrow.")
        return ActionResult("completed", "Your demo alarms", "\n".join(f"{i + 1}. {item}" for i, item in enumerate(self.alarms)))

    def _audio_down(self, _text):
        self.audio["volume"] = max(0, int(self.audio["volume"]) - 10)
        self.audio["muted"] = False
        return ActionResult("completed", "Demo volume lowered", f"Volume is now {self.audio['volume']}%. No physical speaker is connected.")

    def _audio_mute(self, _text):
        self.audio["muted"] = True
        return ActionResult("completed", "Demo audio muted", "The virtual audio state is muted. No physical speaker is connected.")

    def _audio_other(self, text):
        match = re.search(r"(\d{1,3})\s*%?", text)
        if match:
            self.audio["volume"] = min(100, max(0, int(match.group(1))))
        elif re.search(r"\b(up|increase|raise|louder)\b", text, re.I):
            self.audio["volume"] = min(100, int(self.audio["volume"]) + 10)
        self.audio["muted"] = False
        return ActionResult("completed", "Demo audio updated", f"Volume is {self.audio['volume']}%. No physical speaker is connected.")

    def _calendar_set(self, text):
        detail = self._clean_after_prefix(text, r"^(please\s+)?(add|create|schedule|set)\s+")
        if not detail:
            return ActionResult("needs_input", "What should I schedule?", "Include the event and time, for example: schedule project review tomorrow at 3 PM.")
        self.calendar.append(detail)
        return ActionResult("completed", "Event saved for this demo", f"{detail}. It is not synced to an external calendar.")

    def _cooking_query(self, text):
        query = self._clean_after_prefix(text, r"^(find|show|suggest|what can i cook|cooking)\s+") or text
        link = self._query_link("https://www.themealdb.com/browse/search/?s=", query)
        return ActionResult("handoff", "Meal ideas", f"Find meal ideas for: {query}.", link, "Browse recipes")

    def _cooking_recipe(self, text):
        query = re.sub(r"\b(recipe|how (?:do|can) i cook|how to (?:make|cook))\b", " ", text, flags=re.I).strip(" .?!")
        if not query:
            return ActionResult("needs_input", "Which dish?", "Example: give me a recipe for vegetable pasta")
        try:
            data = self._json("https://www.themealdb.com/api/json/v1/1/search.php?s=" + urllib.parse.quote(query))
            meal = (data.get("meals") or [])[0]
            instructions = re.sub(r"\s+", " ", meal.get("strInstructions", "")).strip()
            return ActionResult("completed", meal.get("strMeal", query), instructions[:900], meal.get("strSource") or meal.get("strYoutube"), "Open full recipe")
        except Exception:
            return ActionResult("handoff", "Recipe search", f"Open recipe results for {query}.", self._query_link("https://www.google.com/search?q=recipe+", query), "Find recipe")

    def _math(self, text):
        expression = text.lower().replace("calculate", "").replace("what is", "")
        expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
        expression = re.sub(r"[^0-9+\-*/().% ]", "", expression).strip()
        if not expression:
            return ActionResult("needs_input", "Calculator", "Include an expression, for example: calculate (245 * 18) / 7")
        allowed = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        def evaluate(node):
            if isinstance(node, ast.Expression): return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                left, right = evaluate(node.left), evaluate(node.right)
                if abs(left) > 1e12 or abs(right) > 1e12: raise ValueError("Numbers are too large")
                return allowed[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](evaluate(node.operand))
            raise ValueError("Unsupported expression")
        try:
            value = evaluate(ast.parse(expression, mode="eval"))
            if not math.isfinite(float(value)): raise ValueError("Non-finite result")
            return ActionResult("completed", "Calculation complete", f"{expression} = {value:g}")
        except Exception as exc:
            return ActionResult("error", "Could not calculate", str(exc))

    def _datetime(self, _text):
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return ActionResult("completed", "Current date and time", now.strftime("%A, %d %B %Y · %I:%M %p IST"))

    def _datetime_convert(self, text):
        zones = {
            "india": ("Asia/Kolkata", "IST"), "delhi": ("Asia/Kolkata", "IST"),
            "london": ("Europe/London", "London"), "tokyo": ("Asia/Tokyo", "Tokyo"),
            "dubai": ("Asia/Dubai", "Dubai"), "singapore": ("Asia/Singapore", "Singapore"),
            "new york": ("America/New_York", "New York"), "los angeles": ("America/Los_Angeles", "Los Angeles"),
            "sydney": ("Australia/Sydney", "Sydney"),
        }
        match = next(((zone, label) for name, (zone, label) in zones.items() if name in text.lower()), None)
        if not match:
            return ActionResult("needs_input", "Which city or time zone?", "Try: what time is it in London? Supported demo cities include London, Tokyo, Dubai, Singapore, New York and Sydney.")
        zone, label = match
        now = datetime.now(ZoneInfo(zone))
        return ActionResult("completed", f"Time in {label}", now.strftime("%A, %d %B %Y · %I:%M %p"))

    def _weather(self, text):
        match = re.search(r"(?:in|at|for)\s+([A-Za-z .-]{2,40})", text, re.I)
        city = match.group(1).strip(" ?.!") if match else ""
        if not city:
            return ActionResult("needs_input", "Weather", "Mention a city, for example: weather in Bengaluru")
        try:
            geo = self._json("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(city))
            place = geo.get("results", [])[0]
            url = (
                "https://api.open-meteo.com/v1/forecast?current=temperature_2m,apparent_temperature,"
                "relative_humidity_2m,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max&forecast_days=3&timezone=auto&latitude=" + str(place["latitude"]) +
                "&longitude=" + str(place["longitude"])
            )
            forecast = self._json(url)
            if "tomorrow" in text.lower():
                daily = forecast["daily"]
                message = (
                    f"{place['name']} tomorrow: {daily['temperature_2m_min'][1]}–"
                    f"{daily['temperature_2m_max'][1]}°C, maximum rain probability "
                    f"{daily['precipitation_probability_max'][1]}%."
                )
                return ActionResult("completed", "Tomorrow's forecast", message)
            current = forecast["current"]
            message = (
                f"{place['name']} now: {current['temperature_2m']}°C, feels like "
                f"{current['apparent_temperature']}°C, humidity {current['relative_humidity_2m']}%, "
                f"wind {current['wind_speed_10m']} km/h."
            )
            return ActionResult("completed", "Live weather", message)
        except Exception:
            link = self._query_link("https://www.google.com/search?q=weather+", city)
            return ActionResult("handoff", "Weather search", "Live weather service is unavailable. Open the verified search instead.", link, "View weather")

    def _currency(self, text):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]{3})\s+(?:to|in)\s+([A-Za-z]{3})", text)
        if not match:
            return ActionResult("needs_input", "Currency conversion", "Use a clear format such as: convert 250 USD to INR")
        amount, source, target = float(match.group(1)), match.group(2).upper(), match.group(3).upper()
        try:
            data = self._json(f"https://api.frankfurter.app/latest?amount={amount}&from={source}&to={target}")
            value = data["rates"][target]
            return ActionResult("completed", "Live currency conversion", f"{amount:g} {source} = {value:,.2f} {target} · rate date {data['date']}")
        except Exception:
            return ActionResult("error", "Currency service unavailable", "Check the currency codes or internet connection.")

    def _knowledge(self, text):
        query = re.sub(r"^(define|what is|who is|tell me about)\s+", "", text.strip(), flags=re.I).strip(" ?.!")
        if not query:
            return ActionResult("needs_input", "Knowledge search", "Tell me what you want to look up.")
        try:
            data = self._json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(query.replace(" ", "_")))
            extract = data.get("extract")
            if extract:
                return ActionResult("completed", data.get("title", query), extract[:700], data.get("content_urls", {}).get("desktop", {}).get("page"), "Read source")
        except Exception:
            pass
        link = self._query_link("https://en.wikipedia.org/w/index.php?search=", query)
        return ActionResult("handoff", "Knowledge search", "Open Wikipedia results for this query.", link, "Search Wikipedia")

    def _list_add(self, text):
        item = re.sub(r"^(add|put)\s+", "", text, flags=re.I)
        item = re.sub(r"\s+(to|in)\s+(my\s+)?(shopping\s+)?list.*$", "", item, flags=re.I).strip(" .")
        if not item: return ActionResult("needs_input", "Smart list", "Tell me what to add.")
        self.lists["default"].append(item)
        return ActionResult("completed", "Added to your list", item)

    def _list_query(self, _text):
        items = self.lists["default"]
        message = "Your list is empty." if not items else "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        return ActionResult("completed", "Your smart list", message)

    def _list_remove(self, text):
        target = re.sub(r"^(remove|delete)\s+", "", text, flags=re.I)
        target = re.sub(r"\s+from\s+.*list.*$", "", target, flags=re.I).strip(" .")
        for index, item in enumerate(self.lists["default"]):
            if target.lower() in item.lower():
                removed = self.lists["default"].pop(index)
                return ActionResult("completed", "Removed from your list", removed)
        return ActionResult("needs_input", "Item not found", "Check your list and use the item name shown there.")

    def _news(self, text):
        query = re.sub(r"\b(latest|news|show me|about)\b", " ", text, flags=re.I).strip() or "technology"
        return ActionResult("handoff", "Live news", f"Open current news results for {query}.", self._query_link("https://news.google.com/search?q=", query), "Open Google News")

    def _music(self, text):
        query = re.sub(r"\b(play|music|song|radio|podcast|audiobook)\b", " ", text, flags=re.I).strip() or text
        return ActionResult("handoff", "Media search", f"Search for {query}.", self._query_link("https://www.youtube.com/results?search_query=", query), "Open YouTube")

    def _music_like(self, text):
        item = re.sub(r"\b(i|really|do|like|love|this|song|music)\b", " ", text, flags=re.I).strip() or "current track"
        self.music_preferences["liked"].append(item)
        return ActionResult("completed", "Preference saved for this demo", f"Liked: {item}. This is session-local and is not sent to a music account.")

    def _music_dislike(self, text):
        item = re.sub(r"\b(i|really|do not|don't|dislike|hate|this|song|music)\b", " ", text, flags=re.I).strip() or "current track"
        self.music_preferences["disliked"].append(item)
        return ActionResult("completed", "Preference saved for this demo", f"Disliked: {item}. This is session-local and is not sent to a music account.")

    def _music_settings(self, _text):
        return ActionResult(
            "completed", "Demo music preferences",
            f"Liked: {len(self.music_preferences['liked'])} · Disliked: {len(self.music_preferences['disliked'])}. Connect a music account to change real settings.",
        )

    def _game(self, text):
        query = re.sub(r"\b(play|start|game)\b", " ", text, flags=re.I).strip()
        if re.search(r"\b(coin|heads|tails)\b", text, re.I):
            return ActionResult("completed", "Coin toss", random.choice(["Heads", "Tails"]))
        if re.search(r"\b(dice|die|roll)\b", text, re.I):
            return ActionResult("completed", "Dice roll", str(random.randint(1, 6)))
        return ActionResult("handoff", "Game search", f"Find a browser game for {query or 'a quick game'}.", self._query_link("https://www.google.com/search?q=play+online+game+", query), "Find a game")

    def _movies(self, text):
        return ActionResult("handoff", "Movie discovery", "Open current movie recommendations matching your request.", self._query_link("https://www.google.com/search?q=movie+recommendations+", text), "Discover movies")

    def _places(self, text):
        return ActionResult("handoff", "Place discovery", "Open matching places in Maps.", self._query_link("https://www.google.com/maps/search/", text), "Open Maps")

    def _events(self, text):
        return ActionResult("handoff", "Event discovery", "Open current event results.", self._query_link("https://www.google.com/search?q=events+", text), "Find events")

    def _traffic(self, text):
        return ActionResult("handoff", "Live traffic and routes", "Open the request in Maps for current traffic information.", self._query_link("https://www.google.com/maps/search/", text), "Open live map")

    def _travel(self, text):
        return ActionResult("handoff", "Travel search", "Open current public travel results; IntentShield does not purchase tickets.", self._query_link("https://www.google.com/search?q=", text), "Search travel")

    def _food(self, text):
        return ActionResult("handoff", "Food search", "Find matching restaurants or takeaway options. No order is placed automatically.", self._query_link("https://www.google.com/maps/search/", text), "Find food")

    def _social(self, text):
        draft = re.sub(r"^(post|tweet|share)\s+", "", text, flags=re.I).strip()
        return ActionResult("completed", "Social post drafted", draft or text)

    def _email(self, text):
        draft = re.sub(r"^(email|send an email to|write to)\s+", "", text, flags=re.I).strip()
        return ActionResult("draft", "Email request prepared", f"Draft request: {draft or text}. Connect an email account and confirm the recipient before anything is sent.")

    def _greet(self, _text):
        return ActionResult("completed", "Hello — how can I help?", "Try a calculation, weather request, recipe, list, knowledge question, route, media search, or smart-home demo command.")

    def _joke(self, _text):
        jokes = [
            "Why did the NLP model carry a dictionary? It wanted better word embeddings.",
            "My classifier said the meeting was out of scope. Honestly, it may be the smartest model here.",
        ]
        return ActionResult("completed", "A small NLP joke", random.choice(jokes))

    def _quirky(self, text):
        lowered = text.lower()
        if "who are you" in lowered or "what are you" in lowered:
            message = "I am IntentShield, a bounded assistant demo built on a frozen multilingual intent classifier. I answer supported tasks and pause when the request is unclear."
        elif "how are you" in lowered:
            message = "Running reliably and ready for a supported task."
        else:
            message = "I can help with 50 supported request types. Try asking for weather, a calculation, a recipe, your list, traffic, or a smart-home demo action."
        return ActionResult("completed", "About IntentShield", message)

    def _device_result(self, title: str, message: str) -> ActionResult:
        return ActionResult("completed", title, message + " This changes only the virtual demo state; no physical device is connected.")

    def _iot_cleaning(self, _text):
        self.devices["cleaner"] = "running"
        return self._device_result("Virtual cleaner started", "Cleaner status: running.")

    def _iot_coffee(self, _text):
        self.devices["coffee"] = "brewing"
        return self._device_result("Virtual coffee maker started", "Coffee maker status: brewing.")

    def _iot_lighton(self, _text):
        self.devices["lights"] = "on"
        return self._device_result("Virtual lights on", f"Lights are on at {self.devices['brightness']}% in {self.devices['colour']}.")

    def _iot_lightoff(self, _text):
        self.devices["lights"] = "off"
        return self._device_result("Virtual lights off", "Lights are off.")

    def _iot_lightdim(self, _text):
        self.devices["lights"] = "on"
        self.devices["brightness"] = max(10, int(self.devices["brightness"]) - 20)
        return self._device_result("Virtual lights dimmed", f"Brightness is {self.devices['brightness']}%.")

    def _iot_lightup(self, _text):
        self.devices["lights"] = "on"
        self.devices["brightness"] = min(100, int(self.devices["brightness"]) + 20)
        return self._device_result("Virtual lights brightened", f"Brightness is {self.devices['brightness']}%.")

    def _iot_lightchange(self, text):
        colours = ["red", "blue", "green", "yellow", "purple", "orange", "white", "warm white", "cool white"]
        colour = next((item for item in colours if item in text.lower()), "warm white")
        self.devices.update({"lights": "on", "colour": colour})
        return self._device_result("Virtual light colour changed", f"Light colour is {colour}.")

    def _iot_plugon(self, _text):
        self.devices["plug"] = "on"
        return self._device_result("Virtual plug on", "Smart-plug status: on.")

    def _iot_plugoff(self, _text):
        self.devices["plug"] = "off"
        return self._device_result("Virtual plug off", "Smart-plug status: off.")


def result_to_html(result: ActionResult) -> str:
    link = ""
    if result.link:
        label = html.escape(result.link_label or "Open")
        link = f'<a class="action-link" href="{html.escape(result.link)}" target="_blank">{label} ↗</a>'
    message = html.escape(result.message).replace("\n", "<br>")
    return f'''<section class="action-card action-{html.escape(result.status)}">
      <div class="action-kicker">ASSISTANT RESPONSE · {html.escape(result.status.upper())}</div>
      <h3>{html.escape(result.title)}</h3><p>{message}</p>{link}</section>'''
