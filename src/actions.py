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
    def __init__(self):
        self.lists: dict[str, list[str]] = {"default": []}

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
            "qa_maths": self._math,
            "datetime_query": self._datetime,
            "datetime_convert": self._datetime,
            "weather_query": self._weather,
            "qa_currency": self._currency,
            "qa_definition": self._knowledge,
            "qa_factoid": self._knowledge,
            "lists_createoradd": self._list_add,
            "lists_query": self._list_query,
            "lists_remove": self._list_remove,
            "news_query": self._news,
            "music_query": self._music,
            "music_likeness": self._music,
            "play_radio": self._music,
            "play_podcasts": self._music,
            "play_audiobook": self._music,
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
        }
        if intent in handlers:
            return handlers[intent](text)
        if intent.startswith("iot_"):
            return ActionResult(
                "needs_connection",
                "Device command understood",
                "The intent is valid, but no physical device is linked. IntentShield will not simulate a successful device action.",
            )
        return ActionResult(
            "understood",
            "Request understood",
            f"IntentShield routed this to '{intent.replace('_', ' ')}'. A production connector is required to complete the action.",
        )

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
                "relative_humidity_2m,wind_speed_10m&latitude=" + str(place["latitude"]) +
                "&longitude=" + str(place["longitude"])
            )
            current = self._json(url)["current"]
            message = (
                f"{place['name']}: {current['temperature_2m']}°C, feels like "
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
        return ActionResult("understood", "Email request understood", "For safety, connect an email account and confirm the recipient before sending anything.")

    def _greet(self, _text):
        return ActionResult("completed", "IntentShield is ready", "Ask in English, Hindi, Kannada, or Romanized text. The reliability firewall checks confidence before acting.")

    def _joke(self, _text):
        jokes = [
            "Why did the NLP model carry a dictionary? It wanted better word embeddings.",
            "My classifier said the meeting was out of scope. Honestly, it may be the smartest model here.",
        ]
        return ActionResult("completed", "A small NLP joke", random.choice(jokes))


def result_to_html(result: ActionResult) -> str:
    link = ""
    if result.link:
        label = html.escape(result.link_label or "Open")
        link = f'<a class="action-link" href="{html.escape(result.link)}" target="_blank">{label} ↗</a>'
    message = html.escape(result.message).replace("\n", "<br>")
    return f'''<section class="action-card action-{html.escape(result.status)}">
      <div class="action-kicker">ASSISTANT ACTION · {html.escape(result.status.upper())}</div>
      <h3>{html.escape(result.title)}</h3><p>{message}</p>{link}</section>'''
