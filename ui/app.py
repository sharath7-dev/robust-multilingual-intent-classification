"""IntentShield: professional Gradio research and assistant demonstration."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.actions import ActionEngine, result_to_html  # noqa: E402
from src.inference import IntentClassifier  # noqa: E402
from src.voice import transcribe_audio  # noqa: E402

classifier = IntentClassifier()
actions = ActionEngine()

PRESETS = [
    ("Wethr tmrw", "whts the wethr in Bengaluru tmrw"),
    ("Calclate", "calclate (245 * 18) / 7"),
    ("Add to lst", "add milk and medcine to my shoping list"),
    ("Recpie", "give me a recpie for veg pasta"),
    ("Time in Londn", "wht time is it in London rn"),
    ("Trffic", "hows the trffic to Kempegowda airport"),
    ("Ligths on", "turn the bedroom ligths on"),
    ("Knowlege", "define retrival augmented generation"),
]


def _intent_name(intent: str) -> str:
    return intent.replace("_", " ").title()


def _decision_html(prediction: dict, resolved_intent: str | None = None, used_cue: bool = False) -> str:
    accepted = prediction["status"] == "accepted" or used_cue
    status = "REQUEST UNDERSTOOD" if accepted else "MORE DETAIL NEEDED"
    title = _intent_name(resolved_intent or prediction["top_candidate"]) if accepted else "Clarification required"
    description = (
        "A clear command cue routed this request while the frozen model score remains shown below."
        if used_cue
        else "The frozen classifier passed its confidence threshold."
        if accepted
        else "The frozen classifier is below its confidence threshold, so no action was taken."
    )
    confidence = prediction["confidence"] * 100
    threshold = prediction["threshold"] * 100
    state = "accepted" if accepted else "paused"
    return f"""
    <article class="decision {state}">
      <div class="decision-top"><span class="status-dot"></span><span>{status}</span></div>
      <h2>{html.escape(title)}</h2><p>{description}</p>
      <div class="confidence-head"><span>Calibrated confidence</span><strong>{confidence:.1f}%</strong></div>
      <div class="confidence-track"><span style="width:{min(confidence, 100):.1f}%"></span><i style="left:{threshold:.1f}%"></i></div>
      <div class="threshold-note">Frozen acceptance threshold · {threshold:.1f}%</div>
    </article>"""


def _candidates_html(prediction: dict) -> str:
    rows = []
    for rank, item in enumerate(prediction["top_predictions"], start=1):
        probability = item["probability"] * 100
        rows.append(f"""<div class="candidate-row"><span class="candidate-rank">{rank:02d}</span>
          <span class="candidate-name">{html.escape(_intent_name(item['intent']))}</span>
          <span class="candidate-bar"><i style="width:{probability:.1f}%"></i></span><strong>{probability:.1f}%</strong></div>""")
    return '<section class="candidate-card"><div class="section-label">MODEL INTERPRETATION</div>' + "".join(rows) + "</section>"


def _metadata_html(prediction: dict) -> str:
    return f"""<div class="audit-strip">
      <div><span>MODEL</span><strong>{html.escape(prediction['model'])}</strong></div>
      <div><span>INPUT TOKENS</span><strong>{prediction['token_count']}</strong></div>
      <div><span>TRUNCATED</span><strong>{'Yes' if prediction['truncated'] else 'No'}</strong></div>
      <div><span>TEMPERATURE</span><strong>{prediction['temperature']:.4f}</strong></div></div>"""


def run_command(text: str):
    try:
        prediction = classifier.predict(text, top_k=5)
        explicit_intent = actions.detect_explicit_intent(text)
        accepted = prediction["status"] == "accepted"
        used_cue = bool(explicit_intent and (not accepted or explicit_intent != prediction["top_candidate"]))
        resolved_intent = explicit_intent or prediction["top_candidate"]
        if accepted or explicit_intent:
            action_html = result_to_html(actions.execute(resolved_intent, text))
        else:
            action_html = result_to_html(actions.clarification(prediction["top_predictions"]))
        return action_html, _decision_html(prediction, resolved_intent, used_cue), _candidates_html(prediction), _metadata_html(prediction)
    except Exception as error:
        message = html.escape(str(error))
        return f'<article class="action-card action-error"><div class="action-kicker">REQUEST ERROR</div><h3>Unable to process</h3><p>{message}</p></article>', "", "", ""


def run_preset(text: str):
    return (text, *run_command(text))


def use_voice(audio_path: str | None):
    try:
        text = transcribe_audio(audio_path)
        return text, "Voice converted to text. Check it, then select **Ask IntentShield**."
    except Exception as error:
        return "", f"Voice input failed: {html.escape(str(error))}"


def clear_results():
    return EMPTY_ANSWER, "", "", ""


def clear_all():
    return "", EMPTY_ANSWER, "", "", ""


EMPTY_ANSWER = """<article class="action-card action-waiting"><div class="action-kicker">READY</div>
  <h3>How can I help?</h3><p>Ask for a calculation, weather, recipe, list, time, knowledge, route, media search, or a virtual smart-home action.</p>
  <div class="empty-line"></div><div class="empty-line short"></div></article>"""

CSS = r"""
:root{--bg:#f4f6f8;--surface:#fff;--ink:#17202a;--muted:#687482;--line:#dce2e7;--navy:#102a43;--teal:#007f73;--amber:#b96a16;--shadow:0 12px 32px rgba(16,42,67,.08)}
html,body{width:100%!important;min-height:100%!important;margin:0!important;background:var(--bg)!important;color-scheme:light!important}gradio-app{display:block!important;width:100%!important;max-width:none!important;min-height:100vh!important;background:var(--bg)!important}.gradio-container{width:100%!important;max-width:none!important;min-height:100vh!important;margin:0!important;background:var(--bg)!important;color:var(--ink)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif!important;padding:0 32px 24px!important;box-sizing:border-box!important}
.app-header{height:64px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:rgba(255,255,255,.94);padding:0 28px;margin:0 -24px 8px;position:sticky;top:0;z-index:20;backdrop-filter:blur(14px)}.brand{display:flex;align-items:center;gap:12px}.brand-mark{width:36px;height:36px;border-radius:10px;background:var(--navy);display:grid;place-items:center;color:#fff!important;font-weight:800;font-size:15px}.brand-name{font-size:18px;font-weight:760;color:var(--ink)!important}.brand-sub{font-size:9px;color:var(--muted)!important;letter-spacing:.06em}.header-badge{display:flex;gap:8px;align-items:center;background:#e8f6f3;color:var(--teal)!important;border:1px solid #b9ded7;border-radius:999px;padding:7px 11px;font-size:11px;font-weight:700}.header-badge i{width:7px;height:7px;border-radius:50%;background:#10a37f}
.intro{display:flex;align-items:end;justify-content:space-between;gap:24px;padding:8px 4px 14px}.intro h1{font-size:29px;letter-spacing:-.035em;margin:0 0 5px;color:var(--navy)!important}.intro p{margin:0;color:var(--muted)!important;max-width:760px;line-height:1.4;font-size:13px}.intro-note{font-size:11px;color:var(--muted)!important;text-align:right;white-space:nowrap}
.tabs{border:0!important}.tab-nav{background:transparent!important;border-bottom:1px solid var(--line)!important;min-height:38px!important}.tab-nav button{color:#526171!important;font-weight:650!important;padding:8px 14px!important}.tab-nav button.selected{color:var(--teal)!important;border-bottom:2px solid var(--teal)!important;background:#fff!important}.workspace{padding-top:10px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:var(--shadow)}.section-label{font-size:9px;letter-spacing:.14em;font-weight:800;color:var(--teal)!important;margin-bottom:7px}.panel-title{font-size:18px;font-weight:750;color:var(--navy)!important;margin:0 0 3px}.panel-copy{font-size:12px;color:var(--muted)!important;margin:0 0 10px}
#command-input textarea{background:#f8fafb!important;color:var(--ink)!important;border:1px solid #cfd8df!important;border-radius:9px!important;min-height:100px!important;font-size:14px!important;line-height:1.45!important;padding:12px!important;box-shadow:none!important}#command-input label{color:var(--navy)!important;font-weight:700!important}.run-btn{background:var(--navy)!important;color:#fff!important;border:1px solid var(--navy)!important;height:40px!important}.clear-btn{background:#fff!important;color:#536171!important;border:1px solid var(--line)!important;height:40px!important}.preset-title{font-size:11px;font-weight:750;color:var(--navy)!important;margin:10px 0 6px}.preset-grid button{background:#f7f9fa!important;color:#3f4d5a!important;border:1px solid var(--line)!important;border-radius:8px!important;font-size:11px!important;min-height:31px!important;padding:4px 7px!important}.preset-grid button:hover{border-color:var(--teal)!important;color:var(--teal)!important;background:#f0faf8!important}
.decision{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 21px;min-height:190px;box-shadow:var(--shadow);position:relative;overflow:hidden}.decision:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:#aab6c0}.decision.accepted:before{background:var(--teal)}.decision.paused:before{background:var(--amber)}.decision.error:before{background:#b42318}.decision-top{display:flex;gap:8px;align-items:center;font-size:9px;letter-spacing:.14em;font-weight:850;color:#647382!important}.decision.accepted .decision-top{color:var(--teal)!important}.decision.paused .decision-top{color:var(--amber)!important}.status-dot{width:7px;height:7px;border-radius:50%;background:#9ca8b2}.accepted .status-dot{background:var(--teal)}.paused .status-dot{background:var(--amber)}.decision h2{font-size:24px;letter-spacing:-.025em;color:var(--navy)!important;margin:12px 0 5px}.decision p{color:var(--muted)!important;line-height:1.4;font-size:12px}.confidence-head{display:flex;justify-content:space-between;margin-top:15px;font-size:11px;color:#536171!important}.confidence-head strong{color:var(--navy)!important;font-size:14px}.confidence-track{height:7px;background:#e7ebee;border-radius:20px;position:relative;margin-top:6px}.confidence-track span{height:100%;display:block;background:var(--teal);border-radius:20px}.paused .confidence-track span{background:var(--amber)}.confidence-track i{position:absolute;top:-4px;width:2px;height:15px;background:var(--navy)}.threshold-note{font-size:9px;color:#84909b!important;margin-top:5px;text-align:right}.empty-line{height:7px;width:90%;background:#edf0f2;border-radius:8px;margin-top:18px}.empty-line.short{width:58%;margin-top:8px}
.action-card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px 24px;min-height:220px;box-shadow:var(--shadow);border-top:4px solid var(--teal)}.action-card.action-paused,.action-card.action-needs_input{border-top-color:var(--amber)}.action-kicker{font-size:9px;letter-spacing:.13em;font-weight:800;color:var(--teal)!important}.action-paused .action-kicker,.action-needs_input .action-kicker{color:var(--amber)!important}.action-card h3{font-size:24px;letter-spacing:-.02em;color:var(--navy)!important;margin:12px 0 8px}.action-card p{font-size:14px;color:#4e5e6d!important;line-height:1.65}.action-link{display:inline-flex;margin-top:12px;background:var(--navy);color:#fff!important;text-decoration:none;padding:9px 13px;border-radius:8px;font-size:12px;font-weight:700}
.candidate-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:13px 15px;min-height:145px;box-shadow:var(--shadow)}.candidate-row{display:grid;grid-template-columns:25px minmax(95px,1fr) 1fr 43px;gap:7px;align-items:center;padding:4px 0;border-bottom:1px solid #edf0f2;font-size:10px}.candidate-row:last-child{border:0}.candidate-rank{font-family:ui-monospace,monospace;color:#95a0aa!important}.candidate-name{color:#344454!important;font-weight:650}.candidate-bar{height:4px;background:#e9edef;border-radius:8px;overflow:hidden}.candidate-bar i{display:block;height:100%;background:var(--teal);border-radius:8px}.candidate-row strong{text-align:right;color:var(--navy)!important}.audit-strip{display:grid;grid-template-columns:2.2fr .8fr .8fr .8fr;background:#edf2f5;border:1px solid #d8e0e6;border-radius:10px;overflow:hidden;margin-top:7px}.audit-strip div{padding:8px 11px;border-right:1px solid #d8e0e6}.audit-strip span{font-size:8px;letter-spacing:.1em;color:#82909c!important;display:block}.audit-strip strong{font-size:10px;color:#344454!important;display:block;margin-top:2px}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0 22px}.metric{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.metric span{font-size:10px;letter-spacing:.1em;color:var(--muted)!important}.metric strong{display:block;font-size:26px;color:var(--navy)!important;margin:6px 0 3px}.metric small{color:var(--teal)!important;font-size:11px}.evidence-copy{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}.evidence-copy h2{color:var(--navy)!important;margin:0 0 8px}.evidence-copy h3{color:var(--navy)!important;margin:18px 0 6px}.evidence-copy p,.evidence-copy li{color:var(--muted)!important;line-height:1.6}.capability-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}.capability-grid div{border:1px solid var(--line);border-radius:11px;padding:14px;background:#f8fafb}.capability-grid b{display:block;color:var(--navy)!important;margin-bottom:5px}.capability-grid span{font-size:12px;color:var(--muted)!important;line-height:1.5}.figure-card{background:#fff!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:8px!important;box-shadow:var(--shadow)}.footer{display:flex;justify-content:space-between;padding:24px 4px 4px;color:#84909b!important;font-size:11px}.footer b{color:#536171!important}
@media(max-width:850px){.app-header{padding:0 18px}.intro{display:block}.intro-note{display:none}.metric-grid,.capability-grid{grid-template-columns:1fr}.audit-strip{grid-template-columns:1fr 1fr}.candidate-row{grid-template-columns:26px 1fr 70px}.candidate-bar{display:none}.header-badge{display:none}.gradio-container{padding:0 14px 28px!important}}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IntentShield · Practical AI Assistant") as demo:
        gr.HTML("""<header class="app-header"><div class="brand"><div class="brand-mark">IS</div><div><div class="brand-name">IntentShield</div><div class="brand-sub">PRACTICAL AI ASSISTANT</div></div></div><div class="header-badge"><i></i> FROZEN RESEARCH MODEL</div></header>
        <section class="intro"><div><h1>Ask IntentShield.</h1><p>Useful answers and safe task handling across 50 supported everyday request types.</p></div><div class="intro-note">Answers first<br>Model details on demand</div></section>""")
        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("Assistant"):
                with gr.Column(elem_classes="workspace"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes="panel"):
                            gr.HTML('<div class="section-label">YOUR REQUEST</div><div class="panel-title">What do you need?</div><p class="panel-copy">Use a clear everyday request. Minor spelling errors are okay.</p>')
                            command = gr.Textbox(label="Ask anything supported", lines=5, placeholder="Example: what will the weather be in Bengaluru tomorrow?", elem_id="command-input")
                            with gr.Row():
                                run = gr.Button("Ask IntentShield", variant="primary", elem_classes="run-btn")
                                clear = gr.Button("Clear", elem_classes="clear-btn")
                            with gr.Accordion("Speak instead · optional beta", open=False):
                                voice = gr.Audio(sources=["microphone"], type="filepath", label="Record your request")
                                voice_button = gr.Button("Convert voice to text", size="sm")
                                voice_status = gr.Markdown("The first use downloads a small open-source Whisper model.")
                            gr.HTML('<div class="preset-title">TRY A ROBUSTNESS EXAMPLE · INTENTIONAL TYPOS</div>')
                            preset_buttons = []
                            for start in range(0, len(PRESETS), 4):
                                with gr.Row(elem_classes="preset-grid"):
                                    for label, text in PRESETS[start:start + 4]:
                                        preset_buttons.append((gr.Button(label, size="sm"), text))
                        with gr.Column(scale=7):
                            action_output = gr.HTML(EMPTY_ANSWER)
                    with gr.Accordion("How the model interpreted this request", open=False):
                        decision = gr.HTML()
                        candidates = gr.HTML()
                        metadata = gr.HTML()
            with gr.Tab("Results — simple view"):
                gr.HTML("""<section class="evidence-copy"><div class="section-label">FINAL FROZEN TEST</div><h2>What did the model achieve?</h2><p>In simple terms: it chose the correct intent for about 86 of every 100 clean test requests. Its average score across represented intent classes was 82%. For unseen-intent detection, an AUROC of 83.29% means known and unknown requests were separated reasonably well, but not perfectly.</p></section>
                <section class="metric-grid"><div class="metric"><span>CLEAN ACCURACY</span><strong>85.68%</strong><small>correct intent predictions</small></div><div class="metric"><span>MACRO-F1</span><strong>82.04%</strong><small>average across observed classes</small></div><div class="metric"><span>UNSEEN-INTENT AUROC</span><strong>83.29%</strong><small>ability to separate unknown requests</small></div></section>""")
                with gr.Accordion("Show the one-page final test chart", open=False):
                    gr.Image(str(REPO_ROOT / "results/figures/final_test_summary.png"), show_label=False, elem_classes="figure-card")
            with gr.Tab("Capabilities & limits"):
                gr.HTML("""<section class="evidence-copy"><div class="section-label">WHAT THIS DEMO CAN DO</div><h2>A useful bounded assistant—not a general chatbot</h2><p>The trained NLP model recognizes 50 intent types. The assistant layer then gives a result, updates safe session-local demo state, or opens the right verified service.</p><div class="capability-grid"><div><b>Answers</b><span>Math, time, definitions, factual lookup, weather, currency and recipes.</span></div><div><b>Organises</b><span>Lists, alarms, calendar drafts and preferences during the current session.</span></div><div><b>Finds</b><span>News, media, movies, places, events, food, routes, traffic and tickets.</span></div><div><b>Virtual controls</b><span>Lights, plugs, cleaner, coffee maker and audio state without claiming a real device changed.</span></div><div><b>Safety</b><span>Low-confidence requests ask for clarification instead of silently doing the wrong task.</span></div><div><b>Limits</b><span>No purchases, posts, emails or physical actions occur without a connected account and confirmation.</span></div></div></section>""")
        gr.HTML('<footer class="footer"><span><b>IntentShield</b> · Robust and Calibrated Multilingual Intent Classification</span><span>MASSIVE 1.1 · Combined-Augmented XLM-R</span></footer>')
        outputs = [action_output, decision, candidates, metadata]
        run.click(run_command, inputs=command, outputs=outputs)
        command.submit(run_command, inputs=command, outputs=outputs)
        command.input(clear_results, outputs=outputs)
        clear.click(clear_all, outputs=[command, *outputs])
        voice_button.click(use_voice, inputs=voice, outputs=[command, voice_status])
        for button, text in preset_buttons:
            button.click(lambda value=text: run_preset(value), outputs=[command, *outputs])
    return demo


demo = build_demo()

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "1").lower() not in {"0", "false", "no"}
    demo.launch(share=share, show_error=True, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS,
                allowed_paths=[str(REPO_ROOT / "results/figures")])
