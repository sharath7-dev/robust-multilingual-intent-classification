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

classifier = IntentClassifier()
actions = ActionEngine()

PRESETS = [
    ("Calculate", "calculate (245 * 18) / 7"),
    ("Currency", "convert 250 USD to INR"),
    ("Weather", "what will the weather be in Bengaluru tomorrow"),
    ("My list", "add milk, medicine and coffee to my shopping list"),
    ("Knowledge", "define retrieval augmented generation"),
    ("Traffic", "how is the traffic to Kempegowda airport"),
    ("Hindi", "kal bengaluru ka mausam kaisa hai"),
    ("Kannada", "nale bengaluru havaamana hegide"),
]


def _intent_name(intent: str) -> str:
    return intent.replace("_", " ").title()


def _decision_html(prediction: dict) -> str:
    accepted = prediction["status"] == "accepted"
    status = "READY TO ASSIST" if accepted else "ACTION PAUSED"
    title = _intent_name(prediction["top_candidate"]) if accepted else "Clarification required"
    description = (
        "The request passed the frozen confidence threshold and may continue to the bounded action layer."
        if accepted
        else "Confidence is below the frozen threshold. No action was executed. Add more detail and try again."
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
        if prediction["status"] == "accepted":
            action_html = result_to_html(actions.execute(prediction["top_candidate"], text))
        else:
            action_html = """<article class="action-card action-paused"><div class="action-kicker">CONFIDENCE FIREWALL</div>
              <h3>No action executed</h3><p>The system stopped safely. Clarify the task, location, item, amount or destination.</p></article>"""
        return _decision_html(prediction), action_html, _candidates_html(prediction), _metadata_html(prediction)
    except Exception as error:
        message = html.escape(str(error))
        return f'<article class="decision error"><div class="decision-top">REQUEST ERROR</div><h2>Unable to process</h2><p>{message}</p></article>', "", "", ""


def run_preset(text: str):
    return (text, *run_command(text))


def clear_results():
    return "", "", "", ""


def clear_all():
    return "", "", "", "", ""


EMPTY_DECISION = """<article class="decision waiting"><div class="decision-top"><span class="status-dot"></span><span>AWAITING REQUEST</span></div>
  <h2>Reliability first.</h2><p>Enter a command. IntentShield will classify it, measure confidence and either assist or pause safely.</p>
  <div class="empty-line"></div><div class="empty-line short"></div></article>"""

CSS = r"""
:root{--bg:#f4f6f8;--surface:#fff;--ink:#17202a;--muted:#687482;--line:#dce2e7;--navy:#102a43;--teal:#007f73;--amber:#b96a16;--shadow:0 12px 32px rgba(16,42,67,.08)}
html,body{width:100%!important;min-height:100%!important;margin:0!important;background:var(--bg)!important;color-scheme:light!important}gradio-app{display:block!important;width:100%!important;max-width:none!important;min-height:100vh!important;background:var(--bg)!important}.gradio-container{width:100%!important;max-width:none!important;min-height:100vh!important;margin:0!important;background:var(--bg)!important;color:var(--ink)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif!important;padding:0 32px 24px!important;box-sizing:border-box!important}
.app-header{height:64px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:rgba(255,255,255,.94);padding:0 28px;margin:0 -24px 8px;position:sticky;top:0;z-index:20;backdrop-filter:blur(14px)}.brand{display:flex;align-items:center;gap:12px}.brand-mark{width:36px;height:36px;border-radius:10px;background:var(--navy);display:grid;place-items:center;color:#fff!important;font-weight:800;font-size:15px}.brand-name{font-size:18px;font-weight:760;color:var(--ink)!important}.brand-sub{font-size:9px;color:var(--muted)!important;letter-spacing:.06em}.header-badge{display:flex;gap:8px;align-items:center;background:#e8f6f3;color:var(--teal)!important;border:1px solid #b9ded7;border-radius:999px;padding:7px 11px;font-size:11px;font-weight:700}.header-badge i{width:7px;height:7px;border-radius:50%;background:#10a37f}
.intro{display:flex;align-items:end;justify-content:space-between;gap:24px;padding:8px 4px 14px}.intro h1{font-size:29px;letter-spacing:-.035em;margin:0 0 5px;color:var(--navy)!important}.intro p{margin:0;color:var(--muted)!important;max-width:760px;line-height:1.4;font-size:13px}.intro-note{font-size:11px;color:var(--muted)!important;text-align:right;white-space:nowrap}
.tabs{border:0!important}.tab-nav{background:transparent!important;border-bottom:1px solid var(--line)!important;min-height:38px!important}.tab-nav button{color:#526171!important;font-weight:650!important;padding:8px 14px!important}.tab-nav button.selected{color:var(--teal)!important;border-bottom:2px solid var(--teal)!important;background:#fff!important}.workspace{padding-top:10px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:var(--shadow)}.section-label{font-size:9px;letter-spacing:.14em;font-weight:800;color:var(--teal)!important;margin-bottom:7px}.panel-title{font-size:18px;font-weight:750;color:var(--navy)!important;margin:0 0 3px}.panel-copy{font-size:12px;color:var(--muted)!important;margin:0 0 10px}
#command-input textarea{background:#f8fafb!important;color:var(--ink)!important;border:1px solid #cfd8df!important;border-radius:9px!important;min-height:100px!important;font-size:14px!important;line-height:1.45!important;padding:12px!important;box-shadow:none!important}#command-input label{color:var(--navy)!important;font-weight:700!important}.run-btn{background:var(--navy)!important;color:#fff!important;border:1px solid var(--navy)!important;height:40px!important}.clear-btn{background:#fff!important;color:#536171!important;border:1px solid var(--line)!important;height:40px!important}.preset-title{font-size:11px;font-weight:750;color:var(--navy)!important;margin:10px 0 6px}.preset-grid button{background:#f7f9fa!important;color:#3f4d5a!important;border:1px solid var(--line)!important;border-radius:8px!important;font-size:11px!important;min-height:31px!important;padding:4px 7px!important}.preset-grid button:hover{border-color:var(--teal)!important;color:var(--teal)!important;background:#f0faf8!important}
.decision{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 21px;min-height:190px;box-shadow:var(--shadow);position:relative;overflow:hidden}.decision:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:#aab6c0}.decision.accepted:before{background:var(--teal)}.decision.paused:before{background:var(--amber)}.decision.error:before{background:#b42318}.decision-top{display:flex;gap:8px;align-items:center;font-size:9px;letter-spacing:.14em;font-weight:850;color:#647382!important}.decision.accepted .decision-top{color:var(--teal)!important}.decision.paused .decision-top{color:var(--amber)!important}.status-dot{width:7px;height:7px;border-radius:50%;background:#9ca8b2}.accepted .status-dot{background:var(--teal)}.paused .status-dot{background:var(--amber)}.decision h2{font-size:24px;letter-spacing:-.025em;color:var(--navy)!important;margin:12px 0 5px}.decision p{color:var(--muted)!important;line-height:1.4;font-size:12px}.confidence-head{display:flex;justify-content:space-between;margin-top:15px;font-size:11px;color:#536171!important}.confidence-head strong{color:var(--navy)!important;font-size:14px}.confidence-track{height:7px;background:#e7ebee;border-radius:20px;position:relative;margin-top:6px}.confidence-track span{height:100%;display:block;background:var(--teal);border-radius:20px}.paused .confidence-track span{background:var(--amber)}.confidence-track i{position:absolute;top:-4px;width:2px;height:15px;background:var(--navy)}.threshold-note{font-size:9px;color:#84909b!important;margin-top:5px;text-align:right}.empty-line{height:7px;width:90%;background:#edf0f2;border-radius:8px;margin-top:18px}.empty-line.short{width:58%;margin-top:8px}
.action-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-height:145px;box-shadow:var(--shadow);border-top:3px solid var(--teal)}.action-card.action-paused,.action-card.action-needs_input{border-top-color:var(--amber)}.action-kicker{font-size:8px;letter-spacing:.13em;font-weight:800;color:var(--teal)!important}.action-paused .action-kicker,.action-needs_input .action-kicker{color:var(--amber)!important}.action-card h3{font-size:18px;color:var(--navy)!important;margin:7px 0 4px}.action-card p{font-size:11px;color:var(--muted)!important;line-height:1.4}.action-link{display:inline-flex;margin-top:5px;background:var(--navy);color:#fff!important;text-decoration:none;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:700}
.candidate-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:13px 15px;min-height:145px;box-shadow:var(--shadow)}.candidate-row{display:grid;grid-template-columns:25px minmax(95px,1fr) 1fr 43px;gap:7px;align-items:center;padding:4px 0;border-bottom:1px solid #edf0f2;font-size:10px}.candidate-row:last-child{border:0}.candidate-rank{font-family:ui-monospace,monospace;color:#95a0aa!important}.candidate-name{color:#344454!important;font-weight:650}.candidate-bar{height:4px;background:#e9edef;border-radius:8px;overflow:hidden}.candidate-bar i{display:block;height:100%;background:var(--teal);border-radius:8px}.candidate-row strong{text-align:right;color:var(--navy)!important}.audit-strip{display:grid;grid-template-columns:2.2fr .8fr .8fr .8fr;background:#edf2f5;border:1px solid #d8e0e6;border-radius:10px;overflow:hidden;margin-top:7px}.audit-strip div{padding:8px 11px;border-right:1px solid #d8e0e6}.audit-strip span{font-size:8px;letter-spacing:.1em;color:#82909c!important;display:block}.audit-strip strong{font-size:10px;color:#344454!important;display:block;margin-top:2px}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0 22px}.metric{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.metric span{font-size:10px;letter-spacing:.1em;color:var(--muted)!important}.metric strong{display:block;font-size:26px;color:var(--navy)!important;margin:6px 0 3px}.metric small{color:var(--teal)!important;font-size:11px}.evidence-copy{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}.evidence-copy h2{color:var(--navy)!important;margin:0 0 8px}.evidence-copy p,.evidence-copy li{color:var(--muted)!important;line-height:1.6}.figure-card{background:#fff!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:8px!important;box-shadow:var(--shadow)}.footer{display:flex;justify-content:space-between;padding:24px 4px 4px;color:#84909b!important;font-size:11px}.footer b{color:#536171!important}
@media(max-width:850px){.app-header{padding:0 18px}.intro{display:block}.intro-note{display:none}.metric-grid{grid-template-columns:1fr 1fr}.audit-strip{grid-template-columns:1fr 1fr}.candidate-row{grid-template-columns:26px 1fr 70px}.candidate-bar{display:none}.header-badge{display:none}.gradio-container{padding:0 14px 28px!important}}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IntentShield · Multilingual Reliability Console") as demo:
        gr.HTML("""<header class="app-header"><div class="brand"><div class="brand-mark">IS</div><div><div class="brand-name">IntentShield</div><div class="brand-sub">MULTILINGUAL RELIABILITY CONSOLE</div></div></div><div class="header-badge"><i></i> FROZEN RESEARCH MODEL</div></header>
        <section class="intro"><div><h1>From request to safe action.</h1><p>Classify multilingual commands, measure uncertainty and assist only when the frozen model is sufficiently confident.</p></div><div class="intro-note">English · Hindi · Kannada<br>Native and Romanized input</div></section>""")
        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("Assistant Console"):
                with gr.Column(elem_classes="workspace"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes="panel"):
                            gr.HTML('<div class="section-label">01 · COMMAND</div><div class="panel-title">What do you need?</div><p class="panel-copy">Natural language and imperfect spelling are welcome.</p>')
                            command = gr.Textbox(label="Your request", lines=5, placeholder="Example: convert 250 USD to INR", elem_id="command-input")
                            with gr.Row():
                                run = gr.Button("Analyze & assist", variant="primary", elem_classes="run-btn")
                                clear = gr.Button("Clear", elem_classes="clear-btn")
                            gr.HTML('<div class="preset-title">Try a real task</div>')
                            preset_buttons = []
                            for start in range(0, len(PRESETS), 4):
                                with gr.Row(elem_classes="preset-grid"):
                                    for label, text in PRESETS[start:start + 4]:
                                        preset_buttons.append((gr.Button(label, size="sm"), text))
                        with gr.Column(scale=7):
                            decision = gr.HTML(EMPTY_DECISION)
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=5): action_output = gr.HTML()
                                with gr.Column(scale=7): candidates = gr.HTML()
                            metadata = gr.HTML()
            with gr.Tab("Research Evidence"):
                gr.HTML("""<section class="evidence-copy"><div class="section-label">FROZEN FINAL TEST</div><h2>Evidence behind the assistant</h2><p>The final model was selected using validation data, then evaluated once with frozen calibration and rejection settings. No post-test model or threshold tuning was performed.</p></section>
                <section class="metric-grid"><div class="metric"><span>CLEAN ACCURACY</span><strong>85.70%</strong><small>6,831 known-intent tests</small></div><div class="metric"><span>OBSERVED MACRO-F1</span><strong>0.8204</strong><small>49 represented intents</small></div><div class="metric"><span>CHALLENGE SELECTIVE ACC.</span><strong>95.83%</strong><small>accepted diagnostic requests</small></div><div class="metric"><span>UNSAFE EXECUTION</span><strong>0%</strong><small>36-request diagnostic</small></div></section>""")
                with gr.Accordion("Final test overview", open=True): gr.Image(str(REPO_ROOT / "results/figures/final_test_summary.png"), show_label=False, elem_classes="figure-card")
                with gr.Accordion("Intent confusion matrix", open=False): gr.Image(str(REPO_ROOT / "results/figures/final_test_confusion_matrix.png"), show_label=False, elem_classes="figure-card")
                with gr.Row():
                    with gr.Accordion("Spelling-noise robustness", open=False): gr.Image(str(REPO_ROOT / "results/figures/spelling_noise_robustness.png"), show_label=False, elem_classes="figure-card")
                    with gr.Accordion("Romanized-input robustness", open=False): gr.Image(str(REPO_ROOT / "results/figures/transliteration_robustness.png"), show_label=False, elem_classes="figure-card")
            with gr.Tab("Method & Boundaries"):
                gr.HTML("""<section class="evidence-copy"><div class="section-label">HOW TO READ THIS DEMO</div><h2>A reliability gateway—not a general-purpose chatbot</h2><ul><li><b>50 known intents:</b> the model routes requests supported by the MASSIVE-derived task.</li><li><b>Confidence firewall:</b> requests below 72.8% confidence are paused.</li><li><b>Bounded actions:</b> calculations and lists run locally; current information uses public services.</li><li><b>Honest limits:</b> orders, email, posting and devices are not reported as completed without a connected service.</li><li><b>Research status:</b> the 36-request diagnostic is team-authored and exploratory.</li></ul></section>""")
        gr.HTML('<footer class="footer"><span><b>IntentShield</b> · Robust and Calibrated Multilingual Intent Classification</span><span>MASSIVE 1.1 · Combined-Augmented XLM-R</span></footer>')
        outputs = [decision, action_output, candidates, metadata]
        run.click(run_command, inputs=command, outputs=outputs)
        command.submit(run_command, inputs=command, outputs=outputs)
        command.input(clear_results, outputs=outputs)
        clear.click(clear_all, outputs=[command, *outputs])
        for button, text in preset_buttons:
            button.click(lambda value=text: run_preset(value), outputs=[command, *outputs])
    return demo


demo = build_demo()

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "1").lower() not in {"0", "false", "no"}
    demo.launch(share=share, show_error=True, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS,
                allowed_paths=[str(REPO_ROOT / "results/figures")])
