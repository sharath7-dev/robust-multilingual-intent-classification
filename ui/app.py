"""Professional Gradio demonstration for the IntentShield research project."""

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


EXAMPLES = [
    ["calculate (245 * 18) / 7"],
    ["convert 250 US dollars to Indian rupees"],
    ["what is the weather in Bengaluru"],
    ["add milk, medicine and coffee to my shopping list"],
    ["show my shopping list"],
    ["what is retrieval augmented generation"],
    ["show me the latest artificial intelligence news"],
    ["how is the traffic to Kempegowda airport"],
    ["recommend a science fiction movie"],
    ["kal bengaluru ka mausam kaisa hai"],
    ["250 dollar ko rupees me convert karo"],
    ["bengaluru nalli nale havaamana hegide"],
    ["calclate 455 divided by 7"],
    ["find trafic near majestic bus stand"],
]


def _decision_html(prediction: dict) -> str:
    accepted = prediction["status"] == "accepted"
    color = "#176b55" if accepted else "#a34723"
    background = "#edf7f2" if accepted else "#fff3e9"
    label = "REQUEST ACCEPTED" if accepted else "RELIABILITY FIREWALL ACTIVATED"
    title = (
        prediction["top_candidate"].replace("_", " ").title()
        if accepted
        else "Action paused — clarification needed"
    )
    explanation = (
        "The intent passed the frozen confidence threshold, so a bounded assistant action was attempted."
        if accepted
        else "The model was not confident enough. No action was executed; revise or clarify the request."
    )
    confidence = prediction["confidence"] * 100
    threshold = prediction["threshold"] * 100
    width = max(0.0, min(100.0, confidence))
    return f"""
    <section class="decision-card" style="--decision:{color};--decision-bg:{background}">
      <div class="decision-kicker">{label}</div>
      <h2>{html.escape(title)}</h2>
      <p>{explanation}</p>
      <div class="meter"><span style="width:{width:.1f}%"></span></div>
      <div class="score-row"><strong>{confidence:.1f}% confidence</strong><span>Frozen threshold: {threshold:.1f}%</span></div>
    </section>
    """


def _metadata_html(prediction: dict) -> str:
    truncated = "Yes" if prediction["truncated"] else "No"
    return f"""
    <div class="meta-grid">
      <div><span>MODEL</span><strong>{html.escape(prediction['model'])}</strong></div>
      <div><span>TOKENS</span><strong>{prediction['token_count']}</strong></div>
      <div><span>TRUNCATED</span><strong>{truncated}</strong></div>
      <div><span>TEMPERATURE</span><strong>{prediction['temperature']:.4f}</strong></div>
    </div>
    """


def run_command(text: str):
    try:
        prediction = classifier.predict(text, top_k=5)
        candidates = {
            item["intent"].replace("_", " ").title(): item["probability"]
            for item in prediction["top_predictions"]
        }
        if prediction["status"] == "accepted":
            action = actions.execute(prediction["top_candidate"], text)
            action_html = result_to_html(action)
        else:
            action_html = """
            <section class="action-card paused">
              <div class="action-kicker">SAFE ABSTENTION</div>
              <h3>Nothing was executed</h3>
              <p>Try adding the task, subject, location or quantity. The closest known intent is shown for transparency.</p>
            </section>
            """
        return _decision_html(prediction), action_html, candidates, _metadata_html(prediction)
    except Exception as error:
        message = html.escape(str(error))
        return (
            f'<section class="decision-card error"><div class="decision-kicker">ERROR</div><h2>Request could not be processed</h2><p>{message}</p></section>',
            "",
            {},
            "",
        )


def clear_console():
    return "", "", "", {}, ""


CSS = r"""
:root {
  --ink: #22201c; --muted: #716c63; --paper: #f7f3eb; --card: #fffdf8;
  --line: #d9d0c2; --rust: #b55234; --green: #176b55; --charcoal: #292622;
}
body, .gradio-container { background: #e9e3d8 !important; color: var(--ink) !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important; }
.page-shell { background: var(--paper); border-left: 1px solid #d4cabc; border-right: 1px solid #d4cabc; min-height: 100vh; }
.hero { padding: 44px 48px 30px; border-bottom: 1px solid var(--line); position: relative; overflow: hidden; }
.eyebrow, .decision-kicker, .action-kicker { font-size: 11px; letter-spacing: .16em; font-weight: 800; color: var(--rust); }
.hero h1 { font-family: Georgia, serif; font-size: clamp(44px, 6vw, 76px); line-height: .94; margin: 14px 0 18px; letter-spacing: -.04em; max-width: 820px; }
.hero h1 em { color: var(--rust); font-weight: 400; }
.hero p { max-width: 730px; color: #514d46; font-size: 17px; line-height: 1.65; }
.hero-orbit { position:absolute; width:340px; height:340px; border:1px solid #cbbca8; border-radius:50%; right:-80px; top:-110px; }
.hero-orbit:before,.hero-orbit:after { content:""; position:absolute; inset:45px; border:1px solid #d5c8b8; border-radius:50%; transform:rotate(55deg); }
.hero-orbit:after { inset:95px; background:var(--rust); border:0; }
.stat-strip { display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--line); }
.stat-strip div { padding:18px 22px; border-right:1px solid var(--line); }
.stat-strip span,.meta-grid span { display:block; font-size:10px; letter-spacing:.12em; color:var(--muted); }
.stat-strip strong { font-family:Georgia,serif; font-size:24px; }
.content-pad { padding: 24px 38px 42px; }
.command-box textarea { background:#282521 !important; color:#fffaf1 !important; min-height:150px !important; border-radius:4px !important; font-size:17px !important; padding:18px !important; }
.primary-btn { background:var(--rust) !important; color:white !important; border:0 !important; }
.secondary-btn { background:#eee7db !important; color:var(--ink) !important; border:1px solid var(--line) !important; }
.decision-card,.action-card { background:var(--card); border:1px solid var(--line); border-top:4px solid var(--decision,var(--green)); padding:22px; border-radius:8px; min-height:190px; }
.decision-card { background:var(--decision-bg,var(--card)); }
.decision-card h2,.action-card h3 { font-family:Georgia,serif; font-size:28px; margin:10px 0; }
.decision-card p,.action-card p { color:#5b554c; line-height:1.55; }
.meter { height:6px; background:#ded8ce; margin-top:20px; border-radius:20px; overflow:hidden; }
.meter span { display:block; height:100%; background:var(--decision,var(--green)); }
.score-row { display:flex; justify-content:space-between; margin-top:8px; font-size:12px; }
.action-card a { display:inline-block; background:var(--charcoal); color:white !important; padding:10px 14px; margin-top:10px; text-decoration:none; border-radius:4px; }
.meta-grid { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); background:#f0eadf; }
.meta-grid div { padding:14px; border-right:1px solid var(--line); }
.research-intro { padding:14px 4px 22px; max-width:850px; }
.research-intro h2,.about-card h2 { font-family:Georgia,serif; font-size:34px; }
.research-intro p,.about-card p,.about-card li { color:#5b554c; line-height:1.65; }
.figure-card { background:white; padding:10px; border:1px solid var(--line); }
.about-card { background:var(--card); border:1px solid var(--line); padding:28px; border-radius:6px; }
.examples table { font-size:13px !important; }
footer { padding:24px 44px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
@media(max-width:800px){.hero{padding:32px 24px}.content-pad{padding:18px}.stat-strip{grid-template-columns:1fr 1fr}.meta-grid{grid-template-columns:1fr 1fr}.hero-orbit{display:none}}
"""


HEAD = """
<meta name="description" content="IntentShield: robust multilingual intent classification research demonstration">
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IntentShield — Multilingual Reliability Console") as demo:
        gr.HTML("""
        <div class="page-shell"><header class="hero">
          <div class="hero-orbit"></div>
          <div class="eyebrow">MULTILINGUAL NLP · RESEARCH DEMONSTRATION</div>
          <h1>Understand the request.<br><em>Know when to pause.</em></h1>
          <p>A calibrated assistant gateway for English, Hindi and Kannada—evaluated under spelling noise, Romanized input and unseen intents.</p>
        </header>
        <section class="stat-strip">
          <div><span>LANGUAGES</span><strong>3</strong></div>
          <div><span>KNOWN INTENTS</span><strong>50</strong></div>
          <div><span>FROZEN TEST MACRO-F1</span><strong>0.821</strong></div>
          <div><span>ACCEPTANCE THRESHOLD</span><strong>72.8%</strong></div>
        </section></div>
        """)

        with gr.Tabs():
            with gr.Tab("Command Center"):
                with gr.Column(elem_classes="content-pad"):
                    gr.Markdown("### Issue a meaningful request\nThe model first identifies intent. Only confident requests reach the bounded action layer.")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5):
                            command = gr.Textbox(
                                label="Assistant request",
                                placeholder="Try: convert 250 US dollars to Indian rupees",
                                lines=5,
                                elem_classes="command-box",
                            )
                            with gr.Row():
                                run = gr.Button("Analyze and assist", variant="primary", elem_classes="primary-btn")
                                clear = gr.Button("Clear", elem_classes="secondary-btn")
                        with gr.Column(scale=5):
                            decision = gr.HTML()
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5):
                            action_output = gr.HTML()
                        with gr.Column(scale=5):
                            candidates = gr.Label(label="Top five known-intent candidates", num_top_classes=5)
                    metadata = gr.HTML()
                    gr.Markdown("### Request gallery")
                    gr.Examples(EXAMPLES, inputs=command, examples_per_page=14, elem_classes="examples")

            with gr.Tab("Reliability Lab"):
                with gr.Column(elem_classes="content-pad"):
                    gr.HTML("""
                    <section class="research-intro"><div class="eyebrow">FROZEN EVALUATION</div>
                    <h2>Research evidence, not a staged chatbot.</h2>
                    <p>Four systems were evaluated on clean requests, controlled spelling corruption, synthetic Romanization and held-out intents. The final test protocol was frozen before test inference.</p></section>
                    """)
                    with gr.Row():
                        gr.Image(str(REPO_ROOT / "results/figures/final_test_summary.png"), label="Final frozen test summary", elem_classes="figure-card")
                        gr.Image(str(REPO_ROOT / "results/figures/spelling_noise_robustness.png"), label="Spelling-noise robustness", elem_classes="figure-card")
                    with gr.Row():
                        gr.Image(str(REPO_ROOT / "results/figures/transliteration_robustness.png"), label="Romanized-input robustness", elem_classes="figure-card")
                        gr.Image(str(REPO_ROOT / "results/figures/oos_calibration_summary.png"), label="Calibration and unseen intents", elem_classes="figure-card")

            with gr.Tab("About & Safety"):
                with gr.Column(elem_classes="content-pad"):
                    gr.HTML("""
                    <section class="about-card"><div class="eyebrow">WHAT THIS DEMO PROVES</div>
                    <h2>A reliability layer for multilingual assistants.</h2>
                    <p>The research contribution is not a new alarm application. It is a controlled study of how intent models behave when real user language is messy, Romanized or outside the system's supported intents.</p>
                    <ul>
                      <li><b>Useful actions:</b> calculations, time, public knowledge, weather, currency, lists and safe search handoffs.</li>
                      <li><b>Confidence firewall:</b> low-confidence inputs are paused instead of being silently executed.</li>
                      <li><b>Honest boundaries:</b> device, email, ordering and posting requests are never reported as completed without a connected service and confirmation.</li>
                      <li><b>Frozen settings:</b> model, calibration temperature and acceptance threshold are loaded from the published experiment configuration.</li>
                    </ul></section>
                    """)

        gr.HTML("<footer>IntentShield · B.Tech NLP research prototype · MASSIVE 1.1 · Combined-Augmented XLM-R</footer>")

        run.click(run_command, inputs=command, outputs=[decision, action_output, candidates, metadata])
        command.submit(run_command, inputs=command, outputs=[decision, action_output, candidates, metadata])
        clear.click(clear_console, outputs=[command, decision, action_output, candidates, metadata])
    return demo


demo = build_demo()


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "1").lower() not in {"0", "false", "no"}
    demo.launch(
        share=share,
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
        head=HEAD,
        allowed_paths=[str(REPO_ROOT / "results/figures")],
    )
