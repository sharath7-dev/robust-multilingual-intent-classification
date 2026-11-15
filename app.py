"""Vercel ASGI entrypoint for the IntentShield Gradio application."""

from fastapi import FastAPI
import gradio as gr

from ui.app import REPO_ROOT, demo


api = FastAPI(
    title="IntentShield",
    description="Robust and calibrated multilingual intent classification demo.",
)


@api.get("/healthz", include_in_schema=False)
def healthcheck():
    """Small readiness endpoint used by deployment checks."""
    return {"status": "ok", "model": "Combined-Augmented XLM-R"}


app = gr.mount_gradio_app(
    api,
    demo,
    path="/",
    allowed_paths=[str(REPO_ROOT / "results" / "figures")],
    show_error=True,
    footer_links=[],
    run_history=False,
)
