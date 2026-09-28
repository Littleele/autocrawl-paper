"""Runtime configuration shared by the paper-enhancement job."""

from urllib.parse import urlparse


def build_chat_openai_kwargs(
    model_name: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str = "",
) -> dict:
    """Build provider-safe ChatOpenAI settings from workflow configuration."""
    model_name = model_name.strip()
    base_url = base_url.strip().rstrip("/")
    api_key = api_key.strip()

    missing = [
        name
        for name, value in (
            ("MODEL_NAME", model_name),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }
    hostname = urlparse(base_url).hostname or ""
    if hostname.endswith("volces.com"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # Non-reasoning models reject `reasoning_effort` outright, so only send it when set.
    reasoning_effort = reasoning_effort.strip()
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        # /v1/chat/completions refuses function tools while reasoning is active;
        # /v1/responses is the endpoint where structured output and reasoning coexist.
        if reasoning_effort != "none" and hostname.endswith("openai.com"):
            kwargs["use_responses_api"] = True
    return kwargs


def raise_if_processing_failed(errors: list[str]) -> None:
    """Make a failed AI batch fail the workflow instead of publishing placeholders."""
    if errors:
        raise RuntimeError(f"{len(errors)} paper(s) failed AI enhancement: {errors[0]}")
