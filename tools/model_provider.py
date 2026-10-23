import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()

def get_provider_model_name(
    provider: str,
    model_name: str,
) -> str:
    """
    Translate the project's logical model name into
    the identifier expected by the selected provider.
    """

    if provider == "cloudflare":
        cloudflare_models = {
            "openai/gpt-oss-120b": (
                "@cf/openai/gpt-oss-120b"
            ),
            "openai/gpt-oss-20b": (
                "@cf/openai/gpt-oss-20b"
            ),
        }

        return cloudflare_models.get(
            model_name,
            model_name,
        )

    return model_name

def create_chat_model(
    model_name: str,
    temperature: float,
    timeout: int = 30,
    max_retries: int = 0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
):
    """
    Create the project's configured chat model.

    The agents depend on one model interface.
    Python decides which inference provider hosts it.
    """

    provider = os.getenv(
        "MODEL_PROVIDER",
        "groq",
    ).strip().casefold()

    provider_model_name = (
        get_provider_model_name(
            provider=provider,
            model_name=model_name,
        )
    )

    effective_timeout = timeout

    if provider == "cloudflare":
        effective_timeout = max(
            timeout,
            120,
        )

    common_kwargs = {
        "model": provider_model_name,
        "temperature": temperature,
        "timeout": effective_timeout,
        "max_retries": max_retries,
    }

    if max_tokens is not None:
        common_kwargs["max_tokens"] = max_tokens

    elif provider == "cloudflare":
        common_kwargs["max_tokens"] = 4096

    if reasoning_effort is not None:
        common_kwargs["reasoning_effort"] = (
            reasoning_effort
        )

    if provider == "groq":
        return ChatGroq(
            **common_kwargs,
        )

    if provider == "cloudflare":
        api_token = os.getenv(
            "CLOUDFLARE_API_TOKEN"
        )

        account_id = os.getenv(
            "CLOUDFLARE_ACCOUNT_ID"
        )

        if not api_token:
            raise ValueError(
                "CLOUDFLARE_API_TOKEN is missing."
            )

        if not account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID is missing."
            )

        base_url = (
            "https://api.cloudflare.com/client/v4/"
            f"accounts/{account_id}/ai/v1"
        )

        return ChatOpenAI(
            api_key=api_token,
            base_url=base_url,
            **common_kwargs,
        )

    raise ValueError(
        "Unsupported MODEL_PROVIDER: "
        f"{provider!r}"
    )