import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


# ---------------------------------------------------------
# Test Schema
# ---------------------------------------------------------


class CloudflareTestResult(BaseModel):
    status: Literal["pass", "fail"] = Field(
        description=(
            "Whether the simple validation task passed."
        )
    )

    value: int = Field(
        description=(
            "The numerical answer to the supplied arithmetic task."
        )
    )


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------


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


# ---------------------------------------------------------
# Cloudflare Through LangChain
# ---------------------------------------------------------


base_url = (
    "https://api.cloudflare.com/client/v4/"
    f"accounts/{account_id}/ai/v1"
)


model = ChatOpenAI(
    model="@cf/openai/gpt-oss-120b",
    api_key=api_token,
    base_url=base_url,
    temperature=0,
    max_tokens=256,
    timeout=60,
    max_retries=0,
)


structured_model = model.with_structured_output(
    CloudflareTestResult,
    method="json_mode",
    include_raw=True,
)


# ---------------------------------------------------------
# One Tiny Structured Call
# ---------------------------------------------------------


result = structured_model.invoke(
    """
Calculate 6 * 7.

Return ONLY valid JSON using exactly these fields:

{
    "status": "pass",
    "value": 42
}

Rules:
- status must be either "pass" or "fail".
- value must be an integer.
"""
)


print(
    "Parsed type:",
    type(result["parsed"]).__name__,
)

print(
    "Parsed result:",
    result["parsed"],
)

print(
    "Parsing error:",
    result["parsing_error"],
)


raw = result["raw"]

print(
    "Raw response present:",
    raw is not None,
)