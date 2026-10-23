import os

import requests
from dotenv import load_dotenv


load_dotenv()


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


url = (
    "https://api.cloudflare.com/client/v4/"
    f"accounts/{account_id}/ai/v1/chat/completions"
)


headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
}


payload = {
    "model": "@cf/openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": (
                "Reply with exactly: "
                "CLOUDFLARE TEST PASS"
            ),
        }
    ],
    "max_tokens": 128,
    "temperature": 0,
}


response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60,
)


print(
    "HTTP status:",
    response.status_code,
)


if not response.ok:
    print(
        "Response:",
        response.text,
    )

    response.raise_for_status()


data = response.json()

choice = data["choices"][0]
message = choice["message"]

print(
    "Finish reason:",
    choice.get("finish_reason"),
)

print(
    "Model response:",
    message.get("content"),
)

print(
    "Reasoning present:",
    bool(message.get("reasoning_content")),
)

print(
    "Usage:",
    data.get("usage"),
)