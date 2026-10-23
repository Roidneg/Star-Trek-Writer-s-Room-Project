import math
import random
import re
import time
from typing import Any

import groq
import openai


# ---------------------------------------------------------
# Rate Limit Helpers
# ---------------------------------------------------------


def _get_retry_after_seconds(
    error: Any,
) -> float | None:
    """
    Read Groq's retry-after header from a 429 response.

    Falls back to parsing Groq's error message if the
    response header is unavailable through a wrapper.
    """

    response = getattr(error, "response", None)

    if response is not None:
        headers = getattr(response, "headers", None)

        if headers:
            retry_after = headers.get("retry-after")

            if retry_after is not None:
                try:
                    seconds = float(retry_after)
                    if math.isfinite(seconds) and seconds >= 0:
                        return seconds
                except (TypeError, ValueError):
                    pass


    # -----------------------------------------------------
    # Fallback:
    # Parse messages such as:
    #
    # "Please try again in 2.1675s"
    # "Please try again in 10m5.232s"
    # -----------------------------------------------------

    message = str(error)

    match = re.search(
        r"Please try again in "
        r"(?:(\d+(?:\.\d+)?)m)?"
        r"(\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE,
    )

    if not match:
        return None

    minutes_text = match.group(1)
    seconds_text = match.group(2)

    minutes = (
        float(minutes_text)
        if minutes_text is not None
        else 0.0
    )

    seconds = float(seconds_text)

    return (minutes * 60.0) + seconds

def _is_daily_quota_exhausted(
    error: Any,
) -> bool:
    """
    Return True when a provider reports a daily quota
    that cannot recover through a short automatic retry.
    """

    message = str(error).casefold()

    return (
        "daily free allocation" in message
        and "neurons" in message
    )


# ---------------------------------------------------------
# Safe LLM Invocation
# ---------------------------------------------------------


def safe_invoke(
    runnable: Any,
    input_data: Any,
    *,
    label: str = "LLM call",
    max_retries: int = 6,
    max_wait_seconds: float = 60.0,
    max_total_wait_seconds: float = 120.0,
) -> Any:
    """
    Invoke a LangChain runnable while handling short Groq
    rate-limit interruptions.

    Behavior:
    - Runs normally when no rate limit occurs.
    - Catches Groq HTTP 429 RateLimitError.
    - Uses Groq's retry-after value when available.
    - Adds a small jitter buffer.
    - Retries short rate-limit windows automatically.
    - Refuses to silently wait through long quota resets.
    - Allows at most six retries by default.
    - Caps cumulative scheduled sleep at 120 seconds per invocation.
      This excludes time spent inside runnable.invoke().
    - Re-raises the original exception after retries fail.
    """

    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    for name, value in (("max_wait_seconds", max_wait_seconds),
                        ("max_total_wait_seconds", max_total_wait_seconds)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    retries_used = 0
    total_wait_seconds = 0.0

    while True:

        try:
            return runnable.invoke(input_data)


        except (

                groq.RateLimitError,

                openai.RateLimitError,

        ) as error:

            if _is_daily_quota_exhausted(error):
                print(
                    f"\n[Rate Limit] {label}: "
                    "daily provider allocation exhausted."
                )

                print(
                    "[Rate Limit] This condition cannot recover "
                    "through a short automatic retry."
                )

                print(
                    "[Rate Limit] Stopping immediately without "
                    "sleeping or retrying."
                )

                raise

            # ---------------------------------------------
            # No retries remaining
            # ---------------------------------------------

            if retries_used >= max_retries:
                print(
                    f"\n[Rate Limit] {label}: "
                    "maximum automatic retries reached."
                )
                raise


            # ---------------------------------------------
            # Determine Groq's requested wait
            # ---------------------------------------------

            retry_after = _get_retry_after_seconds(error)

            if retry_after is None:

                # Exponential fallback:
                # 2 sec, 4 sec, 8 sec...
                wait_seconds = min(
                    2 ** (retries_used + 1),
                    max_wait_seconds,
                )

            else:
                wait_seconds = retry_after


            # ---------------------------------------------
            # Add small jitter so we do not retry exactly
            # on the rate-limit boundary
            # ---------------------------------------------

            wait_seconds += random.uniform(
                0.25,
                0.75,
            )


            # ---------------------------------------------
            # Do not unexpectedly freeze the application
            # for a large quota reset
            # ---------------------------------------------

            if wait_seconds > max_wait_seconds:

                print(
                    f"\n[Rate Limit] {label}: "
                    f"Provider requested approximately "
                    f"{wait_seconds:.1f} seconds before retrying."
                )

                print(
                    "[Rate Limit] Wait exceeds the "
                    f"{max_wait_seconds:.0f}-second automatic "
                    "retry limit."
                )

                print(
                    "[Rate Limit] The workflow will stop "
                    "instead of waiting silently."
                )

                raise


            # ---------------------------------------------
            # Retry
            # ---------------------------------------------

            if total_wait_seconds + wait_seconds > max_total_wait_seconds:
                print(
                    f"\n[Rate Limit] {label}: next wait would exceed the "
                    f"{max_total_wait_seconds:.0f}-second cumulative sleep budget. "
                    "The workflow will stop."
                )
                raise

            retries_used += 1
            total_wait_seconds += wait_seconds

            print(
                f"\n[Rate Limit] {label}: "
                f"waiting {wait_seconds:.2f} seconds."
            )

            print(
                f"[Rate Limit] Automatic retry "
                f"{retries_used}/{max_retries}."
            )

            time.sleep(wait_seconds)