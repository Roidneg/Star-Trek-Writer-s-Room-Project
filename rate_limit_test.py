import groq
import httpx
import openai
import tools.rate_limit as rate_limit_module

from tools.rate_limit import safe_invoke

class FakeDailyQuotaRunnable:

    def __init__(self):
        self.invoke_count = 0

    def invoke(self, input_data):
        self.invoke_count += 1

        request = httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/"
            "accounts/test/ai/v1/chat/completions",
        )

        response = httpx.Response(
            status_code=429,
            request=request,
        )

        raise openai.RateLimitError(
            (
                "you have used up your daily free "
                "allocation of 10,000 neurons"
            ),
            response=response,
            body={
                "errors": [
                    {
                        "message": (
                            "you have used up your daily free "
                            "allocation of 10,000 neurons"
                        ),
                        "code": 4006,
                    }
                ]
            },
        )

class FakeLongWaitRunnable:

    def invoke(self, input_data):

        print("Fake long-wait runnable called.")

        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        )

        response = httpx.Response(
            status_code=429,
            request=request,
            headers={
                "retry-after": "120",
            },
        )

        raise groq.RateLimitError(
            "Synthetic long rate limit.",
            response=response,
            body={
                "error": {
                    "message": (
                        "Synthetic long rate limit."
                    )
                }
            },
        )


def test_daily_quota_fails_immediately():
    fake_runnable = FakeDailyQuotaRunnable()

    sleep_count = 0
    original_sleep = rate_limit_module.time.sleep

    def fake_sleep(seconds):
        nonlocal sleep_count
        sleep_count += 1

    rate_limit_module.time.sleep = fake_sleep

    try:
        try:
            safe_invoke(
                fake_runnable,
                {
                    "message": "Daily quota test."
                },
                label="Cloudflare daily quota test",
                max_retries=6,
            )

        except openai.RateLimitError:
            pass

        else:
            raise AssertionError(
                "Expected openai.RateLimitError."
            )

    finally:
        rate_limit_module.time.sleep = original_sleep

    assert fake_runnable.invoke_count == 1, (
        "Daily quota exhaustion must not retry."
    )

    assert sleep_count == 0, (
        "Daily quota exhaustion must not sleep."
    )

    print(
        "Cloudflare daily quota fail-fast: PASS"
    )

if __name__ == "__main__":

    test_daily_quota_fails_immediately()
    fake_runnable = FakeLongWaitRunnable()

    try:

        safe_invoke(
            fake_runnable,
            {
                "message": "Test long wait."
            },
            label="Long-wait wrapper test",
            max_wait_seconds=30,
        )

    except groq.RateLimitError:

        print("\n--- Expected Result ---")
        print(
            "Long rate limit was correctly rejected."
        )