"""Summarizing the conversation history: once when it gets too big, then continue with the summary."""

from app import chat
from app.chat import HISTORY_TOKEN_BUDGET, Message, fit_history, history_tokens


def question_and_answer(i: int) -> list[Message]:
    # about the size of a typical question and answer (~370 tokens)
    return [
        Message(role="user", content=f"Frage {i}: Was hat Benjamin bei Codext gemacht?"),
        Message(role="assistant", content=f"Antwort {i}: " + "Benjamin arbeitet mit Angular und Ionic. " * 45),
    ]


async def fake_summarize(messages):
    return "- The visitor asked about Benjamin's work at Codext.", type("R", (), {"usage": None})()


async def test_summarizes_only_when_the_budget_is_exceeded_again(monkeypatch):
    monkeypatch.setattr(chat, "summarize", fake_summarize)
    monkeypatch.setattr(chat, "usage_info", lambda response: {})

    history: list[Message] = []
    summarized_at = []
    for turn in range(1, 21):
        compacted, summary = await fit_history(history)
        if summary:
            summarized_at.append(turn)
            # like NestJS: the summary replaces the older messages, the conversation continues with it
            history = [Message(role="system", content=summary["content"])] + history[summary["replacedMessages"] :]
            assert history == compacted
            # clearly below the budget afterwards, so the next question does not summarize again
            assert history_tokens(history) < HISTORY_TOKEN_BUDGET / 2
        history += question_and_answer(turn)

    assert summarized_at, "a long conversation is summarized at some point"
    gaps = [b - a for a, b in zip(summarized_at, summarized_at[1:])]
    assert all(gap >= 3 for gap in gaps), f"summarized too often: {summarized_at}"


async def test_short_history_is_left_alone():
    history = question_and_answer(1)
    compacted, summary = await fit_history(history)
    assert summary is None and compacted == history


def test_waiting_times_and_daily_limit():
    import litellm

    from app.chat import error_event, is_daily_limit, retry_after_seconds

    def rate_limit(text):
        return litellm.RateLimitError(message=text, llm_provider="openai", model="gpt-5.4-mini")

    minute = rate_limit("Rate limit reached on tokens per min (TPM): Limit 100000. Please try again in 7.15s.")
    day = rate_limit("Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 22m10.56s.")
    assert retry_after_seconds(minute) == 7.15
    assert retry_after_seconds(day) == 22 * 60 + 10.56
    assert retry_after_seconds(rate_limit("Please try again in 1h2m3s")) == 3723
    # OpenAI names short waits in milliseconds – not to be read as 120 minutes
    assert retry_after_seconds(rate_limit("Please try again in 120ms.")) == 0.12
    assert not is_daily_limit(minute) and is_daily_limit(day)
    assert error_event(minute)["code"] == "rate_limit"
    assert error_event(day) == {"error": "The daily budget of the chat is used up.", "code": "daily_limit", "retryAfterMinutes": 23}


def test_used_up_credit_is_an_error_not_a_limit():
    import litellm

    from app.chat import error_event

    err = litellm.RateLimitError(
        message="You exceeded your current quota. {'code': 'insufficient_quota'}", llm_provider="openai", model="gpt-5.4-mini"
    )
    assert error_event(err)["code"] == "error"
