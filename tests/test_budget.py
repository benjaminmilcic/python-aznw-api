"""Daily budget: counts the costs of the model calls, survives restarts and stops the chat once it is used up."""

import json

from app import budget as budget_module
from app import chat
from app.budget import DailyBudget
from app.chat import ChatRequest, stream_answer


def test_counts_costs_and_survives_a_restart(tmp_path):
    path = tmp_path / "daily-budget.json"
    budget = DailyBudget(path, 2.0)
    budget.add(1.5)
    budget.add(None)  # model without a price
    assert not budget.exhausted()
    budget.add(0.6)
    assert budget.exhausted()

    restarted = DailyBudget(path, 2.0)
    assert restarted.spent() == 2.1 and restarted.exhausted()


def test_a_new_day_starts_from_zero(tmp_path):
    path = tmp_path / "daily-budget.json"
    path.write_text(json.dumps({"day": "2020-01-01", "spentUsd": 5}))
    budget = DailyBudget(path, 2.0)
    assert budget.spent() == 0 and not budget.exhausted()

    budget.add(1.0)
    budget._day = "2020-01-01"  # as if the service had been running since yesterday
    assert budget.spent() == 0


def test_zero_switches_the_budget_off(tmp_path):
    budget = DailyBudget(tmp_path / "daily-budget.json", 0)
    budget.add(100)
    assert not budget.exhausted()


def test_minutes_until_reset_are_within_a_day():
    assert 1 <= DailyBudget.minutes_until_reset() <= 24 * 60


async def test_chat_stops_with_daily_limit_when_the_budget_is_used_up(tmp_path, monkeypatch):
    used_up = DailyBudget(tmp_path / "daily-budget.json", 1.0)
    used_up.add(1.0)
    monkeypatch.setattr(chat, "budget", used_up)

    async def no_model_call(**kwargs):
        raise AssertionError("no model call once the budget is used up")

    monkeypatch.setattr(chat.litellm, "acompletion", no_model_call)
    events = [e async for e in stream_answer(ChatRequest(message="Hallo"))]
    assert len(events) == 1
    event = json.loads(events[0].removeprefix("data: "))
    assert event["code"] == "daily_limit" and event["retryAfterMinutes"] >= 1


async def test_model_calls_use_the_configured_settings_and_are_counted(tmp_path, monkeypatch):
    budget = DailyBudget(tmp_path / "daily-budget.json", 2.0)
    monkeypatch.setattr(chat, "budget", budget)
    # Independent of LLM_MODEL in a local .env
    monkeypatch.setattr(chat, "LLM_MODEL", "openai/gpt-5.4-mini")
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        message = type("M", (), {"content": "- summary"})()
        usage = type("U", (), {"prompt_tokens": 10_000, "completion_tokens": 1_000, "total_tokens": 11_000})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()], "usage": usage})()

    monkeypatch.setattr(chat.litellm, "acompletion", fake_completion)
    text, response = await chat.summarize([chat.Message(role="user", content="Hallo")])
    info = chat.track(response)

    assert calls[0]["model"] == chat.LLM_MODEL
    assert calls[0]["reasoning_effort"] == chat.REASONING_EFFORT
    assert calls[0]["max_completion_tokens"] == chat.MAX_COMPLETION_TOKENS
    # GPT-5 models with reasoning reject any temperature but the default
    assert "temperature" not in calls[0]
    # 10,000 input and 1,000 output tokens of GPT-5.4 mini: 0.0075 + 0.0045 US dollars
    assert round(info["cost"], 6) == 0.012
    assert round(budget.spent(), 6) == 0.012


def test_the_default_budget_lives_in_the_state_directory():
    assert budget_module.budget.path.parent == budget_module.STATE_DIR
