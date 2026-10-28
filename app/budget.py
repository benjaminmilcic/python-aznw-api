"""Daily budget of the chat in US dollars, for all visitors together.

LiteLLM prices every model call. The costs of the current day (UTC) are added up and kept in a small file in the state
directory, so a restart doesn't reset the count. Once the budget is used up, the chat answers with "daily_limit" until
midnight UTC. Protects against visitors who get around the per-IP limits of NestJS (many IPs, bots).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import DAILY_BUDGET_USD, STATE_DIR

log = logging.getLogger(__name__)


class DailyBudget:
    def __init__(self, path: Path, limit_usd: float):
        self.path = path
        # 0 or less switches the budget off
        self.limit_usd = limit_usd
        self._day: str | None = None
        self._spent = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _sync(self) -> None:
        """Loads the count from the file once, and starts from zero on a new day."""
        today = self._today()
        if self._day is None:
            try:
                data = json.loads(self.path.read_text())
                if data.get("day") == today:
                    self._spent = float(data.get("spentUsd", 0))
            except (OSError, ValueError):  # no file yet or unreadable – start from zero
                pass
            self._day = today
        elif self._day != today:
            self._day, self._spent = today, 0.0

    def spent(self) -> float:
        self._sync()
        return self._spent

    def exhausted(self) -> bool:
        return self.limit_usd > 0 and self.spent() >= self.limit_usd

    def add(self, cost: float | None) -> None:
        if not cost:
            return
        self._sync()
        self._spent += cost
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write and rename, so a crash never leaves a half-written file
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"day": self._day, "spentUsd": round(self._spent, 6)}))
            tmp.replace(self.path)
        except OSError as err:  # the count still works in memory
            log.warning("Could not save the daily budget: %s", err)

    @staticmethod
    def minutes_until_reset() -> int:
        now = datetime.now(timezone.utc)
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
        return max(1, int((midnight - now).total_seconds() // 60) + 1)


budget = DailyBudget(STATE_DIR / "daily-budget.json", DAILY_BUDGET_USD)
