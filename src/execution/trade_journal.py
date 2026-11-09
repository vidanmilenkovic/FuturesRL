"""Append-only local journal for paper/live benchmark runs.

The runner writes two plain CSV files under ``artifacts/live_trades/`` by default:

``trades.csv``
    One row per broker order acknowledgement (filled or rejected), with the
    order intent, fill price/quantity, commission and the account snapshot at
    that moment.

``equity.csv``
    One row per processed rebalance state with net liquidation value, cash,
    buying power, margin usage and excess liquidity. This is the benchmark
    equity curve.

The files are deliberately human-readable so they can be opened in Excel,
pandas, DuckDB or any charting tool, and are served read-only through
``GET /api/runner/trade-journal``. No synthetic values are written: fields the
broker does not report stay blank.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.types import OrderReceipt, ensure_utc

if TYPE_CHECKING:
    from configs.settings import Settings

__all__ = [
    "EQUITY_COLUMNS",
    "TRADE_COLUMNS",
    "TRADE_JOURNAL_DIR_ENV",
    "TradeJournal",
]

logger = logging.getLogger(__name__)

TRADE_JOURNAL_DIR_ENV: str = "FUTURESRL_TRADE_JOURNAL_DIR"

TRADE_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "mode",
    "account_id",
    "order_id",
    "instrument_id",
    "side",
    "requested_quantity",
    "filled_quantity",
    "average_price",
    "commission",
    "status",
    "message",
    "contract_symbol",
    "expiration",
    "net_liquidation",
    "cash_balance",
    "buying_power",
    "initial_margin_used",
    "maintenance_margin_used",
    "excess_liquidity",
    "margin_utilization",
)

EQUITY_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "mode",
    "account_id",
    "net_liquidation",
    "cash_balance",
    "buying_power",
    "initial_margin_used",
    "maintenance_margin_used",
    "excess_liquidity",
    "margin_utilization",
    "daily_pnl",
    "cumulative_pnl",
)

_NUMERIC_TRADE_FIELDS = {
    "requested_quantity",
    "filled_quantity",
    "average_price",
    "commission",
    "net_liquidation",
    "cash_balance",
    "buying_power",
    "initial_margin_used",
    "maintenance_margin_used",
    "excess_liquidity",
    "margin_utilization",
}
_NUMERIC_EQUITY_FIELDS = {
    "net_liquidation",
    "cash_balance",
    "buying_power",
    "initial_margin_used",
    "maintenance_margin_used",
    "excess_liquidity",
    "margin_utilization",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _format_number(value: object) -> str:
    number = _number(value)
    if number is None:
        return ""
    return f"{number:.10g}"


class TradeJournal:
    """Thread-safe append-only CSV journal for live/paper execution."""

    def __init__(
        self,
        directory: Path,
        *,
        mode: str,
        account_id: str | None = None,
        clock: Any = None,
    ) -> None:
        self.directory = Path(directory)
        self.mode = mode
        self.account_id = account_id
        self._clock = clock if callable(clock) else _utc_now
        self._lock = threading.Lock()
        self._baseline_loaded = False
        self._first_equity: float | None = None
        self._day: str | None = None
        self._day_start_equity: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> TradeJournal:
        """Build the journal from runtime settings (env override wins)."""
        raw = os.environ.get(TRADE_JOURNAL_DIR_ENV, "").strip()
        directory = Path(raw) if raw else Path(settings.report_dir) / "live_trades"
        return cls(directory, mode=settings.environment, account_id=settings.ibkr_account or None)

    @property
    def trades_path(self) -> Path:
        """CSV path containing one row per order acknowledgement."""
        return self.directory / "trades.csv"

    @property
    def equity_path(self) -> Path:
        """CSV path containing the account equity curve."""
        return self.directory / "equity.csv"

    def ensure_files(self) -> None:
        """Create header-only CSV files so an operator can locate them immediately."""
        for path, columns in (
            (self.trades_path, TRADE_COLUMNS),
            (self.equity_path, EQUITY_COLUMNS),
        ):
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size > 0:
                    continue
                with self._lock, path.open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
            except OSError as exc:
                logger.error("could not initialise trade journal %s: %s", path, exc)

    def _append(self, path: Path, columns: tuple[str, ...], row: Mapping[str, object]) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock:
                write_header = not path.exists() or path.stat().st_size == 0
                with path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
                    if write_header:
                        writer.writeheader()
                    writer.writerow({column: row.get(column, "") for column in columns})
        except OSError as exc:
            # Journal failures must never abort trading; surface them in logs.
            logger.error("could not append trade journal %s: %s", path, exc)

    def record_trade(
        self,
        receipt: OrderReceipt,
        *,
        account: Mapping[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append one order receipt together with the current account snapshot."""
        moment = ensure_utc(timestamp) if timestamp is not None else ensure_utc(self._clock())
        account = account or {}
        expiration = receipt.expiration.isoformat() if receipt.expiration is not None else ""
        row: dict[str, object] = {
            "timestamp_utc": moment.isoformat(),
            "mode": self.mode,
            "account_id": self.account_id or "",
            "order_id": receipt.order_id,
            "instrument_id": receipt.instrument_id or "",
            "side": receipt.side or "",
            "requested_quantity": _format_number(receipt.requested_quantity),
            "filled_quantity": receipt.filled_quantity,
            "average_price": _format_number(receipt.average_price),
            "commission": _format_number(receipt.commission),
            "status": receipt.status.value,
            "message": receipt.message,
            "contract_symbol": receipt.contract_symbol or "",
            "expiration": expiration,
            "net_liquidation": _format_number(account.get("net_liquidation")),
            "cash_balance": _format_number(account.get("cash_balance")),
            "buying_power": _format_number(account.get("buying_power")),
            "initial_margin_used": _format_number(account.get("initial_margin_used")),
            "maintenance_margin_used": _format_number(account.get("maintenance_margin_used")),
            "excess_liquidity": _format_number(account.get("excess_liquidity")),
            "margin_utilization": _format_number(account.get("margin_utilization")),
        }
        self._append(self.trades_path, TRADE_COLUMNS, row)

    def record_account(
        self,
        account: Mapping[str, object] | None,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Append one equity/account snapshot (the benchmark curve)."""
        if not account:
            return
        moment = ensure_utc(timestamp) if timestamp is not None else ensure_utc(self._clock())
        nlv = _number(account.get("net_liquidation"))
        self._load_equity_baseline()
        day = moment.date().isoformat()
        if nlv is not None:
            if self._first_equity is None:
                self._first_equity = nlv
            if day != self._day:
                self._day = day
                self._day_start_equity = nlv
        daily_pnl = (
            nlv - self._day_start_equity
            if nlv is not None and self._day_start_equity is not None
            else None
        )
        cumulative_pnl = (
            nlv - self._first_equity if nlv is not None and self._first_equity is not None else None
        )
        row: dict[str, object] = {
            "timestamp_utc": moment.isoformat(),
            "mode": self.mode,
            "account_id": self.account_id or "",
            "net_liquidation": _format_number(nlv),
            "cash_balance": _format_number(account.get("cash_balance")),
            "buying_power": _format_number(account.get("buying_power")),
            "initial_margin_used": _format_number(account.get("initial_margin_used")),
            "maintenance_margin_used": _format_number(account.get("maintenance_margin_used")),
            "excess_liquidity": _format_number(account.get("excess_liquidity")),
            "margin_utilization": _format_number(account.get("margin_utilization")),
            "daily_pnl": _format_number(daily_pnl),
            "cumulative_pnl": _format_number(cumulative_pnl),
        }
        self._append(self.equity_path, EQUITY_COLUMNS, row)

    def _load_equity_baseline(self) -> None:
        """Seed first-equity/day baselines from an existing journal file once."""
        if self._baseline_loaded:
            return
        self._baseline_loaded = True
        rows = self._read(self.equity_path, _NUMERIC_EQUITY_FIELDS)
        if not rows:
            return
        first = _number(rows[0].get("net_liquidation"))
        if first is not None:
            self._first_equity = first
        last = rows[-1]
        last_nlv = _number(last.get("net_liquidation"))
        last_stamp = str(last.get("timestamp_utc") or "")
        if last_nlv is not None:
            self._day = last_stamp[:10] or None
            # Day start is the first row on the latest journal date.
            for row in rows:
                if str(row.get("timestamp_utc") or "")[:10] == self._day:
                    start = _number(row.get("net_liquidation"))
                    self._day_start_equity = start
                    break

    def _read(self, path: Path, numeric: set[str]) -> list[dict[str, object]]:
        if not path.is_file():
            return []
        rows: list[dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    row: dict[str, object] = {}
                    for key, value in raw.items():
                        if key is None:
                            continue
                        if key in numeric:
                            row[key] = _number(value)
                        else:
                            row[key] = value if value not in (None, "") else None
                    rows.append(row)
        except (OSError, csv.Error) as exc:
            logger.error("could not read trade journal %s: %s", path, exc)
            return []
        return rows

    def snapshot(self, *, limit: int = 500) -> dict[str, object]:
        """Return recent trades, the equity curve and a benchmark summary."""
        trades = self._read(self.trades_path, _NUMERIC_TRADE_FIELDS)
        equity = self._read(self.equity_path, _NUMERIC_EQUITY_FIELDS)
        if limit > 0:
            trades = trades[-limit:]
        first_equity = _number(equity[0].get("net_liquidation")) if equity else None
        latest_equity = _number(equity[-1].get("net_liquidation")) if equity else None
        total_pnl = (
            latest_equity - first_equity
            if latest_equity is not None and first_equity is not None
            else None
        )
        total_return = total_pnl / first_equity if total_pnl is not None and first_equity else None
        peak: float | None = None
        max_drawdown = 0.0
        day_start: dict[str, float] = {}
        for row in equity:
            nlv = _number(row.get("net_liquidation"))
            if nlv is None:
                row["daily_pnl"] = None
                row["cumulative_pnl"] = None
                continue
            peak = nlv if peak is None else max(peak, nlv)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - nlv) / peak)
            stamp = str(row.get("timestamp_utc") or "")
            day = stamp[:10]
            day_start.setdefault(day, nlv)
            row["daily_pnl"] = nlv - day_start[day]
            row["cumulative_pnl"] = nlv - first_equity if first_equity is not None else None
        filled = sum(1 for row in trades if row.get("status") == "FILLED")
        return {
            "enabled": True,
            "directory": str(self.directory),
            "trades_path": str(self.trades_path),
            "equity_path": str(self.equity_path),
            "trades": trades,
            "equity": equity,
            "summary": {
                "starting_equity": first_equity,
                "latest_equity": latest_equity,
                "total_pnl": total_pnl,
                "total_return": total_return,
                "max_drawdown": max_drawdown if equity else None,
                "trade_rows": len(trades),
                "filled_trades": filled,
                "equity_points": len(equity),
                "last_trade_at": trades[-1].get("timestamp_utc") if trades else None,
            },
        }
