"""Central configuration: ``pydantic-settings`` settings + typed accessors.

Configuration is layered: environment variables (``FUTURESRLX_*``), an optional
``.env`` file, then the committed ``configs/*.yaml`` instrument/strategy masters.
Secrets are held as :class:`pydantic.SecretStr` so they can never leak into logs
or the telemetry API by accident.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.instruments import InstrumentsConfig, load_instruments

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIGS_DIR: Path = REPO_ROOT / "configs"

_IBKR_PAPER_PORTS: frozenset[int] = frozenset({7497, 4002})
_IBKR_LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})


class RiskSettings(BaseModel):
    """All quantitative and LLM risk parameters in one typed object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- ContractTranslator / margin -------------------------------------------------
    max_margin_utilization: float = Field(default=0.60, gt=0.0, le=1.0)
    margin_warning_threshold: float = Field(default=0.60, gt=0.0, le=1.0)
    margin_critical_threshold: float = Field(default=0.80, gt=0.0, le=1.0)
    max_gross_leverage: float = Field(default=3.0, gt=0.0)

    # -- Quant risk overlay ----------------------------------------------------------
    target_annual_volatility: float = Field(default=0.12, gt=0.0)
    covariance_lookback_days: int = Field(default=60, ge=5)
    ewma_halflife_days: float = Field(default=20.0, gt=0.0)
    max_single_instrument_weight: float = Field(default=0.50, gt=0.0)
    max_sector_weight: float = Field(default=0.40, gt=0.0)
    daily_max_loss_pct: float = Field(default=0.03, gt=0.0, le=1.0)
    trailing_stop_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    drawdown_limit_pct: float = Field(default=0.20, gt=0.0, le=1.0)

    # -- LLM Risk-First engine (Tier A) ----------------------------------------------
    llm_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    llm_semantic_variance_max: float = Field(default=0.25, gt=0.0)
    llm_sample_count: int = Field(default=3, ge=1, le=10)
    llm_safe_score: float = Field(default=1.5, ge=1.0, le=5.0)
    llm_critical_score: float = Field(default=4.5, ge=1.0, le=5.0)
    llm_risk_penalty_lambda: float = Field(default=0.5, ge=0.0)
    macro_lockout_minutes_before: int = Field(default=30, ge=0)
    macro_lockout_minutes_after: int = Field(default=30, ge=0)

    # -- OMS pre-trade gate ------------------------------------------------------------
    max_order_contracts: int = Field(default=200, gt=0)
    order_rate_limit_per_second: float = Field(default=5.0, gt=0.0)
    duplicate_order_window_seconds: float = Field(default=1.0, ge=0.0)
    reconciliation_interval_seconds: float = Field(default=30.0, gt=0.0)
    #: Grace window during which in-flight working orders may explain position drift.
    reconcile_grace_seconds: float = Field(default=30.0, ge=0.0)

    # -- LLM Risk-First async cache ---------------------------------------------------
    #: Time-to-live of cached LLM assessments (sync pipeline never blocks on LLM I/O).
    llm_cache_ttl_seconds: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> RiskSettings:
        if self.margin_critical_threshold <= self.margin_warning_threshold:
            raise ValueError("margin_critical_threshold must exceed margin_warning_threshold")
        if self.llm_critical_score <= self.llm_safe_score:
            raise ValueError("llm_critical_score must exceed llm_safe_score")
        return self


class Settings(BaseSettings):
    """Runtime settings for the full FuturesRL-X stack."""

    model_config = SettingsConfigDict(
        env_prefix="FUTURESRLX_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # -- application ---------------------------------------------------------------
    app_name: str = "FuturesRL-X"
    environment: Literal["development", "paper", "live", "backtest"] = "development"
    log_level: str = "INFO"
    repo_root: Path = Field(default=REPO_ROOT)
    configs_dir: Path = Field(default=CONFIGS_DIR)
    data_dir: Path = Field(default=REPO_ROOT / "data")
    parquet_dir: Path = Field(default=REPO_ROOT / "data" / "parquet")
    duckdb_path: Path = Field(default=REPO_ROOT / "data" / "futuresrl_x.duckdb")
    report_dir: Path = Field(default=REPO_ROOT / "artifacts")
    instruments_path: Path | None = None

    # -- Databento -----------------------------------------------------------------
    databento_api_key: SecretStr | None = None
    databento_dataset: str = "GLBX.MDP3"
    databento_default_schema: str = "ohlcv-1d"
    databento_live_enabled: bool = False

    # -- IBKR ----------------------------------------------------------------------
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 42
    ibkr_account: str | None = None
    ibkr_paper: bool = True
    ibkr_connect_timeout_seconds: float = 10.0
    ibkr_reconnect_initial_delay_seconds: float = 1.0
    ibkr_reconnect_max_delay_seconds: float = 60.0
    ibkr_reconnect_max_attempts: int = 10
    ibkr_heartbeat_interval_seconds: float = 5.0
    ibkr_nightly_reset_start_hour_et: int = 23
    ibkr_nightly_reset_end_hour_et: int = 0

    # -- API / telemetry -----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")
    telemetry_interval_seconds: float = Field(default=1.0, gt=0.0)

    # -- Phase-7 production hardening -------------------------------------------------
    #: IBKR account base currency; ``MarketState.equity`` is denominated in it.
    account_currency: str = "USD"
    #: ``currency -> USD per one unit`` rates for non-USD base accounts.
    fx_rates: dict[str, float] = Field(default_factory=dict)
    #: Durable risk/kill-switch state backend: "" / "memory:" / file path / "duckdb:" / "redis:".
    risk_state_store_uri: str = ""
    #: Historical lookback window used to warm indicators before the first live bar.
    hydration_lookback_days: int = Field(default=400, ge=260)
    #: Allow the cold-start hydrator to backfill missing history (Databento/IBKR).
    hydration_backfill_enabled: bool = True

    # -- risk ------------------------------------------------------------------------
    risk: RiskSettings = Field(default_factory=RiskSettings)

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.strip().upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @field_validator("account_currency")
    @classmethod
    def _valid_account_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if len(cleaned) != 3 or not cleaned.isalpha():
            raise ValueError(f"account_currency must be a 3-letter ISO-4217 code, got {value!r}")
        return cleaned

    @field_validator("fx_rates")
    @classmethod
    def _valid_fx_rates(cls, rates: dict[str, float]) -> dict[str, float]:
        checked: dict[str, float] = {}
        for currency, rate in rates.items():
            code = currency.strip().upper()
            if len(code) != 3 or not code.isalpha():
                raise ValueError(f"fx_rates key must be a 3-letter code, got {currency!r}")
            if not isinstance(rate, (int, float)) or isinstance(rate, bool):
                raise ValueError(f"fx_rates[{code!r}] must be a number")
            if not (rate == rate) or rate <= 0.0 or rate == float("inf"):
                raise ValueError(f"fx_rates[{code!r}] must be finite and strictly positive")
            if code == "USD" and float(rate) != 1.0:
                raise ValueError("fx_rates['USD'] must be exactly 1.0")
            checked[code] = float(rate)
        return checked

    @field_validator("databento_api_key", mode="before")
    @classmethod
    def _secret_or_none(cls, value: object) -> object:
        if value in (None, "", "none", "null"):
            return None
        return value

    @model_validator(mode="after")
    def _ibkr_port_matches_environment(self) -> Settings:
        if self.environment == "live":
            if self.ibkr_port not in _IBKR_LIVE_PORTS:
                raise ValueError(
                    f"live environment requires an IBKR live port {sorted(_IBKR_LIVE_PORTS)}, "
                    f"got {self.ibkr_port}"
                )
            if self.ibkr_paper:
                raise ValueError("live environment requires ibkr_paper=False")
        elif self.environment == "paper":
            if self.ibkr_port not in _IBKR_PAPER_PORTS:
                raise ValueError(
                    f"paper environment requires an IBKR paper port {sorted(_IBKR_PAPER_PORTS)}, "
                    f"got {self.ibkr_port}"
                )
            if not self.ibkr_paper:
                raise ValueError("paper environment requires ibkr_paper=True")
        if self.ibkr_reconnect_max_delay_seconds < self.ibkr_reconnect_initial_delay_seconds:
            raise ValueError("ibkr_reconnect_max_delay must be >= initial delay")
        return self

    # -- derived accessors -----------------------------------------------------------

    @property
    def is_live(self) -> bool:
        """Whether the session is configured for live capital."""
        return self.environment == "live"

    @property
    def instruments(self) -> InstrumentsConfig:
        """Load the instrument master configured for this session."""
        path = self.instruments_path or (self.configs_dir / "instruments.yaml")
        return load_instruments(path)

    @property
    def strategies_dir(self) -> Path:
        """Directory containing declarative strategy YAML files."""
        return self.configs_dir / "strategies"

    @property
    def api_base_url(self) -> str:
        """Base URL for the local telemetry API."""
        host = "127.0.0.1" if self.api_host in {"0.0.0.0", "::"} else self.api_host
        return f"http://{host}:{self.api_port}"

    @property
    def databento_api_key_value(self) -> str | None:
        """Plain API key value when configured (never logged)."""
        return self.databento_api_key.get_secret_value() if self.databento_api_key else None

    def ensure_directories(self) -> None:
        """Create the writable data/report directories if they are missing."""
        for path in (self.data_dir, self.parquet_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset: str | None = None) -> Path:
        """Root of the partitioned Parquet cache for a Databento dataset."""
        return self.parquet_dir / (dataset or self.databento_dataset)

    def model_dump_redacted(self) -> dict[str, object]:
        """Dump settings with secrets redacted (safe for logs/telemetry)."""
        dumped = self.model_dump(mode="json")
        dumped["databento_api_key"] = "***" if self.databento_api_key else None
        return dumped


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (tests / hot configuration reload)."""
    get_settings.cache_clear()
