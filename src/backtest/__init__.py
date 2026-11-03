"""Futures-native backtesting: costs, margin, walk-forward, broker and engine.

Public API
----------
* :class:`BacktestEngine` -- unified vectorised/event-driven runner returning a
  typed :class:`BacktestResult`.
* :class:`EventBacktester` -- bar-by-bar simulation through :class:`SimulatedBroker`.
* :class:`CostModel` -- deterministic tick-aware spread/impact/commission/fee model.
* :class:`MarginSimulator` -- daily MTM settlement and forced-liquidation logic.
* :class:`PurgedWalkForwardSplit` -- leakage-free walk-forward folds.
* :func:`run_from_config` / :func:`main` -- declarative runner and offline CLI.
* :func:`run_walk_forward` / :class:`WalkForwardReport` -- purged/embargoed
  walk-forward validation API for the Backtest Lab (same JSON as the CLI).

The heavyweight engine exports are resolved lazily through PEP 562
``__getattr__``. This keeps ``import src.backtest`` cheap (no cost/engine
side-effects) and, critically, stops the package ``__init__`` from pre-importing
``src.backtest.engine``. Without that, ``python -m src.backtest.engine`` triggers
runpy's ``<frozen runpy>:128: RuntimeWarning: 'src.backtest.engine' found in
sys.modules ...`` warning before the module executes. Non-engine exports (cost
model, margin simulator, broker, walk-forward splitter) stay eager because they
are lightweight and dependency-free.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from src.backtest.cost_model import (
    ASSET_CLASS_FEE_SCHEDULES,
    DEFAULT_FEE_SCHEDULE,
    EXCHANGE_FEE_SCHEDULES,
    CostBreakdown,
    CostModel,
    FeeSchedule,
    RollCostBreakdown,
    SlippageModel,
    default_fee_schedule,
)
from src.backtest.margin_simulator import (
    ForcedLiquidationPlan,
    MarginCall,
    MarginRequirements,
    MarginSimulator,
    SettlementResult,
)
from src.backtest.simulated_broker import (
    BarQuote,
    FillPriceMode,
    FillRecord,
    SimulatedBroker,
    TradeKind,
)
from src.backtest.walk_forward import PurgedWalkForwardSplit, WalkForwardFold

if TYPE_CHECKING:  # pragma: no cover - static typing surface only
    from src.backtest.engine import (
        EQUITY_CURVE_SCHEMA,
        WEIGHT_HISTORY_SCHEMA,
        AccountingIdentityError,
        BacktestEngine,
        BacktestMode,
        BacktestResult,
        EventBacktester,
        ForcedLiquidationEvent,
        OfflineMomentumPipeline,
        RebalanceSchedule,
        RollExecution,
        SimpleContractTranslator,
        StateBuilder,
        TradeRecord,
        WalkForwardReport,
        build_states_from_bars,
        default_state_builder,
        main,
        run_from_config,
        run_walk_forward,
    )

#: Mapping of lazily-exported engine names to their defining module.
_ENGINE_EXPORTS: dict[str, str] = {
    "EQUITY_CURVE_SCHEMA": "src.backtest.engine",
    "WEIGHT_HISTORY_SCHEMA": "src.backtest.engine",
    "AccountingIdentityError": "src.backtest.engine",
    "BacktestEngine": "src.backtest.engine",
    "BacktestMode": "src.backtest.engine",
    "BacktestResult": "src.backtest.engine",
    "EventBacktester": "src.backtest.engine",
    "ForcedLiquidationEvent": "src.backtest.engine",
    "OfflineMomentumPipeline": "src.backtest.engine",
    "RebalanceSchedule": "src.backtest.engine",
    "RollExecution": "src.backtest.engine",
    "SimpleContractTranslator": "src.backtest.engine",
    "StateBuilder": "src.backtest.engine",
    "TradeRecord": "src.backtest.engine",
    "WalkForwardReport": "src.backtest.engine",
    "build_states_from_bars": "src.backtest.engine",
    "default_state_builder": "src.backtest.engine",
    "main": "src.backtest.engine",
    "run_from_config": "src.backtest.engine",
    "run_walk_forward": "src.backtest.engine",
}

__all__ = [
    "ASSET_CLASS_FEE_SCHEDULES",
    "DEFAULT_FEE_SCHEDULE",
    "EQUITY_CURVE_SCHEMA",
    "EXCHANGE_FEE_SCHEDULES",
    "WEIGHT_HISTORY_SCHEMA",
    "AccountingIdentityError",
    "BacktestEngine",
    "BacktestMode",
    "BacktestResult",
    "BarQuote",
    "CostBreakdown",
    "CostModel",
    "EventBacktester",
    "FeeSchedule",
    "FillPriceMode",
    "FillRecord",
    "ForcedLiquidationEvent",
    "ForcedLiquidationPlan",
    "MarginCall",
    "MarginRequirements",
    "MarginSimulator",
    "OfflineMomentumPipeline",
    "PurgedWalkForwardSplit",
    "RebalanceSchedule",
    "RollCostBreakdown",
    "RollExecution",
    "SettlementResult",
    "SimpleContractTranslator",
    "SimulatedBroker",
    "SlippageModel",
    "StateBuilder",
    "TradeKind",
    "TradeRecord",
    "WalkForwardFold",
    "WalkForwardReport",
    "build_states_from_bars",
    "default_fee_schedule",
    "default_state_builder",
    "main",
    "run_from_config",
    "run_walk_forward",
]


def __getattr__(name: str) -> object:
    """Resolve engine exports on first access (PEP 562)."""
    module_name = _ENGINE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Include lazy exports in ``dir(src.backtest)``."""
    return sorted(set(globals()) | set(__all__))
