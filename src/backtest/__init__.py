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
"""

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
    build_states_from_bars,
    default_state_builder,
    main,
    run_from_config,
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
    "build_states_from_bars",
    "default_fee_schedule",
    "default_state_builder",
    "main",
    "run_from_config",
]
