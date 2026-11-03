"""Phase-11A runner rebalance preview/execute service (dashboard rebalance book).

The service is intentionally independent of the API process wiring: it composes the
already-live pipeline/translator, the active execution adapter's book (IBKR paper or
the deterministic simulated book) and the runner bridge store. ``preview`` never
submits orders; ``execute`` submits only gate-approved delta orders, refreshes the
book and publishes a bridge snapshot.

Honest-mode rules:
* live environment -> execute refuses with 409 (preview is allowed read-only);
* IBKR adapter enabled but not connected -> no simulated substitution (409/503);
* offline/simulated adapter -> ``mode="offline"`` and execution through the
  simulated broker; a fixture-only adapter refuses execution honestly.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import cast

from configs.instruments import InstrumentsConfig
from configs.settings import Settings

from src.api import fixtures
from src.api.phase11_schemas import (
    RebalanceExecuteRequest,
    RebalanceExecuteResponse,
    RebalanceFillRow,
    RebalanceMode,
    RebalanceOrderRow,
    RebalancePositionRow,
    RebalancePreviewRequest,
    RebalancePreviewResponse,
    RebalanceTargetRow,
)
from src.api.protocols import MarketDataService, StrategyCatalog
from src.api.services import build_pipeline_parts, discretization_details
from src.core.types import (
    AccountSnapshot,
    ContractTargetSet,
    InstrumentId,
    MarketState,
    OrderEvent,
    OrderReceipt,
    OrderStatus,
    Position,
)

logger = logging.getLogger(__name__)

__all__ = ["RebalanceError", "RebalanceService"]

#: Margin policy cap (matches the contract translator default; surfaced to the UI).
MARGIN_UTILIZATION_CAP: float = 0.60


class RebalanceError(RuntimeError):
    """API-mappable rebalance error (``status_code`` + actionable message)."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ExecutionBook:
    """Normalised current book from an execution adapter (never mutated here)."""

    def __init__(
        self,
        *,
        account: AccountSnapshot,
        positions: Mapping[InstrumentId, Position],
        currency: str,
    ) -> None:
        self.account = account
        self.positions = positions
        self.currency = currency

    @property
    def current_quantities(self) -> dict[InstrumentId, int]:
        return {
            instrument_id: int(position.quantity)
            for instrument_id, position in self.positions.items()
            if int(position.quantity) != 0
        }


class RebalanceService:
    """Compose pipeline -> translator -> delta orders for preview/execute."""

    def __init__(
        self,
        *,
        settings: Settings,
        market: MarketDataService,
        catalog: StrategyCatalog,
        execution: object,
        clock: fixtures.Clock = fixtures.utc_now,
    ) -> None:
        self._settings = settings
        self._market = market
        self._catalog = catalog
        self._execution = execution
        self._clock = clock
        self._idempotent_key: tuple[object, ...] | None = None
        self._idempotent_response: RebalanceExecuteResponse | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def instrument_master(self) -> InstrumentsConfig:
        return cast(InstrumentsConfig, getattr(self._market, "instrument_master", None))

    def _ibkr_enabled(self) -> bool:
        return bool(getattr(self._execution, "ibkr_enabled", False))

    def _ibkr_connected(self) -> bool:
        return bool(getattr(self._execution, "connected", False))

    def _mode(self) -> RebalanceMode:
        if self._ibkr_enabled() and self._ibkr_connected():
            return "ibkr-paper"
        return "offline"

    # -- book / state ----------------------------------------------------------

    async def _fetch_book(self) -> _ExecutionBook:
        account_getter = getattr(self._execution, "account", None)
        positions_getter = getattr(self._execution, "positions", None)
        if not callable(account_getter) or not callable(positions_getter):
            raise RebalanceError(
                "active execution adapter does not expose account/positions", status_code=503
            )
        account_provider = cast("Callable[[], Awaitable[AccountSnapshot]]", account_getter)
        positions_provider = cast(
            "Callable[[], Awaitable[Mapping[InstrumentId, Position]]]", positions_getter
        )
        try:
            account = await account_provider()
            positions = await positions_provider()
        except Exception as exc:
            if self._ibkr_enabled():
                raise RebalanceError(
                    "IBKR paper book is unavailable; refusing to substitute the simulated "
                    f"book (connect TWS/IB Gateway and retry): {type(exc).__name__}: {exc}",
                    status_code=503,
                ) from exc
            raise RebalanceError(
                f"offline execution book unavailable: {type(exc).__name__}: {exc}",
                status_code=503,
            ) from exc
        return _ExecutionBook(
            account=account,
            positions=positions,
            currency=self._settings.account_currency,
        )

    def _state_for_request(self, roots: Sequence[str] | None, book: _ExecutionBook) -> MarketState:
        state = self._market.latest_state()
        master = self.instrument_master
        if roots:
            requested = tuple(dict.fromkeys(root.strip().upper() for root in roots if root.strip()))
            missing = [root for root in requested if not master.has(root)]
            if missing:
                raise RebalanceError(
                    f"unknown futures root(s) in request: {missing!r}", status_code=422
                )
            available = {
                root: snapshot for root, snapshot in state.instruments.items() if root in requested
            }
            if not available:
                raise RebalanceError(
                    "none of the requested roots are present in the latest hydrated market state",
                    status_code=422,
                )
            covariance = state.covariance
            covariance_instruments = state.covariance_instruments
            if covariance is not None and set(covariance_instruments) != set(available):
                covariance = None
                covariance_instruments = ()
            state = dataclasses.replace(
                state,
                instruments=available,
                covariance=covariance,
                covariance_instruments=covariance_instruments,
                returns={
                    root: series for root, series in state.returns.items() if root in available
                },
            )
        return dataclasses.replace(
            state,
            equity=book.account.net_liquidation,
            account_currency=book.currency,
        )

    # -- plan ------------------------------------------------------------------

    def _build_plan(
        self, request_strategy_id: str, roots: Sequence[str] | None, book: _ExecutionBook
    ) -> tuple[
        MarketState,
        ContractTargetSet,
        list[OrderEvent],
        list[RebalanceTargetRow],
        list[RebalanceOrderRow],
        float,
        float,
        float,
        list[str],
    ]:
        config = self._catalog.core_config(request_strategy_id)
        if config is None:
            raise RebalanceError(f"unknown strategy {request_strategy_id!r}", status_code=404)
        state = self._state_for_request(roots, book)
        pipeline, translator = build_pipeline_parts(
            config,
            self.instrument_master,
            self._settings,
            clock=self._clock,
            state=state,
        )
        result = pipeline.run(state)
        targets = result.targets
        equity = state.equity or book.account.net_liquidation
        if targets is None:
            targets = translator.translate(result.final_weights, state, equity=equity)
        tracking_error, _routing, _achieved, _source = discretization_details(
            translator, result.final_weights, targets, state, equity
        )
        current = book.current_quantities
        orders = self._orders_from_targets(targets, current, state.timestamp)
        front_symbols = self._front_month_symbols(targets, state.timestamp)
        gate_rows, projected_util, _cap = self._gate_report(
            orders, targets, book, front_symbols=front_symbols
        )
        target_rows: list[RebalanceTargetRow] = [
            RebalanceTargetRow(
                instrument_id=target.instrument_id,
                quantity=target.quantity,
                price=target.price,
                multiplier=target.multiplier,
                initial_margin=target.initial_margin,
                signed_notional=target.signed_notional,
                contract_symbol=target.contract_symbol,
                expiration=target.expiration,
            )
            for target in targets
        ]
        warnings: list[str] = []
        if state.equity is None:
            warnings.append("market state carried no equity; using the broker NLV")
        return (
            state,
            targets,
            orders,
            target_rows,
            gate_rows,
            tracking_error,
            projected_util,
            equity,
            warnings,
        )

    @staticmethod
    def _orders_from_targets(
        targets: ContractTargetSet,
        current_positions: Mapping[InstrumentId, int],
        timestamp: datetime,
    ) -> list[OrderEvent]:
        from src.execution.oms import build_orders_from_targets

        return build_orders_from_targets(
            targets, current_positions, timestamp, tag="rebalance-preview"
        )

    def _front_month_symbols(
        self, targets: ContractTargetSet, reference: datetime
    ) -> dict[InstrumentId, str]:
        """Cheap front-month ``ROOT YYYYMM`` labels for preview symbols (no network)."""
        try:
            from src.data.ibkr_contracts import front_contract_month
        except ImportError:  # pragma: no cover - task-58 helper optional
            return {}
        labels: dict[InstrumentId, str] = {}
        for target in targets:
            try:
                month, _expiration = front_contract_month(
                    target.instrument_id,
                    reference=reference,
                    instrument_master=self.instrument_master,
                )
            except Exception:
                continue
            if isinstance(month, str) and month:
                labels[target.instrument_id] = f"{target.instrument_id} {month}"
        return labels

    def _gate_report(
        self,
        orders: Sequence[OrderEvent],
        targets: ContractTargetSet,
        book: _ExecutionBook,
        *,
        front_symbols: Mapping[InstrumentId, str] | None = None,
    ) -> tuple[list[RebalanceOrderRow], float, float]:
        locked = bool(getattr(self._execution, "is_locked", False))
        max_contracts = int(self._settings.risk.max_order_contracts)
        cap = float(
            getattr(
                getattr(self._execution, "translator", None),
                "max_margin_utilization",
                MARGIN_UTILIZATION_CAP,
            )
        )
        if not cap or cap <= 0:
            cap = MARGIN_UTILIZATION_CAP
        by_instrument = targets.by_instrument
        projected_margin = float(book.account.initial_margin_used)
        rows: list[RebalanceOrderRow] = []
        for order in orders:
            target = by_instrument.get(order.instrument_id)
            per_contract_margin = float(target.initial_margin) if target is not None else 0.0
            impact = per_contract_margin * abs(order.quantity)
            projected_margin += impact
            utilization = projected_margin / book.account.net_liquidation
            rejection: str | None = None
            if locked:
                rejection = "kill switch engaged"
            elif order.quantity > max_contracts:
                rejection = (
                    f"fat-finger: quantity {order.quantity} exceeds max_order_contracts "
                    f"{max_contracts}"
                )
            elif utilization > cap:
                rejection = (
                    f"margin: projected utilization {utilization:.2%} exceeds the {cap:.0%} cap"
                )
            front_symbol = (front_symbols or {}).get(order.instrument_id) if front_symbols else None
            symbol = (
                (target.contract_symbol if target is not None else None)
                or front_symbol
                or (
                    target.expiration.strftime("%Y%m")
                    if target is not None and target.expiration
                    else None
                )
                or order.instrument_id
            )
            rows.append(
                RebalanceOrderRow(
                    order_id=order.order_id,
                    symbol=str(symbol),
                    instrument_id=order.instrument_id,
                    side=order.side.value,
                    quantity=order.quantity,
                    order_type=order.order_type.value,
                    est_price=order.limit_price or (target.price if target is not None else None),
                    est_margin_impact=round(impact, 6),
                    pre_trade_gate_passed=rejection is None,
                    rejection_reason=rejection,
                )
            )
        return rows, round(projected_margin / book.account.net_liquidation, 6), cap

    # -- preview ---------------------------------------------------------------

    async def preview(self, request: RebalancePreviewRequest) -> RebalancePreviewResponse:
        book = await self._fetch_book()
        (
            _state,
            _targets,
            _orders,
            target_rows,
            order_rows,
            tracking_error,
            projected_util,
            _equity,
            warnings,
        ) = self._build_plan(request.strategy_id, request.roots, book)
        cap = self._gate_cap()
        mode = self._mode()
        if self._ibkr_enabled() and not self._ibkr_connected():
            warnings.append(
                "IBKR adapter is enabled but not connected; preview uses the offline "
                "deterministic state and simulated book"
            )
        return RebalancePreviewResponse(
            strategy_id=request.strategy_id,
            account_nlv=book.account.net_liquidation,
            account_currency=book.currency,
            current_positions=self._position_rows(book.positions),
            target_contracts=target_rows,
            proposed_orders=[row for row in order_rows if row.pre_trade_gate_passed],
            projected_margin_utilization=projected_util,
            margin_utilization_cap=cap,
            tracking_error_pct=tracking_error,
            mode=mode,
            generated_at=self._clock(),
            strategy_roots=sorted(cast(list[str], request.roots or [])),
            warnings=warnings,
        )

    def _gate_cap(self) -> float:
        translator = getattr(self._execution, "translator", None)
        cap = getattr(translator, "max_margin_utilization", None)
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
            return float(cap)
        return MARGIN_UTILIZATION_CAP

    @staticmethod
    def _position_rows(positions: Mapping[InstrumentId, Position]) -> list[RebalancePositionRow]:
        return [
            RebalancePositionRow(
                instrument_id=position.instrument_id,
                quantity=int(position.quantity),
                average_price=position.average_price,
                market_price=position.market_price,
                multiplier=position.multiplier,
                signed_notional=round(
                    int(position.quantity) * position.market_price * position.multiplier, 6
                ),
                initial_margin=float(position.initial_margin) * abs(int(position.quantity)),
                contract_symbol=position.contract_symbol,
            )
            for position in sorted(positions.values(), key=lambda row: row.instrument_id)
        ]

    # -- execute ---------------------------------------------------------------

    async def execute(self, request: RebalanceExecuteRequest) -> RebalanceExecuteResponse:
        if self._settings.environment == "live":
            raise RebalanceError(
                "refusing to execute rebalances in the live environment; use paper/offline",
                status_code=409,
            )
        if not request.confirm:
            raise RebalanceError(
                "confirmation required: set confirm=true to submit the rebalance", status_code=400
            )
        book = await self._fetch_book()
        mode = self._mode()
        if self._ibkr_enabled() and not self._ibkr_connected():
            raise RebalanceError(
                "IBKR paper session is not connected; refusing to substitute the simulated "
                "broker for a live-mode execution",
                status_code=409,
            )
        (
            _state,
            _targets,
            orders,
            _target_rows,
            order_rows,
            tracking_error,
            projected_util,
            _equity,
            _warnings,
        ) = self._build_plan(request.strategy_id, request.roots, book)
        approved = {row.order_id: row for row in order_rows if row.pre_trade_gate_passed}
        approved_orders = [order for order in orders if order.order_id in approved]
        rejected = [row for row in order_rows if not row.pre_trade_gate_passed]
        key = (
            request.strategy_id,
            tuple(sorted(request.roots or ())),
            tuple(
                (order.order_id, order.instrument_id, order.side.value, order.quantity)
                for order in approved_orders
            ),
            tuple(sorted(book.current_quantities.items())),
            round(book.account.net_liquidation, 6),
            mode,
        )
        if (
            approved_orders
            and key == self._idempotent_key
            and self._idempotent_response is not None
        ):
            return self._idempotent_response.model_copy(
                update={"status": "idempotent-replay", "generated_at": self._clock()}
            )
        if not approved_orders:
            return RebalanceExecuteResponse(
                strategy_id=request.strategy_id,
                mode=mode,
                status="no-op",
                submitted_order_ids=[],
                orders=order_rows,
                fills=[],
                rejected_orders=rejected,
                account_nlv=book.account.net_liquidation,
                account_currency=book.currency,
                projected_margin_utilization=projected_util,
                margin_utilization_cap=self._gate_cap(),
                tracking_error_pct=tracking_error,
                updated_positions=self._position_rows(book.positions),
                bridge_updated=False,
                message="no gate-approved orders; nothing submitted",
                generated_at=self._clock(),
            )
        receipts = await self._submit_orders(approved_orders)
        submitted_ids = [
            receipt.order_id for receipt in receipts if receipt.status is not OrderStatus.REJECTED
        ]
        rejected_receipts = [
            receipt for receipt in receipts if receipt.status is OrderStatus.REJECTED
        ]
        for receipt in rejected_receipts:
            for row in order_rows:
                if row.order_id == receipt.order_id:
                    row = row.model_copy(
                        update={
                            "pre_trade_gate_passed": False,
                            "rejection_reason": receipt.message or "pre-trade gate rejected",
                        }
                    )
                    rejected.append(row)
        fills = [
            RebalanceFillRow(
                order_id=receipt.order_id,
                instrument_id=order.instrument_id,
                side=order.side.value,
                quantity=receipt.filled_quantity or order.quantity,
                price=float(receipt.average_price or 0.0),
                commission=float(receipt.commission or 0.0),
                status=receipt.status.value,
                timestamp=receipt.timestamp,
            )
            for receipt, order in zip(receipts, approved_orders, strict=False)
        ]
        try:
            refreshed = await self._fetch_book()
        except Exception:
            refreshed = book
        bridge_updated = self._publish_bridge(
            request.strategy_id,
            mode,
            book=refreshed,
            orders=order_rows,
            receipts=receipts,
            tracking_error=tracking_error,
            projected_util=projected_util,
        )
        response = RebalanceExecuteResponse(
            strategy_id=request.strategy_id,
            mode=mode,
            status="submitted",
            submitted_order_ids=submitted_ids,
            orders=order_rows,
            fills=fills,
            rejected_orders=rejected,
            account_nlv=refreshed.account.net_liquidation,
            account_currency=refreshed.currency,
            projected_margin_utilization=projected_util,
            margin_utilization_cap=self._gate_cap(),
            tracking_error_pct=tracking_error,
            updated_positions=self._position_rows(refreshed.positions),
            bridge_updated=bridge_updated,
            message=(
                f"submitted {len(submitted_ids)} of {len(approved_orders)} approved "
                f"delta order(s) in {mode} mode"
            ),
            generated_at=self._clock(),
        )
        self._idempotent_key = key
        self._idempotent_response = response
        return response

    async def _submit_orders(self, orders: Sequence[OrderEvent]) -> list[OrderReceipt]:
        submit_orders = getattr(self._execution, "submit_orders", None)
        if callable(submit_orders):
            return list(await cast("object", submit_orders(list(orders))))  # type: ignore[misc]
        submit_targets = getattr(self._execution, "submit_targets", None)
        if callable(submit_targets):
            # Adapter without per-order submission: best effort via target submission.
            raise RebalanceError(
                "active execution adapter exposes no per-order submission surface; "
                "refusing to submit an unfiltered target set",
                status_code=409,
            )
        raise RebalanceError("active execution adapter cannot submit orders", status_code=409)

    # -- bridge ----------------------------------------------------------------

    def _publish_bridge(
        self,
        strategy_id: str,
        mode: RebalanceMode,
        *,
        book: _ExecutionBook,
        orders: Sequence[RebalanceOrderRow],
        receipts: Sequence[OrderReceipt],
        tracking_error: float,
        projected_util: float,
    ) -> bool:
        if not self._settings.runner_bridge_enabled:
            return False
        try:
            from src.execution.runner_bridge import (
                BRIDGE_SCHEMA_VERSION,
                DEFAULT_STALE_AFTER_SECONDS,
                BridgeFill,
                BridgePosition,
                BridgeWorkingOrder,
                RunnerBridgeSnapshot,
                RunnerBridgeStore,
            )
        except Exception:  # pragma: no cover - bridge package always present
            return False
        store = RunnerBridgeStore(self._settings)
        previous = store.load()
        sequence = (previous.sequence + 1) if previous is not None else 1
        positions = tuple(
            BridgePosition(
                instrument_id=position.instrument_id,
                contract_symbol=position.contract_symbol,
                quantity=int(position.quantity),
                average_price=position.average_price,
                market_price=position.market_price,
                multiplier=position.multiplier,
                unrealized_pnl=0.0,
                initial_margin=float(position.initial_margin) * abs(int(position.quantity)),
            )
            for position in sorted(book.positions.values(), key=lambda row: row.instrument_id)
        )
        fills = tuple(
            BridgeFill(
                order_id=receipt.order_id,
                instrument_id=order.instrument_id,
                side=order.side,
                quantity=receipt.filled_quantity or order.quantity,
                price=float(receipt.average_price or order.est_price or 1.0),
                commission=float(receipt.commission or 0.0),
                slippage_ticks=0.0,
                timestamp=receipt.timestamp or self._clock(),
            )
            for receipt, order in zip(receipts, orders, strict=False)
            if receipt.status is not OrderStatus.REJECTED
        )
        working = tuple(
            BridgeWorkingOrder(
                order_id=receipt.order_id,
                instrument_id=order.instrument_id,
                contract_symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=receipt.filled_quantity,
                remaining_quantity=max(0, order.quantity - receipt.filled_quantity),
                order_type=order.order_type,
                limit_price=None,
                status=receipt.status.value,
                updated_at=receipt.timestamp or self._clock(),
            )
            for receipt, order in zip(receipts, orders, strict=False)
            if receipt.status not in (OrderStatus.FILLED, OrderStatus.REJECTED)
        )
        snapshot = RunnerBridgeSnapshot(
            schema_version=BRIDGE_SCHEMA_VERSION,
            runner_id="api-rebalance",
            mode=cast("object", "paper" if mode == "ibkr-paper" else "offline"),  # type: ignore[arg-type]
            sequence=sequence,
            updated_at=self._clock(),
            stale_after_seconds=DEFAULT_STALE_AFTER_SECONDS,
            status={
                "source": "api-rebalance",
                "strategy_id": strategy_id,
                "orders": len(receipts),
                "tracking_error_pct": tracking_error,
                "projected_margin_utilization": projected_util,
            },
            account={
                "net_liquidation": book.account.net_liquidation,
                "cash_balance": book.account.cash_balance,
                "initial_margin_used": book.account.initial_margin_used,
                "margin_utilization": book.account.margin_utilization,
            },
            pipeline={
                "strategy_id": strategy_id,
                "tracking_error_pct": tracking_error,
                "projected_margin_utilization": projected_util,
            },
            positions=positions,
            working_orders=working,
            fills=fills,
            tracking_error_pct=tracking_error,
            tracking_error_source="translator",
        )
        try:
            store.save(snapshot)
        except Exception:
            logger.exception("failed to publish rebalance bridge snapshot")
            return False
        return True
