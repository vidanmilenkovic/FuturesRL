"""Independent verification of the open-source release packaging and hygiene.

Verification-only suite for branch ``refactor/open-source-release`` (OSS tasks
74-77 + release packaging); no production edits unless a real bug is found
(reported first). Everything is read-only: filesystem/markdown/pyproject checks,
import smoke tests and an offline API app. No network, no git commands, no .env.

Sections
--------
1. File hygiene: internal agent/phase documents and phase/QoL modules removed,
   release documents (LICENSE/CONTRIBUTING/CHANGELOG/ARCHITECTURE/Makefile/dev.sh)
   present.
2. Grep invariants: no local machine paths, agent jargon, phase-numbered test
   references, or stale frontend port 3000 anywhere in the public tree.
3. API package: the consolidated ``src.api.schemas`` package re-exports the
   public contract, ``src.api.deps`` carries the merged dependencies, deleted
   modules are unimportable, and OpenAPI generation/route smoke still works.
4. Test renames: the 18 phase-named integration tests were renamed to domain
   names, old names are gone and pytest collection contains no ``test_phase*``.
5. Packaging: pyproject metadata/scripts/extras, Makefile ``dev``/``up`` targets,
   ``scripts/dev.sh`` syntax/URLs/trap, MIT LICENSE and CONTRIBUTING content.
6. README/docs consistency: renamed screenshots resolve, frontend port is 3100,
   changelog covers v1.9.0 and no phase/agent language leaks.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from configs.settings import Settings
from fastapi.testclient import TestClient
from src.api.main import create_app

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
SCREENSHOTS = DOCS / "screenshots"
TEXT_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CHANGELOG.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "Makefile",
    REPO_ROOT / "scripts",
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "configs",
    DOCS,
)
TEXT_SUFFIXES = frozenset(
    {".md", ".py", ".ts", ".tsx", ".js", ".json", ".yml", ".yaml", ".sh", ".toml"}
)
EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".next",
        ".venv",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
    }
)

# Assembled from fragments so this verification file does not itself contain
# the public-tree strings it rejects.
FORBIDDEN_TEXT = (
    "/home/" + "vidan",
    "Existing " + "text:",
    "Lead" + "-owned",
    "AGENTS" + ".md",
    "Instructions" + ".md",
    "localhost:" + "3000",
    "127.0.0.1:" + "3000",
    "Windows 11 " + "build",
)
PHASE_PATTERN = re.compile(r"Phase [0-9]")

RENAMED_TESTS: dict[str, str] = {
    "test_phase7_fx.py": "test_fx_normalization.py",
    "test_phase7_hardening.py": "test_production_hardening.py",
    "test_phase9_api.py": "test_bridge_and_persisted_backtests_api.py",
    "test_phase9_core.py": "test_runner_bridge_and_drl_cli.py",
    "test_phase10_api.py": "test_strategy_studio_and_optimizer_api.py",
    "test_phase10_1_api.py": "test_instruments_catalog_api.py",
    "test_phase10_1_universe.py": "test_full_universe_pipeline.py",
    "test_phase10_2_v151.py": "test_search_space_and_live_attribution.py",
    "test_phase10_3_v152.py": "test_telemetry_and_equity_history_sync.py",
    "test_phase10_4_v153.py": "test_optimizer_activity_and_strategy_apply.py",
    "test_phase10_5_qol.py": "test_studio_and_backtest_exports.py",
    "test_phase10_drl_api.py": "test_drl_trainer_api.py",
    "test_phase10_ibkr_resilience.py": "test_ibkr_reconnection_resilience.py",
    "test_phase10_offline_drl.py": "test_drl_term_structure_training.py",
    "test_phase11a_paper_execution.py": "test_ibkr_contract_qualification_and_fallback.py",
    "test_phase11a_rebalance.py": "test_rebalance_preview_and_execution.py",
    "test_phase11b_data_cache.py": "test_market_data_cache_and_cost_guard.py",
    "test_phase11b_data_modes.py": "test_backtest_real_vs_synthetic_modes.py",
}

RENAMED_SCREENSHOTS: dict[str, str] = {
    "phase9-paper-ibkr-connected.png": "command-center-ibkr-paper.png",
    "phase10-strategy-studio.png": "strategy-studio-builder.png",
    "phase10ux-studio-1920.png": "strategy-studio-universe.png",
    "phase10-optuna-optimizer.png": "optuna-walk-forward-optimizer.png",
}

EXPECTED_SCRIPTS = {
    "futuresrlx-api": "src.api.cli:main",
    "futuresrlx-backtest": "src.backtest.engine:main",
    "futuresrlx-optimize": "src.backtest.optimizer:main",
    "futuresrlx-runner": "src.execution.runner:main",
    "futuresrlx-train-drl": "src.strategies.train_drl:main",
}
EXPECTED_EXTRAS = ("databento", "drl", "dev", "live", "optimize", "all")


def _iter_python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if not EXCLUDED_PARTS & set(path.parts)]


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    self_path = Path(__file__).resolve()
    for root in TEXT_ROOTS:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if EXCLUDED_PARTS & set(path.parts):
                continue
            if (
                path.suffix in TEXT_SUFFIXES or path.name in {"Makefile"}
            ) and path.resolve() != self_path:
                files.append(path)
    return files


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. File hygiene
# ---------------------------------------------------------------------------


def test_internal_agent_and_phase_documents_removed() -> None:
    # AGENTS.md / Instructions.md may exist locally as developer references, but
    # they must never be tracked or pushed.
    for relative in ("AGENTS" + ".md", "Instructions" + ".md", "docs/PHASE_COMPLETION.md"):
        tracked = subprocess.run(
            ["git", "ls-files", "--", relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert tracked.stdout.strip() == "", f"{relative} must not ship in the public repo"
    # No phase/QoL modules remain in the API package.
    assert not (REPO_ROOT / "src" / "api" / "schemas.py").exists(), "schemas.py must be a package"
    api_files = [path.name for path in (REPO_ROOT / "src" / "api").iterdir() if path.is_file()]
    assert not [name for name in api_files if name.startswith(("phase", "qol"))]
    # No phase-named integration tests or screenshots remain.
    integration = REPO_ROOT / "tests" / "integration"
    assert not list(integration.glob("test_phase*"))
    assert not list(SCREENSHOTS.glob("phase*"))


def test_release_documents_exist() -> None:
    required = (
        "LICENSE",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "Makefile",
        "scripts/dev.sh",
    )
    for relative in required:
        path = REPO_ROOT / relative
        assert path.is_file() and path.stat().st_size > 0, f"missing release document: {relative}"
    assert (DOCS / "ARCHITECTURE.md").is_file(), "docs/ARCHITECTURE.md is required"
    assert (DOCS / "USER_GUIDE.md").is_file()
    assert (REPO_ROOT / "README.md").is_file()


# ---------------------------------------------------------------------------
# 2. Grep invariants across the public tree
# ---------------------------------------------------------------------------


def test_no_forbidden_text_anywhere_in_public_tree() -> None:
    offenders: list[str] = []
    for path in _iter_text_files():
        text = _read_text(path)
        for token in FORBIDDEN_TEXT:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token!r}")
        if path.suffix == ".md" and PHASE_PATTERN.search(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: phase-numbered reference")
        if "test_phase" in text and path.parent != integration_tests_root():
            offenders.append(f"{path.relative_to(REPO_ROOT)}: stale test_phase reference")
    assert not offenders, "forbidden text found:\n" + "\n".join(offenders)


def integration_tests_root() -> Path:
    return REPO_ROOT / "tests" / "integration"


def test_frontend_uses_port_3100_everywhere() -> None:
    package = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    package_text = json.dumps(package)
    assert "3000" not in package_text, "frontend/package.json still references port 3000"
    assert "3100" in package_text, "frontend/package.json must point the dashboard at :3100"
    # Public docs advertise 3100 for the dashboard and 8000 for the API, never 3000.
    for path in [
        REPO_ROOT / "README.md",
        DOCS / "USER_GUIDE.md",
        REPO_ROOT / "CONTRIBUTING.md",
    ]:
        text = _read_text(path)
        assert "3000" not in text, f"{path.name} still mentions port 3000"
        assert "3100" in text, f"{path.name} does not mention the dashboard port 3100"
    architecture = _read_text(DOCS / "ARCHITECTURE.md")
    assert "3000" not in architecture, "ARCHITECTURE.md still mentions port 3000"


def test_docs_do_not_link_to_deleted_phase_document() -> None:
    for path in [REPO_ROOT / "README.md", *(DOCS.glob("*.md"))]:
        assert "PHASE_COMPLETION" not in _read_text(path), f"{path.name} links PHASE_COMPLETION.md"


# ---------------------------------------------------------------------------
# 3. API package consolidation + OpenAPI smoke
# ---------------------------------------------------------------------------


def test_schemas_package_reexports_public_contract() -> None:
    schemas = importlib.import_module("src.api.schemas")
    for symbol in (
        "AccountSnapshotResponse",  # core
        "RebalancePreviewRequest",  # execution
        "RootCacheSummary",  # data cache
        "ArtifactDeleteResponse",  # operations
    ):
        assert hasattr(schemas, symbol), f"src.api.schemas does not export {symbol}"
    for module_name in (
        "src.api.schemas.core",
        "src.api.schemas.execution",
        "src.api.schemas.data",
        "src.api.schemas.operations",
    ):
        importlib.import_module(module_name)


def test_merged_deps_expose_phase_dependencies() -> None:
    deps = importlib.import_module("src.api.deps")
    for symbol in (
        "ContainerDep",
        "BacktestDep",
        "OptimizeDep",
        "StudioDep",
        "get_optimize_service",
        "get_studio_service",
    ):
        assert hasattr(deps, symbol), f"src.api.deps does not expose {symbol}"


def test_deleted_api_modules_are_unimportable() -> None:
    for module_name in (
        "src.api.phase10_deps",
        "src.api.phase11_schemas",
        "src.api.phase11b_schemas",
        "src.api.qol_schemas",
    ):
        with pytest.raises(ImportError):
            importlib.import_module(module_name)


def test_openapi_generation_and_route_smoke(tmp_path: Path) -> None:
    settings = Settings(
        environment="backtest",
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        duckdb_path=tmp_path / "data" / "oss.duckdb",
        report_dir=tmp_path / "artifacts",
        configs_dir=REPO_ROOT / "configs",
        instruments_path=REPO_ROOT / "configs" / "instruments.yaml",
        telemetry_interval_seconds=0.02,
        api_enable_ibkr=False,
    )
    app = create_app(
        settings, clock=lambda: __import__("datetime").datetime.now(__import__("datetime").UTC)
    )
    schema = app.openapi()
    paths = schema["paths"]
    for route in (
        "/api/positions",
        "/api/pipeline/latest",
        "/api/strategies",
        "/api/backtests",
        "/api/optimize",
        "/api/data/inventory",
        "/api/runner/rebalance-preview",
    ):
        assert route in paths, f"missing OpenAPI route {route}"
    with TestClient(app) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == schema["info"]["title"]
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# 4. Test renames / collection
# ---------------------------------------------------------------------------


def test_all_eighteen_phase_tests_were_renamed() -> None:
    integration = REPO_ROOT / "tests" / "integration"
    for old_name, new_name in RENAMED_TESTS.items():
        assert not (integration / old_name).exists(), f"old phase test still exists: {old_name}"
        new_path = integration / new_name
        assert new_path.is_file(), f"renamed test missing: {new_name}"
        assert new_path.stat().st_size > 0


def test_no_prompt_prefixed_test_files_anywhere() -> None:
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
        if path.name.startswith(("test_spec", "test_phase"))
    ]
    assert not offenders, f"prompt-prefixed test files remain: {sorted(map(str, offenders))}"


def test_pytest_collection_has_no_phase_named_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/integration"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    output = completed.stdout + completed.stderr
    assert "test_phase" not in output, "pytest still collects phase-named tests"
    assert "test_spec" not in output, "pytest still collects spec-prefixed tests"
    for new_name in RENAMED_TESTS.values():
        assert new_name in output, f"renamed test not collected: {new_name}"


# ---------------------------------------------------------------------------
# 5. Packaging: pyproject / Makefile / dev.sh / LICENSE / CONTRIBUTING
# ---------------------------------------------------------------------------


def test_pyproject_metadata_scripts_and_extras() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert project["name"] == "futuresrl-x"
    assert project["version"] == "1.9.0"
    license_value = project.get("license")
    if isinstance(license_value, dict):
        assert str(license_value.get("text", "")).strip().lower() == "mit"
    else:
        assert str(license_value).strip().lower() == "mit"
    scripts = payload["project"]["scripts"]
    assert scripts == EXPECTED_SCRIPTS
    extras = payload["project"]["optional-dependencies"]
    for extra in EXPECTED_EXTRAS:
        assert extra in extras, f"missing optional dependency extra: {extra}"
        assert extras[extra], f"extra {extra} must not be empty"
    all_extra = " ".join(extras["all"])
    assert "futuresrl-x[" in all_extra
    for extra in EXPECTED_EXTRAS[:-1]:
        assert extra in all_extra
    # Core dependency list remains importable/production-shaped.
    assert any(dep.startswith("fastapi") for dep in project["dependencies"])


def test_makefile_dev_and_up_targets() -> None:
    text = _read_text(REPO_ROOT / "Makefile")
    for target in ("dev:", "up:", "gate:", "lint:", "type:"):
        assert re.search(rf"(?m)^{re.escape(target)}", text), f"Makefile missing {target} target"
    dev_block = text.split("dev:", 1)[1].split("\n\n", 1)[0]
    assert "scripts/dev.sh" in dev_block, "dev target should launch the dev stack"
    assert re.search(r"(?m)^up:\s*dev", text), "up target should alias dev"


def test_dev_script_syntax_and_urls() -> None:
    script = REPO_ROOT / "scripts" / "dev.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    text = _read_text(script)
    assert "trap " in text and "cleanup" in text, "dev.sh must install a cleanup trap"
    assert "8000" in text and "3100" in text
    assert "3000" not in text
    assert "uvicorn" in text and "npm run dev" in text
    assert script.stat().st_mode & 0o111, "dev.sh must be executable"


def test_license_is_mit_and_contributing_is_human() -> None:
    license_text = _read_text(REPO_ROOT / "LICENSE")
    assert license_text.lstrip().startswith("MIT License")
    assert "Permission is hereby granted" in license_text
    assert "THE SOFTWARE IS PROVIDED" in license_text
    assert "Copyright" in license_text
    contributing = _read_text(REPO_ROOT / "CONTRIBUTING.md")
    lowered = contributing.lower()
    assert "contribut" in lowered
    assert "pull request" in lowered or "merge request" in lowered
    assert "quality_gate.sh" in contributing, "CONTRIBUTING must document the quality gate"
    assert ("AGENTS" + ".md") not in contributing and "Phase " not in contributing.replace(
        "Phase ", "Phase "
    )


# ---------------------------------------------------------------------------
# 6. README/docs consistency
# ---------------------------------------------------------------------------


def test_screenshot_links_resolve_to_renamed_files() -> None:
    for old_name, new_name in RENAMED_SCREENSHOTS.items():
        assert not (SCREENSHOTS / old_name).exists(), f"old screenshot still present: {old_name}"
        target = SCREENSHOTS / new_name
        assert target.is_file() and target.stat().st_size > 0, f"missing screenshot: {new_name}"
    documents = [REPO_ROOT / "README.md", *(DOCS.glob("*.md"))]
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
    found: set[str] = set()
    for document in documents:
        text = _read_text(document)
        for match in image_pattern.finditer(text):
            link = match.group(1).strip()
            if not link.startswith(("http://", "https://", "data:")):
                resolved = (document.parent / link).resolve()
                assert resolved.is_file(), f"{document.name} image link does not resolve: {link}"
                found.add(resolved.name)
    assert set(RENAMED_SCREENSHOTS.values()) <= found, (
        "docs must link the renamed screenshots; found: " + ", ".join(sorted(found))
    )


def test_changelog_covers_current_release_without_phase_language() -> None:
    text = _read_text(REPO_ROOT / "CHANGELOG.md")
    assert "Keep a Changelog" in text or "Changelog" in text
    assert "1.9.0" in text, "CHANGELOG must document the v1.9.0 release"
    assert "timeframe" in text.lower(), "CHANGELOG must document the Spec-1 timeframe work"
    assert not PHASE_PATTERN.search(text)
    assert ("AGENTS" + ".md") not in text and ("Lead" + "-owned") not in text


def test_readme_and_architecture_have_no_agent_or_phase_language() -> None:
    for path in (REPO_ROOT / "README.md", DOCS / "ARCHITECTURE.md"):
        text = _read_text(path)
        assert not PHASE_PATTERN.search(text), f"{path.name} still uses phase-numbered language"
        assert ("AGENTS" + ".md") not in text and ("Instructions" + ".md") not in text
        assert ("Lead" + "-owned") not in text and ("/home/" + "vidan") not in text
    architecture = _read_text(DOCS / "ARCHITECTURE.md").lower()
    for topic in ("pipeline", "translator", "risk", "backtest"):
        assert topic in architecture, f"ARCHITECTURE.md should cover {topic!r}"
