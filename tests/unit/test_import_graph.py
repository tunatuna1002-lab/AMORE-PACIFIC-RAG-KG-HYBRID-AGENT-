"""
Static import-graph contract (Phase 5)
======================================
Parses every module under ``src/`` with ``ast`` (no imports executed) and asserts the
architectural invariants the refactor plan fixes:

1. No import cycles between ``src`` modules (strongly connected component size 1).
2. Clean Architecture direction: ``domain`` depends on nothing inside ``src`` except
   ``src.domain`` and ``src.shared``; ``application`` does not import ``src.api``.
3. No layer inversion from the tool/infrastructure side into the web layer
   (``src.tools`` / ``src.rag`` / ``src.ontology`` must not import ``src.api``).

Only TOP-LEVEL imports count. A deferred import inside a function body is how this
codebase breaks cycles on purpose (lazy ``__init__`` loading, optional dependencies),
so those are collected separately and reported, not failed.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = "src"


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: str, node: ast.ImportFrom, is_package: bool) -> str | None:
    """``from ..x import y`` inside ``module`` -> absolute dotted name."""
    parts = module.split(".")
    if not is_package:
        parts = parts[:-1]  # a module's "." is its package
    up = node.level - 1
    if up:
        parts = parts[:-up] if up <= len(parts) else []
    if not parts:
        return None
    base = ".".join(parts)
    return f"{base}.{node.module}" if node.module else base


def _top_level_and_deferred(tree: ast.AST) -> tuple[set[ast.AST], set[ast.AST]]:
    """Import nodes reachable at module import time vs. inside function bodies."""
    deferred: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import | ast.ImportFrom):
                    deferred.add(inner)
    top: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom) and node not in deferred:
            top.add(node)
    return top, deferred


def _build_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(top-level edges, deferred edges) between ``src.*`` modules."""
    top_edges: dict[str, set[str]] = defaultdict(set)
    deferred_edges: dict[str, set[str]] = defaultdict(set)

    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(path)
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top, deferred = _top_level_and_deferred(tree)

        for bucket, nodes in ((top_edges, top), (deferred_edges, deferred)):
            for node in nodes:
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                else:
                    if node.level:
                        resolved = _resolve_relative(module, node, is_package)
                        targets = [resolved] if resolved else []
                    else:
                        targets = [node.module] if node.module else []
                for target in targets:
                    if target and target.split(".")[0] == PACKAGE_ROOT and target != module:
                        bucket[module].add(target)

    return dict(top_edges), dict(deferred_edges)


def _known_modules() -> set[str]:
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        if "__pycache__" not in path.parts:
            names.add(_module_name(path))
    return names


def _normalize(edges: dict[str, set[str]], known: set[str]) -> dict[str, set[str]]:
    """``src.pkg.mod.Symbol`` -> ``src.pkg.mod``; drop targets that are not modules."""
    out: dict[str, set[str]] = defaultdict(set)
    for src_mod, targets in edges.items():
        for target in targets:
            node = target
            while node and node not in known:
                node = node.rpartition(".")[0]
            if node and node != src_mod:
                out[src_mod].add(node)
    return dict(out)


def _strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC; returns components with more than one member."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    nodes = set(graph) | {t for ts in graph.values() for t in ts}

    for root in sorted(nodes):
        if root in index:
            continue
        # iterative Tarjan (recursion depth is unbounded on a 200-module graph)
        work: list[tuple[str, list[str]]] = [(root, sorted(graph.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, pending = work[-1]
            if pending:
                child = pending.pop()
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack[child] = True
                    work.append((child, sorted(graph.get(child, ()))))
                elif on_stack.get(child):
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        result.append(sorted(component))
    return result


@pytest.fixture(scope="module")
def graph() -> dict[str, set[str]]:
    known = _known_modules()
    top, _ = _build_graph()
    return _normalize(top, known)


def test_no_import_cycles(graph: dict[str, set[str]]) -> None:
    """Top-level imports must form a DAG (deferred imports are exempt by design)."""
    cycles = _strongly_connected(graph)
    assert cycles == [], "import cycles (top-level):\n" + "\n".join(" -> ".join(c) for c in cycles)


def test_domain_depends_on_nothing_but_domain_and_shared(graph: dict[str, set[str]]) -> None:
    violations = {
        module: sorted(t for t in targets if not t.startswith(("src.domain", "src.shared")))
        for module, targets in graph.items()
        if module.startswith("src.domain")
    }
    violations = {k: v for k, v in violations.items() if v}
    assert violations == {}, f"domain must not depend on outer layers: {violations}"


def test_application_does_not_import_the_web_layer(graph: dict[str, set[str]]) -> None:
    violations = {
        module: sorted(t for t in targets if t.startswith("src.api"))
        for module, targets in graph.items()
        if module.startswith("src.application")
    }
    violations = {k: v for k, v in violations.items() if v}
    assert violations == {}, f"application must not import src.api: {violations}"


@pytest.mark.parametrize("layer", ["src.tools", "src.rag", "src.ontology", "src.infrastructure"])
def test_inner_layers_do_not_import_the_web_layer(graph: dict[str, set[str]], layer: str) -> None:
    violations = {
        module: sorted(t for t in targets if t.startswith("src.api"))
        for module, targets in graph.items()
        if module.startswith(layer)
    }
    violations = {k: v for k, v in violations.items() if v}
    assert violations == {}, f"{layer} must not import src.api: {violations}"
