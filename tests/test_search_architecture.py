import ast
from pathlib import Path


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
        elif isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
    return result


def test_search_domain_and_application_keep_clean_boundaries() -> None:
    root = Path("app/modules/search")
    violations: list[str] = []
    for path in (root / "domain").rglob("*.py"):
        for imported in _imports(path):
            if imported.startswith("app.") and not imported.startswith(
                "app.modules.search.domain"
            ):
                violations.append(f"{path}: {imported}")
    for path in (root / "application").rglob("*.py"):
        for imported in _imports(path):
            if imported.startswith("app.") and not imported.startswith(
                (
                    "app.modules.search.domain",
                    "app.modules.search.application",
                )
            ):
                violations.append(f"{path}: {imported}")
    assert violations == []
