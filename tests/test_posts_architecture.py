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


def test_posts_domain_and_application_do_not_import_infrastructure() -> None:
    root = Path("app/modules/posts")
    forbidden = ("app.shared", "app.modules.posts.infrastructure", "app.modules.posts.api")
    violations: list[str] = []
    for layer in ("domain", "application"):
        for path in (root / layer).rglob("*.py"):
            for imported in _imports(path):
                if imported.startswith(forbidden):
                    violations.append(f"{path}: {imported}")
    assert violations == []


def test_legacy_posts_api_and_application_ports_are_gone() -> None:
    assert not any(
        path.name != "__init__.py" for path in Path("app/modules/posts/api").glob("*.py")
    )
    assert not any(
        path.name != "__init__.py"
        for path in Path("app/modules/posts/application/ports").glob("*.py")
    )
