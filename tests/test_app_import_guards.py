import ast
from pathlib import Path


OPTIONAL_ENGINE_SYMBOLS = {
    "recommend_cross_asset_tickers",
    "ASSET_CLASS_UNIVERSES",
    "ASSET_CLASS_DEFAULT_CONVICTIONS",
}


def test_cross_asset_symbols_are_loaded_with_getattr_fallbacks():
    tree = ast.parse(Path("app.py").read_text())
    direct_engine_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "engine"
        for alias in node.names
    }

    assert not (OPTIONAL_ENGINE_SYMBOLS & direct_engine_imports)
