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

def test_cross_asset_loader_filters_optional_kwargs_by_signature():
    source = Path("app.py").read_text()

    assert "inspect.signature(recommend_cross_asset_tickers)" in source
    assert "if key in sig.parameters" in source
    assert "recommend_cross_asset_tickers(**filtered)" in source


def test_cross_asset_display_rows_are_normalized_before_column_selection():
    source = Path("app.py").read_text()

    assert "CROSS_ASSET_DISPLAY_COLUMNS" in source
    assert "def _prepare_cross_asset_display_df" in source
    assert "if col not in df.columns" in source
    assert "display_rows = _prepare_cross_asset_display_df(rows)" in source
