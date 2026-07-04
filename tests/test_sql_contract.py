from pathlib import Path


def test_hybrid_search_dynamic_sql_escapes_format_percent_signs() -> None:
    contract = Path("sql/aws_pgvector_contract.sql").read_text(encoding="utf-8")
    dynamic_sql = contract.split("RETURN QUERY EXECUTE format($query$", maxsplit=1)[1]
    dynamic_sql = dynamic_sql.split("$query$, target_table)", maxsplit=1)[0]

    remaining_percent_signs = dynamic_sql.replace("%%", "").replace("%I", "")

    assert "%" not in remaining_percent_signs
