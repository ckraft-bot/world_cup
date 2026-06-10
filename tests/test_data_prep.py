from __future__ import annotations

import pytest


def test_normalize_text_handles_missing_tokens(data_prep_module) -> None:
    assert data_prep_module.normalize_text("  ") is None
    assert data_prep_module.normalize_text(" n/a ") is None
    assert data_prep_module.normalize_text(" Brazil ") == "Brazil"


def test_to_int_parses_with_separators(data_prep_module) -> None:
    assert data_prep_module.to_int("1,234") == 1234
    assert data_prep_module.to_int("1.234") == 1234
    assert data_prep_module.to_int(None) is None


def test_parse_match_datetime_valid_and_invalid(data_prep_module) -> None:
    assert data_prep_module.parse_match_datetime("12 Jun 2014 - 16:00") == "2014-06-12"
    assert (
        data_prep_module.parse_match_datetime("12 June 2014 - 16:00", output_type="datetime")
        == "2014-06-12 16:00"
    )

    with pytest.raises(ValueError):
        data_prep_module.parse_match_datetime("2014/06/12 16:00")


def test_transform_with_schema_applies_types_and_uppercase(data_prep_module) -> None:
    rows = [
        {
            "Team": "spain",
            "Goals": "2",
            "Kickoff": "11 Jul 2010 - 20:30",
        }
    ]
    schema = {
        "columns": [
            {"source": "Team", "target": "Team", "type": "str", "uppercase": True},
            {"source": "Goals", "target": "Goals", "type": "int"},
            {"source": "Kickoff", "target": "Kickoff", "type": "datetime"},
        ]
    }

    transformed, fieldnames = data_prep_module.transform_with_schema(rows, schema)

    assert fieldnames == ["Team", "Goals", "Kickoff"]
    assert transformed == [{"Team": "SPAIN", "Goals": 2, "Kickoff": "2010-07-11 20:30"}]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
