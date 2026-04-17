from tesla_qa.utils import parse_numeric


def test_parse_numeric():
    assert parse_numeric('$1,234') == 1234.0
    assert parse_numeric('(567)') == -567.0
    assert parse_numeric('—') is None
