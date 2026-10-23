from __future__ import annotations

import pytest

from minimal_nn_competition.__main__ import main


def test_main_callable() -> None:
    assert callable(main)


def test_main_lists_benchmarks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["chess-eval", "fashion-mnist", "mnist", "snake"]
