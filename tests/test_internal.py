import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from minimal_nn_competition.__main__ import main
from minimal_nn_competition.internal import (
    SCOPES,
    InternalResult,
    Scope,
    load_internal_results,
    measure_scope,
    rank_internal_results,
    save_internal_results,
    scope_files,
    scope_rank,
    upsert_internal_result,
)
from minimal_nn_competition.internal.render import (
    render_internal_html,
    render_internal_site,
)
from minimal_nn_competition.internal.scoring import RULES_VERSION, count_tokens


def test_rules_version_is_pinned() -> None:
    assert RULES_VERSION == 1


def test_score_is_deterministic() -> None:
    source = "def f(a):\n    # comment\n    return a + 1\n"
    assert count_tokens(source) == count_tokens(source)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("x = 1", "long_variable_name = 1"),
        ("def f(a):\n    return a", "def g(b):\n    return b"),
        ("class A:\n    pass", "class BoringName:\n    pass"),
        ("import os", "import operating_system"),
        ("from x import y", "from module import thing"),
        ("a.b = 1", "a.attribute = 1"),
        ("def f(*args, **kwargs):\n    pass", "def f(*a, **k):\n    pass"),
    ],
)
def test_identifier_renames_do_not_change_score(first: str, second: str) -> None:
    assert count_tokens(first) == count_tokens(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("x = 1", "x = 1  # trailing comment"),
        ("x = 1", "# full line comment\nx = 1"),
        ("x = 1\n", "\n\n\nx = 1\n\n"),
        ("x=1", "x = 1"),
        ("def f():\n  pass", "def f():\n        pass"),
        ("x = 1\n", "x = 1\r\n"),
        ("x = 1", '"""Module docstring."""\nx = 1'),
        ("x = 1", '"""Module docstring."""\nx = 1  # comment\n'),
    ],
)
def test_comments_whitespace_and_docstrings_do_not_change_score(first: str, second: str) -> None:
    assert count_tokens(first) == count_tokens(second)


def test_class_and_function_docstrings_are_ignored() -> None:
    assert count_tokens('class A:\n    """Class doc."""\n    pass') == count_tokens(
        "class A:\n    pass"
    )
    assert count_tokens('def f():\n    """Fn doc."""\n    return 1') == count_tokens(
        "def f():\n    return 1"
    )


def test_implicitly_concatenated_docstrings_are_ignored() -> None:
    assert count_tokens('"""a""" """b"""\nx = 1\n') == 2
    assert count_tokens('def f():\n    """a""" """b"""\n    return 1\n') == count_tokens(
        "def f():\n    return 1\n"
    )


def test_nondocstring_strings_count() -> None:
    assert count_tokens('x = """not a docstring"""\n') == 2
    assert count_tokens('s = "# not a comment"') == count_tokens('s = "x"') == 2
    assert count_tokens('s = "a" "b"') == 3  # implicit concat: one token per literal


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = 1", 2),
        ("if a == 1:\n    b = 2", 6),
        ("import os", 1),
        ("from x import y", 2),
        ("def f(a):\n    return a", 5),
        ("class A:\n    pass", 3),
        ("@deco\ndef f():\n    pass", 6),
        ("y = -1", 3),
        ('d = {"k": 1}', 6),
        ("y = lambda a: a + 1", 5),
        ("z = a if b else c", 3),
    ],
)
def test_exact_token_counts(source: str, expected: int) -> None:
    assert count_tokens(source) == expected


def test_syntax_changes_change_the_score() -> None:
    assert count_tokens("x = 1") < count_tokens("x = 1 + 2")
    assert count_tokens("x = 1") < count_tokens("x = 1\nprint(x)")
    assert count_tokens("if a:\n    pass") < count_tokens("if a and b:\n    pass")
    # Same shape, different literal values: no change.
    assert count_tokens("x = 1") == count_tokens("x = 999")
    assert count_tokens('s = "a"') == count_tokens('s = "abc"')


def test_soft_keywords_count_only_as_syntax() -> None:
    assert count_tokens("match = 5") == 2
    assert count_tokens("match cmd:\n    case 1:\n        pass") == 6
    assert count_tokens("case = 1") == 2
    assert count_tokens("obj.match(x)") == 3  # ., (, ) — match is an attribute
    assert count_tokens("match cmd:\n    case _:\n        pass") == 6
    assert count_tokens("type X = int") == 2
    assert count_tokens("type = 5") == 2
    assert count_tokens("match x:\n    case 1:\n        _ = 5\n") == 7
    assert (
        count_tokens("for _ in range(3):\n    pass")
        == count_tokens("match x:\n    case 1:\n        for _ in range(3):\n            pass\n") - 5
    )


def test_fstring_counts_once() -> None:
    assert count_tokens('f = f"{x + 1}"') == 2
    assert count_tokens('f = f"{x + 1}"') == count_tokens('f = "plain"')
    assert count_tokens('f = f"outer {f"inner {y}"} end"') == 2


def test_unparseable_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        count_tokens("def broken(:\n")


def test_scope_files_partition_src() -> None:
    root = Path(__file__).resolve().parent.parent
    covered: dict[Path, str] = {}
    for scope in SCOPES:
        for path in scope_files(scope, root):
            assert path not in covered, f"{path} in scopes {covered[path]} and {scope.name}"
            covered[path] = scope.name
    every = {path for path in (root / "src").rglob("*.py") if "__pycache__" not in path.parts}
    assert set(covered) == every


def test_scope_tests_cover_suite_exactly_once() -> None:
    from minimal_nn_competition.internal import expand_scope_tests

    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    collected = {line.strip() for line in completed.stdout.splitlines() if "::" in line}
    mapped = [node for scope in SCOPES for node in expand_scope_tests(scope, collected)]
    assert sorted(mapped) == sorted(collected)


def _mini_tree(root: Path, failing: bool = False) -> Scope:
    (root / ".python-version").write_text("3.12\n")
    package = root / "pkg"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n")
    tests = root / "tests"
    tests.mkdir()
    body = "assert False" if failing else "assert True"
    (tests / "test_a.py").write_text(f"def test_ok():\n    {body}\n")
    return Scope(name="mini", sources=("pkg",), tests=("tests/test_a.py::test_ok",))


def test_failing_validation_scores_nothing(tmp_path: Path) -> None:
    scope = _mini_tree(tmp_path, failing=True)
    result = measure_scope(scope, tmp_path)
    assert result.status == "fail"
    assert result.tokens == 0
    assert result.failure_reason != ""


def test_passing_validation_scores_sources(tmp_path: Path) -> None:
    scope = _mini_tree(tmp_path)
    result = measure_scope(scope, tmp_path)
    assert result.status == "pass"
    assert result.tokens == 2
    assert result.rules_version == RULES_VERSION
    assert result.python_version != ""


def test_ranking_shares_ties_and_drops_failures() -> None:
    def entry(scope: str, tokens: int, commit: str, status: str = "pass") -> InternalResult:
        return InternalResult(scope, tokens, commit, False, "", "", "", status=status)

    results = [
        entry("core", 100, "c1"),
        entry("core", 90, "c2"),
        entry("core", 90, "c3"),
        entry("core", 0, "c4", status="fail"),
        entry("snake", 50, "c1"),
    ]
    ranked = rank_internal_results(results)
    assert [(item.scope, item.tokens) for item in ranked] == [
        ("core", 90),
        ("core", 90),
        ("core", 100),
        ("core", 0),
        ("snake", 50),
    ]
    assert scope_rank(ranked[0], ranked) == 1
    assert scope_rank(ranked[1], ranked) == 1
    assert scope_rank(ranked[2], ranked) == 3


def test_upsert_replaces_same_scope_and_commit() -> None:
    old = InternalResult("core", 100, "c1", False, "", "", "")
    new = InternalResult("core", 90, "c1", False, "", "", "")
    other = InternalResult("snake", 50, "c1", False, "", "", "")
    assert upsert_internal_result([old, other], new) == [other, new]


def test_leaderboard_round_trip(tmp_path: Path) -> None:

    result = InternalResult("core", 10, "c1", True, "now", "arm64", "3.12.1")
    path = tmp_path / "internal-leaderboard.json"
    save_internal_results(path, [result])
    assert load_internal_results(path) == [result]


def test_render_internal_site(tmp_path: Path) -> None:
    results = [
        InternalResult("core", 90, "abc123", False, "", "arm64", "3.12.1"),
        InternalResult("core", 100, "def456", False, "", "arm64", "3.12.1"),
        InternalResult(
            "core", 0, "bad789", False, "", "", "", status="fail", failure_reason="1 failed"
        ),
    ]
    index = render_internal_site(results, tmp_path)
    page = index.read_text()
    assert (tmp_path / "internal" / "index.html") == index
    assert "CORE" in page
    assert "1 failed" in page
    assert "failed validation" in page
    assert "Main leaderboard" in page
    assert 'content="noindex,nofollow"' in page
    raw = json.loads((tmp_path / "internal" / "leaderboard.json").read_text())
    assert [item["tokens"] for item in raw] == [90, 100, 0]


def test_render_internal_html_empty() -> None:
    assert "No internal results" in render_internal_html([])


def test_validation_subprocess_sees_nesting_marker(tmp_path: Path) -> None:
    from minimal_nn_competition.internal import run_scope_tests

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_env.py").write_text(
        "import os\ndef test_marker():\n"
        "    assert os.environ.get('MINIMAL_NN_COMPETITION_INTERNAL') == '1'\n"
    )
    scope = Scope(name="env", sources=(), tests=("tests/test_env.py::test_marker",))
    passed, reason = run_scope_tests(scope, tmp_path)
    assert passed, reason


def test_internal_cli(tmp_path: Path) -> None:
    if os.environ.get("MINIMAL_NN_COMPETITION_INTERNAL"):
        pytest.skip("nested validation run: the outer run already covers this")
    root = Path(__file__).resolve().parent.parent
    output = tmp_path / "internal-leaderboard.json"
    assert main(["internal", "--root", str(root), "--output", str(output)]) == 0
    results = load_internal_results(output)
    assert {result.scope for result in results} == {
        "chess-eval",
        "core",
        "fashion-mnist",
        "mnist",
        "snake",
    }
    assert all(result.status == "pass" for result in results)
    assert all(result.tokens > 0 for result in results)


def test_render_internal_cli(tmp_path: Path) -> None:

    path = tmp_path / "internal-leaderboard.json"
    save_internal_results(path, [InternalResult("core", 10, "c1", False, "", "arm64", "3.12.1")])
    output = tmp_path / "docs"
    assert main(["render-internal", str(path), str(output)]) == 0
    assert (output / "internal" / "index.html").is_file()
