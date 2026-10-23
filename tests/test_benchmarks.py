import json
from pathlib import Path
from typing import Any

import pytest
import torch

from minimal_nn_competition.benchmarks.chess_eval.benchmark import (
    BLACK_BETTER,
    DRAW,
    WHITE_BETTER,
    ChessEvalBenchmark,
    label_of,
    material_of,
    sample_dataset,
)
from minimal_nn_competition.benchmarks.fashion_mnist.benchmark import FashionMnistBenchmark
from minimal_nn_competition.benchmarks.mnist.benchmark import MnistBenchmark
from minimal_nn_competition.benchmarks.register import create_benchmark
from minimal_nn_competition.benchmarks.snake import benchmark as snake_benchmark
from minimal_nn_competition.benchmarks.snake.benchmark import SnakeBenchmark
from minimal_nn_competition.model.entry import Submission, materialize_submission


def _write_submission(
    tmp_path: Path, config: dict[str, Any], state: dict[str, torch.Tensor]
) -> Submission:
    directory = tmp_path / "example"
    directory.mkdir()
    config = {"name": "example", "author": "tester", **config}
    (directory / "config.json").write_text(json.dumps(config))
    torch.save(state, directory / "model.pt")
    return Submission.from_directory(directory)


def test_snake_partial_run_does_not_pass(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    result = SnakeBenchmark(seeds=(0, 1, 2)).evaluate(materialize_submission(submission))
    score, score_std, passed, details = result
    assert score == 1.0
    assert score_std == 0.0
    assert passed is False
    assert details["seed_lengths"] == [1, 1, 1]
    assert details["seed_moves"] == [5, 5, 5]


def test_snake_full_board_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FullBoardEnvironment:
        size = 8

        def __init__(self, seed: int, max_moves: int, input_indices: object = None) -> None:
            self.snake = [(0, 0)] * 64

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", FullBoardEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeBenchmark(seeds=(0, 1, 2)).evaluate(
        materialize_submission(submission)
    )
    assert score == 64.0
    assert score_std == 0.0
    assert passed is True
    assert details["seed_lengths"] == [64, 64, 64]
    assert details["seed_moves"] == [1, 1, 1]


def test_snake_defaults_to_100_seeds() -> None:
    assert SnakeBenchmark().seeds == tuple(range(100))


def test_snake_score_std_across_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class VariableEnvironment:
        size = 8

        def __init__(self, seed: int, max_moves: int, input_indices: object = None) -> None:
            self.snake = [(0, 0)] * (64 if seed == 0 else 1)

        def reset(self) -> tuple[int, ...]:
            return (0.0,) * 24

        def step(self, action: int) -> tuple[tuple[int, ...], float, bool]:
            return (0.0,) * 24, 1.0, True

    monkeypatch.setattr(snake_benchmark, "SnakeEnvironment", VariableEnvironment)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 24,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 24)), "bias": torch.zeros(4)},
    )
    score, score_std, passed, details = SnakeBenchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert score == 32.5
    assert score_std == 31.5
    assert passed is False
    assert details["seed_lengths"] == [64, 1]
    assert details["seed_moves"] == [1, 1]


def test_snake_rejects_wrong_config(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 8,
            "output_size": 4,
        },
        {"weight": torch.zeros((4, 8)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_size 24"):
        SnakeBenchmark().load_submission(submission)


def test_snake_selected_input_indices_build_smaller_model(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 5,
            "output_size": 4,
            "input_indices": [0, 1, 2, 4, 7],
        },
        {"weight": torch.zeros((4, 5)), "bias": torch.zeros(4)},
    )
    loaded = SnakeBenchmark().load_submission(submission)
    assert loaded.model.in_features == 5


def test_snake_selected_inputs_evaluate(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 5,
            "output_size": 4,
            "input_indices": [0, 1, 2, 4, 7],
        },
        {"weight": torch.zeros((4, 5)), "bias": torch.zeros(4)},
    )
    result = SnakeBenchmark(seeds=(0,)).evaluate(materialize_submission(submission))
    score, score_std, passed, details = result
    assert score == 1.0
    assert score_std == 0.0
    assert passed is False
    assert details["seed_lengths"] == [1]


def test_snake_rejects_out_of_range_input_indices(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "snake",
            "architecture": "linear",
            "input_size": 1,
            "output_size": 4,
            "input_indices": [24],
        },
        {"weight": torch.zeros((4, 1)), "bias": torch.zeros(4)},
    )
    with pytest.raises(ValueError, match="input_indices"):
        SnakeBenchmark().load_submission(submission)


def _write_mnist_cache(path: Path) -> None:
    inputs = torch.zeros((2, 784), dtype=torch.uint8)
    inputs[0, 0] = 255
    inputs[1, 1] = 255
    torch.save((inputs, torch.tensor([0, 1])), path)


def test_mnist_evaluation_and_threshold(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist-test.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = MnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}

    score, _, passed, _ = MnistBenchmark(data_path=data_path, threshold=1.01).evaluate(loaded)
    assert score == 1.0
    assert passed is False


def test_mnist_normalisation_flag(tmp_path: Path) -> None:
    data_path = tmp_path / "mnist-test.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
            "normalisation": True,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = MnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": True}


def test_mnist_requires_data_file(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    with pytest.raises(FileNotFoundError, match="MNIST test data is unavailable"):
        MnistBenchmark(data_path=tmp_path / "missing.pt").evaluate(loaded)


def test_mnist_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "mnist",
            "architecture": "linear",
            "input_size": 10,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 10)), "bias": torch.zeros(10)},
    )
    with pytest.raises(ValueError, match="input_size 784"):
        MnistBenchmark().load_submission(submission)


def test_fashion_mnist_evaluation_and_threshold(tmp_path: Path) -> None:
    data_path = tmp_path / "fashion-mnist-test.pt"
    _write_mnist_cache(data_path)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    loaded.state_dict["weight"][0, 0] = 1
    loaded.state_dict["weight"][1, 1] = 1
    loaded.model.load_state_dict(loaded.state_dict)
    score, score_std, passed, details = FashionMnistBenchmark(data_path=data_path).evaluate(loaded)
    assert score == 1.0
    assert score_std == 0.0
    assert passed is True
    assert details == {"samples": 2, "normalisation": False}


def test_fashion_mnist_requires_data_file(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 784,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 784)), "bias": torch.zeros(10)},
    )
    loaded = materialize_submission(submission)
    with pytest.raises(FileNotFoundError, match="Fashion-MNIST test data is unavailable"):
        FashionMnistBenchmark(data_path=tmp_path / "missing.pt").evaluate(loaded)


def test_fashion_mnist_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "fashion-mnist",
            "architecture": "linear",
            "input_size": 10,
            "output_size": 10,
        },
        {"weight": torch.zeros((10, 10)), "bias": torch.zeros(10)},
    )
    with pytest.raises(ValueError, match="Fashion-MNIST submissions"):
        FashionMnistBenchmark().load_submission(submission)


def test_chess_eval_registers_by_name() -> None:
    assert create_benchmark("chess-eval").name == "chess-eval"
    assert ChessEvalBenchmark().seeds == tuple(range(20))


def test_chess_eval_material_sign_known_vectors() -> None:
    assert material_of((0,) * 64) == 0
    assert label_of((0,) * 64) == DRAW
    queen = (5,) + (0,) * 63
    assert material_of(queen) == 9
    assert label_of(queen) == WHITE_BETTER
    black_rook = (-4,) + (0,) * 63
    assert material_of(black_rook) == -5
    assert label_of(black_rook) == BLACK_BETTER
    trade = (1,) + (-1,) + (0,) * 62
    assert material_of(trade) == 0
    assert label_of(trade) == DRAW


def test_chess_eval_evaluate_dummy_returns_tuple(tmp_path: Path) -> None:
    assert sample_dataset(0) == sample_dataset(0)
    assert sample_dataset(0) != sample_dataset(1)
    inputs, labels = sample_dataset(0)
    assert len(inputs) == len(labels) == 512
    position = tuple(int(round(value * 6.0)) for value in inputs[0])
    assert labels[0] == label_of(position)
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "chess-eval",
            "architecture": "linear",
            "input_size": 64,
            "output_size": 3,
        },
        {"weight": torch.zeros((3, 64)), "bias": torch.zeros(3)},
    )
    score, score_std, passed, details = ChessEvalBenchmark(seeds=(0, 1)).evaluate(
        materialize_submission(submission)
    )
    assert isinstance(score, float)
    assert isinstance(score_std, float)
    assert isinstance(passed, bool)
    assert 0.0 <= score <= 1.0
    assert score_std >= 0.0
    assert passed is False
    assert details["seeds"] == 2
    assert len(details["seed_accuracies"]) == 2


def test_chess_eval_rejects_wrong_shape(tmp_path: Path) -> None:
    submission = _write_submission(
        tmp_path,
        {
            "benchmark": "chess-eval",
            "architecture": "linear",
            "input_size": 8,
            "output_size": 3,
        },
        {"weight": torch.zeros((3, 8)), "bias": torch.zeros(3)},
    )
    with pytest.raises(ValueError, match="input_size 64"):
        ChessEvalBenchmark().load_submission(submission)
    second = tmp_path / "second"
    second.mkdir()
    bad_outputs = _write_submission(
        second,
        {
            "benchmark": "chess-eval",
            "architecture": "linear",
            "input_size": 64,
            "output_size": 2,
        },
        {"weight": torch.zeros((2, 64)), "bias": torch.zeros(2)},
    )
    with pytest.raises(ValueError, match="output_size 3"):
        ChessEvalBenchmark().load_submission(bad_outputs)
