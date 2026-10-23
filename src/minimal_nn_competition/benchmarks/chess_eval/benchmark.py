"""Material-advantage benchmark: predict the sign of the piece-value sum."""

import random
import statistics

from minimal_nn_competition.model.entry import materialize_submission

# Standard values P=1 N=3 B=3 R=5 Q=9.
VALUES = (0, 1, 3, 3, 5, 9)
WHITE_BETTER, DRAW, BLACK_BETTER = range(3)
DEFAULT_SEEDS = tuple(range(20))
SAMPLES_PER_SEED = 512


def material_of(position):
    """Net white-minus-black material for a 64-tuple position."""
    total = 0
    for piece in position:
        if piece > 0:
            total += VALUES[piece]
        elif piece < 0:
            total -= VALUES[-piece]
    return total


def label_of(position):
    """Map a position to WHITE_BETTER, DRAW, or BLACK_BETTER."""
    score = material_of(position)
    if score:
        return WHITE_BETTER if score > 0 else BLACK_BETTER
    return DRAW


def sample_dataset(seed):
    """Sample a deterministic eval set of positions and material labels."""
    rng = random.Random(seed * 65537 + 7919)
    inputs = []
    labels = []
    for _ in range(SAMPLES_PER_SEED):
        position = [0] * 64
        for square in rng.sample(range(64), rng.randint(2, 12)):
            kind = rng.randint(1, 5)
            position[square] = kind if rng.getrandbits(1) else -kind
        labels.append(label_of(position))
        inputs.append([piece / 6.0 for piece in position])
    return inputs, labels


class ChessEvalBenchmark:
    name = "chess-eval"

    def __init__(self, seeds=DEFAULT_SEEDS):
        if not seeds:
            raise ValueError("Chess-eval benchmark requires at least one seed")
        self.seeds = seeds

    def load_submission(self, submission):
        if submission.config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match Chess-eval")
        if submission.config.input_size != 64:
            raise ValueError("Chess-eval submissions must declare input_size 64")
        if submission.config.output_size != 3:
            raise ValueError("Chess-eval submissions must declare output_size 3")
        return materialize_submission(submission)

    def evaluate(self, submission):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Chess-eval evaluation") from exc
        accuracies = []
        with torch.no_grad():
            for seed in self.seeds:
                inputs, labels = sample_dataset(seed)
                batch = torch.tensor(inputs)
                expected = torch.tensor(labels)
                predicted = torch.argmax(submission.model(batch), dim=1)
                accuracies.append((predicted == expected).float().mean().item())
        return (
            statistics.fmean(accuracies),
            statistics.pstdev(accuracies),
            min(accuracies) == 1.0,
            {"seed_accuracies": accuracies, "seeds": len(self.seeds)},
        )
