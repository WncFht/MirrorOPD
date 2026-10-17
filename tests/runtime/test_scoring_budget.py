"""Sparse support metadata must fit the same staging budget as dense evidence."""

import pytest

from mirroropd.core.config import Estimator, Objective
from mirroropd.runtime.memory import Memory, cache_capacity, scoring_tile_size


def test_large_support_cannot_fit_in_the_old_fixed_allowance():
    objective = Objective(estimator=Estimator(topj=64), diagnostics="full")
    # Anchor IDs and probabilities alone require 64*(8+4) bytes, above 512.
    with pytest.raises(MemoryError, match="one scoring predictor"):
        scoring_tile_size(objective, Memory(scoring_buffer_bytes=512), vocab_size=128)
    memory = Memory(scoring_buffer_bytes=3000)
    tile = scoring_tile_size(objective, memory, vocab_size=128)
    assert tile * cache_capacity(objective, prefixes=1, vocab_size=128)["total_bytes"] <= 3000


def test_default_real_vocabulary_tile_is_unchanged():
    objective = Objective(estimator=Estimator("full", "full", 16), diagnostics="none")
    assert scoring_tile_size(objective, Memory(), vocab_size=151665) == 55
