import hashlib

import pytest

from onepass.model.layout import TokenizerAdapter


class FakeTok:
    """Deterministic word-level tokenizer over the real special-token ids; no network, no files."""

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 50281, 50282, 50284, 50283

    def encode(self, text, add_special_tokens=False):
        return [100 + int(hashlib.md5(w.encode()).hexdigest(), 16) % 40000 for w in text.split()]


@pytest.fixture(scope="session")
def tok():
    return TokenizerAdapter(FakeTok())
