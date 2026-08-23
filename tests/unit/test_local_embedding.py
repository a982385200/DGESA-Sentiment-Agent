import numpy as np
import pytest

from sentiment_agent.embeddings.local_bge import DisabledEmbedding, LocalBGEEmbedding


class FakeEncoder:
    def encode(self, texts, **kwargs):
        assert texts == ["a", "b"]
        return np.array([[3.0, 4.0], [0.0, 2.0]])


def test_local_embedding_batches_and_normalizes() -> None:
    backend = LocalBGEEmbedding(model_id="local/test", encoder=FakeEncoder(), batch_size=8)
    vectors = backend.embed(["a", "b"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_local_embedding_rejects_empty_text() -> None:
    backend = LocalBGEEmbedding(model_id="local/test", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="empty"):
        backend.embed([""])


def test_disabled_embedding_never_loads_a_model() -> None:
    vectors = DisabledEmbedding().embed(["a", "b"])
    assert vectors.shape == (2, 1)
    np.testing.assert_array_equal(vectors, np.ones((2, 1), dtype=np.float32))
