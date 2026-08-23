# Local ModelScope Embeddings Design

## Goal

Download `BAAI/bge-m3` and `AI-ModelScope/multilingual-e5-small` from ModelScope into the project and let experiments select either model for fully local embedding generation while retaining API-based chat completion.

## Storage and Download

Model weights live under `models/embeddings/bge-m3/` and `models/embeddings/multilingual-e5-small/`. The `models/` directory is ignored by Git. A reproducible script uses ModelScope `snapshot_download` with explicit model IDs and local directories. Re-running the script is safe and reuses completed files.

## Configuration

`ModelConfig` gains `embedding_provider` with `openai` and `local` choices, plus local path, device, and batch-size fields. Existing OpenAI embedding configuration remains backward compatible. Local configuration must identify an existing model directory and never falls back to a remote Hugging Face download.

Example:

```yaml
model:
  embedding_provider: local
  embedding_model: bge-m3
  embedding_model_path: models/embeddings/bge-m3
  embedding_device: cpu
  embedding_batch_size: 16
```

## Runtime Architecture

A focused embedding factory constructs either `OpenAIEmbeddings` or LangChain `HuggingFaceEmbeddings`. `OpenAICompatibleClient` continues to expose the unchanged `embed()` method and accepts injected embedding objects for offline unit tests. Chat access remains `ChatOpenAI` and still requires the configured chat API key.

Local models use normalized dense vectors. BGE-M3 encodes text directly. Multilingual E5-small applies the `query: ` prompt consistently because this project performs symmetric semantic-similarity retrieval between samples and stored experiences.

## Errors and Offline Guarantees

Local mode validates the directory before model construction. Missing models raise a clear error containing the ModelScope download command. Loading always uses the explicit local filesystem path, so experiment execution cannot silently fetch weights from Hugging Face.

## Dependencies

Add `langchain-huggingface`, `sentence-transformers`, and `modelscope` through `uv`, and commit the updated `uv.lock`. Model weights are artifacts and remain untracked.

## Testing and Verification

Tests are written before production changes. Unit tests cover configuration validation, provider selection, E5 prefixing, BGE direct encoding, missing-directory errors, and injected-client behavior without network access. The full offline unit and integration suite must pass. After downloading, each real local model receives a small embedding smoke test that checks vector count, dimensions, finite values, and normalization.
