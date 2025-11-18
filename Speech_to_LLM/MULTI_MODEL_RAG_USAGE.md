# Multi_model_RAG.py Usage Guide

## Tokenizer Support Commands

### Auto-uses token-based chunking with OpenAI

```bash
python Multi_Model_RAG.py --openai --documents ./documents
```

### Auto-uses character-based chunking with Ollama

```bash
python Multi_Model_RAG.py --documents ./documents
```

### Force token-based even with Ollama

```bash
python Multi_Model_RAG.py --use-tokenizer --documents ./documents
```

### Custom chunk sizes

```bash
python Multi_Model_RAG.py --openai --chunk-size 2000 --chunk-overlap 400 --documents ./documents
```

