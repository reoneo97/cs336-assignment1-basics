.PHONY: bpe-test lint

bpe-tr:
	uv run pytest tests/test_train_bpe.py
bpe-to:
	uv run pytest tests/test_tokenizer.py

lint: 
	ruff format