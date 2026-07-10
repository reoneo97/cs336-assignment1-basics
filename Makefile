.PHONY: bpe-test lint

bpe-test:
	uv run pytest tests/test_tokenizer.py
	uv run pytest tests/test_train_bpe.py
model-test:
	uv run pytest tests/test_model.py
lint: 
	ruff format