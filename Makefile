.PHONY: bpe-test

bpe-test:
	uv run pytest tests/test_train_bpe.py -vv