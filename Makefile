prepare-shakespeare:
	uv run data/shakespeare_char/prepare.py

train-shakespeare:
	uv run scripts/train.py config/train_shakespeare_char.py

infer-shakespeare:
	uv run scripts/sample.py --out_dir=out-shakespeare-char

prepare-gpt2:
	uv run data/openwebtext/prepare.py

train-gpt2:
	# on macbook pro
	export OMP_NUM_THREADS=4; uv run scripts/train.py config/train_gpt2.py --compile=False