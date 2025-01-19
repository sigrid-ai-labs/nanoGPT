prepare-shakespeare-char:
	uv run data/shakespeare_char/prepare.py

prepare-shakespeare:
	uv run data/shakespeare/prepare.py

train-shakespeare:
	export OMP_NUM_THREADS=4; uv run scripts/train.py config/train_shakespeare_char.py

infer-shakespeare:
	uv run scripts/sample.py --out_dir=./data/out-shakespeare-char

prepare-gpt2:
	uv run data/openwebtext/prepare.py

train-gpt2:
	# on macbook pro
	export OMP_NUM_THREADS=4; uv run scripts/train.py config/train_gpt2.py --compile=False

train-shakespeare-gpt2:
	# python train.py config/finetune_shakespeare.py
	export OMP_NUM_THREADS=4; uv run scripts/train.py config/finetune_shakespeare.py --compile=False

infer-shakespeare-gpt2:
	uv run scripts/sample.py --out_dir=./data/out-shakespeare

sample-inference-gpt2:
	uv run ./scripts/sample.py \
    --init_from=resume \
    --start="What is the answer to the president of South Korea?" \
    --num_samples=5 --max_new_tokens=100