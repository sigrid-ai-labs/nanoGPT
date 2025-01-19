# Running nanoGPT on `uv` and `Macbook Pro M3 Max`
* https://github.com/karpathy/nanoGPT/issues/28

## Command
```
# configuration
make init

# step 1
make prepare-shakespeare-char
make train-shakespeare
make infer-shakespeare

# step 2
make prepare-gpt2
make train-gpt2
make sample-inference-gpt2

# step 3
make prepare-shakespeare
make train-shakespeare-gpt2
make infer-shakespeare-gpt2

# appendix
make transformer-analysis
make chinchilla_scaling
```