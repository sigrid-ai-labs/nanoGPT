#!/usr/bin/env python3
"""
transformer_analysis.py

https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb

이 스크립트는 GPT-2(Transformer) 계열 모델의 파라미터 개수, 체크포인트 크기,
FLOPs(연산량), 모델 FLOPs 활용도(MFU) 등을 이론적으로 계산하고 분석하기 위한 예시 코드입니다.

주요 기능:
1) 모델 파라미터 개수 계산 (params 함수)
2) 체크포인트 예상 크기 계산
3) Forward/Backward 시 FLOPs 계산 (flops 함수)
4) A100 GPU 성능 대비 모델 연산 활용도(MFU) 추정
5) 데이터셋 전체 학습에 필요한 시간 추정 (6*N*D 공식 활용)

작성일: 2025-01-19
"""

from collections import OrderedDict

# -----------------------------
# 모델 구성 파라미터 설정
# -----------------------------
block_size = 1024  # sequence length(맥스 토큰 시퀀스 길이)
vocab_size = 50257  # GPT-2 vocab size
n_layer = 12  # Transformer block 개수
n_head = 12  # multi-head attention 개수
n_embd = 768  # 임베딩 차원 수
bias = False  # 이론 계산 단순화를 위해 bias=False 가정
assert not bias, "이 예시에서는 bias=False로 가정합니다."


# -----------------------------
# 1) 파라미터 개수 계산 함수
# -----------------------------
def params():
    """
    estimates the number of parameters in the model
    - 임베딩, Attention, MLP 레이어 등 구성을 나눠서 개별 파라미터 수를 계산하고
      합산합니다.
    - OrderedDict에 부분별 파라미터 수를 저장해 최종 total을 반환합니다.
    """
    out = OrderedDict()

    # (1) 임베딩 파트
    #    - position embedding: n_embd x block_size
    #    - token embedding: n_embd x vocab_size
    out["emebedding/position"] = n_embd * block_size
    out["embedding/token"] = n_embd * vocab_size
    out["embedding"] = out["emebedding/position"] + out["embedding/token"]

    # (2) Attention 레이어당 파라미터
    #    - LayerNorm (bias=False로 가정) => n_embd
    #    - k, q, v projection: n_embd -> 3 * n_embd
    #    - output projection: n_embd -> n_embd
    out["attention/ln"] = n_embd
    out["attention/kqv"] = n_embd * 3 * n_embd
    out["attention/proj"] = n_embd * n_embd
    out["attention"] = (
        out["attention/ln"] + out["attention/kqv"] + out["attention/proj"]
    )

    # (3) MLP 레이어당 파라미터
    #     - feed-forward hidden size: 4 * n_embd
    #     - 첫 번째 matmul: n_embd -> 4*n_embd
    #     - 두 번째 matmul: 4*n_embd -> n_embd
    out["mlp/ln"] = n_embd
    ffw_size = 4 * n_embd
    out["mlp/ffw"] = n_embd * ffw_size
    out["mlp/proj"] = ffw_size * n_embd
    out["mlp"] = out["mlp/ln"] + out["mlp/ffw"] + out["mlp/proj"]

    # (4) Transformer block (Attention + MLP)
    out["block"] = out["attention"] + out["mlp"]
    out["transformer"] = n_layer * out["block"]

    # (5) 최종 레이어노름 및 출력 로지틱스
    out["ln_f"] = n_embd  # final layernorm(가중치만, bias=False)
    out["dense"] = 0  # GPT-2는 output layer가 token embedding과 파라미터 공유

    # (6) 총 파라미터 수
    out["total"] = out["embedding"] + out["transformer"] + out["ln_f"] + out["dense"]

    return out


# -----------------------------
# 2) FLOPs 계산 함수
# -----------------------------
def flops():
    """
    이 함수는 한 번의 forward pass에서 소비되는 부동소수점 연산량(FLOPs)을 추정합니다.
    - Attention: q, k, v 연산 + 어텐션 스코어 계산 + 값(value) 결합 + 최종 proj
    - MLP: 두 개의 큰 matmul
    - forward + backward의 총 FLOPs도 계산
    """
    out = OrderedDict()
    head_size = n_embd // n_head

    # Attention 부분
    # 1) q, k, v projection
    out["attention/kqv"] = 2 * block_size * (n_embd * 3 * n_embd)
    # 2) attention scores 계산 => (T x T x n_embd)
    out["attention/scores"] = 2 * block_size * block_size * n_embd
    # 3) values reduction => (T x T x head_size) * n_head
    out["attention/reduce"] = 2 * n_head * (block_size * block_size * head_size)
    # 4) 최종 proj => (n_embd x n_embd)
    out["attention/proj"] = 2 * block_size * (n_embd * n_embd)

    # attention 총합
    out["attention"] = sum(
        out["attention/" + k] for k in ["kqv", "scores", "reduce", "proj"]
    )

    # MLP 부분
    ffw_size = 4 * n_embd
    out["mlp/ffw1"] = 2 * block_size * (n_embd * ffw_size)
    out["mlp/ffw2"] = 2 * block_size * (ffw_size * n_embd)
    out["mlp"] = out["mlp/ffw1"] + out["mlp/ffw2"]

    # Transformer block과 output
    out["block"] = out["attention"] + out["mlp"]
    out["transformer"] = n_layer * out["block"]

    # 최종 dense (어휘 분류) => (n_embd x vocab_size)
    out["dense"] = 2 * block_size * (n_embd * vocab_size)

    # Forward total: transformer + dense
    out["forward_total"] = out["transformer"] + out["dense"]

    # Backward total: 보통 forward의 2배로 추정
    out["backward_total"] = 2 * out["forward_total"]

    # 전체(F+B)
    out["total"] = out["forward_total"] + out["backward_total"]

    return out


# -----------------------------
# 3) PaLM paper식 FLOPs 추정
# -----------------------------
def palm_flops():
    """
    PaLM 논문에서 언급된 6*N + 12*L*H*Q*T 공식을 기반으로
    total FLOPs(일회 forward-pass용)를 추정하는 함수.
    """
    p = params()
    N = p["total"] - p["emebedding/position"]  # non-embedding model parameters
    L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
    mf_per_token = 6 * N + 12 * L * H * Q * T
    return mf_per_token * block_size


# -----------------------------
# 메인 함수
# -----------------------------
def main():
    # (A) 파라미터 수 계산
    p = params()
    params_total = p["total"]
    print(f"[params()] 파라미터 총합: {params_total}")

    # 참고로 GPT-2 small의 공식 파라미터 수는 약 124,337,664
    expected_gpt2_small = 124337664
    print(
        f"비교 => GPT-2 small 공식 값: {expected_gpt2_small}, "
        f"둘이 일치?: {params_total == expected_gpt2_small}\n"
    )

    # 각 파트별 파라미터 비율 출력
    print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
    for k, v in p.items():
        ratio = 100.0 * v / params_total
        print(f"{k:20s} {v:10d} {ratio:10.4f}")

    # 체크포인트 크기 추정 (float32 가정, AdamW의 2개 통계 버퍼)
    params_bytes = params_total * 4
    params_and_buffers_bytes = params_bytes + 2 * params_bytes
    print(f"\n체크포인트 예상 크기: {params_and_buffers_bytes / 1e9:.2f} GB")

    # (B) FLOPs 계산
    f = flops()
    flops_total_forward = f["forward_total"]
    print("\n[FLOPs] Forward pass 총 연산량")
    print(f"{'name':20s} {'flops':14s} {'ratio (%)':10s}")
    for k, v in f.items():
        ratio = 100.0 * v / flops_total_forward
        print(f"{k:20s} {v:14d} {ratio:10.4f}")

    # (C) PaLM식 FLOPs 비교
    palm_val = palm_flops()
    ratio = palm_val / f["total"]
    print(f"\n[PaLM 방정식 비교]")
    print(f"palm_flops: {palm_val}, flops()['total']: {f['total']}, ratio: {ratio:.4f}")

    # (D) A100 대비 추정 MFU 계산 예시
    #   - batch_size, iteration 수행 시간을 가정하고, 이론적 TFLOPS 대비
    #     실제 FLOPs 활용률(%) 추정
    batch_size_effective = 20 * 5  # batch_size=20, grad_accum=5
    measured_time = 0.755  # 한 iteration당 걸린 시간(초)
    measured_throughput = batch_size_effective / measured_time

    # forward+backward FLOPs
    flops_achieved = f["total"] * measured_throughput

    a100_flops_promised = 312e12  # A100: 312 TFLOPS (bfloat16)
    fraction_used = flops_achieved / a100_flops_promised * 100.0
    print(f"\n추정 MFU: {fraction_used:.2f}% (A100 대비)")

    # (E) 전체 학습 시간(6*N*D 공식)
    tokens_num = 300e9  # 300B tokens
    a100_flops = 312e12  # 312 TFLOPS
    assumed_mfu = 0.3  # 추정 활용도 30%
    # 8장 A100 시스템에서 30%만 사용된다고 가정
    flops_throughput = a100_flops * 8 * assumed_mfu
    # 6 * N * D 계산 (N은 total 파라미터, D는 토큰 수)
    flops_needed = 6 * params_total * tokens_num
    time_needed_s = flops_needed / flops_throughput
    time_needed_days = time_needed_s / 3600 / 24
    print(f"\n[학습 시간 추정]")
    print(f"6*N*D 기준으로 추정: 약 {time_needed_days:.2f} 일 소요 예상")


if __name__ == "__main__":
    main()
