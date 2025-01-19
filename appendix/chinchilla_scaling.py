#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chinchilla Scaling Laws Demonstration Script

이 스크립트는 Chinchilla 논문의 스케일링 법칙을 재현하기 위한 예시 코드를 담고 있습니다.
- GPT/Chinchilla 계열 모델의 파라미터 수를 계산하는 함수
- FLOPs 계산 방식
- Chinchilla 논문 Appendix A4, A9의 재현 예시
- 최종 Loss 함수 L(N, D) 기반의 Compute 최적화(Approach 3)
- Approach 2 (단순 파라미터/토큰 대응)에 대한 선형 회귀 예시

사용하려면:
    python3 chinchilla_scaling.py
등으로 실행해보세요.

https://chatgpt.com/c/678cd446-6fb8-8008-b6b6-4a5ebcd6c470
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
# 1. GPT 파라미터 계산
###############################################################################
def gpt_params(seq_len, vocab_size, d_model, num_heads, num_layers):
    """
    GPT 계열 모델에서 파라미터 수를 추정합니다.

    Arguments:
    - seq_len    : 모델이 처리하는 최대 시퀀스 길이
    - vocab_size : 단어 집합 크기 (토크나이저 구성이 얼마인지)
    - d_model    : 토큰 임베딩 차원(모델 내부 hidden dimension)
    - num_heads  : multi-head attention에서 head 개수
    - num_layers : 트랜스포머 레이어(layer) 개수

    Returns:
    - total_params: 전체 파라미터 수 (int)

    참고: 여기서는 GPT-2 small(124M) 수준과 비교하면 대략적으로 일치함을 확인할 수 있습니다.
    """
    ffw_size = 4 * d_model  # GPT에서 FFN(hidden)이 보통 d_model의 4배
    # 임베딩 부분(단, 여기서는 계산에서 제외)
    embeddings = d_model * vocab_size + d_model * seq_len

    # transformer block 당 계산:
    # 1. Attention (Q, K, V) 가중치와 바이ас
    attention = 3 * d_model**2 + 3 * d_model
    # 2. output projection (Attn -> hidden)
    attproj = d_model**2 + d_model
    # 3. FFN (W, b)
    ffw = d_model * ffw_size + ffw_size
    # 4. FFN output projection
    ffwproj = ffw_size * d_model + d_model
    # 5. layernorm 2개 (각각 d_model에 대해 scale, bias)
    layernorms = 2 * 2 * d_model

    # 최종 레이어 (ln_f, 최종 로지틱 projection)
    ln_f = 2 * d_model
    dense = d_model * vocab_size  # 보통 bias 제외

    # GPT-2 small과 맞추기 위해, 임베딩은 파라미터 수에서 빼는 convention
    total_params = (
        num_layers * (attention + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    )
    return total_params


def demo_gpt2_params():
    """
    GPT-2 small(124M) 파라미터를 계산해본 예시입니다.
    """
    gpt2_cfg = dict(
        seq_len=1024, vocab_size=50257, d_model=768, num_heads=12, num_layers=12
    )
    estimated_params = gpt_params(**gpt2_cfg)
    print("GPT-2 small 추정 파라미터: {:.3f} M".format(estimated_params / 1e6))
    print("OpenAI 보고: ~124M, 일치하는지 확인")


###############################################################################
# 2. Chinchilla 파라미터 계산
###############################################################################
def chinchilla_params(seq_len, vocab_size, d_model, num_heads, num_layers, ffw_size):
    """
    Chinchilla 계열 모델에서 파라미터 수를 추정합니다.
    (상대적 위치 임베딩 사용 등, GPT와 다른 부분 반영)

    Arguments:
    - seq_len    : 최대 시퀀스 길이
    - vocab_size : 어휘 집합 크기
    - d_model    : hidden dimension
    - num_heads  : multi-head attention head 수
    - num_layers : 트랜스포머 레이어 수
    - ffw_size   : FFN hidden 크기 (d_model의 몇 배인지는 모델 설계에 따라)

    Returns:
    - total_params: 전체 파라미터 수 (int)
    """
    # token 임베딩만 (상대적 위치임베딩인 경우, position 임베딩 파라미터가 다름)
    embeddings = d_model * vocab_size

    # Attention (Q, K, V)
    attention = 3 * d_model**2 + 3 * d_model

    # Chinchilla의 상대적 위치 인코딩 관련(논문 Table A.9 등):
    # - relative keys, content bias, relative bias
    relative_pos = d_model**2 + 2 * d_model

    # Attention output projection
    attproj = d_model**2 + d_model

    # Feed-Forward: W, b
    ffw = d_model * ffw_size + ffw_size
    # Feed-Forward output projection
    ffwproj = ffw_size * d_model + d_model

    # LayerNorm
    layernorms = 2 * 2 * d_model

    # 마지막 dense
    ln_f = 2 * d_model
    dense = d_model * vocab_size  # 보통 bias 제외

    # embeddings(토큰) 파라미터는 convention상 제외
    total_params = (
        num_layers * (attention + relative_pos + attproj + ffw + ffwproj + layernorms)
        + ln_f
        + dense
    )
    return total_params


###############################################################################
# 3. Chinchilla 50개 모델(논문 Appendix A9) 파라미터 추정 및 비교
###############################################################################
def demo_chinchilla_table_a9():
    """
    Chinchilla 논문 마지막 페이지(표 A9)에서 보고된 50개 모델의 파라미터 수를
    위 함수(chinchilla_params)로 대략 계산해보고, 논문 값과 비교합니다.
    """
    chinchilla_models_txt = """[[44000000.0, 512, 2048, 64, 8, 8], [57000000.0, 576, 2304, 64, 9, 9],
    [74000000.0, 640, 2560, 64, 10, 10], [90000000.0, 640, 2560, 64, 10, 13], [106000000.0, 640, 2560, 64, 10, 16],
    [117000000.0, 768, 3072, 64, 12, 12], [140000000.0, 768, 3072, 64, 12, 15], [163000000.0, 768, 3072, 64, 12, 18],
    [175000000.0, 896, 3584, 64, 14, 14], [196000000.0, 896, 3584, 64, 14, 16], [217000000.0, 896, 3584, 64, 14, 18],
    [251000000.0, 1024, 4096, 64, 16, 16], [278000000.0, 1024, 4096, 64, 16, 18], [306000000.0, 1024, 4096, 64, 16, 20],
    [425000000.0, 1280, 5120, 128, 10, 18], [489000000.0, 1280, 5120, 128, 10, 21], [509000000.0, 1408, 5632, 128, 11, 18],
    [552000000.0, 1280, 5120, 128, 10, 24], [587000000.0, 1408, 5632, 128, 11, 21], [632000000.0, 1536, 6144, 128, 12, 19],
    [664000000.0, 1408, 5632, 128, 11, 24], [724000000.0, 1536, 6144, 128, 12, 22], [816000000.0, 1536, 6144, 128, 12, 25],
    [893000000.0, 1792, 7168, 128, 14, 20], [1018000000.0, 1792, 7168, 128, 14, 23], [1143000000.0, 1792, 7168, 128, 14, 26],
    [1266000000.0, 2048, 8192, 128, 16, 22], [1424000000.0, 2176, 8704, 128, 17, 22], [1429000000.0, 2048, 8192, 128, 16, 25],
    [1593000000.0, 2048, 8192, 128, 16, 28], [1609000000.0, 2176, 8704, 128, 17, 25], [1731000000.0, 2304, 9216, 128, 18, 24],
    [1794000000.0, 2176, 8704, 128, 17, 28], [2007000000.0, 2304, 9216, 128, 18, 28], [2283000000.0, 2304, 9216, 128, 18, 32],
    [2298000000.0, 2560, 10240, 128, 20, 26], [2639000000.0, 2560, 10240, 128, 20, 30], [2980000000.0, 2560, 10240, 128, 20, 34],
    [3530000000.0, 2688, 10752, 128, 22, 36], [3802000000.0, 2816, 11264, 128, 22, 36], [4084000000.0, 2944, 11776, 128, 22, 36],
    [4516000000.0, 3072, 12288, 128, 24, 36], [6796000000.0, 3584, 14336, 128, 28, 40], [9293000000.0, 4096, 16384, 128, 32, 42],
    [11452000000.0, 4352, 17408, 128, 32, 47], [12295000000.0, 4608, 18432, 128, 36, 44], [12569000000.0, 4608, 18432, 128, 32, 47],
    [13735000000.0, 4864, 19456, 128, 32, 47], [14940000000.0, 4992, 19968, 128, 32, 49], [16183000000.0, 5120, 20480, 128, 40, 47]]"""
    chilchilla_models = json.loads(chinchilla_models_txt)

    print("Chinchilla 표 A9 일부 샘플 (마지막 5개 모델) 비교:")
    for m in chilchilla_models[-5:]:
        p_chi, d, f, k, h, l = m  # 논문 보고값(첫 번째가 공식 파라미터)
        # seq_len=1024, vocab_size=32000은 Chinchilla 논문에서 사용
        estimated = chinchilla_params(
            seq_len=1024,
            vocab_size=32000,
            d_model=d,
            num_heads=h,
            num_layers=l,
            ffw_size=f,
        )
        print(
            f" - our estimated params: {estimated / 1e6:.4f}M, "
            f"paper's: {p_chi / 1e6:.4f}M, "
            f"(d_model={d}, n_heads={h}, n_layers={l})"
        )


###############################################################################
# 4. Chinchilla FLOPs 계산
###############################################################################
def chinchilla_flops(seq_len, vocab_size, d_model, num_heads, num_layers, ffw_size):
    """
    Chinchilla 논문 Appendix F 기반으로 FLOPs(연산량)를 추정합니다.
    (forward + backward 모두 포함)

    주석에 있는 식은 논문 내에 제시된 내용 + Kaplan et al. 2020의
    전방/역전파 2배 계수를 조합한 것입니다.

    Arguments:
    - seq_len, vocab_size, d_model, num_heads, num_layers, ffw_size (위와 동일)
    Returns:
    - total_flops: forward + backward 연산량 (float)
    """
    key_size = d_model // num_heads

    # 논문에서는 Embedding, Logits 연산은 FLOPs 계산에서 제외한다고 밝힘(오류/타이포 언급)
    # => 아래에서 embeddings, logits 부분 제외
    # attention
    # Q, K, V projection
    attention = 2 * 3 * seq_len * d_model * (key_size * num_heads)
    # key @ query (seq_len x seq_len)
    attlogits = 2 * seq_len * seq_len * (key_size * num_heads)
    # softmax
    attsoftmax = 3 * num_heads * seq_len * seq_len
    # softmax 결과 @ value
    attvalue = 2 * seq_len * seq_len * (key_size * num_heads)
    # final linear
    attlinear = 2 * seq_len * (key_size * num_heads) * d_model
    att = attention + attlogits + attsoftmax + attvalue + attlinear

    # feed forward
    # (dense = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)는
    #  아래처럼 합친 것)
    dense = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)

    # forward
    forward_flops = num_layers * (att + dense)
    # backward = forward의 2배(관행)
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    return total_flops


def demo_chinchilla_table_a4():
    """
    Chinchilla 논문 Appendix A4에서 언급된 "Approximate vs Accurate FLOPs(F = 6*N*D)" 결과를
    예시로 재현해봅니다.

    표 A4의 일부 모델 구성에 대해, 정확 계산(chinchilla_flops)과
    단순 근사치(6*N*D)를 비교하고, 그 비율을 확인.
    """
    # (num_layers, d_model, ffw_size, num_heads, kv_size)
    # 논문 표 A4에 예시로 나오는 구성
    chinchilla_models_table4 = [
        [10, 640, 2560, 10, 64],
        [20, 1024, 4096, 16, 64],
        [24, 1280, 5120, 10, 128],
        [26, 1792, 7168, 14, 128],
        [28, 2048, 8192, 16, 128],
        [40, 3584, 14336, 28, 128],
    ]

    rows = []
    for num_layers, d_model, ffw_size, num_heads, kv_size in chinchilla_models_table4:
        args = dict(
            seq_len=2048,
            vocab_size=32000,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffw_size=ffw_size,
        )
        # 파라미터 수
        N = chinchilla_params(**args)
        # 정확 FLOPs
        F = chinchilla_flops(**args)
        # 단순 근사(논문에서 자주 사용하는 F=6*N*D)
        # 여기서 D=seq_len(2048)로 간주하는 경우
        approx_flops = 6 * args["seq_len"] * N
        # 표 A4에서는 seq_len(D) = 2048
        # 실제 논문에서는 더 복잡한 계산이 있지만, 여기서는 단순 비교

        out = {
            "seq_len": args["seq_len"],
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ffw_size": ffw_size,
            "N": N,
            "F": F,
            "approx_flops": approx_flops,
            # chinchilla_flops()는 seq_len * vocab_size * ... 형태이므로
            # Table 4와 동일 조건(2048 길이)을 기준으로 계산
            "chinch_flops": F,
            "ratio": F / approx_flops,
        }
        rows.append(out)

    df = pd.DataFrame(rows)
    print("Chinchilla 표 A4 재현 결과(일부)")
    print(df)
    # 여기서 ratio를 보면 ~1.0 근처에서 근사하는지 확인할 수 있습니다.


###############################################################################
# 5. Chinchilla 논문의 Approach 3: Loss 함수 L(N, D), Compute 최적화
###############################################################################
def L(N, D):
    """
    Chinchilla 논문(Approach 3)에서 제시된 최종 Loss를 근사하는 함수.

    L(N, D) = A / (N^alpha) + B / (D^beta) + E
    - A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69

    Args:
    - N (float): 파라미터 수
    - D (float): 데이터(토큰) 개수
    Returns:
    - 손실값 (float)
    """
    E = 1.69
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    return A / (N**alpha) + B / (D**beta) + E


def demo_approach_3():
    """
    Chinchilla 논문의 Approach 3를 예시로 재현:
    - L(N, D)를 이용하여, 주어진 compute 예산 C에서 N과 D의 조합을 순회
    - flops = 6 * N * D 라고 하고, flops <= C인 (N, D) 중에서 Loss가 최소가 되는 지점을 찾음
    """
    # 예시 Compute 예산(플롭 수)
    c = 2.21e19  # 예: Chinchilla 논문 Table A3의 row 1 (약간 임의)

    # model size (N) 스윕
    ns = 10 ** np.arange(7, 11, step=2**-4)  # 1e7 ~ 1e10
    # D = C / (6*N)
    ds = c / (6 * ns)
    losses = L(ns, ds)
    best = np.argmin(losses)

    print("Approach 3 예시: compute 예산 =", c)
    print(" - best model size (N): {:.2f}M".format(ns[best] / 1e6))
    print(" - best dataset size (D): {:.2f}B".format(ds[best] / 1e9))

    # 그래프(모델 크기에 따른 Loss 변화):
    plt.figure()
    plt.plot(ns, losses, label="Loss")
    plt.xscale("log")
    plt.xlabel("Model Size (N, log scale)")
    plt.ylabel("Loss")
    plt.title(f"Chinchilla Approach 3, C={c:.2e}")
    # 최적점 표시
    plt.axvline(ns[best], color="red", linestyle="--")
    plt.legend()
    plt.show()


###############################################################################
# 6. Chinchilla 논문의 Approach 2: 실험 데이터 기반 (파라미터 vs 토큰) 단순 회귀
###############################################################################
def demo_approach_2():
    """
    Chinchilla 논문(Approach 2)의 테이블에서 추출된 (파라미터, 토큰) 대응값들을
    직선(로그-로그 스케일)으로 근사(fit)하고, 특정 모델 사이즈에 대해 예측 토큰 수를 확인.
    """
    # 실제 논문에 기재된 Approach 2 값(대략):
    raw = [
        [400e6, 7.7e9],
        [1e9, 20.0e9],
        [10e9, 219.5e9],
        [67e9, 1.7e12],
        [175e9, 4.3e12],
        [280e9, 7.1e12],
        [520e9, 13.4e12],
        [1e12, 26.5e12],
        [10e12, 292.0e12],
    ]

    # 로그 변환 후 선형 회귀
    x = np.log10([item[0] for item in raw])  # 파라미터 수
    y = np.log10([item[1] for item in raw])  # 토큰 수
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # y = m*x + c
    print(
        f"Approach 2 fitting result: log10(tokens) = {m:.4f} * log10(params) + {c:.4f}"
    )

    # 시각화
    plt.figure()
    # 원 데이터
    params_raw = [item[0] for item in raw]
    tokens_raw = [item[1] for item in raw]
    plt.scatter(params_raw, tokens_raw, label="data", color="blue")

    # 회귀선을 그리기 위해, params를 10^7 ~ 10^13 범위 등으로 스윕
    params_line = np.logspace(7, 13, 50)
    tokens_line = 10 ** (m * np.log10(params_line) + c)
    plt.plot(params_line, tokens_line, label="fit", color="red")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Parameters (N)")
    plt.ylabel("Tokens (D)")
    plt.title("Approach 2 - Parameter vs Tokens")
    plt.grid()
    plt.legend()
    plt.show()

    # 특정 모델 크기에 대해 예측(예: GPT-2 small 124M 파라미터)
    xquery = 124e6
    yquery = 10 ** (m * np.log10(xquery) + c)
    print(
        f"예: {xquery:.2e} 파라미터 모델일 때, 회귀선에 따르면 ~{yquery:.2e} 토큰이 compute-optimal"
    )


###############################################################################
# 메인 실행 부분
###############################################################################
def main():
    print("===== GPT-2 small(124M) 파라미터 계산 예시 =====")
    demo_gpt2_params()
    print()

    print("===== Chinchilla 표 A9 파라미터 재현 예시 =====")
    demo_chinchilla_table_a9()
    print()

    print("===== Chinchilla 표 A4 FLOPs 재현 예시 =====")
    demo_chinchilla_table_a4()
    print()

    print("===== Chinchilla Approach 3 (Loss 함수) 예시 =====")
    demo_approach_3()
    print()

    print("===== Chinchilla Approach 2 (단순 (N,D) 데이터) =====")
    demo_approach_2()
    print()


if __name__ == "__main__":
    main()
