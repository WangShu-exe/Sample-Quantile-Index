import bisect
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 方法 1：纯抽样 ECDF（阶梯函数）
# ============================================================
def pred_pure_ecdf(sample_sorted, x, N):
    n = len(sample_sorted)
    pos = bisect.bisect_left(sample_sorted, x)
    return pos / n * N

# ============================================================
# 方法 2：分位桶（桶内线性插值）
# ============================================================
class QuantileBuckets:
    def __init__(self, sample_sorted, B):
        self.s = sample_sorted
        self.n = len(sample_sorted)
        self.B = B
        self.idx = [int(round(i * (self.n - 1) / B)) for i in range(B + 1)]
        self.bounds = [self.s[i] for i in self.idx]
        self.bounds[0] = -np.inf
        self.bounds[-1] = np.inf

    def predict(self, x, N):
        bi = max(0, min(bisect.bisect_right(self.bounds, x) - 1, self.B - 1))
        lo, hi = self.idx[bi], self.idx[bi + 1]
        p_lo, p_hi = lo / self.n, hi / self.n
        local = self.s[lo:hi + 1]
        m = len(local)
        if m == 0:
            local_p = 0.0
        elif m == 1:
            local_p = 0.0 if x <= local[0] else 1.0
        elif x <= local[0]:
            local_p = 0.0
        elif x >= local[-1]:
            local_p = 1.0
        else:
            j = bisect.bisect_left(local, x)
            v0, v1 = local[j - 1], local[j]
            frac = (x - v0) / (v1 - v0) if v1 > v0 else 0.0
            local_p = (j - 1 + frac) / (m - 1)
        return (p_lo + (p_hi - p_lo) * local_p) * N

# ============================================================
# 实验：验证 1/√n 缩放 (Log-Log 图)
# ============================================================
if __name__ == "__main__":
    N = 500_000
    print(f"生成总体数据 N={N} (Lognormal 分布)...")
    F = np.sort(np.random.lognormal(mean=3, sigma=0.8, size=N))
    Q = np.random.lognormal(mean=3, sigma=0.8, size=3000)
    true_ranks = np.searchsorted(F, Q)

    ns = [200, 500, 1000, 2000, 5000, 10000]
    bucket_means = []
    R = 30 # 重复 30 次取均值，画误差棒

    print(f"开始实验 (重复 R={R} 次)...")
    for n in ns:
        rels = []
        for _ in range(R):
            idx = np.random.choice(N, n, replace=False)
            sample = np.sort(F[idx])
            model = QuantileBuckets(sample, B=32)
            p = np.array([model.predict(x, N) for x in Q])
            rels.append(np.mean(np.abs(p - true_ranks) / N))
        bucket_means.append(np.mean(rels))
        print(f"n={n:>6}, 相对误差={np.mean(rels):.6f} ± {np.std(rels):.6f}")

    # 画图
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, bucket_means, marker='s', capsize=3, label='Quantile Buckets B=32')
    ref = bucket_means[-1] * np.sqrt(ns[-1]) / np.sqrt(np.array(ns))
    ax.plot(ns, ref, 'k--', alpha=0.5, label=r'\(\propto 1/\sqrt{n}\)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('sample size n')
    ax.set_ylabel('mean relative error')
    ax.set_title(f'Relative rank error vs n (N={N}, R={R})')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('exp1_loglog.png', dpi=150)
    print("→ 完美图已保存: exp1_loglog.png")
    plt.show()