import bisect
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 核心结构：纯抽样 ECDF 线性插值
# ============================================================
class PureInterpECDF:
    def __init__(self, full_data, n_global=2000):
        self.N = len(full_data)
        # 从全量中随机抽样
        idx = np.random.choice(self.N, n_global, replace=False)
        self.samples = np.sort(full_data[idx])
        
    def predict(self, x):
        # 边界处理
        if x <= self.samples[0]: return 0
        if x >= self.samples[-1]: return self.N
        
        # 二分查找
        j = bisect.bisect_left(self.samples, x)
        v0, v1 = self.samples[j-1], self.samples[j]
        
        # 局部线性插值
        frac = (x - v0) / (v1 - v0) if v1 > v0 else 0.0
        
        # 映射为全局概率并返回绝对排名预测
        p = (j - 1 + frac) / (len(self.samples) - 1)
        return p * self.N

# ============================================================
# 实验：相对误差 vs 抽样量 n (N=500k, R=30)
# ============================================================
if __name__ == "__main__":
    N = 500_000
    print(f"Generating dataset: N={N} (Lognormal)...")
    F = np.sort(np.random.lognormal(mean=3, sigma=0.8, size=N))
    Q = np.random.lognormal(mean=3, sigma=0.8, size=3000)
    true_ranks = np.searchsorted(F, Q)
    
    ns = [200, 500, 1000, 2000, 5000, 10000]
    R = 30 # 重复次数
    means, stds = [], []
    
    print("Running experiments...")
    for n in ns:
        rel_errors = []
        for _ in range(R):
            model = PureInterpECDF(F, n_global=n)
            preds = np.array([model.predict(x) for x in Q])
            rel_errors.append(np.mean(np.abs(preds - true_ranks) / N))
        
        mean_err = np.mean(rel_errors)
        std_err = np.std(rel_errors)
        means.append(mean_err)
        stds.append(std_err)
        print(f"n={n:>5} | Mean Relative Error={mean_err:.6f} ± {std_err:.6f}")

    # 计算 log-log 拟合斜率
    log_n = np.log(ns)
    slope = np.polyfit(log_n, np.log(means), 1)[0]
    print(f"\n=> Fitted log-log slope: {slope:.3f} (Theoretical: -0.5)")

    # ============================================================
    # 画图
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, means, yerr=stds, marker='o', capsize=3, label='Subsampled ECDF')
    
    # 参考线 ∝ 1/√n
    ref = means[-1] * np.sqrt(ns[-1]) / np.sqrt(np.array(ns))
    ax.plot(ns, ref, 'k--', alpha=0.5, label=r'$\propto 1/\sqrt{n}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Mean Relative Error')
    ax.set_title(f'Relative Rank Error vs n (N={N}, R={R})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp1_loglog.png', dpi=150)
    print("=> Figure saved as exp1_loglog.png")
    plt.show()
