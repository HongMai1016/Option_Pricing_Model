"""
可视化工具集

包含四类图：
  1. 希腊字母曲线  (Greeks vs Spot) — 观察 Δ/Γ/ν/Θ/ρ 随标的价格的变化
  2. 二叉树收敛图  (CRR Convergence) — 展示 CRR 价格如何收敛至 BS
  3. 二叉树结构图  (Tree Diagram)    — 将小步数的股价 + 期权价值树可视化
  4. 策略收益图    (Strategy Payoff) — 绘制组合策略的到期 payoff 与 profit

图片统一保存到 figures/ 目录。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头后端，无显示器环境也能运行
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

from option_models import OptionContract, OptionType, ExerciseType
from pricing_engines import BinomialTreeEngine, BlackScholesEngine


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- 中文字体（SimHei 黑体） ----
_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_FONT_PATH):
    fontManager.addfont(_FONT_PATH)

_CN_FONT = FontProperties(fname=_FONT_PATH, size=11) if os.path.exists(_FONT_PATH) else None


def _cn(text: str) -> str:
    """返回中文文本（若中文字体不可用则回退为英文 key）。"""
    # 有字体就用中文，没有就用原样
    return text


# ==============================================================================
# 1. 希腊字母 vs 标的价格曲线
# ==============================================================================

def plot_greeks_vs_spot(
    K: float = 100, T: float = 1.0, r: float = 0.05, sigma: float = 0.2,
    S_range: np.ndarray | None = None,
    filename: str = "greeks_vs_spot.png",
) -> str:
    """绘制欧式看涨期权的五个解析希腊字母随 S₀ 的变化曲线。

    Parameters
    ----------
    K, T, r, sigma : float
        市场参数。
    S_range : np.ndarray or None
        标的价格采样范围，默认 K×0.4 ~ K×1.6。
    filename : str
        输出文件名。

    Returns
    -------
    str
        保存路径。
    """
    if S_range is None:
        S_range = np.linspace(K * 0.4, K * 1.6, 200)

    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
    for S in S_range:
        c = OptionContract(S0=S, K=K, r=r, sigma=sigma, T=T,
                           option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)
        eng = BlackScholesEngine(c)
        deltas.append(eng.delta())
        gammas.append(eng.gamma())
        vegas.append(eng.vega())
        thetas.append(eng.theta())
        rhos.append(eng.rho())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [
        ("Delta", deltas, "Δ = N(d₁)"),
        ("Gamma", gammas, "Γ = N'(d₁)/(S₀σ√T)"),
        ("Vega (per 1%)", vegas, "ν = S₀·N'(d₁)·√T / 100"),
        ("Theta (per day)", thetas, "Θ = ∂C/∂t / 365"),
        ("Rho (per 1%)", rhos, "ρ = K·T·e^{-rT}·N(d₂) / 100"),
    ]
    for ax, (name, vals, label) in zip(axes.flat, metrics):
        ax.plot(S_range, vals, linewidth=2)
        ax.axvline(K, color="gray", linestyle=":", linewidth=1, label=f"K={K}")
        ax.set_title(name)
        ax.set_xlabel("Spot S₀")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")

    fig.suptitle(f"Greeks of European Call  (K={K}, T={T}, r={r}, σ={sigma})",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ==============================================================================
# 2. 二叉树收敛图
# ==============================================================================

def plot_binomial_convergence(
    contract: OptionContract,
    N_max: int = 200,
    filename: str = "crr_convergence.png",
) -> str:
    """绘制 CRR 二叉树价格随步数 N 的收敛过程。

    展示经典的"奇偶振荡 + O(1/N) 收敛"曲线，与 BS 价格对比。

    Parameters
    ----------
    contract : OptionContract
        期权合约（通常用欧式，以便与 BS 比较）。
    N_max : int
        最大步数。
    filename : str
        输出文件名。

    Returns
    -------
    str
        保存路径。
    """
    import warnings, sys, logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    _old = sys.stderr
    sys.stderr = open(os.devnull, "w")

    Ns = list(range(2, N_max + 1))
    crr_prices = [BinomialTreeEngine(contract, N=N).price() for N in Ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Ns, crr_prices, label="CRR Binomial Price", linewidth=1.2)

    if not contract.is_american:
        bs_price = BlackScholesEngine(contract).price()
        ax.axhline(bs_price, color="red", linestyle="--",
                   label=f"BS Price = {bs_price:.4f}")

    ax.set_xlabel("Number of Steps N")
    ax.set_ylabel("Option Price")
    opt_type = "Call" if contract.is_call else "Put"
    ax.set_title(
        f"CRR Convergence — European {opt_type}  "
        f"(S₀={contract.S0}, K={contract.K}, T={contract.T}, "
        f"r={contract.r}, σ={contract.sigma})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    sys.stderr = _old
    return path


# ==============================================================================
# 3. 二叉树结构图
# ==============================================================================

def plot_binomial_tree(
    contract: OptionContract,
    N: int = 4,
    filename: str = "binomial_tree.png",
) -> str:
    """将 CRR 二叉树的股价 + 期权价值可视化。

    每个节点显示「S=股价 / V=期权价值」。
    N 不宜过大，否则节点挤在一起看不清楚。

    Parameters
    ----------
    contract : OptionContract
        期权合约。
    N : int
        步数（建议 ≤ 6）。
    filename : str
        输出文件名。

    Returns
    -------
    str
        保存路径。
    """
    eng = BinomialTreeEngine(contract, N=N)
    _, stock_tree, value_tree = eng.price(return_tree=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(N + 1):
        for j in range(i + 1):
            x = i
            y = j - i / 2.0
            S_val = stock_tree[i, j]
            V_val = value_tree[i, j]
            ax.scatter(x, y, s=600, color="lightblue", edgecolor="navy", zorder=3)
            ax.text(x, y, f"S={S_val:.2f}\nV={V_val:.2f}",
                    ha="center", va="center", fontsize=7, zorder=4)
            if i < N:
                ax.plot([x, x + 1], [y, y + 0.5], color="gray", linewidth=0.7, zorder=1)
                ax.plot([x, x + 1], [y, y - 0.5], color="gray", linewidth=0.7, zorder=1)

    ax.set_xlabel("Time Step")
    opt_type = "Call" if contract.is_call else "Put"
    ex_type = "American" if contract.is_american else "European"
    ax.set_title(
        f"CRR Binomial Tree (N={N}) — {ex_type} {opt_type}\n"
        f"S₀={contract.S0}, K={contract.K}, T={contract.T}, "
        f"r={contract.r}, σ={contract.sigma}"
    )
    ax.set_yticks([])
    ax.set_xticks(range(N + 1))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ==============================================================================
# 4. 策略收益图
# ==============================================================================

def plot_strategy_payoff(
    name: str,
    legs: list[tuple[int, OptionContract]],
    S_min: float | None = None,
    S_max: float | None = None,
    n_points: int = 300,
    filename: str | None = None,
) -> str:
    """绘制期权组合策略的到期收益图与利润图。

    Parameters
    ----------
    name : str
        策略名称。
    legs : list[tuple[int, OptionContract]]
        每条腿 (position, OptionContract)，position: +1=long, -1=short。
    S_min, S_max : float or None
        x 轴范围，默认自动根据行权价确定。
    n_points : int
        采样点数。
    filename : str or None
        输出文件名。

    Returns
    -------
    str
        保存路径。
    """
    # 自动确定 x 轴范围
    Ks = [leg[1].K for leg in legs]
    if S_min is None:
        S_min = max(0.01, min(Ks) * 0.5)
    if S_max is None:
        S_max = max(Ks) * 1.5

    S = np.linspace(S_min, S_max, n_points)
    payoff = np.zeros_like(S)
    initial_cost = 0.0
    for pos, opt in legs:
        payoff += pos * opt.payoff(S)
        if not opt.is_american:
            initial_cost += pos * BlackScholesEngine(opt).price()
        else:
            initial_cost += pos * BinomialTreeEngine(opt, N=500).price()

    profit = payoff - initial_cost

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S, payoff, label="Payoff at expiry", linewidth=2)
    ax.plot(S, profit, label="Profit (payoff − cost)", linewidth=2, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    for Kv in sorted(set(Ks)):
        ax.axvline(Kv, color="gray", linewidth=0.5, linestyle=":")
        ax.text(Kv, ax.get_ylim()[1] * 0.95, f"K={Kv:g}",
                rotation=90, va="top", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("Underlying Price at Expiry S_T")
    ax.set_ylabel("P&L")
    ax.set_title(f"{name}  (initial cost = {initial_cost:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    fname = filename or f"strategy_{name.replace(' ', '_').lower()}.png"
    path = os.path.join(FIG_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
