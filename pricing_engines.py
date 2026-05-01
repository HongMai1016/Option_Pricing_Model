"""
定价引擎 —— 二叉树模型 & Black-Scholes 解析解

理论依据：Shreve《金融随机分析 第1卷》二叉树资产定价模型。

核心思想：在无套利假设下，存在唯一的风险中性概率测度 P̃，
使得所有衍生品价格等于其折现期望 payoff。

二叉树 CRR 参数化（Cox-Ross-Rubinstein, 1979）：
    Δt = T / N
    u  = exp( σ √Δt )          —— 上升乘数
    d  = exp(-σ √Δt ) = 1/u    —— 下降乘数（保证 recombining tree）
    p̃ = (e^{rΔt} - d) / (u - d) —— 风险中性上升概率
    q̃ = 1 - p̃                    —— 风险中性下降概率

向后递推定价公式（Backward Induction）：
    V_n(j) = e^{-rΔt} [ p̃ · V_{n+1}(j+1) + q̃ · V_{n+1}(j) ]

美式期权扩展 —— 最优停时（Optimal Stopping Time）：
    V_n(j) = max( Intrinsic_n(j), Continuation_n(j) )
在每个节点比较"立即行权价值"与"继续持有价值"。

Black-Scholes 解析解（仅适用于欧式期权）用于收敛性验证。
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm

from option_models import OptionContract, OptionType, ExerciseType, PricingResult


# ==============================================================================
# 二叉树定价引擎
# ==============================================================================

class BinomialTreeEngine:
    """基于 CRR 参数化的二叉树期权定价引擎。

    支持欧式与美式、看涨与看跌期权的定价。

    Parameters
    ----------
    contract : OptionContract
        期权合约参数。
    N : int
        二叉树步数。N 越大，定价越精确，收敛到 Black-Scholes。
    """

    def __init__(self, contract: OptionContract, N: int) -> None:
        if N < 1:
            raise ValueError("二叉树步数 N 必须 ≥ 1。")
        self.contract = contract
        self.N = N
        # 预计算 CRR 参数
        self._dt: float = 0.0
        self._u: float = 0.0
        self._d: float = 0.0
        self._p_tilde: float = 0.0   # 风险中性上升概率 p̃
        self._q_tilde: float = 0.0   # 风险中性下降概率 q̃
        self._discount: float = 0.0  # 单步折现因子
        self._compute_crr_params()

    # ---- CRR 参数计算 -------------------------------------------------------

    def _compute_crr_params(self) -> None:
        """计算 Cox-Ross-Rubinstein 二叉树参数。

        根据无套利假设计算风险中性概率 p̃, q̃：
            p̃ = (e^{rΔt} - d) / (u - d)
            q̃ = (u - e^{rΔt}) / (u - d) = 1 - p̃

        上升/下降乘数满足 u·d = 1，保证树的 recombining 性质。

        Raises
        ------
        ValueError
            若 p̃ 不在 [0,1] 区间内（参数不满足无套利条件），
            通常需要增大 N 使得 Δt 足够小。
        """
        c = self.contract
        self._dt = c.T / self.N
        self._u = math.exp(c.sigma * math.sqrt(self._dt))
        self._d = 1.0 / self._u   # u·d = 1, recombining tree
        growth = math.exp(c.r * self._dt)  # e^{rΔt}
        self._p_tilde = (growth - self._d) / (self._u - self._d)
        self._q_tilde = 1.0 - self._p_tilde
        self._discount = math.exp(-c.r * self._dt)

        # 健壮性检查：p̃ 应在 [0,1] 内（无套利条件）
        if not (0.0 <= self._p_tilde <= 1.0):
            raise ValueError(
                f"风险中性概率 p̃={self._p_tilde:.4f} 不在 [0,1] 区间。"
                f"请检查参数或增大 N（当前 N={self.N}）。"
            )

    # ---- 股票价格生成 -------------------------------------------------------

    def _stock_prices_at_step(self, n: int) -> np.ndarray:
        """生成第 n 步所有节点的股票价格。

        第 n 步有 n+1 个节点 (0..n)，节点 j 表示 j 次上升。
            S_n(j) = S₀ · u^j · d^{n-j}

        Parameters
        ----------
        n : int
            时间步索引。

        Returns
        -------
        np.ndarray of shape (n+1,)
        """
        j = np.arange(n + 1)
        return self.contract.S0 * (self._u ** j) * (self._d ** (n - j))

    # ---- 定价（向后递推）---------------------------------------------------

    def price(self, return_tree: bool = False):
        """执行二叉树向后递推定价。

        步骤：
        1. 生成到期日 (N) 所有节点的标的价格
        2. 计算到期日 payoff 作为终端条件
        3. 从 N-1 步向 0 步递推：
           - 欧式：V_n(j) = e^{-rΔt} [p̃·V_{n+1}(j+1) + q̃·V_{n+1}(j)]
           - 美式：V_n(j) = max(Intrinsic(j), Continuation(j))

        Parameters
        ----------
        return_tree : bool
            若为 True，额外返回 (price, stock_tree, value_tree) 三元组，
            用于可视化和教学。

        Returns
        -------
        float 或 tuple
            期权公允价格；若 return_tree=True 则返回 (price, stock_tree, value_tree)。
        """
        N = self.N
        c = self.contract

        # --- 全树股价矩阵（N+1 × N+1，下三角有效） ---
        if return_tree:
            stock_tree = np.zeros((N + 1, N + 1))
            for i in range(N + 1):
                for jj in range(i + 1):
                    stock_tree[i, jj] = c.S0 * (self._u ** jj) * (self._d ** (i - jj))
            value_tree = np.zeros((N + 1, N + 1))

        # --- 到期日 payoff ---
        S_T = self._stock_prices_at_step(N)
        V = c.payoff(S_T)

        if return_tree:
            value_tree[N, : N + 1] = V

        # --- 向后递推 ---
        for n in range(N - 1, -1, -1):
            continuation = self._discount * (
                self._p_tilde * V[1:n+2] + self._q_tilde * V[0:n+1]
            )
            if c.is_american:
                S_n = self._stock_prices_at_step(n)
                intrinsic = c.payoff(S_n)
                V = np.maximum(intrinsic, continuation)
            else:
                V = continuation

            if return_tree:
                value_tree[n, : n + 1] = V

        if return_tree:
            return float(V[0]), stock_tree, value_tree
        return float(V[0])

    # ---- 收敛性研究 ---------------------------------------------------------

    @staticmethod
    def convergence_study(
        contract: OptionContract,
        N_list: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """对一系列步数 N 计算二叉树价格，用于收敛性分析。

        Parameters
        ----------
        contract : OptionContract
            期权合约。
        N_list : list[int] or None
            步数列表，默认为 [5, 10, 20, 50, 100, 200, 500, 1000]。

        Returns
        -------
        list[tuple[int, float]]
            [(N, price), ...] 列表。
        """
        if N_list is None:
            N_list = [5, 10, 20, 50, 100, 200, 500, 1000]
        return [(N, BinomialTreeEngine(contract, N=N).price()) for N in N_list]

    # ---- 属性 ---------------------------------------------------------------

    @property
    def p_tilde(self) -> float:
        """风险中性上升概率 p̃。"""
        return self._p_tilde

    @property
    def q_tilde(self) -> float:
        """风险中性下降概率 q̃ = 1 - p̃。"""
        return self._q_tilde

    @property
    def u(self) -> float:
        """上升乘数 u = exp(σ√Δt)。"""
        return self._u

    @property
    def d(self) -> float:
        """下降乘数 d = 1/u。"""
        return self._d


# ==============================================================================
# Black-Scholes 解析解（欧式期权）
# ==============================================================================

class BlackScholesEngine:
    """Black-Scholes 解析定价引擎（仅适用于欧式期权）。

    Black-Scholes 公式（无分红）：
        d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T

        Call:  C = S₀·N(d₁) - K·e^{-rT}·N(d₂)
        Put:   P = K·e^{-rT}·N(-d₂) - S₀·N(-d₁)

    其中 N(·) 为标准正态累积分布函数。

    该引擎用于验证二叉树模型在 N → ∞ 时的收敛性。

    Parameters
    ----------
    contract : OptionContract
        期权合约参数（必须是欧式）。
    """

    def __init__(self, contract: OptionContract) -> None:
        if contract.exercise_type != ExerciseType.EUROPEAN:
            raise ValueError("Black-Scholes 解析解仅适用于欧式期权。")
        self.contract = contract

    def _d1_d2(self) -> tuple[float, float]:
        """计算 Black-Scholes 中间变量 d₁ 和 d₂。

        处理 T=0 或 sigma=0 的边界情况。

        Returns
        -------
        tuple[float, float]
            (d₁, d₂)
        """
        c = self.contract
        if c.T == 0 or c.sigma == 0:
            # 边界情况：期权等价于 intrinsic value
            eps = 1e-12
            d1 = (math.log(c.S0 / c.K) + c.r * c.T) / (c.sigma * math.sqrt(c.T) + eps) \
                if c.T > 0 else (float("inf") if c.S0 > c.K else float("-inf"))
            return d1, d1

        denom = c.sigma * math.sqrt(c.T)
        d1 = (math.log(c.S0 / c.K) + (c.r + 0.5 * c.sigma**2) * c.T) / denom
        d2 = d1 - denom
        return d1, d2

    def price(self) -> float:
        """计算欧式期权的 Black-Scholes 价格。

        Returns
        -------
        float
            Black-Scholes 公式给出的期权公允价值。
        """
        c = self.contract
        if c.T == 0:
            return float(c.payoff(c.S0))

        d1, d2 = self._d1_d2()
        discount = math.exp(-c.r * c.T)

        if c.is_call:
            return c.S0 * norm.cdf(d1) - c.K * discount * norm.cdf(d2)
        else:
            return c.K * discount * norm.cdf(-d2) - c.S0 * norm.cdf(-d1)

    # ---- 解析希腊字母 --------------------------------------------------------
    # 使用解析公式而非有限差分，精度更高、速度更快。
    # Vega/Theta/Rho 已按行业惯例归一化。
    # -------------------------------------------------------------------------

    def delta(self) -> float:
        """Δ = ∂C/∂S₀（解析公式）。

        Call: N(d₁)
        Put:  N(d₁) - 1
        """
        d1, _ = self._d1_d2()
        if self.contract.is_call:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0  # = -N(-d₁)

    def gamma(self) -> float:
        """Γ = ∂²C/∂S₀²（解析公式，看涨看跌相同）。

        Γ = N'(d₁) / (S₀·σ·√T)
        """
        c = self.contract
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (c.S0 * c.sigma * math.sqrt(c.T))

    def vega(self) -> float:
        """ν = ∂C/∂σ（解析公式，已除以 100）。

        返回 σ 上升 1 个百分点（如 20%→21%）对应的价格变动。
        """
        c = self.contract
        d1, _ = self._d1_d2()
        return c.S0 * norm.pdf(d1) * math.sqrt(c.T) / 100.0

    def theta(self) -> float:
        """Θ = ∂C/∂t（解析公式，已除以 365）。

        返回每过 1 个自然日的期权价值衰减。
        """
        c = self.contract
        d1, d2 = self._d1_d2()
        first = -(c.S0 * norm.pdf(d1) * c.sigma) / (2.0 * math.sqrt(c.T))
        if c.is_call:
            second = -c.r * c.K * math.exp(-c.r * c.T) * norm.cdf(d2)
        else:
            second = c.r * c.K * math.exp(-c.r * c.T) * norm.cdf(-d2)
        return (first + second) / 365.0

    def rho(self) -> float:
        """ρ = ∂C/∂r（解析公式，已除以 100）。

        返回 r 上升 1 个百分点（如 5%→6%）对应的价格变动。
        """
        c = self.contract
        _, d2 = self._d1_d2()
        disc = c.K * c.T * math.exp(-c.r * c.T)
        if c.is_call:
            return disc * norm.cdf(d2) / 100.0
        else:
            return -disc * norm.cdf(-d2) / 100.0

    def all_greeks(self) -> dict:
        """打包返回全部解析希腊字母。"""
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }
