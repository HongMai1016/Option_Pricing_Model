"""
希腊字母（Greeks）计算 —— 有限差分法

本模块通过有限差分近似计算期权的希腊字母（Greeks），
反映期权价格对各风险因子的敏感度，是对冲（Delta Hedging）
与风险管理的基础。

计算方式：
    Δ (Delta)  = ∂V/∂S   —— 标的价格敏感度
    Γ (Gamma)  = ∂²V/∂S² —— Delta 的变化率
    Θ (Theta)  = -∂V/∂T  —— 时间衰减（注意符号约定）
    ν (Vega)   = ∂V/∂σ   —— 波动率敏感度
    ρ (Rho)    = ∂V/∂r   —— 利率敏感度

全部使用中心差分（Theta 除外，使用前向差分以保证 T>0），
步长可配置以适应不同精度需求。

使用方式：
    from option_models import OptionContract, OptionType, ExerciseType
    from pricing_engines import BinomialTreeEngine, BlackScholesEngine

    contract = OptionContract(S0=100, K=100, r=0.05, sigma=0.2, T=1,
                              option_type=OptionType.CALL,
                              exercise_type=ExerciseType.EUROPEAN)
    engine = BinomialTreeEngine(contract, N=200)
    greeks = GreeksCalculator()
    result = greeks.compute_all(engine)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from option_models import OptionContract, OptionType, ExerciseType, PricingResult
from pricing_engines import BinomialTreeEngine, BlackScholesEngine


# ---------------------------------------------------------------------------
# 引擎协议：允许 GreeksCalculator 接受任意可定价对象
# ---------------------------------------------------------------------------

class PricingEngine(Protocol):
    """可定价引擎的结构协议（duck typing）。"""
    contract: OptionContract
    def price(self) -> float: ...


# ---------------------------------------------------------------------------
# Greeks 计算器
# ---------------------------------------------------------------------------

@dataclass
class GreeksCalculator:
    """通过有限差分法计算期权 Greeks。

    适用于任何实现了 .price() 方法且持有 .contract 属性的定价引擎
    （BinomialTreeEngine 或 BlackScholesEngine）。

    Attributes
    ----------
    h_S : float
        Delta/Gamma 计算的标的价格扰动步长（默认 0.5）。
    eps_T : float
        Theta 计算的时间扰动步长（默认 1e-4 年 ≈ 52 分钟）。
    eps_sigma : float
        Vega 计算的波动率扰动步长（默认 1e-3）。
    eps_r : float
        Rho 计算的利率扰动步长（默认 1e-4）。
    """

    h_S: float = 1.0        # 标的价格扰动（二叉树需较大步长以减少网格噪声）
    eps_T: float = 1e-4     # 时间扰动（年）
    eps_sigma: float = 1e-3 # 波动率扰动
    eps_r: float = 1e-4     # 利率扰动

    # ------------------------------------------------------------------
    # 内部辅助：创建扰动后的"克隆"合约并重新定价
    # ------------------------------------------------------------------

    @staticmethod
    def _make_contract(contract: OptionContract, **overrides) -> OptionContract:
        """基于原合约创建参数修改后的新合约。

        保持 option_type 和 exercise_type 不变，仅覆盖指定字段。
        """
        return OptionContract(
            S0=overrides.get("S0", contract.S0),
            K=overrides.get("K", contract.K),
            r=overrides.get("r", contract.r),
            sigma=overrides.get("sigma", contract.sigma),
            T=overrides.get("T", contract.T),
            option_type=contract.option_type,
            exercise_type=contract.exercise_type,
        )

    @staticmethod
    def _reprice(engine: PricingEngine, contract: OptionContract) -> float:
        """为扰动后的合约创建新引擎并重新定价。

        自动匹配原引擎类型（二叉树或 Black-Scholes）。
        """
        if isinstance(engine, BinomialTreeEngine):
            new_engine = BinomialTreeEngine(contract, N=engine.N)
        elif isinstance(engine, BlackScholesEngine):
            new_engine = BlackScholesEngine(contract)
        else:
            raise TypeError(f"不支持的引擎类型: {type(engine)}")
        return new_engine.price()

    # ------------------------------------------------------------------
    # 各 Greek 计算
    # ------------------------------------------------------------------

    def delta(self, engine: PricingEngine) -> float:
        r"""计算 Delta: Δ = ∂V/∂S₀。

        使用中心差分：
            Δ ≈ [V(S₀ + h) - V(S₀ - h)] / (2h)

        Δ 反映了期权价格对标的资产价格变化的敏感度，
        是 Delta 对冲（Delta Hedging）的核心参数。
        """
        c = engine.contract
        V_up = self._reprice(engine, self._make_contract(c, S0=c.S0 + self.h_S))
        V_down = self._reprice(engine, self._make_contract(c, S0=c.S0 - self.h_S))
        return (V_up - V_down) / (2.0 * self.h_S)

    def gamma(self, engine: PricingEngine) -> float:
        r"""计算 Gamma: Γ = ∂²V/∂S₀²。

        使用中心差分：
            Γ ≈ [V(S₀ + h) - 2V(S₀) + V(S₀ - h)] / h²

        Γ 衡量 Delta 本身随标的价格变化的速率，大的 |Γ| 意味着
        Delta 对冲需要更频繁地再平衡（rebalancing）。
        """
        c = engine.contract
        V0 = engine.price()
        V_up = self._reprice(engine, self._make_contract(c, S0=c.S0 + self.h_S))
        V_down = self._reprice(engine, self._make_contract(c, S0=c.S0 - self.h_S))
        return (V_up - 2.0 * V0 + V_down) / (self.h_S ** 2)

    def theta(self, engine: PricingEngine) -> float:
        r"""计算 Theta: Θ = -∂V/∂T。

        使用前向差分（保证 T - ε > 0）：
            Θ ≈ -[V(T - ε) - V(T)] / ε

        符号约定：Θ 通常为负（对于多头持有的看涨/看跌期权），
        代表随时间流逝期权价值的衰减（Time Decay）。
        """
        c = engine.contract
        if c.T <= self.eps_T:
            raise ValueError(f"到期时间 T={c.T} 太小，无法计算 Theta（需要 > {self.eps_T}）。")
        V0 = engine.price()
        V_back = self._reprice(engine, self._make_contract(c, T=c.T - self.eps_T))
        return -(V_back - V0) / self.eps_T

    def vega(self, engine: PricingEngine) -> float:
        r"""计算 Vega: ν = ∂V/∂σ。

        使用中心差分：
            ν ≈ [V(σ + ε) - V(σ - ε)] / (2ε)

        注意：Vega 通常除以 100 以表示波动率变化 1% 带来的
        价格变化。此处返回原始值（1 单位 σ 变化，即 100% 波动率变化）。
        实践中常将其除以 100 来解读。
        """
        c = engine.contract
        V_up = self._reprice(engine, self._make_contract(c, sigma=c.sigma + self.eps_sigma))
        V_down = self._reprice(engine, self._make_contract(c, sigma=c.sigma - self.eps_sigma))
        return (V_up - V_down) / (2.0 * self.eps_sigma)

    def rho(self, engine: PricingEngine) -> float:
        r"""计算 Rho: ρ = ∂V/∂r。

        使用中心差分：
            ρ ≈ [V(r + ε) - V(r - ε)] / (2ε)

        注意：ρ 通常除以 100 以表示利率变化 1%（即 100 bp）
        带来的价格变化。此处返回原始值（1 单位 r 变化）。
        """
        c = engine.contract
        V_up = self._reprice(engine, self._make_contract(c, r=c.r + self.eps_r))
        V_down = self._reprice(engine, self._make_contract(c, r=c.r - self.eps_r))
        return (V_up - V_down) / (2.0 * self.eps_r)

    # ------------------------------------------------------------------
    # 批量计算
    # ------------------------------------------------------------------

    def compute_all(self, engine: PricingEngine) -> PricingResult:
        """计算全部 Greeks 并连同价格一起返回。

        Parameters
        ----------
        engine : PricingEngine
            已初始化的定价引擎。

        Returns
        -------
        PricingResult
            包含 price, delta, gamma, theta, vega, rho 的结果对象。
        """
        price = engine.price()
        delta = self.delta(engine)
        gamma = self.gamma(engine)
        theta = self.theta(engine)
        vega = self.vega(engine)
        rho = self.rho(engine)
        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )
