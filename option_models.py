"""
期权合约定价模型 —— 数据类型定义

本模块定义了期权合约的核心数据结构，包括：
  - 期权类型（看涨/看跌）
  - 行权方式（欧式/美式）
  - 期权合约参数容器
  - 定价结果容器（含 Greeks）

理论框架基于 Shreve《金融随机分析 第1卷》二叉树资产定价模型。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class OptionType(Enum):
    """期权类型：看涨 (Call) 或 看跌 (Put)。"""
    CALL = auto()
    PUT = auto()


class ExerciseType(Enum):
    """行权方式：欧式 (European) 只能在到期日行权；
       美式 (American) 可在到期前任一时刻行权。"""
    EUROPEAN = auto()
    AMERICAN = auto()


@dataclass(frozen=True)
class OptionContract:
    """期权合约参数容器。

    Attributes
    ----------
    S0 : float
        当前标的资产价格 (t=0)。
    K : float
        执行价格 (Strike Price)。
    r : float
        连续复利无风险利率，例如 0.05 表示 5%。
    sigma : float
        标的资产年化波动率，例如 0.20 表示 20%。
    T : float
        到期时间（年），例如 1.0 表示 1 年。
    option_type : OptionType
        看涨 (CALL) 或看跌 (PUT)。
    exercise_type : ExerciseType
        欧式 (EUROPEAN) 或美式 (AMERICAN)。
    """

    S0: float
    K: float
    r: float
    sigma: float
    T: float
    option_type: OptionType
    exercise_type: ExerciseType = ExerciseType.EUROPEAN

    def __post_init__(self) -> None:
        if self.S0 <= 0 or self.K <= 0:
            raise ValueError("S0 和 K 必须为正数。")
        if self.sigma < 0:
            raise ValueError("波动率 sigma 不能为负。")
        if self.T < 0:
            raise ValueError("到期时间 T 不能为负。")
        if self.r < 0:
            raise ValueError("无风险利率 r 不能为负。")

    # ---- 便捷属性 ------------------------------------------------------------

    @property
    def is_call(self) -> bool:
        """是否为看涨期权。供引擎内部判断收益函数时使用。"""
        return self.option_type == OptionType.CALL

    @property
    def is_american(self) -> bool:
        """是否为美式期权。"""
        return self.exercise_type == ExerciseType.AMERICAN

    # ---- 收益函数 ------------------------------------------------------------

    def payoff(self, S):
        """到期收益（内在价值）。

        看涨: max(S - K, 0)
        看跌: max(K - S, 0)

        Parameters
        ----------
        S : float 或 np.ndarray
            标的价格（标量或数组），兼容蒙特卡洛向量化调用。

        Returns
        -------
        float 或 np.ndarray
            max(S-K, 0) 或 max(K-S, 0)。
        """
        if self.is_call:
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)


@dataclass
class PricingResult:
    """定价结果容器，包含期权价格及 Greeks。

    Attributes
    ----------
    price : float
        期权在 t=0 时刻的公允价格。
    delta : Optional[float]
        Δ = ∂V/∂S，价格对标的资产的一阶偏导。
    gamma : Optional[float]
        Γ = ∂²V/∂S²，价格对标的资产的二阶偏导。
    theta : Optional[float]
        Θ = -∂V/∂T（注意符号约定），时间衰减。
    vega : Optional[float]
        ν = ∂V/∂σ，价格对波动率的偏导。
        （已除以 100，表示 σ 上升 1 个百分点的价格变动）
    rho : Optional[float]
        ρ = ∂V/∂r，价格对无风险利率的偏导。
        （已除以 100，表示 r 上升 1 个百分点的价格变动）
    """

    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
