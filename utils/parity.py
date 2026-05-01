"""
Put-Call Parity 验证模块

理论
----
对于不分红股票上的欧式期权，以下恒等式恒成立：

    C - P = S₀ - K·e^{-rT}

直觉：持有 call + 空 put 的组合到期收益恰好是 S_T - K，
等价于一份远期合约，现值 = S₀ - K·e^{-rT}。

用途：
1. 自检定价器 —— 若 BS 算出的 C-P 与理论值不符，说明代码有 bug
2. 跨模型验证 —— 用二叉树对同一组参数算 C 和 P，检查是否满足 parity
3. 套利检测 —— 实盘中若偏离过大，意味着存在套利机会
"""

import math
from option_models import OptionContract, OptionType, ExerciseType
from pricing_engines import BinomialTreeEngine, BlackScholesEngine


def parity_lhs(call_price: float, put_price: float) -> float:
    """C - P，平价等式的左边。"""
    return call_price - put_price


def parity_rhs(S0: float, K: float, r: float, T: float) -> float:
    """S₀ - K·e^{-rT}，平价等式的右边。"""
    return S0 - K * math.exp(-r * T)


def check_parity(
    S0: float, K: float, T: float, r: float, sigma: float,
    model: str = "bs",
    N: int = 500,
    tol: float = 1e-2,
) -> dict:
    """对一组参数验证 Put-Call Parity，返回详细诊断信息。

    用指定模型对同一组市场参数分别定价 call 和 put，
    计算 |(C-P) - (S₀ - K·e^{-rT})| 并与容许误差 tol 比较。

    Parameters
    ----------
    S0, K, T, r, sigma : float
        市场参数（不分红）。
    model : str
        定价模型：'bs'（Black-Scholes）或 'crr'（二叉树）。
    N : int
        二叉树步数（仅 model='crr' 时使用）。
    tol : float
        通过判定的绝对误差上限。BS 可达 1e-10，CRR 约 1e-3。

    Returns
    -------
    dict
        包含 call/put 价格、parity 左右侧、绝对差、是否通过的字典。
    """
    call = OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN,
    )
    put = OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN,
    )

    if model == "bs":
        c = BlackScholesEngine(call).price()
        p = BlackScholesEngine(put).price()
    elif model == "crr":
        c = BinomialTreeEngine(call, N=N).price()
        p = BinomialTreeEngine(put, N=N).price()
    else:
        raise ValueError("model 必须是 'bs' 或 'crr'")

    lhs = parity_lhs(c, p)
    rhs = parity_rhs(S0, K, r, T)
    diff = abs(lhs - rhs)

    return {
        "model": model,
        "call": c,
        "put": p,
        "lhs (C-P)": lhs,
        "rhs (S0-Ke^{-rT})": rhs,
        "abs diff": diff,
        "passed": diff < tol,
    }
