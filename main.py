"""
期权定价系统 —— 主入口

两种运行模式：
  1. 交互模式（默认）: python main.py
     按提示输入参数，选择模型，输出价格与 Greeks。

  2. 演示模式: python main.py --demo
     一键运行所有演示：三模型对比 → 美式期权 → Greeks → 收敛性 → 可视化。

依赖：
    numpy, scipy, matplotlib（可选）
"""

import sys
import time
import math

from option_models import OptionContract, OptionType, ExerciseType, PricingResult
from pricing_engines import BinomialTreeEngine, BlackScholesEngine
from greeks import GreeksCalculator


# ======================================================================
# 辅助函数
# ======================================================================

def print_sep(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def print_result(label: str, result: PricingResult) -> None:
    """格式化打印定价结果（含 Greeks）。"""
    print(f"\n  {label}")
    print(f"    价格 (Price):  {result.price:10.6f}")
    if result.delta is not None:
        print(f"    Δ (Delta):     {result.delta:10.6f}")
        print(f"    Γ (Gamma):     {result.gamma:10.6f}")
        print(f"    Θ (Theta):     {result.theta:10.6f}")
        print(f"    ν (Vega):      {result.vega:10.6f}")
        print(f"    ρ (Rho):       {result.rho:10.6f}")


# ======================================================================
# 交互式 CLI
# ======================================================================

def _ask_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("    ! 请输入一个数字。")


def _ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    suffix = f" [{'/'.join(choices)}]"
    if default:
        suffix += f" (默认={default})"
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip().lower()
        if not raw and default:
            return default
        if raw in choices:
            return raw
        print(f"    ! 请输入 {choices} 之一。")


def interactive_mode() -> None:
    """交互式命令行：按提示输入参数 → 选择模型 → 输出价格。"""
    print("=" * 60)
    print("    Option Pricing — 交互式定价")
    print("    按回车使用方括号内的默认值")
    print("=" * 60)

    # ---- 1. 收集参数 ----
    opt_type_str = _ask_choice("期权方向", ["call", "put"], default="call")
    ex_type_str  = _ask_choice("行权方式", ["european", "american"], default="european")
    S0    = _ask_float("当前股价 S₀",   default=100.0)
    K     = _ask_float("行权价 K",      default=100.0)
    T     = _ask_float("到期时间 T (年)", default=1.0)
    r     = _ask_float("无风险利率 r (小数, 如 0.05=5%)", default=0.05)
    sigma = _ask_float("波动率 σ (小数, 如 0.2=20%)",   default=0.20)

    opt_type = OptionType.CALL if opt_type_str == "call" else OptionType.PUT
    ex_type  = ExerciseType.EUROPEAN if ex_type_str == "european" else ExerciseType.AMERICAN

    contract = OptionContract(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=opt_type, exercise_type=ex_type,
    )
    print(f"\n  已构造期权: {contract}\n")

    # ---- 2. 选择引擎 ----
    is_am = ex_type == ExerciseType.AMERICAN
    if is_am:
        print("  美式期权只能使用二叉树模型 (CRR)。")
        models = ["crr"]
    else:
        choice = _ask_choice("使用哪个模型?", ["bs", "crr", "all"], default="all")
        models = ["bs", "crr"] if choice == "all" else [choice]

    # ---- 3. 定价 ----
    print(f"\n  {'Model':<28} {'Price':>12}")
    print("  " + "-" * 42)

    bt_engine = None
    for m in models:
        if m == "bs":
            price = BlackScholesEngine(contract).price()
            print(f"  {'Black-Scholes (闭式解)':<28} {price:>12.6f}")
        elif m == "crr":
            bt_engine = BinomialTreeEngine(contract, N=500)
            price = bt_engine.price()
            print(f"  {'CRR 二叉树 (N=500)':<28} {price:>12.6f}")

    # ---- 4. Greeks ----
    if not is_am:
        print("\n  Greeks（解析公式，Black-Scholes）：")
        bs_eng = BlackScholesEngine(contract)
        for k, v in bs_eng.all_greeks().items():
            if k != "price":
                print(f"    {k:<6} = {v:>10.6f}")

        if bt_engine is not None:
            print("\n  Greeks（有限差分，二叉树 N=500）：")
            greek_calc = GreeksCalculator()
            result = greek_calc.compute_all(bt_engine)
            for k in ["delta", "gamma", "theta", "vega", "rho"]:
                print(f"    {k:<6} = {getattr(result, k):>10.6f}")

    print("\n" + "-" * 60)
    print("  如需完整演示含可视化，请运行: python main.py --demo")


# ======================================================================
# 一键演示
# ======================================================================

def demo_european() -> None:
    """欧式期权定价对比。"""
    print_sep("1. 欧式期权定价 — 二叉树 vs Black-Scholes")
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    N = 300

    for opt_type in (OptionType.CALL, OptionType.PUT):
        name = "看涨 (Call)" if opt_type == OptionType.CALL else "看跌 (Put)"
        print(f"\n  --- {name} ---")
        contract = OptionContract(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            option_type=opt_type, exercise_type=ExerciseType.EUROPEAN,
        )
        bt_price = BinomialTreeEngine(contract, N=N).price()
        bs_price = BlackScholesEngine(contract).price()
        diff = bt_price - bs_price
        print(f"    二叉树 (N={N}):  {bt_price:.6f}")
        print(f"    Black-Scholes:   {bs_price:.6f}")
        print(f"    差值:            {diff:.6f}  ({abs(diff)*100/bs_price:.4f}%)")


def demo_american() -> None:
    """美式看跌期权 — 提前行权溢价。"""
    print_sep("2. 美式看跌期权 — 提前行权溢价")
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    N = 300

    eu = OptionContract(S0=S0, K=K, r=r, sigma=sigma, T=T,
                        option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN)
    am = OptionContract(S0=S0, K=K, r=r, sigma=sigma, T=T,
                        option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    eu_price = BinomialTreeEngine(eu, N=N).price()
    am_price = BinomialTreeEngine(am, N=N).price()
    premium = am_price - eu_price

    print(f"\n  欧式看跌 (European Put):  {eu_price:.6f}")
    print(f"  美式看跌 (American Put):  {am_price:.6f}")
    print(f"  提前行权溢价:              {premium:.6f}  ({premium*100/eu_price:.2f}%)")

    print("\n  不同标的价格下的提前行权溢价:")
    print(f"  {'S0':>8}  {'欧式 Put':>10}  {'美式 Put':>10}  {'溢价':>10}")
    print("  " + "-" * 44)
    for S in [80, 90, 95, 100, 105, 110, 120]:
        eu_c = OptionContract(S0=S, K=K, r=r, sigma=sigma, T=T,
                              option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN)
        am_c = OptionContract(S0=S, K=K, r=r, sigma=sigma, T=T,
                              option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)
        eu_v = BinomialTreeEngine(eu_c, N=N).price()
        am_v = BinomialTreeEngine(am_c, N=N).price()
        print(f"  {S:8.1f}  {eu_v:10.4f}  {am_v:10.4f}  {am_v-eu_v:10.4f}")


def demo_greeks() -> None:
    """Greeks 对比：BS 解析 vs 二叉树有限差分（均用有限差分确保可比）。"""
    print_sep("3. Greeks — 有限差分法对比 (BS引擎 vs 二叉树引擎)")
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    greek_calc = GreeksCalculator()

    for opt_type in (OptionType.CALL, OptionType.PUT):
        name = "看涨 (Call)" if opt_type == OptionType.CALL else "看跌 (Put)"
        contract = OptionContract(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            option_type=opt_type, exercise_type=ExerciseType.EUROPEAN,
        )

        print(f"\n  --- {name} ---")

        # 均使用有限差分法，确保可比
        bs_result = greek_calc.compute_all(BlackScholesEngine(contract))
        bt_result = greek_calc.compute_all(BinomialTreeEngine(contract, N=300))

        print(f"    {'':>12} {'BS(差分)':>10} {'BT(差分)':>10}")
        print(f"    {'Price':>12} {bs_result.price:>10.4f} {bt_result.price:>10.4f}")
        for g_name in ["delta", "gamma", "theta", "vega", "rho"]:
            bs_val = getattr(bs_result, g_name)
            bt_val = getattr(bt_result, g_name)
            print(f"    {g_name:>12} {bs_val:>10.4f} {bt_val:>10.4f}")


def demo_convergence() -> None:
    """收敛性分析 + 生成图。"""
    print_sep("4. 收敛性分析 — 二叉树 vs Black-Scholes (N → ∞)")

    contract = OptionContract(
        S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
        option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN,
    )
    bs_price = BlackScholesEngine(contract).price()
    print(f"\n  Black-Scholes 参考价: {bs_price:.6f}")

    N_values = [5, 10, 20, 50, 100, 200, 300, 500, 800, 1000]
    print(f"\n  {'N':>6}  {'二叉树价格':>12}  {'误差':>12}  {'耗时(ms)':>10}")
    print("  " + "-" * 46)

    results = []
    for N in N_values:
        t0 = time.perf_counter()
        bt_price = BinomialTreeEngine(contract, N=N).price()
        elapsed = (time.perf_counter() - t0) * 1000
        error = bt_price - bs_price
        results.append((N, bt_price, error))
        print(f"  {N:6d}  {bt_price:12.6f}  {error:+12.6f}  {elapsed:8.2f}")

    # 生成收敛图
    try:
        from visualize import plot_binomial_convergence
        path = plot_binomial_convergence(contract, N_max=1000)
        print(f"\n  收敛图已保存至: {path}")
    except ImportError:
        print("\n  [提示] matplotlib 未安装，跳过绘图。")


def demo_parity() -> None:
    """Put-Call Parity 验证。"""
    print_sep("5. Put-Call Parity 验证")
    from utils.parity import check_parity

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    for model in ["bs", "crr"]:
        result = check_parity(S0, K, T, r, sigma, model=model)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"  [{mark}] {model.upper():<4} "
              f"C={result['call']:.6f}  P={result['put']:.6f}  "
              f"|C-P| vs |S0-Ke^(-rT)| = {result['abs diff']:.2e}")


def demo_visualize() -> None:
    """生成教学可视化。"""
    print_sep("6. 生成教学可视化")

    try:
        import visualize as viz
    except ImportError:
        print("  matplotlib 未安装，跳过可视化。")
        return

    paths = []

    # 6.1 二叉树结构图 (N=4)
    c = OptionContract(S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
                       option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)
    paths.append(viz.plot_binomial_tree(c, N=4))

    # 6.2 Greeks vs Spot 曲线
    paths.append(viz.plot_greeks_vs_spot(K=100, T=1.0, r=0.05, sigma=0.20))

    # 6.3 简单策略：Covered Call
    stock = OptionContract(S0=100, K=1e-9, r=0.05, sigma=0.20, T=1.0,
                           option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)
    short_call = OptionContract(S0=100, K=110, r=0.05, sigma=0.20, T=1.0,
                                option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)
    paths.append(viz.plot_strategy_payoff(
        "Covered Call", [(+1, stock), (-1, short_call)]))

    # 6.4 Straddle
    call = OptionContract(S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
                          option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)
    put = OptionContract(S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
                         option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN)
    paths.append(viz.plot_strategy_payoff(
        "Long Straddle", [(+1, call), (+1, put)]))

    print(f"  共生成 {len(paths)} 张图，保存在 figures/：")
    for p in paths:
        print(f"    - {os.path.basename(p)}")


import os  # for basename above


def demo_mode() -> None:
    """一键运行全部演示。"""
    print("+" + "=" * 60 + "+")
    print("|" + "  期权定价系统 —— 基于二叉树模型 & Black-Scholes".center(58) + "|")
    print("|" + "  理论依据：Shreve《金融随机分析 第1卷》".center(56) + "|")
    print("+" + "=" * 60 + "+")

    demo_european()
    demo_american()
    demo_greeks()
    demo_convergence()
    demo_parity()
    demo_visualize()

    print_sep("演示完毕")
    print("\n  项目结构:")
    print("    option_models.py   — 数据类型定义（合约 + 结果容器）")
    print("    pricing_engines.py — 定价引擎（CRR 二叉树 + Black-Scholes）")
    print("    greeks.py          — 有限差分 Greeks 计算器")
    print("    utils/parity.py    — Put-Call Parity 验证")
    print("    visualize.py       — 可视化工具集")
    print("    main.py            — 本文件（交互 CLI + 一键演示）")
    print()


# ======================================================================
# 主入口
# ======================================================================

def main() -> None:
    if "--demo" in sys.argv:
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
