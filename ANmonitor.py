import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings

# --- LPPLS 核心数学模型 ---
def lppls_func(t, tc, m, w, A, B, C, phi):
    dt = np.maximum(tc - t, 1e-6)
    return A + B * (dt**m) + C * (dt**m) * np.cos(w * np.log(dt) + phi)

# --- 多起点鲁棒拟合 ---
def fit_lppls_robust(x_data, y_data, current_t, n_tries=30):
    """
    多起点随机扰动拟合，取残差最小结果。
    避免单点 p0 陷入局部最优的问题。
    """
    best_popt, best_residual = None, np.inf
    rng = np.random.default_rng(42)

    bounds = (
        [current_t + 1,   0.05,  2.0, -np.inf, -np.inf, -np.inf,       0],
        [current_t + 200, 0.95, 15.0,  np.inf,       0,  np.inf, 2*np.pi]
    )

    for _ in range(n_tries):
        tc0  = current_t + rng.integers(10, 180)
        m0   = rng.uniform(0.1, 0.9)
        w0   = rng.uniform(4.0, 12.0)
        phi0 = rng.uniform(0, 2 * np.pi)
        p0   = [tc0, m0, w0, y_data[-1], -0.5, 0.1, phi0]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    lppls_func, x_data, y_data,
                    p0=p0, bounds=bounds, maxfev=10000
                )
            residual = np.sum((lppls_func(x_data, *popt) - y_data) ** 2)
            if residual < best_residual:
                best_residual, best_popt = residual, popt
        except Exception:
            continue

    return best_popt

# --- R² 计算 ---
def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

# --- 主分析函数 ---
def analyze_ticker(ticker):
    print(f"\n[正在分析 {ticker} ... 正在调取全球实时数据]")

    # 1. 获取数据
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        print(f"错误: 无法获取 {ticker} 的数据，请检查代码后缀（如 .SS, .SZ, .HK）。")
        return

    # 兼容 yfinance MultiIndex 列名（新旧版本均适用）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 取 Close / Volume，squeeze 防止单列变成 DataFrame
    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    if len(close) < 60:
        print(f"错误: {ticker} 数据不足 60 个交易日，无法进行有效拟合。")
        return

    # 2. 准备拟合数据
    y_data    = np.log(close.values.astype(float))
    x_data    = np.arange(len(y_data), dtype=float)
    current_t = x_data[-1]

    # 3. 多起点鲁棒拟合
    popt = fit_lppls_robust(x_data, y_data, current_t, n_tries=30)

    if popt is None:
        print("拟合失败: 多次随机初始化后仍无法收敛，当前形态不具备 LPPLS 对数周期特征。")
        print("         → 这不一定是坏消息：未出现泡沫结构，继续观察。")
        return

    tc_val, m_val, w_val, _, _, _, _ = popt
    days_left  = int(tc_val - current_t)
    crash_date = datetime.now() + timedelta(days=days_left)

    # 4. 计算拟合置信度 R²
    y_pred    = lppls_func(x_data, *popt)
    r_squared = calc_r_squared(y_data, y_pred)
    conf_label = (
        "高置信 ✓" if r_squared > 0.92 else
        "中置信 △" if r_squared > 0.80 else
        "低置信 ✗"
    )

    # 5. 换手率与摩擦力分析
    vol_recent      = volume.tail(3).mean()
    vol_avg         = volume.tail(20).mean()
    vol_ratio       = vol_recent / vol_avg if vol_avg > 0 else 0
    price_change_3d = (close.iloc[-1] / close.iloc[-3]) - 1

    # 6. 输出报告
    print("-" * 55)
    print(f"📊 {ticker} 综合压力报告 ({datetime.now().strftime('%Y-%m-%d')})")
    print("-" * 55)

    # 拟合置信度（先展示，帮助用户校准后续信号权重）
    print(f"🎯 拟合置信度 (R²): {r_squared:.3f}  [{conf_label}]")

    # 时间维度判定
    time_icon = "🔴" if days_left < 15 else "🟡" if days_left < 40 else "🟢"
    print(f"{time_icon} 临界日期 (tc):  {crash_date.strftime('%Y-%m-%d')} (剩余 {days_left} 天)")

    # 物理参数
    m_note = "(垂直加速中!)" if m_val < 0.3 else "(稳步推升)"
    print(f"📈 轨道曲率   (m):  {m_val:.3f}  {m_note}")
    print(f"🌀 震荡频率   (ω):  {w_val:.3f}")

    # 摩擦力指标
    print(f"🔥 成交量能倍数:    {vol_ratio:.2f}x")
    print(f"💰 近 3 日涨幅:     {price_change_3d*100:.2f}%")

    print("-" * 55)

    # 7. 核心评估（低置信时降级为提示而非警告）
    if r_squared < 0.80:
        print("⚠️  状态评估: 【信号弱】拟合质量不足，LPPLS 结构尚不清晰，")
        print("             当前输出仅供参考，不建议据此做出决策。")
    elif vol_ratio > 1.8 and price_change_3d < 0.02:
        print("🚩 状态评估: 【疑似派发】放量滞涨，需警惕主力高位减仓。")
        print(f"             结合 tc 窗口（{days_left} 天），风险敞口较高。")
    elif days_left < 10:
        print("🚩 状态评估: 【时间耗尽】已进入 tc 窗口，系统随时发生相变。")
    elif m_val < 0.2:
        print("🚩 状态评估: 【超速警告】正反馈进入不可控状态，谨防垂直崩塌。")
    else:
        print("✅ 状态评估: 系统能量传导尚在安全阈值内，继续监控。")

    print("-" * 55)

    # 8. 低置信度额外提示
    if r_squared < 0.80:
        print("💡 建议: 可等待更多数据积累后重新扫描，或结合其他技术指标交叉验证。")
        print("-" * 55)

if __name__ == "__main__":
    while True:
        target = input(
            "\n请输入股票代码进行扫描 "
            "(例如: 2513.HK, NVDA, 002432.SZ) "
            "[输入 Q 退出]: "
        ).strip().upper()
        if target == 'Q':
            print("已退出。")
            break
        if not target:
            continue
        analyze_ticker(target)
