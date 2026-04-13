import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =========================================
# CONFIG
# =========================================
UNIVERSE = [
    "NVDA","MSFT","AAPL","AMZN","META",
    "TSLA","SMCI","AMD","PLTR","COIN",
    "MRVL","AVGO","TSM","ASML","NFLX"
]

# =========================================
# DATA
# =========================================
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="300d")
        if len(df) < 200:
            return None
        return df
    except:
        return None

def get_price_volume(df):
    close = df["Close"].iloc[-1]
    vol = df["Volume"].iloc[-1]
    return float(close), float(vol)

def trend_template(df):
    close = df["Close"]
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    return (
        close.iloc[-1] > ma50.iloc[-1] > ma150.iloc[-1] > ma200.iloc[-1]
        and ma150.iloc[-1] > ma200.iloc[-1]
    )

def vcp_contraction(df):
    vol = df["Close"].pct_change().rolling(10).std()
    return vol.iloc[-10:].mean() < vol.iloc[-50:-20].mean() * 0.7

# =========================================
# LPPLS - 针对 Launch 阶段优化
# =========================================
def lppls(t, tc, m, w, A, B, C1, C2):
    dt = np.abs(tc - t)
    dt = np.clip(dt, 1e-6, None)
    power = dt ** m
    cos_term = np.cos(w * np.log(dt))
    sin_term = np.sin(w * np.log(dt))
    return A + B * power + C1 * power * cos_term + C2 * power * sin_term

def fit_lppls_launch(log_prices):
    """
    专门优化检测早期泡沫 (m ~ 0.1-0.3)
    """
    log_prices = np.asarray(log_prices, dtype=np.float64)
    log_prices = log_prices[np.isfinite(log_prices)]
    
    if len(log_prices) < 60:
        return None
    
    y = log_prices[-60:]
    t = np.arange(len(y))
    
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-6)
    
    # 重点搜索早期泡沫的参数空间
    init_params = [
        # [tc, m, w, A, B, C1, C2]
        [len(y) * 1.8, 0.20, 5.0, 0.5, -0.08, 0.03, 0.03],  # 典型 Launch
        [len(y) * 1.5, 0.15, 4.0, 0.5, -0.05, 0.02, 0.02],  # 极早期
        [len(y) * 2.0, 0.28, 6.0, 0.5, -0.12, 0.05, 0.05],  # Launch 末期
    ]
    
    # 收紧边界，重点找早期信号
    bounds = (
        [len(y) + 3, 0.10, 3.0, -2, -2, -1, -1],
        [len(y) * 2.5, 0.40, 8.0, 3, 1, 1, 1]  # 限制 m ≤ 0.4
    )
    
    best_params = None
    best_residual = np.inf
    
    for p0 in init_params:
        try:
            popt, _ = curve_fit(
                lppls, t, y_norm,
                p0=p0,
                bounds=bounds,
                maxfev=2000,
                method='trf'
            )
            
            y_pred = lppls(t, *popt)
            residual = np.sum((y_norm - y_pred) ** 2)
            
            # 严格检查 m 范围
            if 0.12 < popt[1] < 0.35 and residual < best_residual:
                popt_orig = popt.copy()
                popt_orig[3] = popt[3] * (y_max - y_min) + y_min
                popt_orig[4] = popt[4] * (y_max - y_min)
                popt_orig[5] = popt[5] * (y_max - y_min)
                popt_orig[6] = popt[6] * (y_max - y_min)
                
                best_params = popt_orig
                best_residual = residual
                
                if residual < 0.005:
                    break
        except:
            continue
    
    # 如果没找到早期信号，回退到标准拟合
    if best_params is None:
        try:
            bounds_std = (
                [len(y) + 2, 0.1, 3.0, -3, -3, -2, -2],
                [len(y) * 3, 0.9, 20.0, 3, 3, 2, 2]
            )
            p0_std = [len(y) * 1.5, 0.4, 6.0, 0.5, -0.1, 0.05, 0.05]
            
            popt, _ = curve_fit(lppls, t, y_norm, p0=p0_std, 
                                bounds=bounds_std, maxfev=3000, method='trf')
            
            popt_orig = popt.copy()
            popt_orig[3] = popt[3] * (y_max - y_min) + y_min
            popt_orig[4] = popt[4] * (y_max - y_min)
            popt_orig[5] = popt[5] * (y_max - y_min)
            popt_orig[6] = popt[6] * (y_max - y_min)
            best_params = popt_orig
        except:
            pass
    
    return best_params

# =========================================
# ACCELERATION & MOMENTUM
# =========================================
def acceleration(df):
    logp = np.log(df["Close"])
    s = np.polyfit(range(5), logp[-5:], 1)[0]
    l = np.polyfit(range(20), logp[-20:], 1)[0]
    return s / abs(l) if abs(l) > 1e-6 else 0

def recent_momentum(df):
    """过去 10 天涨幅"""
    close = df["Close"]
    return (close.iloc[-1] / close.iloc[-10] - 1) * 100

# =========================================
# ANALYZE - 火箭发射评分
# =========================================
def analyze(ticker):
    df = get_data(ticker)
    if df is None:
        return None

    price, vol = get_price_volume(df)
    trend = trend_template(df)
    vcp = vcp_contraction(df)

    logp = np.log(df["Close"].values)
    fit = fit_lppls_launch(logp)

    if fit is not None:
        tc, m, w, A, B, C1, C2 = fit
        if not (0.1 < m < 0.9):
            m, w = 0, 999
    else:
        m, w = 0, 999

    accel = acceleration(df)
    mom = recent_momentum(df)

    # 🚀 火箭发射评分系统
    score = 0
    
    # 1. 趋势 (必须项)
    if trend:
        score += 3
    else:
        score -= 2  # 惩罚无趋势的
    
    # 2. VCP (蓄力)
    if vcp:
        score += 2
    
    # 3. 🎯 核心：m 值 - Launch 阶段检测
    if 0.1 < m < 0.9:
        if 0.10 < m <= 0.20:
            score += 8   # 🚀 极早期 Launch - 最高分
            stage = "🚀 PRE-LAUNCH"
        elif 0.20 < m <= 0.30:
            score += 7   # 🚀 黄金 Launch 窗口
            stage = "🚀 LAUNCH"
        elif 0.30 < m <= 0.45:
            score += 4   # 早期
            stage = "📈 EARLY"
        elif 0.45 < m <= 0.65:
            score += 2   # 中期
            stage = "📊 MID"
        else:
            score += 0   # 后期 - 不给分
            stage = "⚠️ LATE"
    else:
        stage = "❌ NONE"
    
    # 4. ω 值：低频更健康
    if 3 < w < 7:
        score += 2
    elif 7 <= w < 12:
        score += 1
    
    # 5. 加速度：确认但不透支
    if 1.5 < accel < 8:
        score += 2
    elif 8 <= accel < 15:
        score += 1
    
    # 6. 近期动量：不能太弱
    if 5 < mom < 20:
        score += 1
    
    # 7. 成交量确认
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
    if vol > avg_vol * 1.2:
        score += 1  # 放量

    return {
        "Ticker": ticker,
        "Score": score,
        "Stage": stage,
        "Trend": trend,
        "VCP": vcp,
        "m": round(m, 3) if m != 0 else 0,
        "ω": round(w, 2) if w != 999 else 999,
        "Accel": round(accel, 2),
        "Mom%": round(mom, 1),
        "Price": round(price, 2),
        "Vol/Avg": round(vol / avg_vol, 2) if avg_vol > 0 else 1
    }

def run_scan():
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(analyze, t) for t in UNIVERSE]
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("\n🚀 ROCKET LAUNCH DETECTOR v2.0")
    print("Target: Early-stage bubbles (m = 0.10-0.30)")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    df = run_scan()

    if df.empty:
        print("No results.")
    else:
        # 按 Score 排序，重点显示 Stage
        df = df.sort_values("Score", ascending=False)
        
        print("\n🏆 TOP LAUNCH CANDIDATES\n")
        
        # 高亮显示 Launch 阶段
        for _, row in df.iterrows():
            if "LAUNCH" in row["Stage"]:
                prefix = "🚀🚀"
            elif row["Stage"] == "📈 EARLY":
                prefix = "📈  "
            else:
                prefix = "   "
            
            print(f"{prefix} {row['Ticker']:5} | Score:{row['Score']:2} | {row['Stage']:13} | "
                  f"m:{row['m']:.3f} ω:{row['ω']:.1f} | Accel:{row['Accel']:.1f} | "
                  f"Mom:{row['Mom%']:.1f}% | ${row['Price']:.2f}")
        
        print("\n" + "=" * 70)
        print("📊 Full Results:")
        print(df[["Ticker", "Score", "Stage", "m", "ω", "Accel", "Mom%", "Price"]].to_string(index=False))