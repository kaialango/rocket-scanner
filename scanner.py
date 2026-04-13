import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit
from lppls import lppls_func
import warnings

def compute_rocket_score(df):
    """
    计算火箭起飞评分
    """
    # 确保有 Close 列
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            close = df['Adj Close']
        else:
            return None
    else:
        close = df['Close']
    
    # 确保有 Volume 列
    if 'Volume' not in df.columns:
        volume = pd.Series(0, index=df.index)
    else:
        volume = df['Volume']
    
    if len(close) < 100:
        return None

    # --- 趋势分析 ---
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma150.iloc[-1]):
        trend_ok = False
    else:
        trend_ok = close.iloc[-1] > ma50.iloc[-1] > ma150.iloc[-1]

    # --- 波动率模式 ---
    returns = close.pct_change()
    vol = returns.rolling(20).std()
    
    if len(vol) >= 30 and not pd.isna(vol.iloc[-30]):
        vol_contract = vol.iloc[-10:-3].mean() < vol.iloc[-30:-10].mean()
        vol_expand = vol.iloc[-3:].mean() > vol.iloc[-10:-3].mean()
        vol_pattern = vol_contract and vol_expand
    else:
        vol_pattern = False

    # --- 成交量质量 ---
    up_days = close.diff() > 0
    up_volume = volume[up_days].tail(10).mean() if up_days.tail(10).any() else volume.tail(10).mean()
    down_volume = volume[~up_days].tail(10).mean() if (~up_days).tail(10).any() else volume.tail(10).mean()
    volume_quality = up_volume / (down_volume + 1e-6)

    vol_ratio = volume.tail(3).mean() / (volume.tail(30).mean() + 1e-6)

    # --- 动量 ---
    if len(close) >= 20:
        price_20d = close.iloc[-1] / close.iloc[-20] - 1
    else:
        price_20d = 0

    # --- LPPLS 拟合 ---
    y = np.log(close.values + 1e-6)
    x = np.arange(len(y))
    current_t = x[-1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p0 = [current_t + 100, 0.9, 4.0, y[-1], -0.5, 0.01, 0]
            bounds = ([current_t + 10, 0.5, 2.0, -np.inf, -np.inf, -np.inf, 0],
                      [current_t + 500, 0.99, 10.0, np.inf, 0, np.inf, 2*np.pi])

            popt, _ = curve_fit(lppls_func, x, y, p0=p0, bounds=bounds, maxfev=5000)

            y_pred = lppls_func(x, *popt)
            residual = np.abs(y - y_pred)
            stability = np.mean(residual[-10:]) if len(residual) >= 10 else np.mean(residual)
            m_val = popt[1]

    except Exception as e:
        return None

    # --- 评分系统 ---
    score = 0
    score += 2 if 0.8 < m_val < 0.95 else 0
    score += 2 if stability < 0.02 else 0
    score += 2 if vol_ratio > 1.5 else 0
    score += 2 if volume_quality > 1.3 else 0
    score += 2 if vol_pattern else 0
    score += 2 if trend_ok else 0
    score += 2 if 0.2 < price_20d < 1.0 else 0

    return {
        "score": score,
        "m": round(m_val, 4),
        "vol_ratio": round(vol_ratio, 2),
        "volume_quality": round(volume_quality, 2),
        "price_20d": round(price_20d * 100, 1),
        "trend_ok": trend_ok,
        "vol_pattern": vol_pattern,
        "stability": round(stability, 4),
        "popt": popt
    }


def scan_market(tickers, period="6mo", progress_callback=None):
    """扫描市场"""
    results = []
    total = len(tickers)
    
    for i, t in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, t)
        
        try:
            df = yf.download(t, period=period, interval="1d", progress=False, auto_adjust=False)
            
            if df.empty or len(df) < 50:
                continue
                
            res = compute_rocket_score(df)
            if res:
                results.append((t, res, df))
                
        except Exception as e:
            print(f"Error scanning {t}: {e}")
            continue
    
    results.sort(key=lambda x: x[1]["score"], reverse=True)
    return results


def get_rocket_label(score):
    """根据分数返回火箭标签"""
    if score >= 12:
        return ("CRITICAL", "#ff4444", "极高风险！泡沫即将破裂，建议立即减仓")
    elif score >= 10:
        return ("EARLY ROCKET", "#ff8800", "早期火箭，结构完整，建议分批建仓")
    elif score >= 8:
        return ("WATCH", "#ffaa00", "中等信号，密切观察，等待量能确认")
    elif score >= 6:
        return ("MONITOR", "#44aaff", "低度信号，加入关注列表")
    else:
        return ("QUIET", "#888888", "静默期，暂无起飞信号")