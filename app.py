import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Rocket Scanner Pro", page_icon="🚀")

# --------------------------
# LPPLS 核心公式
# --------------------------
def lppls_func(x, tc, m, w, a, b, c1, c2):
    dt = tc - x
    dt = np.where(dt > 0, dt, 1e-6)
    power = np.power(dt, m)
    log_dt = np.log(dt)
    return a + b * power + power * (c1 * np.cos(w * log_dt) + c2 * np.sin(w * log_dt))

# --------------------------
# 股票代码自动分类
# --------------------------
def classify_code(code):
    code = str(code).strip().upper()
    if code.isdigit():
        if len(code) == 6:
            if code.startswith(('60', '68', '69')):
                return f"{code}.SS", "沪市"
            elif code.startswith('30'):
                return f"{code}.SZ", "创业板"
            elif code.startswith(('00', '001', '002', '003')):
                return f"{code}.SZ", "深市"
        elif len(code) == 5:
            return f"{code}.HK", "港股"
    if re.match(r'^[A-Z]{1,5}$', code):
        return code, "美股"
    return None, "未知"

# --------------------------
# 火箭评分模型
# --------------------------
def compute_rocket_score(df):
    try:
        close = df['Close'].dropna()
        volume = df['Volume'].dropna()
        if len(close) < 100:
            return None, None, None, None, None

        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        trend_ok = close.iloc[-1] > ma50.iloc[-1] > ma150.iloc[-1]

        ret = close.pct_change()
        vol = ret.rolling(20).std().dropna()
        if len(vol) < 30:
            return None, None, None, None, None

        vol_contract = vol.iloc[-10:-3].mean() < vol.iloc[-30:-10].mean()
        vol_expand = vol.iloc[-3:].mean() > vol.iloc[-10:-3].mean()
        vol_pattern = vol_contract and vol_expand

        up_days = close.diff() > 0
        v_up = volume[up_days].tail(10).mean() if sum(up_days.tail(10)) > 0 else 1
        v_dn = volume[~up_days].tail(10).mean() if sum(~up_days.tail(10)) > 0 else 1
        vol_quality = v_up / (v_dn + 1e-6)
        vol_ratio = volume.tail(3).mean() / (volume.tail(30).mean() + 1e-6)

        price_20d = (close.iloc[-1] / close.iloc[-20]) - 1 if len(close) >= 20 else 0

        y = np.log(close.values)
        x = np.arange(len(y))
        t_current = x[-1]

        p0 = [t_current + 100, 0.9, 4.0, y[-1], -0.5, 0.01, 0.01]
        lb = [t_current + 10, 0.5, 2.0, -np.inf, -np.inf, -np.inf, -np.inf]
        ub = [t_current + 500, 0.99, 10.0, np.inf, 0, np.inf, np.inf]

        popt, _ = curve_fit(lppls_func, x, y, p0=p0, bounds=(lb, ub), maxfev=8000)
        m_val = popt[1]
        y_fit = lppls_func(x, *popt)
        stability = np.mean(np.abs(y[-10:] - y_fit[-10:]))

        score = 0
        score += 2 if 0.8 < m_val < 0.95 else 0
        score += 2 if stability < 0.02 else 0
        score += 2 if vol_ratio > 1.5 else 0
        score += 2 if vol_quality > 1.3 else 0
        score += 2 if vol_pattern else 0
        score += 2 if trend_ok else 0
        score += 2 if 0.2 < price_20d < 1.0 else 0

        return round(score, 1), popt, price_20d, vol_ratio, vol_quality
    except Exception:
        return None, None, None, None, None

# --------------------------
# 阶段分类
# --------------------------
def get_phase(score, p20):
    if score >= 11 and p20 < 0.6:
        return "🚀 EARLY ROCKET"
    elif score >= 10:
        return "🔥 MID ROCKET"
    else:
        return "👀 WATCH"

# --------------------------
# 画图
# --------------------------
def plot_lppls(df, popt, ticker):
    close = df['Close']
    x = np.arange(len(close))
    y = np.log(close.values)
    y_fit = lppls_func(x, *popt)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(close.index, close, color='#0078ff', linewidth=2, label='Price')
    ax.plot(close.index, np.exp(y_fit), '--', color='#ff4444', linewidth=2, label='LPPLS Fit')
    ax.set_title(f"{ticker} - Rocket Structure", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.2)
    return fig

# --------------------------
# 界面
# --------------------------
st.title("🚀 Rocket Scanner Pro｜稳定版")
st.markdown("支持：Excel | Finviz | 手动 | 预设股票池 | LPPLS筛选")

with st.sidebar:
    st.header("数据源")
    mode = st.radio("选择输入方式", [
        "预设股票池",
        "同花顺Excel(xlsx)",
        "Finviz",
        "手动输入",
    ])
    tickers = []

    if mode == "预设股票池":
        tickers = ["NVDA", "SMCI", "AEHR", "COHR", "002273.SZ", "300308.SZ", "600105.SS"]
        st.success(f"已加载 {len(tickers)} 只")

    elif mode == "同花顺Excel(xlsx)":
        f = st.file_uploader("仅支持 .xlsx 格式（现代Excel）", type=["xlsx"])
        if f:
            try:
                # 仅用openpyxl，彻底抛弃有问题的xlrd
                df = pd.read_excel(f, engine='openpyxl')
                code_col = None
                for c in df.columns:
                    if '代码' in str(c):
                        code_col = c
                        break
                if code_col:
                    raw = df[code_col].astype(str).str.replace("'", "").tolist()
                    for r in raw:
                        t, _ = classify_code(r)
                        if t:
                            tickers.append(t)
                    tickers = sorted(list(set(tickers)))
                    st.success(f"解析 {len(tickers)} 只")
                else:
                    st.warning("未找到【代码】列，请检查Excel")
            except Exception as e:
                st.error("❌ 上传的不是标准Excel文件！请导出真正的 .xlsx 文件")

    elif mode == "Finviz":
        txt = st.text_area("粘贴Finviz代码列表", height=160)
        if txt:
            raw = re.split(r'[\s,]+', txt.strip())
            for r in raw:
                t, _ = classify_code(r)
                if t:
                    tickers.append(t)
            tickers = sorted(list(set(tickers)))
            st.success(f"识别 {len(tickers)} 只")

    elif mode == "手动输入":
        txt = st.text_input("股票代码，逗号分隔")
        if txt:
            for p in txt.split(","):
                t, _ = classify_code(p.strip())
                if t:
                    tickers.append(t)
            st.success(f"已添加 {len(tickers)} 只")

    st.markdown("---")
    top_n = st.number_input("最大扫描数量", 10, 200, 50)
    scan_btn = st.button("开始扫描", type='primary', use_container_width=True)

# --------------------------
# 执行扫描
# --------------------------
if scan_btn and tickers:
    scan_list = tickers[:top_n]
    bar = st.progress(0)
    output = []

    for i, t in enumerate(scan_list):
        bar.progress((i + 1) / len(scan_list))
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False)
            if len(df) < 100:
                continue
            score, popt, p20, vr, vq = compute_rocket_score(df)
            if not score:
                continue
            phase = get_phase(score, p20)
            mkt = classify_code(t.split('.')[0])[1] if '.' in t else "美股"
            output.append({
                "代码": t,
                "市场": mkt,
                "阶段": phase,
                "火箭分": score,
                "20日涨幅%": round(p20 * 100, 1),
                "量比": round(vr, 1),
                "量质比": round(vq, 1)
            })
        except Exception:
            continue

    bar.empty()
    if not output:
        st.warning("未找到符合条件的火箭股")
    else:
        df_out = pd.DataFrame(output).sort_values("火箭分", ascending=False)
        st.subheader("✅ 今日火箭股排行榜")
        st.dataframe(df_out, use_container_width=True)

        st.subheader("🏛 分类精选")
        for ph in ["🚀 EARLY ROCKET", "🔥 MID ROCKET", "👀 WATCH"]:
            sub = df_out[df_out["阶段"] == ph]
            if len(sub) > 0:
                with st.expander(f"{ph} ({len(sub)})"):
                    st.dataframe(sub, use_container_width=True)

        st.subheader("📈 LPPLS 结构验证")
        selected = st.selectbox("查看拟合曲线", df_out["代码"].tolist())
        if selected:
            df = yf.download(selected, period="1y", progress=False)
            s, popt, _, _, _ = compute_rocket_score(df)
            if popt is not None:
                fig = plot_lppls(df, popt, selected)
                st.pyplot(fig)

else:
    st.info("👈 左侧选择数据源 → 点击【开始扫描】")

st.caption("⚠️ 仅供研究学习，不构成投资建议")