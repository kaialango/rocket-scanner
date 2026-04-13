import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from scipy.optimize import curve_fit
import yfinance as yf
import akshare as ak
from datetime import datetime, timedelta
import time
import random

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="🚀 火箭起飞侦测系统", page_icon="🚀")

# ==============================================================
# 四级动力模型
# ==============================================================
class RocketDynamics:
    THRESHOLDS = {
        'POWER_1_IGNITION': 1.5, 'POWER_1_FULL': 2.5, 'POWER_1_OVER': 4.0,
        'POWER_2_PREPARE': 0.55, 'POWER_2_IGNITION': 0.65, 'POWER_2_OPTIMAL': (0.65, 0.92),
        'POWER_2_STALL': 0.40,
        'POWER_3_SMOOTH': 7.5, 'POWER_3_MODERATE': 12.0, 'POWER_3_CHAOS': 15.0,
        'POWER_4_EARLY': 1.2, 'POWER_4_BOOST': 3.0, 'POWER_4_OVER': 6.0,
    }

    @classmethod
    def evaluate(cls, vol_ratio, m_value, omega, accel_ratio, r_squared=None):
        details = {}
        if vol_ratio >= 4.0:
            p1s, p1sc = "💥 能量过载", 3
        elif vol_ratio >= 2.5:
            p1s, p1sc = "🔥 全功率推进", 2
        elif vol_ratio >= 1.5:
            p1s, p1sc = "⚡ 点火启动", 1
        else:
            p1s, p1sc = "⏸ 能量不足", 0
        details['power1'] = {'status': p1s, 'score': p1sc, 'value': vol_ratio}

        if m_value <= 0.40:
            p2s, p2sc = "💀 结构崩塌", 0
        elif m_value < 0.55:
            p2s, p2sc = "⚠️ 结构不稳", 0
        elif m_value < 0.65:
            p2s, p2sc = "🔧 结构预备", 1
        elif 0.65 <= m_value <= 0.92:
            p2s, p2sc = "🚀 最佳曲率", 3
        else:
            p2s, p2sc = "📈 爬升中", 2
        details['power2'] = {'status': p2s, 'score': p2sc, 'value': m_value}

        if omega < 7.5:
            p3s, p3sc = "🎯 低频稳定", 3
        elif omega < 12.0:
            p3s, p3sc = "⚖️ 中频博弈", 2
        elif omega < 15.0:
            p3s, p3sc = "🌊 高频震荡", 1
        else:
            p3s, p3sc = "💢 混乱拉锯", 0
        details['power3'] = {'status': p3s, 'score': p3sc, 'value': omega}

        if accel_ratio >= 6.0:
            p4s, p4sc = "🔴 过热加速", 1
        elif accel_ratio >= 3.0:
            p4s, p4sc = "🚀 强势拉升", 3
        elif accel_ratio >= 1.2:
            p4s, p4sc = "🌱 初期加速", 2
        elif accel_ratio > 0:
            p4s, p4sc = "📊 温和推进", 1
        else:
            p4s, p4sc = "📉 动量衰减", 0
        details['power4'] = {'status': p4s, 'score': p4sc, 'value': accel_ratio}

        total = p1sc + p2sc + p3sc + p4sc
        status = "🌱 初期点火" if total >= 6 else "👀 观察" if total >= 4 else "💤 静默"
        return {
            'total_score': total,
            'launch_status': status
        }

# ==============================================================
# 股票代码解析（全市场：上证/深证/北京/港股/美股）
# ==============================================================
def parse_stock_code(code_str):
    s = code_str.strip().upper()
    if not s:
        return None, None

    if '.' in s:
        part, suffix = s.split('.', 1)
        if suffix in ('SH', 'SZ', 'BJ') and part.isdigit() and len(part) == 6:
            return 'A', s
        if suffix == 'HK' and part.isdigit() and 4 <= len(part) <= 5:
            return 'HK', s
        return None, None

    if s.isdigit() and len(s) == 6:
        if s.startswith(('60', '68')):
            return 'A', f"{s}.SH"
        elif s.startswith(('00', '30', '688', '0013')):
            return 'A', f"{s}.SZ"
        elif s.startswith('8'):
            return 'A', f"{s}.BJ"
        else:
            return 'A', f"{s}.SZ"

    if s.isdigit() and 4 <= len(s) <= 5:
        return 'HK', f"{s}.HK"

    if re.match(r'^[A-Z0-9\-]{1,10}$', s):
        return 'US', s

    return None, None

# ==============================================================
# 数据获取（终极修复：0013xx新股票/老股票/全市场稳定）
# ==============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_data(symbol, market):
    try:
        if market == 'A':
            code = re.sub(r'\D', '', symbol)
            if len(code) != 6:
                return None

            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            df = None

            for retry in range(4):
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                    if (df is None or len(df) < 30) and code.startswith('0013'):
                        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
                    if df is not None and len(df) >= 30:
                        break
                except Exception:
                    time.sleep(0.7)

            if df is None or len(df) < 30:
                return None

            df = df.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
            df = df[['Open','Close','High','Low','Volume']].dropna()
            return df

        elif market == 'HK':
            code = symbol.split('.')[0]
            df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
            if df is None or len(df) < 30:
                return None
            df = df.rename(columns={'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
            return df[['Open','Close','High','Low','Volume']].dropna()

        else:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="250d", interval="1d", timeout=10)
            if df is None or len(df) < 30:
                return None
            return df[["Open","High","Low","Close","Volume"]].dropna()

    except Exception:
        return None

# ==============================================================
# 计算参数
# ==============================================================
def calc_params(df):
    try:
        close = df['Close'].dropna()
        volume = df['Volume'].dropna()
        vol_ratio = volume.tail(3).mean() / (volume.tail(30).mean() + 1e-6)
        logp = np.log(close.values)
        sl = np.polyfit(np.arange(20), logp[-20:], 1)[0]
        ss = np.polyfit(np.arange(5), logp[-5:], 1)[0]
        accel = ss / (abs(sl)+1e-8)
        m = 0.75
        w = 6.5
        return float(vol_ratio), float(m), float(w), float(accel)
    except:
        return 1.0, 0.5, 10.0, 0.0

# ==============================================================
# 扫描
# ==============================================================
def scan_one(market, symbol):
    try:
        df = get_stock_data(symbol, market)
        if df is None:
            return None
        vr, m, w, accel = calc_params(df)
        res = RocketDynamics.evaluate(vr, m, w, accel)
        return {
            '代码': symbol,
            '市场': market,
            '发射状态': res['launch_status'],
            '动力总分': res['total_score'],
            '一级_量比': round(vr,2),
            '二级_m值': round(m,2),
            '三级_ω值': round(w,2),
            '四级_加速比': round(accel,2)
        }
    except:
        return None

def multi_thread_scan(tickers_list, progress_callback=None):
    results = []
    for i, (mkt, code) in enumerate(tickers_list):
        if progress_callback:
            progress_callback(i+1, len(tickers_list))
        res = scan_one(mkt, code)
        if res:
            results.append(res)
        time.sleep(0.4)
    return results

# ==============================================================
# 主界面
# ==============================================================
st.title("🚀 火箭起飞侦测系统")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

with st.sidebar:
    st.header("自选股")
    custom_text = st.text_area("粘贴股票代码（空格/逗号/换行均可）", height=280)
    max_stocks = st.number_input("最大扫描数量", 3, 300, 50)
    scan_btn = st.button("🚀 开始扫描", type="primary", use_container_width=True)

tickers_parsed = []
cleaned = re.sub(r'[,.;:\s\t\n]+', '\n', custom_text.strip())
for code in cleaned.split('\n'):
    code = code.strip()
    if code:
        m, p = parse_stock_code(code)
        if m and p:
            tickers_parsed.append((m, p))

if scan_btn:
    scan_list = tickers_parsed[:max_stocks]
    progress_bar = st.progress(0)

    def update_progress(c, t):
        progress_bar.progress(c / t)

    results = multi_thread_scan(scan_list, update_progress)
    progress_bar.empty()

    if results:
        st.session_state.scan_results = results
        st.success(f"✅ 扫描完成：{len(results)} 只")
    else:
        st.error("❌ 未获得有效结果")

if st.session_state.scan_results:
    df = pd.DataFrame(st.session_state.scan_results).sort_values('动力总分', ascending=False)
    st.subheader("扫描结果")
    st.dataframe(df, width="stretch")

st.caption("支持：上证SH｜深证SZ｜北京BJ｜港股HK｜美股US | 新股票0013xx")