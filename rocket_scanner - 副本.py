import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import re
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
st.set_page_config(
    layout="wide",
    page_title="🚀 火箭起飞侦测系统 | 四级动力模型",
    page_icon="🚀"
)

# ==============================================================
# 四级动力模型配置
# 新增：第四级 = 价格加速比（初期专用探针）
# ==============================================================
class RocketDynamics:
    """
    四级动力模型
    一级：量能注入（量比）
    二级：结构曲率（LPPLS m 值，修正范围 0~0.99）
    三级：震荡频率（LPPLS ω 值）
    四级：价格加速比（短期斜率/中期斜率），初期捕获核心信号
    """

    THRESHOLDS = {
        # 一级动力（量比）
        'POWER_1_IGNITION': 1.8,
        'POWER_1_FULL':     2.5,
        'POWER_1_OVER':     4.0,

        # 二级动力（m 值，物理有效域 0~0.99）
        'POWER_2_PREPARE':      0.55,
        'POWER_2_IGNITION':     0.65,
        'POWER_2_OPTIMAL':      (0.70, 0.90),   # 收紧上限
        'POWER_2_STALL':        0.40,

        # 三级动力（ω）
        'POWER_3_SMOOTH':   8.0,
        'POWER_3_MODERATE': 12.0,
        'POWER_3_CHAOS':    15.0,

        # 四级动力（加速比）
        'POWER_4_EARLY':    1.5,   # 短期斜率是中期的1.5倍 → 初期加速
        'POWER_4_BOOST':    3.0,   # 3倍以上 → 拉升中
        'POWER_4_OVER':     6.0,   # 6倍以上 → 过热，追高风险

        # 拟合质量
        'R_SQUARED_HIGH':   0.88,
        'R_SQUARED_MID':    0.75,
    }

    @classmethod
    def evaluate(cls, vol_ratio, m_value, omega, accel_ratio, r_squared=None):
        details = {}

        # --- 一级动力：量能 ---
        if vol_ratio >= cls.THRESHOLDS['POWER_1_OVER']:
            p1_status, p1_score = "💥 能量过载", 3
        elif vol_ratio >= cls.THRESHOLDS['POWER_1_FULL']:
            p1_status, p1_score = "🔥 全功率推进", 2
        elif vol_ratio >= cls.THRESHOLDS['POWER_1_IGNITION']:
            p1_status, p1_score = "⚡ 点火启动", 1
        else:
            p1_status, p1_score = "⏸ 能量不足", 0
        details['power1'] = {'status': p1_status, 'score': p1_score, 'value': vol_ratio}

        # --- 二级动力：结构曲率 ---
        if m_value <= cls.THRESHOLDS['POWER_2_STALL']:
            p2_status, p2_score = "💀 结构崩塌", 0
        elif m_value < cls.THRESHOLDS['POWER_2_PREPARE']:
            p2_status, p2_score = "⚠️ 结构不稳", 0
        elif m_value < cls.THRESHOLDS['POWER_2_IGNITION']:
            p2_status, p2_score = "🔧 结构预备", 1
        elif cls.THRESHOLDS['POWER_2_OPTIMAL'][0] <= m_value <= cls.THRESHOLDS['POWER_2_OPTIMAL'][1]:
            p2_status, p2_score = "🚀 最佳曲率", 3
        elif m_value < cls.THRESHOLDS['POWER_2_OPTIMAL'][0]:
            p2_status, p2_score = "📈 爬升中", 2
        else:
            p2_status, p2_score = "🌀 曲率过高", 1
        details['power2'] = {'status': p2_status, 'score': p2_score, 'value': m_value}

        # --- 三级动力：震荡频率 ---
        if omega < cls.THRESHOLDS['POWER_3_SMOOTH']:
            p3_status, p3_score = "🎯 低频扫货", 3
        elif omega < cls.THRESHOLDS['POWER_3_MODERATE']:
            p3_status, p3_score = "⚖️ 中频博弈", 2
        elif omega < cls.THRESHOLDS['POWER_3_CHAOS']:
            p3_status, p3_score = "🌊 高频震荡", 1
        else:
            p3_status, p3_score = "💢 混乱拉锯", 0
        details['power3'] = {'status': p3_status, 'score': p3_score, 'value': omega}

        # --- 四级动力：价格加速比（初期捕获核心） ---
        if accel_ratio >= cls.THRESHOLDS['POWER_4_OVER']:
            p4_status, p4_score = "🔴 过热加速", 1    # 高分但标记风险
        elif accel_ratio >= cls.THRESHOLDS['POWER_4_BOOST']:
            p4_status, p4_score = "🚀 强势拉升", 3
        elif accel_ratio >= cls.THRESHOLDS['POWER_4_EARLY']:
            p4_status, p4_score = "🌱 初期加速", 2    # 初期捕获的黄金区间
        elif accel_ratio > 0:
            p4_status, p4_score = "📊 温和推进", 1
        else:
            p4_status, p4_score = "📉 动量衰减", 0
        details['power4'] = {'status': p4_status, 'score': p4_score, 'value': accel_ratio}

        # --- 置信度 ---
        conf_high = r_squared is not None and r_squared >= cls.THRESHOLDS['R_SQUARED_HIGH']
        conf_mid  = r_squared is not None and r_squared >= cls.THRESHOLDS['R_SQUARED_MID']

        total_score = p1_score + p2_score + p3_score + p4_score  # 最高 12 分

        # --- 阶段判定（新增 EARLY_STAGE） ---
        # 初期捕获优先判定：加速比在黄金区间 + LPPLS 刚成形（中置信）
        is_early = (
            p4_score == 2 and                  # 初期加速
            (conf_mid or r_squared is None) and
            not conf_high and                  # LPPLS 还未完全成形（高置信反而是中后期）
            p1_score >= 1                      # 有量能配合
        )

        if is_early:
            launch_status = "🌱🚀 初期点火 · 起飞窗口开启"
            launch_color  = "#00ccff"
            launch_phase  = "EARLY_STAGE"

        elif total_score >= 10 and conf_high:
            if p1_score >= 2 and p2_score >= 2 and p3_score >= 2:
                launch_status = "🚀🔥 四级全开 · 正在起飞"
                launch_color  = "#ff0000"
                launch_phase  = "TAKEOFF"
            else:
                launch_status = "🔥 三级推进 · 等待全面共振"
                launch_color  = "#ff6600"
                launch_phase  = "BOOST"

        elif total_score >= 7 and conf_mid:
            if p2_score >= 2 and p3_score >= 2:
                launch_status = "🔍 结构+频率共振 · 待量能"
                launch_color  = "#88cc00"
                launch_phase  = "READY"
            else:
                launch_status = "⚡ 一级点火 · 结构确认中"
                launch_color  = "#ffaa00"
                launch_phase  = "IGNITE"

        elif total_score >= 5:
            launch_status = "👀 观察名单 · 动力不完整"
            launch_color  = "#888888"
            launch_phase  = "WATCH"

        else:
            launch_status = "💤 静默期 · 不符合起飞条件"
            launch_color  = "#444444"
            launch_phase  = "SILENT"

        return {
            'total_score':   total_score,
            'launch_status': launch_status,
            'launch_color':  launch_color,
            'launch_phase':  launch_phase,
            'details':       details,
            'conf_high':     conf_high,
            'conf_mid':      conf_mid,
        }


# ==============================================================
# 数据获取
# ==============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_a_share_list_simple():
    try:
        return ak.stock_info_a_code_name()
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)
def get_market_cap_for_stocks(code_list):
    result = {}
    for code in code_list:
        try:
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and len(info) > 0:
                info_dict = dict(zip(info['item'], info['value']))
                def parse_mv(s):
                    s = str(s)
                    if '万亿' in s:
                        return float(s.replace('万亿', '')) * 10000
                    elif '亿' in s:
                        return float(s.replace('亿', ''))
                    else:
                        try:
                            return float(s) / 1e8
                        except:
                            return 0
                result[code] = {
                    'total_mv_yi': parse_mv(info_dict.get('总市值', '0')),
                    'circ_mv_yi':  parse_mv(info_dict.get('流通市值', '0')),
                }
        except:
            continue
    return result


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol):
    try:
        code     = symbol.split('.')[0]
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq"
        )
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '收盘': 'Close',
            '最高': 'High', '最低': 'Low', '成交量': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df[['Open', 'Close', 'High', 'Low', 'Volume']]
    except:
        return None


# ==============================================================
# LPPLS 核心（修复版：多起点鲁棒拟合 + m 上界 0.99）
# ==============================================================
def lppls_func(x, tc, m, w, a, b, c1, c2):
    """
    标准 LPPLS：
    log P(t) = a + b*(tc-t)^m + (tc-t)^m * [c1*cos(w*ln(tc-t)) + c2*sin(w*ln(tc-t))]
    物理约束：b < 0（超指数增长），m ∈ (0, 1)
    """
    dt = tc - x
    dt = np.where(dt > 0, dt, 1e-6)
    power  = np.power(dt, m)
    log_dt = np.log(dt)
    return a + b * power + power * (c1 * np.cos(w * log_dt) + c2 * np.sin(w * log_dt))


def fit_lppls_robust(x_data, y_data, t_current, n_tries=25, window=60):
    """
    多起点随机初始化拟合，取残差最小解。
    window 参数：使用最近 N 日数据（初期用短窗口，信号更清晰）
    """
    # 截取窗口
    if len(x_data) > window:
        x_data = x_data[-window:]
        y_data = y_data[-window:]
        # 重映射 x 轴，但保留与 t_current 的相对关系
        offset = x_data[0]
        x_data = x_data - offset
        t_local = t_current - offset
    else:
        t_local = t_current

    best_popt, best_res = None, np.inf
    rng = np.random.default_rng(0)

    # 物理有效边界：m 上界严格 < 1（用 0.99），b < 0
    bounds = (
        [t_local + 5,  0.10,  3.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [t_local + 180, 0.99, 15.0,  np.inf,      0,  np.inf,  np.inf],
    )

    for _ in range(n_tries):
        tc0  = t_local + rng.integers(10, 150)
        m0   = rng.uniform(0.20, 0.95)
        w0   = rng.uniform(3.0, 12.0)
        phi0 = rng.uniform(0, 2 * np.pi)
        p0   = [tc0, m0, w0, y_data[-1], -0.3, 0.05 * np.cos(phi0), 0.05 * np.sin(phi0)]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    lppls_func, x_data, y_data,
                    p0=p0, bounds=bounds, maxfev=8000
                )
            res = np.sum((lppls_func(x_data, *popt) - y_data) ** 2)
            if res < best_res:
                best_res, best_popt = res, popt
        except:
            continue

    if best_popt is None:
        return None, None, None

    y_fit = lppls_func(x_data, *best_popt)
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return best_popt, y_fit, r2


# ==============================================================
# 四级动力：价格加速比（初期专用探针）
# ==============================================================
def calc_accel_ratio(close, short_window=5, long_window=20):
    """
    slope_acceleration = 短期对数斜率 / 中期对数斜率
    > 1.5 → 初期加速（黄金信号）
    > 3.0 → 拉升中
    > 6.0 → 过热
    < 0   → 方向反转
    """
    if len(close) < long_window + 1:
        return 0.0
    log_price = np.log(close.values.astype(float))
    x_long  = np.arange(long_window)
    x_short = np.arange(short_window)
    slope_long  = np.polyfit(x_long,  log_price[-long_window:],  1)[0]
    slope_short = np.polyfit(x_short, log_price[-short_window:], 1)[0]
    if abs(slope_long) < 1e-8:
        return 0.0
    ratio = slope_short / abs(slope_long)
    return float(np.clip(ratio, -10, 20))


# ==============================================================
# 综合参数计算
# ==============================================================
def compute_parameters(df, lppls_window=60):
    """
    计算四级动力全部参数
    """
    try:
        close  = df['Close'].dropna()
        volume = df['Volume'].dropna()

        if len(close) < 60:
            return None

        # 一级：量比
        vol_ratio = float(volume.tail(3).mean() / (volume.tail(30).mean() + 1e-6))

        # 四级：价格加速比（不依赖 LPPLS，快速计算）
        accel_ratio = calc_accel_ratio(close, short_window=5, long_window=20)

        # LPPLS 拟合
        y_all = np.log(close.values.astype(float))
        x_all = np.arange(len(y_all), dtype=float)
        t_current = x_all[-1]

        popt, y_fit, r_squared = fit_lppls_robust(
            x_all.copy(), y_all.copy(), t_current, n_tries=25, window=lppls_window
        )

        if popt is None:
            # LPPLS 拟合失败仍保留四级动力信号
            return {
                'vol_ratio':   vol_ratio,
                'm_value':     0.0,
                'omega':       0.0,
                'accel_ratio': accel_ratio,
                'r_squared':   None,
                'y_fit':       None,
                'y':           y_all[-lppls_window:],
                'x':           x_all[-lppls_window:],
                'close':       close.iloc[-lppls_window:],
                'price_20d':   float((close.iloc[-1] / close.iloc[-20]) - 1) if len(close) >= 20 else 0,
                'lppls_ok':    False,
            }

        m_value = float(popt[1])
        omega   = float(popt[2])

        # 用于绘图的拟合曲线（对应窗口段）
        win = min(lppls_window, len(close))
        close_win = close.iloc[-win:]
        x_win  = np.arange(win, dtype=float)
        offset = x_all[-win]
        # 重建 y_fit 对应窗口 x（因为 fit_lppls_robust 内部 offset 了 x）
        # 直接用返回的 y_fit（已是窗口内的）
        y_win = y_all[-win:]

        price_20d = float((close.iloc[-1] / close.iloc[-20]) - 1) if len(close) >= 20 else 0

        return {
            'vol_ratio':   vol_ratio,
            'm_value':     m_value,
            'omega':       omega,
            'accel_ratio': accel_ratio,
            'r_squared':   r_squared,
            'popt':        popt,
            'y_fit':       y_fit,
            'y':           y_win,
            'x':           x_win,
            'close':       close_win,
            'price_20d':   price_20d,
            'lppls_ok':    True,
        }

    except Exception as e:
        return None


# ==============================================================
# 多线程扫描
# ==============================================================
def scan_single_stock(ticker, stock_info_cache=None, lppls_window=60):
    try:
        df = get_stock_data(ticker)
        if df is None:
            return None

        params = compute_parameters(df, lppls_window=lppls_window)
        if params is None:
            return None

        rocket_eval = RocketDynamics.evaluate(
            vol_ratio=params['vol_ratio'],
            m_value=params['m_value'],
            omega=params['omega'],
            accel_ratio=params['accel_ratio'],
            r_squared=params['r_squared'],
        )

        code = ticker.split('.')[0]
        market_cap, circ_cap = None, None
        if stock_info_cache and code in stock_info_cache:
            market_cap = stock_info_cache[code].get('total_mv_yi', 0)
            circ_cap   = stock_info_cache[code].get('circ_mv_yi', 0)

        name = code
        if stock_info_cache and code in stock_info_cache:
            name = stock_info_cache[code].get('name', code)

        return {
            '代码':         ticker,
            '名称':         name,
            '发射状态':     rocket_eval['launch_status'],
            '阶段':         rocket_eval['launch_phase'],
            '动力总分':     rocket_eval['total_score'],
            '一级_量比':    round(params['vol_ratio'], 2),
            '二级_m值':     round(params['m_value'], 3),
            '三级_ω值':     round(params['omega'], 2),
            '四级_加速比':  round(params['accel_ratio'], 2),
            '拟合优度_R²':  round(params['r_squared'], 3) if params['r_squared'] is not None else None,
            '20日涨幅%':    round(params['price_20d'] * 100, 1),
            '总市值_亿':    market_cap,
            '流通市值_亿':  circ_cap,
            '动力详情':     rocket_eval['details'],
            'params':       params,
        }
    except:
        return None


def multi_thread_scan(tickers, stock_info_cache=None, max_workers=10,
                      lppls_window=60, progress_callback=None):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_single_stock, t, stock_info_cache, lppls_window): t
            for t in tickers
        }
        for i, future in enumerate(as_completed(futures)):
            if progress_callback:
                progress_callback(i + 1, len(tickers))
            result = future.result()
            if result:
                results.append(result)
    return results


# ==============================================================
# 可视化
# ==============================================================
def plot_rocket_dashboard(result):
    params  = result['params']
    details = result['动力详情']

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '价格与LPPLS拟合（近期窗口）',
            '一级：量能',
            '二级：曲率 m',
            '三级：频率 ω',
            '四级动力雷达',
            '残差分析',
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'scatterpolar'}, {'type': 'scatter'}],
        ]
    )

    # --- 价格图 ---
    close   = params['close']
    y_log   = params['y']
    y_fit   = params['y_fit']

    fig.add_trace(
        go.Scatter(x=close.index, y=close.values, name='价格', line=dict(color='#0078ff')),
        row=1, col=1
    )
    if y_fit is not None:
        fig.add_trace(
            go.Scatter(x=close.index, y=np.exp(y_fit), name='LPPLS拟合',
                       line=dict(color='red', dash='dash')),
            row=1, col=1
        )

    # 标注加速比
    accel = params['accel_ratio']
    accel_color = '#00ccff' if 1.5 <= accel < 3 else '#ff4444' if accel >= 6 else '#44ff44'
    fig.add_annotation(
        x=close.index[-1], y=close.values[-1],
        text=f"加速比 {accel:.1f}x",
        showarrow=True, arrowhead=1,
        font=dict(color=accel_color, size=12),
        row=1, col=1
    )

    # --- 一级仪表：量比 ---
    vol = params['vol_ratio']
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=vol,
        title={"text": "量比"},
        gauge={
            'axis': {'range': [0, 5]},
            'bar':  {'color': '#ff4444' if vol >= 1.8 else '#888888'},
            'steps': [
                {'range': [0, 1.8], 'color': '#333333'},
                {'range': [1.8, 2.5], 'color': '#ffaa44'},
                {'range': [2.5, 5],   'color': '#ff4444'},
            ]
        }
    ), row=1, col=2)

    # --- 二级仪表：m 值（修正范围 0~0.99） ---
    mv = params['m_value']
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=mv,
        title={"text": "曲率 m"},
        gauge={
            'axis': {'range': [0, 1.0]},
            'bar':  {'color': '#44ff44' if 0.70 <= mv <= 0.90 else '#888888'},
            'steps': [
                {'range': [0,    0.55], 'color': '#333333'},
                {'range': [0.55, 0.70], 'color': '#ffaa44'},
                {'range': [0.70, 0.90], 'color': '#44ff44'},
                {'range': [0.90, 1.00], 'color': '#ffaa44'},
            ]
        }
    ), row=1, col=3)

    # --- 三级仪表：ω ---
    om = params['omega']
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=om,
        title={"text": "频率 ω"},
        gauge={
            'axis': {'range': [0, 20]},
            'bar':  {'color': '#44aaff' if om < 8 else '#ffaa44' if om < 15 else '#ff4444'},
            'steps': [
                {'range': [0,  8],  'color': '#44aaff'},
                {'range': [8,  15], 'color': '#ffaa44'},
                {'range': [15, 20], 'color': '#ff4444'},
            ]
        }
    ), row=2, col=1)

    # --- 四维雷达图（含第四级） ---
    scores = [
        details['power1']['score'],
        details['power2']['score'],
        details['power3']['score'],
        details['power4']['score'],
    ]
    labels = ['一级<br>量能', '二级<br>曲率', '三级<br>频率', '四级<br>加速']
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='动力评分',
        line=dict(color='#ff4444', width=2)
    ), row=2, col=2)

    # --- 残差图 ---
    if y_fit is not None:
        residuals = y_log - y_fit
        fig.add_trace(
            go.Scatter(x=close.index, y=residuals, name='残差',
                       line=dict(color='gray')),
            row=2, col=3
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=3)

    r2_str = f"R²={params['r_squared']:.3f}" if params['r_squared'] is not None else "R²=N/A"
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"{result['代码']} {result['名称']} — 火箭动力诊断 | {r2_str}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 3]))
    )
    return fig


# ==============================================================
# Streamlit UI
# ==============================================================
st.title("🚀 火箭起飞侦测系统")
st.markdown(
    "**四级动力模型**：能量注入 → 结构曲率 → 频率一致性 → **价格加速比（初期探针）**  |  支持市值筛选 · 多线程加速"
)

# --- Session State 初始化 ---
for key, default in [
    ('scan_results', None),
    ('stock_info_cache', None),
    ('tickers', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================
# 侧边栏
# ==============================================================
with st.sidebar:
    st.header("⚙️ 扫描配置")

    source_type = st.radio("股票来源", ["全部A股（市值筛选）", "自选股", "手动输入"])
    tickers          = []
    stock_info_cache = None

    # --- 全部A股 ---
    if source_type == "全部A股（市值筛选）":
        st.markdown("### 💰 市值筛选条件")
        col1, col2 = st.columns(2)
        with col1:
            min_market_cap = st.number_input("最小总市值(亿元)", 10, 5000, 100, 10)
        with col2:
            max_market_cap = st.number_input("最大总市值(亿元)（0=不限）", 0, 50000, 0, 100)

        if st.button("加载股票列表", type='primary', use_container_width=True):
            with st.spinner("加载A股列表..."):
                stock_df = get_a_share_list_simple()
                if not stock_df.empty:
                    code_list = stock_df['code'].astype(str).str.zfill(6).tolist()[:500]
                    with st.spinner(f"获取 {len(code_list)} 只市值数据（请稍候）..."):
                        market_cap_dict = get_market_cap_for_stocks(code_list)

                    filtered_stocks = []
                    for _, row in stock_df.iterrows():
                        code = str(row['code']).zfill(6)
                        if code in market_cap_dict:
                            mv = market_cap_dict[code]['total_mv_yi']
                            if mv >= min_market_cap and (max_market_cap == 0 or mv <= max_market_cap):
                                filtered_stocks.append({
                                    'code': code,
                                    'name': row['name'],
                                    'total_mv_yi': mv,
                                    'circ_mv_yi':  market_cap_dict[code]['circ_mv_yi'],
                                })

                    tickers_new = []
                    info_dict   = {}
                    for s in filtered_stocks:
                        c = s['code']
                        if c.startswith(('60', '68')):
                            tk = f"{c}.SH"
                        elif c.startswith(('00', '30')):
                            tk = f"{c}.SZ"
                        else:
                            tk = f"{c}.BJ"
                        tickers_new.append(tk)
                        info_dict[c] = {'name': s['name'],
                                        'total_mv_yi': s['total_mv_yi'],
                                        'circ_mv_yi':  s['circ_mv_yi']}

                    st.session_state.tickers          = tickers_new
                    st.session_state.stock_info_cache = info_dict
                    st.success(f"✅ 已加载 {len(tickers_new)} 只股票（市值≥{min_market_cap}亿）")

                    with st.expander("📊 市值分布预览"):
                        st.dataframe(pd.DataFrame(filtered_stocks[:20]), use_container_width=True)
                else:
                    st.error("获取股票列表失败")

        if st.session_state.tickers:
            tickers          = st.session_state.tickers
            stock_info_cache = st.session_state.stock_info_cache
            st.info(f"当前股票池: {len(tickers)} 只")

    # --- 自选股 ---
    elif source_type == "自选股":
        custom = st.text_area("自选股代码（每行一个）",
                              "000001\n600000\n300750\n002415", height=150)
        for code in re.split(r'[,\n\s]+', custom.strip()):
            code = code.strip()
            if code and code.isdigit() and len(code) == 6:
                if code.startswith(('60', '68')):
                    tickers.append(f"{code}.SH")
                elif code.startswith(('00', '30')):
                    tickers.append(f"{code}.SZ")
                else:
                    tickers.append(f"{code}.BJ")
        st.info(f"自选股: {len(tickers)} 只")

    # --- 手动输入 ---
    else:
        manual = st.text_input("代码（逗号分隔）", "000001,600000,300750")
        for code in manual.split(','):
            code = code.strip()
            if code and code.isdigit() and len(code) == 6:
                if code.startswith(('60', '68')):
                    tickers.append(f"{code}.SH")
                elif code.startswith(('00', '30')):
                    tickers.append(f"{code}.SZ")
                else:
                    tickers.append(f"{code}.BJ")
        st.info(f"已添加: {len(tickers)} 只")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        max_stocks  = st.number_input("最大扫描数", 10, 500, 100)
    with col2:
        max_workers = st.number_input("线程数", 2, 20, 8)

    # LPPLS 窗口（初期用短窗口更精准）
    lppls_window = st.slider(
        "LPPLS 拟合窗口（日）",
        30, 120, 60, 5,
        help="建议 40~60 日捕获初期；80~120 日看中后期结构"
    )

    st.markdown("### 🎚️ 动力阈值调节")
    power1_th  = st.slider("一级点火阈值（量比）",  1.0, 3.0,  1.8, 0.1)
    power2_min = st.slider("二级最佳曲率下限",       0.55, 0.85, 0.70, 0.01)
    power2_max = st.slider("二级最佳曲率上限",       0.80, 0.99, 0.90, 0.01)
    power3_th  = st.slider("三级平滑阈值（ω）",     5.0, 12.0,  8.0, 0.5)
    power4_th  = st.slider("四级初期加速比阈值",     1.0,  3.0,  1.5, 0.1)

    RocketDynamics.THRESHOLDS['POWER_1_IGNITION'] = power1_th
    RocketDynamics.THRESHOLDS['POWER_2_OPTIMAL']  = (power2_min, power2_max)
    RocketDynamics.THRESHOLDS['POWER_3_SMOOTH']   = power3_th
    RocketDynamics.THRESHOLDS['POWER_4_EARLY']    = power4_th

    scan_btn = st.button("🚀 开始扫描", type='primary', use_container_width=True)


# ==============================================================
# 主界面：扫描执行
# ==============================================================
if scan_btn:
    if not tickers:
        st.warning("请先选择股票来源并加载股票列表")
    else:
        scan_list    = tickers[:max_stocks]
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"扫描进度: {current}/{total}")

        results = multi_thread_scan(
            scan_list, stock_info_cache,
            max_workers=max_workers,
            lppls_window=lppls_window,
            progress_callback=update_progress
        )
        progress_bar.empty()
        status_text.empty()

        if results:
            st.session_state.scan_results = results
        else:
            st.warning("未找到符合条件的股票")


# ==============================================================
# 主界面：结果展示
# ==============================================================
if st.session_state.scan_results:
    results    = st.session_state.scan_results
    df_results = pd.DataFrame(results).sort_values('动力总分', ascending=False)

    # --- 统计卡片 ---
    st.subheader("📊 扫描统计")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("扫描总数", len(results))
    with c2:
        n = len(df_results[df_results['阶段'] == 'EARLY_STAGE'])
        st.metric("🌱 初期点火", n)
    with c3:
        n = len(df_results[df_results['阶段'] == 'TAKEOFF'])
        st.metric("🚀 正在起飞", n)
    with c4:
        n = len(df_results[df_results['阶段'] == 'READY'])
        st.metric("🔍 待量能", n)
    with c5:
        n = len(df_results[df_results['阶段'] == 'WATCH'])
        st.metric("👀 观察", n)
    with c6:
        avg = df_results['动力总分'].mean()
        st.metric("平均动力分", f"{avg:.1f}/12")

    # --- 排行榜 ---
    st.subheader("🏆 火箭起飞排行榜")
    display_cols = ['代码', '名称', '发射状态', '动力总分',
                    '一级_量比', '二级_m值', '三级_ω值', '四级_加速比',
                    '拟合优度_R²', '20日涨幅%']
    if '总市值_亿' in df_results.columns and df_results['总市值_亿'].notna().any():
        display_cols.insert(4, '总市值_亿')

    st.dataframe(df_results[display_cols], use_container_width=True, height=400)

    # --- 分组展示 ---
    st.subheader("📂 发射状态分组")
    STATUS_ORDER = ['EARLY_STAGE', 'TAKEOFF', 'BOOST', 'IGNITE', 'READY', 'WATCH', 'SILENT']
    STATUS_NAMES = {
        'EARLY_STAGE': '🌱🚀 初期点火 · 起飞窗口开启',
        'TAKEOFF':     '🚀🔥 四级全开 · 正在起飞',
        'BOOST':       '🔥 三级推进 · 等待全面共振',
        'IGNITE':      '⚡ 一级点火 · 结构确认中',
        'READY':       '🔍 结构+频率共振 · 待量能',
        'WATCH':       '👀 观察名单 · 动力不完整',
        'SILENT':      '💤 静默期',
    }
    for phase in STATUS_ORDER:
        sub = df_results[df_results['阶段'] == phase]
        if len(sub) == 0:
            continue
        expanded = phase in ('EARLY_STAGE', 'TAKEOFF')
        with st.expander(f"{STATUS_NAMES[phase]} ({len(sub)}只)", expanded=expanded):
            st.dataframe(sub[display_cols], use_container_width=True)

    # --- 单股详细诊断 ---
    st.subheader("🔬 火箭动力诊断")
    selected = st.selectbox("选择股票查看详细诊断", df_results['代码'].tolist())

    if selected:
        row = df_results[df_results['代码'] == selected].iloc[0]
        details = row['动力详情']

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("一级：量比",    row['一级_量比'],
                      delta=details['power1']['status'])
        with c2:
            st.metric("二级：m 值",    row['二级_m值'],
                      delta=details['power2']['status'])
        with c3:
            st.metric("三级：ω 值",    row['三级_ω值'],
                      delta=details['power3']['status'])
        with c4:
            st.metric("四级：加速比",  row['四级_加速比'],
                      delta=details['power4']['status'])

        r2_val = row['拟合优度_R²']
        r2_str = f"{r2_val:.3f}" if r2_val is not None else "N/A"
        r2_ok  = r2_val is not None and r2_val >= 0.75

        st.info(f"""
**🎯 综合评估**
- 发射状态：{row['发射状态']}
- 动力总分：{row['动力总分']}/12
- LPPLS 拟合质量：{'✅ 良好' if r2_val and r2_val >= 0.88 else '△ 中等（初期正常）' if r2_ok else '⚠️ 弱（结构未成形，以加速比为主参考）'} (R²={r2_str})
- 四级加速比：{row['四级_加速比']:.2f}x {'← 🌱 初期黄金区间' if 1.5 <= row['四级_加速比'] < 3 else '← 🔴 过热，追高需谨慎' if row['四级_加速比'] >= 6 else ''}
        """)

        fig = plot_rocket_dashboard(row)
        st.plotly_chart(fig, use_container_width=True)

else:
    if not scan_btn:
        st.info("👈 左侧配置扫描参数 → 点击【开始扫描】")

st.markdown("---")
st.caption("⚡ 四级动力模型 | LPPLS 对数周期幂律（多起点鲁棒拟合）| 价格加速比探针 | 支持市值筛选 | 多线程加速")
