import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import warnings
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
st.set_page_config(
    layout="wide",
    page_title="🚀 火箭起飞侦测系统 | 四级动力模型",
    page_icon="🚀"
)

# 预编译正则表达式
TICKER_PATTERN = re.compile(r'[,\n\s]+')

# ==============================================================
# 四级动力模型
# ==============================================================
class RocketDynamics:
    THRESHOLDS = {
        'POWER_1_IGNITION': 1.8,
        'POWER_1_FULL':     2.5,
        'POWER_1_OVER':     4.0,
        'POWER_2_PREPARE':      0.55,
        'POWER_2_IGNITION':     0.65,
        'POWER_2_OPTIMAL':      (0.70, 0.90),
        'POWER_2_STALL':        0.40,
        'POWER_3_SMOOTH':   8.0,
        'POWER_3_MODERATE': 12.0,
        'POWER_3_CHAOS':    15.0,
        'POWER_4_EARLY':    1.5,
        'POWER_4_BOOST':    3.0,
        'POWER_4_OVER':     6.0,
        'R_SQUARED_HIGH':   0.88,
        'R_SQUARED_MID':    0.75,
    }

    @classmethod
    def evaluate(cls, vol_ratio, m_value, omega, accel_ratio, r_squared=None):
        details = {}
        if vol_ratio >= cls.THRESHOLDS['POWER_1_OVER']:
            p1_status, p1_score = "💥 能量过载", 3
        elif vol_ratio >= cls.THRESHOLDS['POWER_1_FULL']:
            p1_status, p1_score = "🔥 全功率推进", 2
        elif vol_ratio >= cls.THRESHOLDS['POWER_1_IGNITION']:
            p1_status, p1_score = "⚡ 点火启动", 1
        else:
            p1_status, p1_score = "⏸ 能量不足", 0
        details['power1'] = {'status': p1_status, 'score': p1_score, 'value': vol_ratio}

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

        if omega < cls.THRESHOLDS['POWER_3_SMOOTH']:
            p3_status, p3_score = "🎯 低频扫货", 3
        elif omega < cls.THRESHOLDS['POWER_3_MODERATE']:
            p3_status, p3_score = "⚖️ 中频博弈", 2
        elif omega < cls.THRESHOLDS['POWER_3_CHAOS']:
            p3_status, p3_score = "🌊 高频震荡", 1
        else:
            p3_status, p3_score = "💢 混乱拉锯", 0
        details['power3'] = {'status': p3_status, 'score': p3_score, 'value': omega}

        if accel_ratio >= cls.THRESHOLDS['POWER_4_OVER']:
            p4_status, p4_score = "🔴 过热加速", 1
        elif accel_ratio >= cls.THRESHOLDS['POWER_4_BOOST']:
            p4_status, p4_score = "🚀 强势拉升", 3
        elif accel_ratio >= cls.THRESHOLDS['POWER_4_EARLY']:
            p4_status, p4_score = "🌱 初期加速", 2
        elif accel_ratio > 0:
            p4_status, p4_score = "📊 温和推进", 1
        else:
            p4_status, p4_score = "📉 动量衰减", 0
        details['power4'] = {'status': p4_status, 'score': p4_score, 'value': accel_ratio}

        conf_high = r_squared is not None and r_squared >= cls.THRESHOLDS['R_SQUARED_HIGH']
        conf_mid  = r_squared is not None and r_squared >= cls.THRESHOLDS['R_SQUARED_MID']
        total_score = p1_score + p2_score + p3_score + p4_score

        is_early = (p4_score == 2 and (conf_mid or r_squared is None) and not conf_high and p1_score >= 1)
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
            'total_score': total_score,
            'launch_status': launch_status,
            'launch_color': launch_color,
            'launch_phase': launch_phase,
            'details': details,
            'conf_high': conf_high,
            'conf_mid': conf_mid,
        }

# ==============================================================
# 数据获取
# ==============================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data_single(symbol, period="400d"):
    """单个股票数据获取"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")
        
        if df is None or len(df) < 100:
            return None
        
        df = df.rename(columns={
            'Open': 'Open', 'Close': 'Close', 
            'High': 'High', 'Low': 'Low', 'Volume': 'Volume'
        })
        return df[['Open', 'Close', 'High', 'Low', 'Volume']].dropna()
    except:
        return None

def get_stock_data_batch(symbols, period="400d", max_workers=10):
    """并行获取多个股票数据"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data_single, symbol, period): symbol 
                   for symbol in symbols}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result(timeout=30)
                if df is not None:
                    results[symbol] = df
            except:
                continue
    
    return results

# ==============================================================
# LPPLS 计算
# ==============================================================
def lppls_func(x, tc, m, w, a, b, c1, c2):
    dt = tc - x
    dt = np.where(dt > 0, dt, 1e-6)
    power = np.power(dt, m)
    log_dt = np.log(dt)
    return a + b * power + power * (c1 * np.cos(w * log_dt) + c2 * np.sin(w * log_dt))

def fit_lppls_fast(x_data, y_data, t_current, window=60):
    """简化版LPPLS拟合"""
    if len(x_data) > window:
        x_data = x_data[-window:]
        y_data = y_data[-window:]
        offset = x_data[0]
        x_data = x_data - offset
        t_local = t_current - offset
    else:
        t_local = t_current
    
    best_popt, best_res = None, np.inf
    
    if len(y_data) > 20:
        y_diff = np.diff(y_data)
        trend_up = np.mean(y_diff[-10:]) > np.mean(y_diff[:10])
    else:
        trend_up = True
    
    if trend_up:
        m_range = (0.40, 0.85)
        tc_range = (t_local + 10, t_local + 120)
    else:
        m_range = (0.20, 0.65)
        tc_range = (t_local + 5, t_local + 80)
    
    bounds = (
        [tc_range[0], m_range[0], 3.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [tc_range[1], m_range[1], 15.0, np.inf, 0, np.inf, np.inf],
    )
    
    n_tries = 8
    rng = np.random.default_rng(42)
    
    for _ in range(n_tries):
        tc0 = t_local + rng.integers(15, 100)
        m0 = rng.uniform(m_range[0], m_range[1])
        w0 = rng.uniform(5.0, 10.0)
        p0 = [tc0, m0, w0, y_data[-1], -0.2, 0.03, 0.03]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    lppls_func, x_data, y_data, 
                    p0=p0, bounds=bounds, 
                    maxfev=3000, xtol=1e-4
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
    
    if r2 < 0.3 or best_popt[1] < 0.1 or best_popt[1] > 0.99:
        return best_popt, y_fit, max(0, min(r2, 0.85))
    
    return best_popt, y_fit, r2

def calc_accel_ratio_fast(close, short_window=5, long_window=20):
    """向量化加速比计算"""
    if len(close) < long_window + 1:
        return 0.0
    
    log_price = np.log(close.values[-long_window:].astype(float))
    
    x_long = np.arange(long_window)
    x_short = np.arange(short_window)
    
    slope_long = np.cov(x_long, log_price)[0, 1] / np.var(x_long) if np.var(x_long) > 0 else 0
    slope_short = np.cov(x_short, log_price[-short_window:])[0, 1] / np.var(x_short) if np.var(x_short) > 0 else 0
    
    if abs(slope_long) < 1e-8:
        return 0.0
    
    return float(np.clip(slope_short / abs(slope_long), -10, 20))

def compute_parameters_fast(df, lppls_window=60):
    """快速参数计算"""
    try:
        close = df['Close'].dropna()
        volume = df['Volume'].dropna()
        
        if len(close) < 60:
            return None
        
        vol_ratio = float(volume.tail(3).mean() / (volume.tail(30).mean() + 1e-6))
        accel_ratio = calc_accel_ratio_fast(close)
        
        y_all = np.log(close.values.astype(float))
        x_all = np.arange(len(y_all), dtype=float)
        t_current = x_all[-1]
        
        popt, y_fit, r_squared = fit_lppls_fast(x_all, y_all, t_current, window=lppls_window)
        
        win = min(lppls_window, len(close))
        close_win = close.iloc[-win:]
        y_win = y_all[-win:]
        
        price_20d = float((close.iloc[-1] / close.iloc[-20]) - 1) if len(close) >= 20 else 0
        
        if popt is None:
            return {
                'vol_ratio': vol_ratio, 'm_value': 0.5, 'omega': 8.0, 
                'accel_ratio': accel_ratio, 'r_squared': None, 
                'y_fit': None, 'y': y_win, 'x': np.arange(win),
                'close': close_win, 'price_20d': price_20d, 'lppls_ok': False
            }
        
        return {
            'vol_ratio': vol_ratio, 
            'm_value': float(np.clip(popt[1], 0.1, 0.99)),
            'omega': float(np.clip(popt[2], 3.0, 15.0)),
            'accel_ratio': accel_ratio,
            'r_squared': r_squared, 'popt': popt, 'y_fit': y_fit,
            'y': y_win, 'x': np.arange(win), 'close': close_win,
            'price_20d': price_20d, 'lppls_ok': True
        }
    except:
        return None

# ==============================================================
# 批量扫描
# ==============================================================
def scan_batch_stocks(tickers, lppls_window=60, max_workers=8, progress_callback=None):
    """批量扫描股票"""
    valid_tickers = [t for t in tickers if t and isinstance(t, str) and len(t) > 0]
    
    if not valid_tickers:
        return []
    
    if progress_callback:
        progress_callback(0, len(valid_tickers))
    
    data_dict = get_stock_data_batch(valid_tickers, max_workers=max_workers)
    
    if not data_dict:
        return []
    
    results = []
    
    def process_single(ticker, df):
        try:
            params = compute_parameters_fast(df, lppls_window)
            if params is None:
                return None
            
            res = RocketDynamics.evaluate(
                params['vol_ratio'], params['m_value'], 
                params['omega'], params['accel_ratio'], 
                params['r_squared']
            )
            
            return {
                '代码': ticker, '名称': ticker, 
                '发射状态': res['launch_status'], '阶段': res['launch_phase'],
                '动力总分': res['total_score'], 
                '一级_量比': round(params['vol_ratio'], 2),
                '二级_m值': round(params['m_value'], 3), 
                '三级_ω值': round(params['omega'], 2),
                '四级_加速比': round(params['accel_ratio'], 2), 
                '拟合优度_R²': round(params['r_squared'], 3) if params['r_squared'] else None,
                '20日涨幅%': round(params['price_20d'] * 100, 1), 
                '总市值_亿': None, '流通市值_亿': None,
                '动力详情': res['details'], 'params': params
            }
        except:
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, ticker, df): ticker
            for ticker, df in data_dict.items()
        }
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            
            result = future.result()
            if result:
                results.append(result)
    
    return results

# ==============================================================
# 绘图（修复版 - 不使用 add_hline）
# ==============================================================
def plot_rocket_dashboard(result):
    params = result['params']
    details = result['动力详情']
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=(
            '价格与LPPLS拟合', '一级：量能', '二级：曲率 m',
            '三级：频率 ω', '四级动力雷达', '残差分析'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'polar'}, {'type': 'scatter'}]
        ]
    )

    close = params['close']
    y_log = params['y']
    y_fit = params['y_fit']

    # 价格图
    fig.add_trace(
        go.Scatter(x=close.index, y=close.values, name='价格', 
                   line=dict(color='#0078ff')),
        row=1, col=1
    )
    
    if y_fit is not None:
        fig.add_trace(
            go.Scatter(x=close.index, y=np.exp(y_fit), name='LPPLS拟合', 
                       line=dict(color='red', dash='dash')),
            row=1, col=1
        )

    # 量比仪表盘
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", 
            value=params['vol_ratio'], 
            title={'text': '量比'},
            gauge={
                'axis': {'range': [0, 5]},
                'steps': [
                    {'range': [0, 1.8], 'color': '#333'},
                    {'range': [1.8, 2.5], 'color': '#fa4'},
                    {'range': [2.5, 5], 'color': '#f44'}
                ]
            }
        ),
        row=1, col=2
    )
    
    # m值仪表盘
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", 
            value=params['m_value'], 
            title={'text': '曲率 m'},
            gauge={
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.55], 'color': '#333'},
                    {'range': [0.55, 0.7], 'color': '#fa4'},
                    {'range': [0.7, 0.9], 'color': '#4f4'},
                    {'range': [0.9, 1], 'color': '#fa4'}
                ]
            }
        ),
        row=1, col=3
    )
    
    # omega仪表盘
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", 
            value=params['omega'], 
            title={'text': '频率 ω'},
            gauge={
                'axis': {'range': [0, 20]},
                'steps': [
                    {'range': [0, 8], 'color': '#4af'},
                    {'range': [8, 15], 'color': '#fa4'},
                    {'range': [15, 20], 'color': '#f44'}
                ]
            }
        ),
        row=2, col=1
    )

    # 雷达图 - 修复：使用正确的 polar 子图类型
    scores = [
        details['power1']['score'], 
        details['power2']['score'], 
        details['power3']['score'], 
        details['power4']['score']
    ]
    labels = ['一级<br>量能', '二级<br>曲率', '三级<br>频率', '四级<br>加速']
    
    fig.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]], 
            theta=labels + [labels[0]], 
            fill='toself',
            name='动力雷达'
        ),
        row=2, col=2
    )
    
    # 设置极坐标范围
    fig.update_polars(
        radialaxis=dict(range=[0, 3], visible=True),
        row=2, col=2
    )

    # 残差图
    if y_fit is not None:
        residuals = y_log - y_fit
        fig.add_trace(
            go.Scatter(x=close.index, y=residuals, name='残差', line=dict(color='#666')),
            row=2, col=3
        )
        
        # 使用 add_shape 代替 add_hline（修复关键错误）
        fig.add_shape(
            type="line",
            x0=close.index[0],
            x1=close.index[-1],
            y0=0,
            y1=0,
            line=dict(dash='dash', color='red'),
            row=2,
            col=3
        )

    # 添加加速比注释
    accel = params['accel_ratio']
    fig.add_annotation(
        x=close.index[-1], 
        y=close.values[-1], 
        text=f"加速比 {accel:.1f}x", 
        showarrow=True, 
        arrowhead=1, 
        font=dict(size=12),
        row=1, 
        col=1
    )

    # 更新布局
    fig.update_layout(
        height=800, 
        showlegend=False, 
        title=f"{result['代码']} 动力诊断"
    )
    
    return fig

# ==============================================================
# UI
# ==============================================================
st.title("🚀 火箭起飞侦测系统")
st.markdown("**四级动力模型**：能量注入 → 结构曲率 → 频率一致性 → **价格加速比（初期探针）**  |  性能优化版")

for key in ['scan_results', 'stock_info_cache', 'tickers']:
    if key not in st.session_state:
        st.session_state[key] = None

# —————— 侧边栏 ——————
with st.sidebar:
    st.header("⚙️ 扫描配置")
    source_type = st.radio("股票来源", ["自选股", "手动输入"])
    tickers = []

    if source_type == "自选股":
        custom = st.text_area("自选股代码（支持空格/逗号/回车）", 
                              "STZ BF-A BF-B AAOI NOK FOXA TS E ET WAB CAR MRVL CRWV RPM AXIA", 
                              height=150)
        for code in TICKER_PATTERN.split(custom.strip()):
            code = code.strip().upper()
            if code:
                tickers.append(code)
        st.info(f"自选股: {len(tickers)} 只")
    else:
        manual = st.text_input("代码（空格/逗号分隔）", "STZ,BF-A,AAOI")
        for code in TICKER_PATTERN.split(manual.strip()):
            code = code.strip().upper()
            if code:
                tickers.append(code)
        st.info(f"已添加: {len(tickers)} 只")

    col1, col2 = st.columns(2)
    max_stocks = col1.number_input("最大扫描数", 10, 500, 100)
    max_workers = col2.number_input("线程数", 2, 20, 8)
    lppls_window = st.slider("LPPLS 拟合窗口（日）", 30, 120, 60, 5)

    st.markdown("### 🎚️ 动力阈值调节")
    power1_th = st.slider("一级点火阈值（量比）", 1.0, 3.0, 1.8, 0.1)
    power2_min = st.slider("二级最佳曲率下限", 0.55, 0.85, 0.70, 0.01)
    power2_max = st.slider("二级最佳曲率上限", 0.80, 0.99, 0.90, 0.01)
    power3_th = st.slider("三级平滑阈值（ω）", 5.0, 12.0, 8.0, 0.5)
    power4_th = st.slider("四级初期加速比阈值", 1.0, 3.0, 1.5, 0.1)

    RocketDynamics.THRESHOLDS['POWER_1_IGNITION'] = power1_th
    RocketDynamics.THRESHOLDS['POWER_2_OPTIMAL'] = (power2_min, power2_max)
    RocketDynamics.THRESHOLDS['POWER_3_SMOOTH'] = power3_th
    RocketDynamics.THRESHOLDS['POWER_4_EARLY'] = power4_th

    scan_btn = st.button("🚀 开始扫描", type='primary')

# —————— 扫描 ——————
if scan_btn:
    if not tickers:
        st.warning("请先输入股票")
    else:
        scan_list = tickers[:max_stocks]
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(cur, total):
            progress_bar.progress(cur / total if total > 0 else 0)
            status_text.text(f"扫描进度: {cur}/{total}")
        
        with st.spinner(f"正在获取 {len(scan_list)} 只股票数据并计算..."):
            results = scan_batch_stocks(scan_list, lppls_window, max_workers, update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.session_state.scan_results = results
            st.success(f"✅ 扫描完成！成功获取 {len(results)} 只股票数据")
        else:
            st.warning("⚠️ 未找到符合条件的股票数据")

# —————— 结果 ——————
if st.session_state.scan_results:
    res = st.session_state.scan_results
    df = pd.DataFrame(res).sort_values('动力总分', ascending=False)

    st.subheader("📊 扫描统计")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("扫描总数", len(res))
    c2.metric("🌱 初期点火", len(df[df['阶段'] == 'EARLY_STAGE']))
    c3.metric("🚀 正在起飞", len(df[df['阶段'] == 'TAKEOFF']))
    c4.metric("🔍 待量能", len(df[df['阶段'] == 'READY']))
    c5.metric("👀 观察", len(df[df['阶段'] == 'WATCH']))
    c6.metric("平均动力分", f"{df['动力总分'].mean():.1f}/12")

    st.subheader("🏆 火箭起飞排行榜")
    cols = ['代码', '名称', '发射状态', '动力总分', '一级_量比', '二级_m值', 
            '三级_ω值', '四级_加速比', '拟合优度_R²', '20日涨幅%']
    st.dataframe(df[cols], use_container_width=True, height=400)

    st.subheader("📂 发射状态分组")
    groups = {
        'EARLY_STAGE': '🌱🚀 初期点火',
        'TAKEOFF': '🚀🔥 正在起飞',
        'BOOST': '🔥 三级推进',
        'IGNITE': '⚡ 一级点火',
        'READY': '🔍 待量能',
        'WATCH': '👀 观察',
        'SILENT': '💤 静默期'
    }
    
    for phase, name in groups.items():
        sub = df[df['阶段'] == phase]
        if len(sub):
            with st.expander(f"{name} ({len(sub)}只)", expanded=phase in ['EARLY_STAGE', 'TAKEOFF']):
                st.dataframe(sub[cols], use_container_width=True)

    st.subheader("🔬 火箭动力诊断")
    selected = st.selectbox("选择股票", df['代码'].tolist())
    
    if selected:
        row = df[df['代码'] == selected].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("一级：量比", row['一级_量比'], delta=row['动力详情']['power1']['status'])
        c2.metric("二级：m 值", row['二级_m值'], delta=row['动力详情']['power2']['status'])
        c3.metric("三级：ω 值", row['三级_ω值'], delta=row['动力详情']['power3']['status'])
        c4.metric("四级：加速比", row['四级_加速比'], delta=row['动力详情']['power4']['status'])
        
        # 绘制诊断图
        try:
            fig = plot_rocket_dashboard(row)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"绘图出错: {e}")
            # 降级显示简化图表
            st.line_chart(row['params']['close'])

else:
    st.info("👈 左侧配置 → 点击【开始扫描】")

st.caption("⚡ 四级动力模型 | LPPLS 对数周期幂律 | 价格加速比探针 | 并行获取 · 快速拟合优化")