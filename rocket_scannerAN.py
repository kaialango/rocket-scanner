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
    def evaluate(cls, vol_ratio, m_value, omega, accel_ratio, r_squared=None, breakout_pct=0.0, price_20d=0.0, rev_accel=None, rev_ttm=None, short_pct=None, short_change=None, inst_pct=None):
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

        # ── 量价联动乘数 (+0~2 加分) ──────────────────────────────────────
        # 逻辑：量能和突破必须同时触发才给高分；单独放量或单独接近高点各降权
        # breakout_pct = 当前价 / 52W高，>0.98 视为突破
        near_breakout = breakout_pct >= 0.98
        vol_firing    = vol_ratio >= cls.THRESHOLDS['POWER_1_IGNITION']  # 量比 >= 1.8
        vol_full      = vol_ratio >= cls.THRESHOLDS['POWER_1_FULL']      # 量比 >= 2.5

        if near_breakout and vol_full:
            multiplier_bonus = 2   # 量价双重确认，最强信号
            multiplier_note  = f"🔗 量价共振 +2 (量比{vol_ratio:.1f}x @ {breakout_pct:.0%})"
        elif near_breakout and vol_firing:
            multiplier_bonus = 1   # 接近突破 + 温和放量
            multiplier_note  = f"🔗 量价配合 +1 (量比{vol_ratio:.1f}x @ {breakout_pct:.0%})"
        elif near_breakout and not vol_firing:
            multiplier_bonus = 0   # 缩量突破，不可信
            multiplier_note  = f"⚠️ 缩量突破 (量比{vol_ratio:.1f}x)"
        elif vol_full and not near_breakout:
            multiplier_bonus = 0   # 放量但未突破，方向不明
            multiplier_note  = f"⚠️ 放量未突破 ({breakout_pct:.0%})"
        else:
            multiplier_bonus = 0
            multiplier_note  = f"52W高 {breakout_pct:.0%}"
        details['multiplier'] = {'bonus': multiplier_bonus, 'note': multiplier_note,
                                  'breakout_pct': breakout_pct}

        total_score = p1_score + p2_score + p3_score + p4_score + multiplier_bonus

        # ── 业绩基本面加分（辅助确认层）-1~+2 ──────────────────────────
        # 设计原则：散户处于信息弱势，股价领先季报 1-2 个季度
        # 因此基本面不作主判断，仅作辅助确认，上限严格控制在 +2
        #
        # 信号优先级（从早到晚）：
        #   1. 做空减仓 + 机构低持仓  ← 比季报早 1-2 季（主要加分来源）
        #   2. 季报营收加速            ← 仅作确认，最多 +1
        #   3. 营收严重减速            ← 警示扣分 -1
        fund_score = 0
        fund_notes = []

        # A. 做空/机构信号（最重要，散户可提前感知机构动向）
        # 做空高且在减仓 = 机构已经开始建仓但还没公告
        if short_pct and short_change is not None:
            if short_pct > 15 and short_change < -0.10:
                fund_score += 2
                fund_notes.append(f"🩳 强烈轧空信号 +2 (空头{short_pct:.0f}%↓{short_change:.0%})")
            elif short_pct > 10 and short_change < -0.05:
                fund_score += 1
                fund_notes.append(f"🩳 空头减仓 +1 (空头{short_pct:.0f}%↓{short_change:.0%})")

        # 机构持仓极低：后续买盘空间大，但注意这可能是机构不感兴趣的信号
        if inst_pct is not None and inst_pct < 20:
            fund_score += 1
            fund_notes.append(f"🏦 机构极低持仓 +1 ({inst_pct:.0f}%，后续买盘空间大)")

        # B. 季报营收加速（确认用，上限 +1）
        # 散户看到季报时价格通常已涨，这层只作确认不作主判断
        if rev_accel is not None:
            if rev_accel > 0.15 and rev_ttm and rev_ttm > 0.20:
                rev_bonus = min(1, fund_score + 1) - fund_score   # 最多让总分+1
                fund_score += 1
                fund_notes.append(f"📊 季报加速确认 +1 (accel={rev_accel:.0%}, TTM={rev_ttm:.0%})")
            elif rev_accel < -0.15:
                fund_score -= 1   # 营收严重减速，扣分警示
                fund_notes.append(f"📉 营收减速警示 -1 (accel={rev_accel:.0%})")

        fund_score = max(-1, min(fund_score, 2))   # 硬限制 -1~+2（总分上限从 17 降到 14）
        details['fundamentals'] = {
            'score': fund_score,
            'notes': fund_notes,
            'rev_accel': rev_accel,
            'rev_ttm': rev_ttm,
            'short_pct': short_pct,
            'inst_pct': inst_pct,
        }
        total_score += fund_score

        # ── REIT 降档（利率驱动，非业绩驱动）──────────────────────────────
        # REIT 的 LPPLS 结构和量比信号在利率预期转向时会批量触发
        # 这是宏观驱动而非个股业绩驱动，对散户而言信号质量打折
        # 自动 -1 分 + 标注，避免 REIT 批量占满排行榜压制成长股
        REIT_INDUSTRIES = {
            'REIT - Retail', 'REIT - Office', 'REIT - Healthcare Facilities',
            'REIT - Industrial', 'REIT - Hotel & Motel', 'REIT - Residential',
            'REIT - Specialty', 'REIT - Mortgage', 'REIT - Diversified',
        }
        # industry 在 evaluate() 中不可用（是扫描后处理阶段加的）
        # 所以这里只打标记，实际降分在 scan_batch_stocks 的后处理里执行
        details['is_reit_candidate'] = False   # 占位，后处理填充

        # ── 已拉升降权（防追顶）─────────────────────────────────────────────
        # 20日涨幅过大说明股票已在飞行中段而非起飞初期
        # 三档降权，超过阈值时同时标记警告，不影响结构指标显示
        already_running = False
        run_warning = ""
        if price_20d >= 0.80:                          # >80%：已经严重过热
            total_score = max(0, total_score - 4)
            already_running = True
            run_warning = f"🔴 已拉升 {price_20d:.0%} (-4分)"
        elif price_20d >= 0.40:                        # 40~80%：拉升中
            total_score = max(0, total_score - 2)
            already_running = True
            run_warning = f"🟠 拉升中 {price_20d:.0%} (-2分)"
        elif price_20d >= 0.25:                        # 25~40%：轻微过热
            total_score = max(0, total_score - 1)
            run_warning = f"🟡 轻微过热 {price_20d:.0%} (-1分)"
        details['run_penalty'] = {'warning': run_warning, 'price_20d': price_20d,
                                   'already_running': already_running}

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
# Industry / Theme 数据获取
# ==============================================================
@st.cache_data(ttl=86400, show_spinner=False)   # 每天缓存一次，不需要实时
def get_industry(symbol: str) -> tuple[str, str]:
    """
    返回 (sector, industry)，获取失败返回 ('Unknown', 'Unknown')
    缓存 24 小时：industry 几乎不变，不需要每次扫描都调用
    """
    try:
        info = yf.Ticker(symbol).info
        sector   = info.get('sector',   'Unknown') or 'Unknown'
        industry = info.get('industry', 'Unknown') or 'Unknown'
        return sector, industry
    except Exception:
        return 'Unknown', 'Unknown'

def get_industries_batch(symbols: list, max_workers: int = 8) -> dict:
    """并行获取所有股票的 industry，返回 {symbol: (sector, industry)}"""
    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(get_industry, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                result[sym] = fut.result(timeout=15)
            except Exception:
                result[sym] = ('Unknown', 'Unknown')
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamentals_us(symbol: str) -> dict:
    """
    获取美股基本面（季报营收加速度 + 机构持仓 + 做空数据）
    缓存 24h：季报数据每季才更新一次，不需要每次扫描都调用

    返回字段：
      rev_accel      — 营收加速度（最新季YoY - 前两季YoY均值）正数=加速
      rev_growth_ttm — 最新季 YoY 增速
      inst_pct       — 机构持仓占比 %
      short_pct      — 做空比例 %
      short_change   — 做空环比变化（负数=空头在减仓=轧空燃料）
    """
    result = {
        'rev_accel': None, 'rev_growth_ttm': None,
        'inst_pct': None, 'short_pct': None, 'short_change': None,
    }
    try:
        tk = yf.Ticker(symbol)

        # ── 营收加速度（因果链第二层）──────────────────────────────────
        try:
            qf = tk.quarterly_financials
            if qf is not None and not qf.empty:
                rev_row = None
                for label in ['Total Revenue', 'Revenue', 'Net Revenue']:
                    if label in qf.index:
                        rev_row = qf.loc[label]
                        break
                if rev_row is not None:
                    rev = rev_row.sort_index().values[::-1]  # 最新在前
                    if len(rev) >= 8 and all(v and v > 0 for v in rev[:8]):
                        g1 = (rev[0] - rev[4]) / rev[4]   # 最新季 YoY
                        g2 = (rev[1] - rev[5]) / rev[5]   # Q-1 YoY
                        g3 = (rev[2] - rev[6]) / rev[6]   # Q-2 YoY
                        result['rev_growth_ttm'] = g1
                        # 加速度 = 最新增速 - 前两期均值（正=加速，这是核心信号）
                        result['rev_accel'] = g1 - (g2 + g3) / 2
        except Exception:
            pass

        # ── 机构持仓（低持仓=后续买盘空间大）──────────────────────────
        try:
            inst = tk.institutional_holders
            if inst is not None and not inst.empty and '% Out' in inst.columns:
                result['inst_pct'] = float(inst['% Out'].sum()) * 100
        except Exception:
            pass

        # ── 做空数据（高做空+减仓=轧空燃料）──────────────────────────
        try:
            info = tk.info
            shares_out   = info.get('sharesOutstanding')
            short_shares = info.get('sharesShort')
            short_prior  = info.get('sharesShortPriorMonth')
            if shares_out and short_shares and shares_out > 0:
                result['short_pct'] = short_shares / shares_out * 100
            if short_shares and short_prior and short_prior > 0:
                result['short_change'] = (short_shares - short_prior) / short_prior
        except Exception:
            pass

    except Exception as e:
        result['_error'] = str(e)[:100]
    return result

def get_fundamentals_batch(symbols: list, max_workers: int = 4) -> dict:
    """
    并行获取基本面，但限制并发数为 4（yfinance 季报接口容易 429）
    返回 {symbol: fund_dict}
    """
    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(get_fundamentals_us, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                result[sym] = fut.result(timeout=20)
            except Exception:
                result[sym] = {'rev_accel': None, 'rev_growth_ttm': None,
                               'inst_pct': None, 'short_pct': None, 'short_change': None}
    return result

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

    # 物理有效性检查：m 超出 (0.1, 0.99) 范围的拟合在数学上成立但无经济意义
    # 此时直接返回 None，不用假 R² 掩盖
    if best_popt[1] < 0.10 or best_popt[1] > 0.99:
        return None, None, None

    return best_popt, y_fit, r2

def calc_accel_ratio_fast(close, short_window=5, mid_window=20, long_window=60):
    """
    加速比 = 短期斜率 / 中期斜率（基准用 60 日而非 20 日）
    
    原版用 20 日作基准，导致已经涨了一段的股票加速比虚高。
    改为：
      - long_slope  = 60 日对数斜率（中期趋势基准）
      - short_slope = 5 日对数斜率（当前动量）
      - accel = short / |long|，> 1.5 = 初期加速区间
    
    同时加入 mid_slope（20日）做方向一致性校验：
      若 short > 0 但 long < 0，说明反弹不是起飞，限制加速比上限
    """
    if len(close) < long_window + 1:
        return 0.0

    log_price = np.log(close.values.astype(float))

    x_long  = np.arange(long_window)
    x_mid   = np.arange(mid_window)
    x_short = np.arange(short_window)

    def slope(x, y):
        vx = np.var(x)
        return np.cov(x, y)[0, 1] / vx if vx > 0 else 0.0

    long_slope  = slope(x_long,  log_price[-long_window:])
    mid_slope   = slope(x_mid,   log_price[-mid_window:])
    short_slope = slope(x_short, log_price[-short_window:])

    # 门槛从 1e-8 提高到 1e-4：防止横盘整理股（CAR型）60日斜率趋近于零
    # 导致 5日任何微小波动被放大成 1200x 级别的虚假加速比
    if abs(long_slope) < 1e-4:
        # 横盘时用 mid_slope（20日）代替，评估近期是否有加速迹象
        if abs(mid_slope) < 1e-4:
            return 1.0  # 完全横盘，返回中性值 1.0
        ratio = short_slope / abs(mid_slope)
    else:
        ratio = short_slope / abs(long_slope)

    # 方向一致性：60日下跌趋势中的短期反弹不是起飞，压制上限
    if long_slope < 0 and short_slope > 0:
        ratio = min(ratio, 0.5)

    # 硬上限 15x：超过此值在经济意义上等价，统一归入过热
    return float(np.clip(ratio, -10, 15))

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
            # LPPLS 失败：m=0.0 → 结构崩塌(0分)；omega=99.0 → 混乱拉锯(0分)
            # 额外计算 breakout_pct 供量价联动乘数使用
            w52_high = float(df['High'].tail(252).max()) if len(df) >= 252 else float(df['High'].max())
            breakout_pct = float(close.iloc[-1]) / w52_high if w52_high > 0 else 0.0
            return {
                'vol_ratio': vol_ratio, 'm_value': 0.0, 'omega': 99.0,
                'accel_ratio': accel_ratio, 'r_squared': None,
                'y_fit': None, 'y': y_win, 'x': np.arange(win),
                'close': close_win, 'price_20d': price_20d,
                'breakout_pct': breakout_pct, 'lppls_ok': False
            }
        
        w52_high = float(df['High'].tail(252).max()) if len(df) >= 252 else float(df['High'].max())
        breakout_pct = float(close.iloc[-1]) / w52_high if w52_high > 0 else 0.0

        return {
            'vol_ratio': vol_ratio,
            'm_value': float(np.clip(popt[1], 0.1, 0.99)),
            'omega': float(np.clip(popt[2], 3.0, 15.0)),
            'accel_ratio': accel_ratio,
            'r_squared': r_squared, 'popt': popt, 'y_fit': y_fit,
            'y': y_win, 'x': np.arange(win), 'close': close_win,
            'price_20d': price_20d, 'breakout_pct': breakout_pct, 'lppls_ok': True
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

    # 并行获取基本面（24h 缓存，速度快；限制并发 4 防 429）
    fund_dict = get_fundamentals_batch(list(data_dict.keys()), max_workers=4)

    results = []
    
    def process_single(ticker, df):
        try:
            params = compute_parameters_fast(df, lppls_window)
            if params is None:
                return None
            
            fund = fund_dict.get(ticker, {})
            # Debug: log if fundamentals came back empty for US-looking tickers
            if not any(fund.get(k) is not None for k in ['rev_accel','short_pct','inst_pct']):
                if re.match(r'^[A-Z]{1,5}$', ticker):   # looks like US ticker
                    pass  # silently skip non-US; real errors surface via exception
            res = RocketDynamics.evaluate(
                params['vol_ratio'], params['m_value'],
                params['omega'], params['accel_ratio'],
                params['r_squared'],
                breakout_pct=params.get('breakout_pct', 0.0),
                price_20d=params.get('price_20d', 0.0),
                rev_accel=fund.get('rev_accel'),
                rev_ttm=fund.get('rev_growth_ttm'),
                short_pct=fund.get('short_pct'),
                short_change=fund.get('short_change'),
                inst_pct=fund.get('inst_pct'),
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
                '52W突破%': round(params.get('breakout_pct', 0) * 100, 1),
                '拉升警告': res['details'].get('run_penalty', {}).get('warning', ''),
                '量价乘数': res['details'].get('multiplier', {}).get('bonus', 0),
                '基本面分': res['details'].get('fundamentals', {}).get('score', 0),
                '营收加速': round(fund.get('rev_accel') * 100, 1) if fund.get('rev_accel') is not None else None,
                'TTM增速%': round(fund.get('rev_growth_ttm') * 100, 1) if fund.get('rev_growth_ttm') is not None else None,
                '做空%': round(fund.get('short_pct'), 1) if fund.get('short_pct') is not None else None,
                '机构持仓%': round(fund.get('inst_pct'), 1) if fund.get('inst_pct') is not None else None,
                '基本面备注': ' | '.join(res['details'].get('fundamentals', {}).get('notes', [])),
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
    
    # ── Theme 共振后处理 ──────────────────────────────────────────────────
    # 在所有个股分数计算完毕后，统一做 industry 分组加分
    # 条件：同 industry 内 ≥2 只独立出信号(score≥6) AND 20日涨幅<30%（非已拉升 sector）
    if results:
        # 并行获取 industry（24h 缓存，速度快）
        all_tickers = [r['代码'] for r in results]
        industry_map = get_industries_batch(all_tickers, max_workers=max_workers)

        # 注入 industry 信息到每条结果
        for r in results:
            sec, ind = industry_map.get(r['代码'], ('Unknown', 'Unknown'))
            r['sector']   = sec
            r['industry'] = ind

        # 统计每个 industry 内的有效信号数
        # 有效信号 = score>=6 AND 20日涨幅<30%（排除已拉升 sector）
        from collections import Counter
        signal_count = Counter()
        for r in results:
            if r['动力总分'] >= 6 and r['20日涨幅%'] < 30 and r['industry'] != 'Unknown':
                signal_count[r['industry']] += 1

        # 对 industry 共振的股票加分
        for r in results:
            ind = r['industry']
            count = signal_count.get(ind, 0)
            if count >= 3 and r['20日涨幅%'] < 30:
                bonus = 2
                note  = f"🔗 行业共振 +2 ({ind}, {count}只信号)"
            elif count >= 2 and r['20日涨幅%'] < 30:
                bonus = 1
                note  = f"🔗 行业共振 +1 ({ind}, {count}只信号)"
            else:
                bonus = 0
                note  = ""
            r['动力总分']   += bonus
            r['行业共振']    = note
            r['行业']        = ind
            r['板块']        = r.get('sector', '')

        # ── REIT 自动降档 ──────────────────────────────────────────────────
        # 在 industry 信息注入后执行，避免 REIT 批量占满排行榜
        REIT_KEYWORDS = ('REIT',)
        for r in results:
            ind = r.get('industry', '')
            if any(kw in ind for kw in REIT_KEYWORDS):
                r['动力总分'] = max(0, r['动力总分'] - 1)
                r['行业']     = ind + ' ⬇'   # 标记已降档
                # 如果降档后阶段不匹配，更新阶段
                if r['动力总分'] < 5 and r['阶段'] in ('WATCH', 'READY', 'IGNITE'):
                    r['阶段']     = 'SILENT'
                    r['发射状态'] = '💤 静默期 · 利率驱动型资产'

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
            code = code.strip().upper()  # 保留连字符：BF-A/BF-B 在 yfinance 必须用连字符
            if code:
                tickers.append(code)
        st.info(f"自选股: {len(tickers)} 只")
    else:
        manual = st.text_input("代码（空格/逗号分隔）", "STZ,BF-A,AAOI")
        for code in TICKER_PATTERN.split(manual.strip()):
            code = code.strip().upper()  # 保留连字符
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

    # 基本面信号统计
    if '基本面分' in df.columns:
        fund_signal   = len(df[df['基本面分'] >= 2])
        fund_nonzero  = len(df[df['基本面分'] != 0])
        fund_total    = len(df)
        if fund_signal > 0:
            st.success(f"📊 **基本面信号** {fund_signal} 只评分≥2（做空减仓/机构低持仓/季报确认）")
        elif fund_nonzero == 0:
            st.warning(
                f"⚠️ 基本面数据全部为空（{fund_total}只）— 可能原因：① 非美股无季报 "
                f"② yfinance 需要≥8季历史 ③ 网络超时。"
                f"技术层信号仍然有效，基本面层暂不计分。"
            )

    # 行业共振统计
    if '行业共振' in df.columns:
        resonance_count = len(df[df['行业共振'] != ''])
        if resonance_count > 0:
            st.info(f"🔗 **行业共振加分** 触发 {resonance_count} 只股票 — 同行业≥2只独立出信号")

    st.subheader("🏆 火箭起飞排行榜")
    cols = ['代码', '名称', '发射状态', '动力总分', '基本面分', '营收加速', 'TTM增速%',
            '做空%', '机构持仓%', '量价乘数', '52W突破%', '一级_量比', '二级_m值',
            '三级_ω值', '四级_加速比', '拟合优度_R²', '20日涨幅%', '拉升警告', '行业共振', '行业']
    safe_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[safe_cols], width="stretch", height=400)

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
                st.dataframe(sub[[c for c in cols if c in sub.columns]], width="stretch")

    st.subheader("🔬 火箭动力诊断")
    selected = st.selectbox("选择股票", df['代码'].tolist())
    
    if selected:
        row = df[df['代码'] == selected].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("一级：量比", row['一级_量比'], delta=row['动力详情']['power1']['status'])
        c2.metric("二级：m 值", row['二级_m值'], delta=row['动力详情']['power2']['status'])
        c3.metric("三级：ω 值", row['三级_ω值'], delta=row['动力详情']['power3']['status'])
        c4.metric("四级：加速比", row['四级_加速比'], delta=row['动力详情']['power4']['status'])

        # 基本面摘要
        fund_detail = row['动力详情'].get('fundamentals', {})
        fund_notes  = fund_detail.get('notes', [])
        fund_sc     = fund_detail.get('score', 0)
        if fund_notes:
            color = "success" if fund_sc >= 2 else "warning" if fund_sc >= 0 else "error"
            getattr(st, color)(f"📊 基本面 ({fund_sc:+d}分)： " + "　".join(fund_notes))
        else:
            st.info("📊 基本面：无季报数据（非美股或数据不足）")
        
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

st.caption("⚡ 量价结构主信号 | 做空/机构辅助 | 季报仅作确认（散户信息弱势，价格领先季报）| REIT自动降档 | 并行获取")