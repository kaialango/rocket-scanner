import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import akshare as ak

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="A股+港股扫描", page_icon="🚀")

# ====================== 股票代码解析 ======================
def parse_code(code_str):
    s = code_str.strip().upper()
    if not s:
        return None, None

    # A股
    if s.endswith((".SZ", ".SH", ".BJ")):
        return "A", s
    if s.isdigit() and len(s) == 6:
        if s.startswith(("60", "68")):
            return "A", f"{s}.SH"
        else:
            return "A", f"{s}.SZ"

    # 港股
    if s.endswith(".HK"):
        return "HK", s
    if s.isdigit() and 4 <= len(s) <= 5:
        return "HK", f"{s}.HK"

    return None, None

# ====================== 数据获取（无缓存 + 最稳定） ======================
def get_data(symbol, market):
    try:
        if market == "A":
            code = re.sub(r"\D", "", symbol)
            # 最简调用，自动获取数据，无多余参数
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            df.columns = ["日期","开盘","最高","最低","收盘","成交量","成交额"]
            return df if len(df) > 30 else None

        elif market == "HK":
            code = symbol.split(".")[0]
            df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
            df.columns = ["日期","开盘","最高","最低","收盘","成交量","成交额"]
            return df if len(df) > 30 else None
    except:
        return None

# ====================== 分析逻辑 ======================
def analyze(market, symbol):
    df = get_data(market, symbol)
    if df is None:
        return None
    
    close = df["收盘"].iloc[-30:]
    vol = df["成交量"].iloc[-30:]
    vol_ratio = vol.iloc[-3:].mean() / (vol.mean() + 1e-6)
    
    return {
        "股票代码": symbol,
        "市场": market,
        "量比": round(vol_ratio, 2),
        "最新价": round(close.iloc[-1], 2),
        "状态": "✅ 数据正常"
    }

# ====================== 主界面 ======================
st.title("🚀 A股 + 港股 扫描工具")
input_text = st.text_area("粘贴股票代码（一行一个）", height=200)

if st.button("开始扫描"):
    code_list = [c.strip() for c in re.split(r"[\n,\s]+", input_text) if c.strip()]
    results = []
    
    for code in code_list:
        mkt, sym = parse_code(code)
        if mkt and sym:
            res = analyze(mkt, sym)
            if res:
                results.append(res)
    
    if results:
        st.success(f"✅ 成功获取 {len(results)} 只股票数据")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.error("❌ 未获取到数据，请检查网络或股票代码")

st.caption("支持：上证SH 深证SZ 北交所BJ 港股HK | 无缓存 纯接口")