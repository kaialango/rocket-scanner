"""
backtest.py — IGNITION 历史信号回测
用法: python backtest.py [--days 30] [--dir scanner_data]
"""

import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "scanner_data"
FORWARD_DAYS = 30          # 默认前向收益窗口
WIN_THRESHOLD = 0.0        # 胜负分界线（收益率%）

# ── 数据加载 ───────────────────────────────────────────────────────────────────
def load_ignition_files(data_dir: str) -> pd.DataFrame:
    """读取所有 ignition_YYYY-MM-DD.csv，合并成一张表"""
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not (fname.startswith("ignition_") and fname.endswith(".csv")):
            continue
        date_str = fname.replace("ignition_", "").replace(".csv", "")
        try:
            scan_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"[SKIP] 文件名格式不匹配: {fname}")
            continue
        df = pd.read_csv(os.path.join(data_dir, fname))
        df["ScanDate"] = scan_date
        df["ScanDateStr"] = date_str
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"在 {data_dir} 中找不到任何 ignition_*.csv 文件")

    result = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] 共读取 {len(result)} 条信号，来自 {len(frames)} 个交易日")
    return result


# ── 前向收益获取 ───────────────────────────────────────────────────────────────
def fetch_forward_return(ticker: str, scan_date: datetime, forward_days: int) -> float | None:
    """返回 scan_date 后 forward_days 天的收益率(%)，数据不足返回 None"""
    end_date = scan_date + timedelta(days=forward_days + 7)   # +7 应对节假日
    if end_date > datetime.now():
        return None  # 未来数据，跳过
    try:
        hist = yf.Ticker(ticker).history(
            start=scan_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True
        )["Close"].dropna()

        if len(hist) < 2:
            return None

        # 取第一个收盘价（信号当日）和第 forward_days 个交易日收盘价
        actual_days = min(forward_days, len(hist) - 1)
        r = (hist.iloc[actual_days] - hist.iloc[0]) / hist.iloc[0] * 100
        return round(float(r), 2)
    except Exception:
        return None


def build_returns(df: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    """批量获取前向收益，附加 ForwardReturn 列"""
    total = len(df)
    returns = []
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  价格获取进度: {i+1}/{total}", end="\r")
        r = fetch_forward_return(row["Ticker"], row["ScanDate"], forward_days)
        returns.append(r)
    print()
    df = df.copy()
    df["ForwardReturn"] = returns
    df = df.dropna(subset=["ForwardReturn"])
    print(f"[OK] 有效回测记录: {len(df)} 条（排除未来/数据不足）")
    return df


# ── 信号组合分析 ───────────────────────────────────────────────────────────────
def analyze_combinations(df: pd.DataFrame, win_threshold: float = 0.0):
    """打印各信号组合的胜率与平均收益"""

    conditions = [
        ("ALL IGNITION",          pd.Series([True] * len(df), index=df.index)),
        ("RS=100",                 df["RS"] == 100),
        ("CV < 2",                 df["Curvature"] < 2),
        ("CV > 3  [HIGH_RISK]",    df["Curvature"] > 3),
        ("M > 30%",                df["Mom10"] > 30),
        ("RS=100 & CV<2",          (df["RS"] == 100) & (df["Curvature"] < 2)),
        ("RS=100 & M>30",          (df["RS"] == 100) & (df["Mom10"] > 30)),
        ("RS=100 & CV<2 & M>30",   (df["RS"] == 100) & (df["Curvature"] < 2) & (df["Mom10"] > 30)),
        ("Score >= 15",            df["Score"] >= 15),
        ("Score >= 15 & CV<2",     (df["Score"] >= 15) & (df["Curvature"] < 2)),
    ]

    header = f"{'条件':<28} {'数量':>4}  {'胜率':>6}  {'平均收益':>8}  {'中位数':>7}  {'最大':>7}  {'最小':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name, mask in conditions:
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {name:<26} {'—':>4}  {'—':>6}  {'—':>8}")
            continue
        wins = (sub["ForwardReturn"] > win_threshold).sum()
        win_rate  = wins / n * 100
        avg_ret   = sub["ForwardReturn"].mean()
        med_ret   = sub["ForwardReturn"].median()
        max_ret   = sub["ForwardReturn"].max()
        min_ret   = sub["ForwardReturn"].min()
        print(f"  {name:<26} {n:>4}  {win_rate:>5.1f}%  {avg_ret:>+7.2f}%  {med_ret:>+6.2f}%  {max_ret:>+6.2f}%  {min_ret:>+6.2f}%")

    print("=" * len(header))


# ── 个股明细（可选） ──────────────────────────────────────────────────────────
def print_top_performers(df: pd.DataFrame, n: int = 10):
    top = df.nlargest(n, "ForwardReturn")[
        ["ScanDateStr", "Ticker", "Score", "Mom10", "RS", "Curvature", "ForwardReturn"]
    ]
    print(f"\n[TOP {n}] 收益最高个股:")
    print(top.to_string(index=False))


# ── 主入口 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IGNITION 回测")
    parser.add_argument("--days",  type=int,  default=FORWARD_DAYS, help="前向收益天数 (默认30)")
    parser.add_argument("--dir",   type=str,  default=DATA_DIR,     help="scanner_data 目录路径")
    parser.add_argument("--save",  action="store_true",             help="保存结果到 backtest_results.csv")
    parser.add_argument("--top",   type=int,  default=0,            help="打印 Top N 个股明细")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  IGNITION BACKTEST  |  前向窗口: {args.days}天")
    print(f"{'='*50}\n")

    raw_df = load_ignition_files(args.dir)
    df     = build_returns(raw_df, args.days)
    analyze_combinations(df)

    if args.top > 0:
        print_top_performers(df, args.top)

    if args.save:
        out = "backtest_results.csv"
        df.to_csv(out, index=False)
        print(f"\n[SAVE] 结果已保存: {out}")


if __name__ == "__main__":
    main()
