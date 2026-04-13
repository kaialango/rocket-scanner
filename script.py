import pandas as pd
import os

def extract_tickers_from_excel(file_path, normalize=True):
    """读取Excel第一列股票代码"""

    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_excel(file_path, engine='xlrd')
    except Exception:
        df = pd.read_excel(file_path)

    # ✅ 强制第一列
    first_col = df.columns[0]

    # 基础清洗
    raw_tickers = df[first_col].dropna().astype(str).tolist()
    raw_tickers = [t.strip() for t in raw_tickers if t.strip()]

    if not normalize:
        return raw_tickers

    # ✅ 规范化代码格式
    normalized = []
    for t in raw_tickers:
        if t.isdigit() and len(t) == 6:
            if t.startswith('6'):
                normalized.append(f"{t}.SS")  # 上海
            else:
                normalized.append(f"{t}.SZ")  # 深圳
        elif t.isdigit() and 1 <= int(t) <= 99999:
            normalized.append(f"{t.zfill(5)}.HK")  # 港股
        else:
            normalized.append(t)  # 美股等

    return normalized


def save_to_txt(tickers, output_file):
    """保存为txt"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tickers))


def main():
    input_file = "2026-04-09.xls"   # 改成你的文件名
    output_file = "tickers.txt"

    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    tickers = extract_tickers_from_excel(input_file, normalize=True)

    save_to_txt(tickers, output_file)

    print(f"完成，共提取 {len(tickers)} 个代码")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()