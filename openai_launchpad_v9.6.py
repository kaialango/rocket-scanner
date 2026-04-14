"""
🚀 ROCKET LAUNCH DETECTOR v9.5 - MONSTER PIPELINE
核心改进：VOODOO早期候选(RS≥55) + 流动性真空 + 早期钩子 + 修正过热惩罚
"""

import pandas as pd
import numpy as np
import yfinance as yf
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
import time
from urllib.parse import quote
warnings.filterwarnings("ignore")

# =========================================
# 配置
# =========================================
yf.set_tz_cache_location("cache")

MIN_PRICE = 5.0
MIN_VOLUME = 5000000
MIN_DOLLAR_VOLUME = 1e9
MAX_WORKERS = 4

# ETF 黑名单
ETF_BLACKLIST = set([
    "SPY", "QQQ", "IWM", "XLK", "XLE", "XLV", "XLF", "XLI", "XLB", "XLU",
    "SOXX", "SMH", "ARKK", "ARKG", "DIA", "VOO", "VTI", "BND", "TLT", "SHY",
    "IEF", "LQD", "HYG", "EWJ", "EFA", "EEM", "VNQ", "GLD", "SLV", "USO"
])

# 防御性板块黑名单
DEFENSIVE_SECTORS = {
    "MO", "KO", "PEP", "KDP", "PG", "CL", "KMB", "GIS", "CAG", "CPB", "HSY", "MDLZ",
    "NEE", "SO", "D", "DUK", "AEP", "PPL", "EXC", "XEL", "WEC", "ED", "EIX", "ES", "ETR", "FE",
    "ENB", "KMI", "WMB", "OKE", "ET", "EPD", "PAA", "MPLX",
    "PM", "BTI",
    "JNJ", "ABT", "MRK", "PFE", "BMY", "GILD",
}

# 贵金属/商品追踪器（RS门槛放宽后的副作用过滤）
COMMODITY_TRACKERS = {
    'PHYS', 'PSLV', 'AG', 'PAAS', 'SVM', 'IAG', 'HBM', 'CDE', 'HL', 'WPM',
    'GOLD', 'NEM', 'GFI', 'AU', 'AEM', 'KGC', 'BVN', 'EGO', 'OR', 'SA',
    'SSRM', 'VALE', 'RIO', 'BHP', 'FCX', 'AA', 'CENX', 'MOS', 'NTR'
}

DATA_DIR = "scanner_data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================================
# 原始股票池（请在此处粘贴你的 1975 只股票列表）
# =========================================
STOCK_UNIVERSE = [
    "AAPL", "OMC", "TTD", "WPP", "ACHR", "AIR", "ATRO", "AVAV", "AXON", "BA", "BETA", "BWXT", "CAE", "CW", "DCO", 
    "DRS", "EMBJ", "ESLT", "FLY", "GD", "GE", "HEI", "HEI-A", "HII", "HWM", "HXL", "KRMN", "KTOS", "LHX", "LMT", 
    "LOAR", "LUNR", "MDA", "MOG-A", "MRCY", "NOC", "PL", "RKLB", "RTX", "SARO", "TDG", "TXT", "VSEC", "VVX", "WWD", 
    "YSS", "CF", "CTVA", "FMC", "ICL", "MOS", "NTR", "SMG", "AAL", "AERO", "ALK", "CPA", "DAL", "LTM", "LUV", "RYAAY", 
    "SKYW", "UAL", "ASR", "CAAP", "JOBY", "OMAB", "PAC", "AA", "CENX", "CSTM", "KALU", "COLM", "FIGS", "GIL", "KTB", 
    "LEVI", "PVH", "RL", "UA", "UAA", "VFC", "ZGN", "AEO", "ANF", "BKE", "BOOT", "BURL", "GAP", "LULU", "ROST", "TJX", 
    "URBN", "VSCO", "AAMI", "AB", "AMG", "AMP", "APAM", "APO", "ARCC", "ARES", "BAM", "BEN", "BLK", "BN", "BX", "BXSL", 
    "CEF", "CG", "CNS", "CRBG", "DBRG", "EQH", "FHI", "FSK", "GBDC", "HASI", "HLNE", "HTGC", "IVZ", "JHG", "KKR", "MAIN", 
    "NTRS", "OBDC", "OTF", "OWL", "PFG", "PHYS", "PSLV", "RJF", "SEIC", "SII", "STEP", "STT", "TPG", "TROW", "VCTR", 
    "WT", "ABG", "AN", "BGSI", "CARG", "CVNA", "DRVN", "GPI", "KMX", "LAD", "MCW", "OPLN", "PAG", "RUSHA", "SAH", "VVV", 
    "F", "GM", "HMC", "LCID", "LI", "NIO", "RACE", "RIVN", "STLA", "TM", "TSLA", "XPEV", "AAP", "ALSN", "ALV", "APTV", 
    "ATMU", "AZO", "BWA", "DAN", "DORM", "GNTX", "GPC", "GT", "GTX", "HSAI", "LEA", "LKQ", "MBLY", "MGA", "MOD", "ORLY", 
    "PHIN", "QS", "VC", "BAC", "BBVA", "BCS", "BK", "BMO", "BNS", "C", "CM", "HSBC", "ING", "JPM", "MUFG", "NTB", "RY", 
    "SAN", "SMFG", "TD", "UBS", "WFC", "ABCB", "ASB", "AUB", "AX", "BANC", "BANF", "BANR", "BAP", "BBAR", "BBT", "BCH", 
    "BKU", "BMA", "BOH", "BOKF", "BPOP", "BSAC", "BSBR", "BUSE", "CASH", "CATY", "CBC", "CBSH", "CBU", "CFG", "CFR", 
    "CIB", "COLB", "CUBI", "CVBF", "DB", "EBC", "EFSC", "EWBC", "FBK", "FBNC", "FBP", "FCNCA", "FFBC", "FFIN", "FHB", 
    "FHN", "FIBK", "FITB", "FLG", "FNB", "FRME", "FULT", "GBCI", "GGAL", "HBAN", "HDB", "HOMB", "HWC", "IBN", "IBOC", 
    "IFS", "INDB", "INTR", "ITUB", "KB", "KEY", "LYG", "MBIN", "MCHB", "MFG", "MTB", "NBTB", "NIC", "NU", "NWG", "ONB", 
    "OZK", "PB", "PFS", "PNC", "PNFP", "PRK", "RF", "RNST", "SBCF", "SFBS", "SFNC", "SHG", "SSB", "SYBT", "TBBK", "TCBI", 
    "TFC", "TFSL", "TOWN", "TRMK", "UBSI", "UCB", "UMBF", "USB", "VLY", "WAFD", "WAL", "WBS", "WF", "WSBC", "WSFS", 
    "WTFC", "ZION", "BUD", "CCU", "FMX", "SAM", "STZ", "TAP", "AKO-B", "CCEP", "CELH", "COCO", "COKE", "FIZZ", "KDP", 
    "KO", "KOF", "MNST", "PEP", "PRMB", "BF-A", "BF-B", "DEO", "AAPG", "ABVX", "ACAD", "ACLX", "ADMA", "ALMS", "ALNY", 
    "ANAB", "ANABV", "APGE", "APLS", "ARGX", "ARQT", "ARWR", "ASND", "AUPH", "AXSM", "BBIO", "BEAM", "BLTE", "BMRN", 
    "BNTX", "CAI", "CELC", "CGON", "CLDX", "CNTA", "COGT", "CORT", "CPRX", "CRNX", "CRSP", "CYTK", "DAWN", "DFTX", 
    "DNLI", "DNTH", "DYN", "ELVN", "ERAS", "EWTX", "EXEL", "FOLD", "GMAB", "GPCR", "HALO", "IBRX", "IDYA", "IMNM", 
    "IMVT", "INCY", "INSM", "IONS", "IRON", "JAZZ", "KOD", "KRYS", "KYMR", "LEGN", "LGND", "MANE", "MDGL", "MIRM", 
    "MLYS", "MRNA", "NAMS", "NKTR", "NUVL", "ONC", "ORKA", "PCVX", "PRAX", "PTCT", "PTGX", "RARE", "RCUS", "REGN", 
    "RLAY", "ROIV", "RPRX", "RVMD", "RYTM", "SLNO", "SMMT", "SNDX", "SRPT", "SRRK", "STOK", "SYRE", "TARS", "TECH", 
    "TERN", "TGTX", "TLX", "TNGX", "TVTX", "VERA", "VKTX", "VRTX", "XENE", "ZLAB", "NXST", "AMRZ", "CRH", "CX", "EXP", 
    "JHX", "KNF", "MLM", "TTAM", "USLM", "VMC", "AAON", "AWI", "BLDR", "CARR", "CSL", "FBIN", "GFF", "JCI", "LII", 
    "LPX", "MAS", "OC", "SPXC", "TREX", "TT", "WMS", "BGC", "BMNR", "CLSK", "CRCL", "ETOR", "EVR", "FIGR", "FUTU", 
    "GLXY", "GS", "HLI", "HOOD", "HUT", "IBKR", "IREN", "JEF", "LAZ", "LPLA", "MARA", "MC", "MIAX", "MKTX", "MRX", 
    "MS", "NMR", "PIPR", "PJT", "RIOT", "SCHW", "SF", "SNEX", "TW", "VIRT", "WULF", "XP", "CE", "DOW", "HUN", "MEOH", 
    "OLN", "AMR", "HCC", "AAOI", "ASTS", "BDC", "CIEN", "CSCO", "DGII", "ERIC", "EXTR", "HPE", "LITE", "MSI", "NOK", 
    "ONDS", "UI", "VIAV", "VISN", "VSAT", "ZBRA", "ANET", "DELL", "HPQ", "IONQ", "LOGI", "PSTG", "QBTS", "RGTI", "SMCI", 
    "SNDK", "STX", "WDC", "HSY", "MDLZ", "TR", "BBUC", "HON", "MMM", "OTTR", "PAM", "SEB", "VMI", "BAH", "EFX", "FCN", 
    "VRSK", "SONY", "ERO", "FCX", "HBM", "IE", "SCCO", "TGB", "AFRM", "ALLY", "AXP", "BFH", "CACC", "COF", "ENVA", 
    "FCFS", "MA", "NNI", "OMF", "PYPL", "SEZL", "SLM", "SOFI", "SYF", "UPST", "V", "WU", "DDS", "M", "PLBL", "A", "BLLN", 
    "CRL", "DGX", "DHR", "GH", "ICLR", "IDXX", "ILMN", "IQV", "LH", "MEDP", "MTD", "NTRA", "QGEN", "RDNT", "RVTY", 
    "SHC", "TMO", "TWST", "VCYT", "WAT", "BJ", "COST", "DG", "DLTR", "OLLI", "PSMT", "TBBB", "TGT", "WMT", "ABBV", 
    "AMGN", "AZN", "BIIB", "BMY", "GILD", "GRFS", "GSK", "JNJ", "LLY", "MRK", "NVO", "NVS", "OGN", "PFE", "SNY", 
    "ALKS", "AMRX", "BCRX", "BHC", "ELAN", "HCM", "HIMS", "HLN", "INDV", "KNSA", "LNTH", "LQDA", "NBIX", "PAHC", "PBH", 
    "RDY", "RGC", "SUPN", "TAK", "TEVA", "UTHR", "VTRS", "ZTS", "CVSA", "EDU", "GHC", "LAUR", "LOPE", "LRN", "MH", 
    "PRDO", "TAL", "AEIS", "AMPX", "ATKR", "AYI", "BE", "ENS", "FPS", "HAYW", "HUBB", "NVT", "POWL", "VRT", "APH", 
    "BELFA", "BELFB", "BHE", "CLS", "FLEX", "FN", "GLW", "JBL", "KN", "LFUS", "OLED", "OSIS", "PLXS", "RAL", "ROG", 
    "SANM", "TEL", "TTMI", "VICR", "EA", "NTES", "RBLX", "TTWO", "ARW", "AVT", "NSIT", "SNX", "ACA", "ACM", "AGX", 
    "APG", "BLD", "CDLR", "DY", "ECG", "EME", "EXPO", "FER", "FIX", "FLR", "GVA", "IESC", "J", "KBR", "LGN", "MTZ", 
    "MYRG", "PRIM", "PWR", "ROAD", "STN", "STRL", "TPC", "TTEK", "BATRA", "BATRK", "CNK", "DIS", "FOX", "FOXA", 
    "FWONA", "FWONK", "IMAX", "LLYVA", "LLYVK", "LYV", "MANU", "MSGS", "NFLX", "NWS", "NWSA", "PSKY", "ROKU", "SIRI", 
    "SPHR", "TKO", "VSNT", "WBD", "WMG", "AGCO", "ALG", "CAT", "CNH", "DE", "OSK", "PCAR", "TEX", "ADM", "AGRO", "BG", 
    "CALM", "TSN", "FRHC", "HTH", "IX", "VOYA", "CBOE", "CME", "COIN", "FDS", "ICE", "MCO", "MORN", "MSCI", "NDAQ", 
    "SPGI", "TRU", "ANDE", "CHEF", "PFGC", "SYY", "UNFI", "USFD", "BIRK", "CROX", "DECK", "NKE", "ONON", "SHOO", "ALH", 
    "HNI", "MHK", "SGI", "SN", "WHR", "BRSL", "CHDN", "DKNG", "FLUT", "RSI", "SGHC", "AAUC", "AEM", "AGI", "ARIS", "AU", 
    "AUGO", "B", "CDE", "CGAU", "DRD", "EGO", "EQX", "FNV", "FSM", "GFI", "HMY", "HYMC", "IAG", "KGC", "NEM", "NG", 
    "OGC", "OR", "ORLA", "PAAS", "RGLD", "SA", "SSRM", "WPM", "ACI", "KR", "SFM", "BTSG", "DOCS", "HNGE", "HQY", "HTFL", 
    "PRVA", "TEM", "TXG", "VEEV", "WAY", "ALHC", "CI", "CNC", "CVS", "ELV", "HUM", "MOH", "OSCR", "UNH", "FND", "HD", 
    "LOW", "CHD", "CL", "CLX", "EL", "ELF", "IPAR", "KMB", "KVUE", "PG", "UL", "AIT", "CNM", "DNOW", "DXPE", "FAST", 
    "FERG", "GWW", "MSM", "POOL", "QXO", "REZI", "SITE", "WCC", "WSO", "XMTR", "ACN", "APLD", "BR", "CACI", "CDW", 
    "CIFR", "CTSH", "EPAM", "EXLS", "FIS", "FISV", "G", "GDS", "GIB", "IBM", "INFY", "INGM", "IT", "JKHY", "KD", "LDOS", 
    "PONY", "PSN", "SAIC", "VNET", "VRRM", "ACGL", "AEG", "AIG", "BNT", "BRK-A", "BRK-B", "HIG", "SLF", "AFL", "BHF", 
    "CNO", "FG", "GL", "GNW", "JXN", "LNC", "MET", "MFC", "PRI", "PRU", "PUK", "UNM", "AFG", "AIZ", "ALL", "CB", "CINF", 
    "CNA", "HGTY", "KNSL", "L", "LMND", "MCY", "MKL", "ORI", "PGR", "PLMR", "RLI", "SIGI", "SKWD", "SLDE", "THG", 
    "TRV", "WRB", "WTM", "EG", "HG", "RGA", "RNR", "SPNT", "ACT", "AGO", "AXS", "ESNT", "FAF", "FNF", "MTG", "NMIH", 
    "RDN", "RYAN", "AJG", "AON", "ARX", "BRO", "BWIN", "CRVL", "ERIE", "MRSH", "NP", "WTW", "CHRW", "EXPD", "FDX", 
    "GXO", "HUBG", "JBHT", "LSTR", "UPS", "ZTO", "ATHM", "BIDU", "BILI", "BZ", "DJT", "GOOG", "GOOGL", "IAC", "JOYY", 
    "META", "MTCH", "PINS", "RDDT", "SPOT", "STUB", "TME", "Z", "ZG", "AMZN", "BABA", "CART", "CHWY", "CPNG", "DASH", 
    "EBAY", "ETSY", "GLBE", "JD", "MELI", "PDD", "SE", "VIPS", "W", "AS", "CALY", "FUN", "GOLF", "HAS", "LTH", "MAT", 
    "MSGE", "OSW", "PLNT", "YETI", "ATAT", "CHH", "H", "HLT", "HTHT", "IHG", "MAR", "WH", "BCC", "SSD", "UFPI", "WFG", 
    "CPRI", "SIG", "TPR", "BWLP", "CMRE", "DAC", "HAFN", "KEX", "MATX", "SBLK", "ZIM", "ACHC", "BKD", "CHE", "CON", 
    "DVA", "EHC", "ENSG", "FMS", "GRDN", "HCA", "LFST", "NHC", "OPCH", "PACS", "SEM", "THC", "UHS", "ABT", "BIO", 
    "BRKR", "BSX", "DXCM", "EW", "GEHC", "GKOS", "GMED", "HAE", "IRTC", "ITGR", "LIVN", "MASI", "MDT", "NEOG", "PEN", 
    "PHG", "PODD", "SNN", "STE", "SYK", "TMDX", "ZBH", "CAH", "COR", "HSIC", "MCK", "ALC", "ALGN", "ATR", "AVTR", "BAX", 
    "BDX", "BLCO", "COO", "HOLX", "ICUI", "ISRG", "LMAT", "MDLN", "MMSI", "NNNN", "NVST", "RGEN", "RMD", "SOLV", "STVN", 
    "TFX", "WRBY", "WST", "XRAY", "ATI", "CMC", "CRS", "ESAB", "GPGI", "MLI", "WOR", "PFSI", "RKT", "HP", "NE", "PTEN", 
    "RIG", "SDRL", "APA", "AR", "BKV", "BSM", "CHRD", "CNQ", "CNX", "COP", "CRC", "CRGY", "2088.HK", "CRK", "CTRA", 
    "DVN", "EOG", "EQT", "EXE", "FANG", "GPOR", "MGY", "MNR", "MTDR", "MUR", "NOG", "OVV", "OXY", "PR", "RRC", "SM", 
    "TALO", "TPL", "VIST", "WDS", "AROC", "BKR", "EFXT", "FLOC", "FTI", "HAL", "KGS", "LB", "LBRT", "NESR", "NOV", 
    "OII", "SEI", "SLB", "TDW", "TS", "USAC", "VAL", "WFRD", "WHD", "WTTR", "BP", "CVE", "CVX", "E", "EC", "EQNR", 
    "IMO", "NFG", "PBR", "PBR-A", "SHEL", "SU", "TGS", "TTE", "XOM", "YPF", "AM", "CMBT", "CQP", "DHT", "DTM", "EE", 
    "ENB", "EPD", "ET", "FRO", "GEL", "GLNG", "HESM", "INSW", "KMI", "KNTK", "LNG", "MPLX", "OKE", "PAA", "PAGP", 
    "PBA", "SOBO", "STNG", "SUNC", "TNK", "TRGP", "TRMD", "TRP", "VG", "VNOM", "WES", "WMB", "CVI", "DINO", "DK", "DKL", 
    "IEP", "MPC", "PARR", "PBF", "PSX", "SUN", "UGP", "VLO", "ALM", "BHP", "EMAT", "MP", "MTRN", "RIO", "SKE", "TECK", 
    "USAR", "VALE", "BVN", "HL", "PPTA", "SBSW", "TFPM", "CAG", "CENT", "CENTA", "CPB", "DAR", "FRPT", "GIS", "HRL", 
    "INGR", "JBS", "KHC", "LW", "MICC", "MKC", "MZTI", "POST", "PPC", "SFD", "SJM", "AMCR", "AVY", "BALL", "CCK", "GEF", 
    "GPK", "IP", "PKG", "REYN", "SEE", "SLGN", "SON", "SW", "SUZ", "ANDG", "BFAM", "FTDR", "HRB", "ROL", "SCI", "CECO", 
    "FSS", "VLTO", "ZWS", "NYT", "PSO", "CNI", "CP", "CSX", "NSC", "TRN", "UNP", "WAB", "VTMX", "HHH", "JOE", "BEKE", 
    "CBRE", "CIGI", "COMP", "CSGP", "CWK", "FSV", "IHS", "JLL", "NMRK", "BC", "DOO", "HOG", "LCII", "PATK", "PII", 
    "THO", "BNL", "GNL", "VICI", "WPC", "AHR", "CTRE", "DOC", "HR", "NHI", "OHI", "SBRA", "VTR", "WELL", "APLE", "DRH", 
    "HST", "PK", "RHP", "COLD", "CUBE", "EGP", "EXR", "FR", "LINE", "LXP", "NSA", "PLD", "PSA", "REXR", "STAG", "TRNO", 
    "AGNC", "ARR", "BXMT", "DX", "NLY", "RITM", "STWD", "ARE", "BXP", "CDP", "CUZ", "HIW", "KRC", "SLG", "VNO", "AMH", 
    "AVB", "CPT", "ELS", "EQR", "ESS", "INVH", "IRT", "MAA", "MRP", "SUI", "UDR", "ADC", "AKR", "BRX", "CURB", "EPRT", 
    "FCPT", "FRT", "GTY", "IVT", "KIM", "KRG", "MAC", "NNN", "NTST", "O", "PECO", "REG", "SKT", "SPG", "UE", "AMT", 
    "CCI", "DLR", "EPR", "EQIX", "GLPI", "IRM", "LAMR", "OUT", "RYN", "SBAC", "UNIT", "WY", "AER", "AL", "CAR", "EQPT", 
    "FTAI", "GATX", "HRI", "MGRC", "R", "SUNB", "UHAL", "UHAL-B", "URI", "WSC", "CVCO", "DHI", "GRBK", "IBP", "KBH", 
    "LEN", "MHO", "MTH", "NVR", "PHM", "SKY", "TMHC", "TOL", "TPH", "BYD", "CZR", "HGV", "LVS", "MGM", "MLCO", "MTN", 
    "PENN", "RRR", "VAC", "WYNN", "BROS", "CAKE", "CAVA", "CMG", "DPZ", "DRI", "EAT", "MCD", "QSR", "SBUX", "SHAK", 
    "TXRH", "WING", "YUM", "YUMC", "BMI", "CGNX", "COHR", "ESE", "FTV", "GRMN", "ITRI", "KEYS", "MKSI", "NOVT", "ST", 
    "TDY", "TRMB", "VNT", "ADT", "ALLE", "BCO", "BRC", "GEO", "MSA", "ACLS", "ACMR", "AEHR", "AMAT", "AMBA", "AMKR", 
    "ASML", "AXTI", "CAMT", "ENTG", "FORM", "IPGP", "KLAC", "KLIC", "LRCX", "NVMI", "ONTO", "PLAB", "Q", "TER", "UCTT", 
    "VECO", "ADI", "ALAB", "ALGM", "AMD", "ARM", "ASX", "AVGO", "CRDO", "CRUS", "DIOD", "GFS", "INTC", "LASR", "LSCC", 
    "MCHP", "MPWR", "MRVL", "MTSI", "MU", "NVDA", "NVTS", "NXPI", "ON", "PI", "POWI", "QCOM", "QRVO", "RMBS", "SIMO", 
    "SITM", "SLAB", "SMTC", "STM", "SWKS", "SYNA", "TSEM", "TSM", "TXN", "UMC", "VSH", "LION", "XXI", "AG", "EXK", 
    "SVM", "ADBE", "ADEA", "ADP", "ADSK", "ALRM", "APPF", "BILL", "BRZE", "BSY", "BTDR", "CDNS", "CHYM", "CRM", "CVLT", 
    "CWAN", "DAVE", "DBD", "DDOG", "DOCU", "DSGX", "DT", "DUOL", "ESTC", "FICO", "FIG", "FROG", "FRSH", "FSLY", "GRND", 
    "GWRE", "HUBS", "IDCC", "INTU", "KC", "KVYO", "LIF", "LYFT", "MANH", "MNDY", "MSTR", "NATL", "NAVN", "NICE", "NIQ", 
    "NOW", "OTEX", "PAYC", "PAYX", "PCOR", "PCTY", "PEGA", "PLUS", "PTC", "PTRN", "QTWO", "RNG", "ROP", "SAP", "SHOP", 
    "SNOW", "SOUN", "SRAD", "SSNC", "STRC", "TEAM", "TTAN", "TYL", "U", "UBER", "WDAY", "WK", "WRD", "YMM", "YOU", "ZM", 
    "ACIW", "AKAM", "BLSH", "BOX", "CALX", "CHKP", "CLBT", "CORZ", "CPAY", "CRWD", "CRWV", "CSGS", "DBX", "DLO", "DOCN", 
    "DOX", "EEFT", "FFIV", "FOUR", "FTNT", "GDDY", "GEN", "GPN", "GTLB", "INFQ", "IOT", "KLAR", "KSPI", "MDB", "MSFT", 
    "NBIS", "NET", "NN", "NTAP", "NTCT", "NTNX", "NTSK", "NYAX", "OKTA", "ORCL", "PAGS", "PANW", "PATH", "PAY", "PAYP", 
    "PLTR", "QLYS", "RBRK", "RELY", "S", "SAIL", "SNPS", "STNE", "TDC", "TOST", "TWLO", "VRNS", "VRSN", "WEX", "WIX", 
    "XYZ", "ZETA", "ZS", "ENPH", "FSLR", "NXT", "RUN", "SEDG", "ABM", "AMTM", "ARMK", "AZZ", "CPRT", "CTAS", "DLB", 
    "MMS", "RBA", "RELX", "RTO", "TRI", "ULS", "UNF", "ALB", "APD", "ASH", "AVNT", "AXTA", "BCPC", "CBT", "CC", "CLMT", 
    "DD", "ECL", "EMN", "ESI", "FUL", "HWKN", "IFF", "KWR", "LIN", "LYB", "MTX", "NEU", "NGVT", "PPG", "PRM", "RPM", 
    "SHW", "SOLS", "SQM", "SSL", "SXT", "WDFC", "WLK", "AME", "AOS", "ATS", "BW", "CMI", "CR", "CSW", "CXT", "DCI", 
    "DOV", "EMR", "ETN", "FELE", "FLS", "GEV", "GGG", "GNRC", "GTES", "GTLS", "HLIO", "IEX", "IR", "ITT", "ITW", "JBTM", 
    "KAI", "MIDD", "MIR", "MWA", "NDSN", "NPO", "OTIS", "PH", "PNR", "ROK", "RRX", "SMR", "SXI", "SYM", "WTS", "XYL", 
    "ASO", "BBWI", "BBY", "CASY", "DKS", "EYE", "FIVE", "GME", "MNSO", "MUSA", "RH", "TSCO", "ULTA", "WSM", "KFY", "RHI", 
    "CLF", "MT", "NUE", "PKX", "RS", "SIM", "STLD", "TX", "AD", "AMX", "BCE", "CHT", "CHTR", "CMCSA", "GSAT", "IRDM", 
    "KT", "KYIV", "LBRDA", "LBRDK", "LBTYA", "LBTYK", "LUMN", "PHI", "RCI", "SATS", "SKM", "T", "TDS", "TIGO", "TIMB", 
    "TKC", "TLK", "TMUS", "TU", "VEON", "VIV", "VOD", "VZ", "ARLP", "BTU", "CNR", "BTI", "MO", "PM", "KMT", "LECO", 
    "RBC", "SNA", "SWK", "TKR", "TTC", "ABNB", "BKNG", "CCL", "CUK", "EXPE", "GBTG", "MMYT", "NCLH", "RCL", "TCOM", 
    "TNL", "VIK", "ARCB", "KNX", "ODFL", "RXO", "SAIA", "SNDR", "TFII", "XPO", "CCJ", "LEU", "NXE", "UEC", "UUUU", 
    "AES", "AQN", "AVA", "BIP", "ELPC", "SRE", "CEG", "KEN", "NRG", "OKLO", "TAC", "TLN", "VST", "AEE", "AEP", "CEPU", 
    "CMS", "CNP", "D", "DTE", "DUK", "ED", "EIX", "EMA", "ES", "ETR", "EVRG", "EXC", "FE", "FTS", "HE", "IDA", "KEP", 
    "LNT", "MGEE", "NEE", "NGG", "NWE", "OGE", "PCG", "PEG", "PNW", "POR", "PPL", "PPLC", "SO", "TXNM", "WEC", "XEL", 
    "ATO", "BIPC", "BKH", "CPK", "CTRI", "MDU", "NI", "NJR", "NWN", "OGS", "SR", "SWX", "UGI", "AWK", "AWR", "CWT", 
    "HTO", "SBS", "WTRG", "AXIA", "BEP", "BEPC", "CWEN", "CWEN-A", "ENLT", "FLNC", "MWH", "ORA", "CLH", "CWST", "GFL", 
    "RSG", "WCN", "WM"
]

STOCK_UNIVERSE = list(set(STOCK_UNIVERSE))
STOCK_UNIVERSE = [t for t in STOCK_UNIVERSE if t not in ETF_BLACKLIST]

# =========================================
# 工具函数
# =========================================
def safe_yf_download(ticker, period="6mo", retries=3):
    for i in range(retries):
        try:
            time.sleep(0.1)
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if len(df) > 0:
                return df
        except:
            if i == retries - 1:
                return None
            time.sleep(0.5)
    return None

def safe_yf_info(ticker, retries=2):
    for i in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            stock = yf.Ticker(ticker)
            info = stock.info
            if info and 'regularMarketPrice' in info:
                return info
        except:
            if i == retries - 1:
                return {}
            time.sleep(0.5)
    return {}

def get_basic_info(ticker):
    try:
        info = safe_yf_info(ticker)
        if not info:
            return None
        
        price = info.get("regularMarketPrice", info.get("currentPrice", 0))
        if price < MIN_PRICE:
            return None
        
        avg_vol = info.get("averageVolume", 0)
        avg_vol_10d = info.get("averageDailyVolume10Day", avg_vol)
        volume = max(avg_vol, avg_vol_10d)
        dollar_volume = volume * price
        
        if (volume >= MIN_VOLUME) or (dollar_volume >= MIN_DOLLAR_VOLUME):
            return {"Ticker": ticker, "Price": price}
        return None
    except:
        return None

def pre_screen(tickers):
    results = []
    print(f"[SEARCH] 预筛选 {len(tickers)} 只... (成交量>{MIN_VOLUME/1e6:.0f}M 或 成交额>${MIN_DOLLAR_VOLUME/1e9:.0f}B)", flush=True)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_basic_info, t): t for t in tickers}
        completed = 0
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
            completed += 1
            if completed % 200 == 0:
                print(f"   预筛选进度: {completed}/{len(tickers)}", flush=True)
    
    print(f"[OK] 通过: {len(results)} 只\n", flush=True)
    return [r["Ticker"] for r in results]

def get_spy_return():
    try:
        df = safe_yf_download("SPY", period="3mo")
        if df is not None and len(df) > 0:
            return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    except:
        pass
    return 0

def calc_curvature(close):
    if len(close) < 20:
        return 0
    short_slope = (close.iloc[-1] - close.iloc[-5]) / 5 if len(close) >= 5 else 0
    long_slope = (close.iloc[-1] - close.iloc[-20]) / 20
    if long_slope > 0:
        curvature = short_slope / long_slope
        return min(curvature, 5)
    return 0

def calc_metrics(df, spy_ret):
    close = df["Close"]
    vol = df["Volume"]
    price = float(close.iloc[-1])
    
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    
    mom5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    mom10 = float((close.iloc[-1] / close.iloc[-10] - 1) * 100)
    mom20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0
    accel = mom5 - mom20
    
    if len(close) >= 60:
        stock_ret = (close.iloc[-1] / close.iloc[-60] - 1) * 100
        rs = stock_ret - spy_ret
        rs_rating = min(100, max(0, 50 + rs * 2))
    else:
        rs_rating = 50
    
    vol_hist = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    volatility = float(vol_hist.iloc[-1]) if not pd.isna(vol_hist.iloc[-1]) else 50
    
    avg_vol = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1
    avg_vol_5 = float(vol.rolling(5).mean().iloc[-1])
    volume_ratio_5 = float(vol.iloc[-1]) / avg_vol_5 if avg_vol_5 > 0 else 1
    
    trend = 0
    if price > float(ma20.iloc[-1]): trend += 1
    if float(ma20.iloc[-1]) > float(ma50.iloc[-1]): trend += 1
    
    high20 = float(close[-20:].max())
    near_high = (price / high20) > 0.95 if high20 > 0 else False
    
    vol10 = float(close.pct_change().rolling(10).std().iloc[-1])
    vol30_series = close.pct_change().rolling(30).std()
    vol30 = float(vol30_series.iloc[-20:].mean()) if len(vol30_series) >= 20 else vol10
    vcp = vol10 < vol30 * 0.8
    
    vol5 = close.pct_change().rolling(5).std().iloc[-1]
    vol20 = close.pct_change().rolling(20).std().iloc[-1]
    if vol20 > 0 and not np.isnan(vol5) and not np.isnan(vol20):
        contraction = 1 - (vol5 / vol20)
        contraction = max(0, min(1, contraction))
    else:
        contraction = 0
    
    high20_price = close.rolling(20).max().iloc[-1]
    breakout_proximity = price / high20_price if high20_price > 0 else 0
    breakout_proximity = max(0, min(1, breakout_proximity))
    
    ma20_price = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else 0
    ma20_dist = price / ma20_price if ma20_price > 0 else 1
    
    if len(ma20) >= 10 and ma20.iloc[-10] > 0:
        slope = (ma20.iloc[-1] - ma20.iloc[-10]) / ma20.iloc[-10]
    else:
        slope = 0
    
    curvature = calc_curvature(close)
    
    return {
        "Price": price,
        "Mom10d": mom10,
        "Accel": accel,
        "Volatility": volatility,
        "VolRatio": vol_ratio,
        "VolumeRatio5": volume_ratio_5,
        "TrendScore": trend,
        "NearHigh": near_high,
        "VCP": vcp,
        "RS": round(rs_rating, 1),
        "Contraction": round(contraction, 3),
        "BreakoutProx": round(breakout_proximity, 3),
        "Ma20_Dist": round(ma20_dist, 3),
        "Slope": round(slope, 4),
        "Curvature": round(curvature, 2)
    }

def get_risk_label(curvature):
    if curvature > 3.0:
        return "[HIGH_RISK]"
    elif curvature < 1.5:
        return "[CLEAN]"
    return ""

def get_sector_tag(ticker):
    return "[DEFENSIVE]" if ticker in DEFENSIVE_SECTORS else ""

# =========================================
# v9.5 评分函数
# =========================================
def calc_score(m, ticker):
    score = 0
    signals = []
    
    is_voodoo_seed = m["VolRatio"] < 0.75 or m["Contraction"] > 0.45
    
    # v9.5 改进: VOODOO 早期候选 RS 门槛降至 55
    if is_voodoo_seed:
        min_rs_required = 55   # 允许早期 leader
    else:
        min_rs_required = 80
    
    if m["RS"] < min_rs_required:
        return 0, [], "[SKIP]"
    
    # 背离检测
    if m["Mom10d"] > 20 and m["VolumeRatio5"] < 0.8:
        signals.append("DIVERGENCE")
    
    # v9.5 新增: 流动性真空 (DRY_UP)
    vol_dry = m["VolumeRatio5"] < 0.5 and m["VolRatio"] < 0.7
    if vol_dry:
        signals.append("DRY_UP")
        score += 8
    
    # VOODOO 评分
    v_score = 0
    if m["VolRatio"] < 0.75:
        v_score += 15
        signals.append("VOODOO")
    if m["Contraction"] > 0.45:
        v_score += 10
        signals.append("TIGHT")
    if 0.985 <= m["Ma20_Dist"] <= 1.03:
        v_score += 8
        signals.append("ENTRY_ZONE")
    
    # IGNITION 评分
    i_score = 0
    if m["VolRatio"] > 1.4 and m["BreakoutProx"] >= 0.99:
        i_score += 20
        signals.append("LAUNCH")
    
    # v9.5 改进: 早期钩子 (曲率 >1.5 且斜率向上)
    if m["Curvature"] > 1.5 and m["Slope"] > 0:
        i_score += 6
        signals.append("EARLY_HOOK")
    elif m["Curvature"] > 1.8:
        i_score += 8
        signals.append("HOOK")
    
    if m["Mom10d"] > 8:
        i_score += 5
        signals.append("THRUST")
    
    score = v_score + i_score
    
    if m["Mom10d"] < 3 and m["Curvature"] < 1.0 and not is_voodoo_seed:
        return 0, [], "[UTILITY_FILTER]"
    
    # v9.5 改进: 修正过热惩罚 (只惩罚弱延伸)
    if m["Ma20_Dist"] > 1.15 and m["VolRatio"] < 1.2:
        score -= 10
        signals.append("OVERHEAT")
    if m["Ma20_Dist"] < 0.97:
        score -= 20
        signals.append("BROKEN")
    
    if m["Ma20_Dist"] > 1.08:
        score -= 10
        signals.append("OFF_MA20")
    
    # 防御性板块降权
    if ticker in DEFENSIVE_SECTORS:
        score = score * 0.6
        signals.append("DEFENSIVE")

    # Commodity tracker penalty
    if ticker in COMMODITY_TRACKERS:
        score = int(score * 0.7)
    
    stage = "[WATCH]"
    if "LAUNCH" in signals:
        stage = "[FIRE]"
    elif "VOODOO" in signals and "TIGHT" in signals:
        stage = "[VOODOO_ENTRY]"
    elif score >= 15:
        stage = "[ROCKET]"
    
    return max(0, round(score, 1)), signals, stage

def get_search_keywords(ticker, score, mom10, accel, vol, vol_ratio, contraction, breakout_prox, rs, slope, curvature, signals):
    keywords = [ticker, "stock"]
    
    if "LAUNCH" in signals:
        keywords.append("breakout")
        keywords.append("high volume")
    if "VOODOO" in signals:
        keywords.append("compression")
        keywords.append("VCP")
    if "EARLY_HOOK" in signals or "HOOK" in signals:
        keywords.append("inflection")
    if "THRUST" in signals:
        keywords.append("momentum")
    if "DIVERGENCE" in signals:
        keywords.append("divergence")
    if "DRY_UP" in signals:
        keywords.append("volume dry up")
    
    if score >= 15:
        keywords.append("top pick")
    
    keywords.append("technical")
    keywords.append("analysis")
    
    unique_keywords = list(dict.fromkeys(keywords))
    search_query = " ".join(unique_keywords)
    
    google_url = f"https://www.google.com/search?q={quote(search_query)}"
    reddit_url = f"https://www.reddit.com/search/?q={quote(ticker + ' stock')}"
    twitter_url = f"https://twitter.com/search?q={quote(ticker + ' stock')}&f=live"
    
    return {
        "query": search_query,
        "google": google_url,
        "reddit": reddit_url,
        "twitter": twitter_url
    }

def analyze(ticker, spy_ret):
    df = safe_yf_download(ticker, period="6mo")
    if df is None or len(df) < 100:
        return None
    
    m = calc_metrics(df, spy_ret)
    
    vr = m["VolumeRatio5"]
    if vr < 0.3 or vr > 10:
        return None
    
    score, signals, stage = calc_score(m, ticker)
    
    if score < 5:
        return None
    
    search = get_search_keywords(
        ticker, score, m["Mom10d"], m["Accel"], 
        m["Volatility"], m["VolumeRatio5"], 
        m["Contraction"], m["BreakoutProx"], m["RS"],
        m["Slope"], m["Curvature"], signals
    )
    
    return {
        "Ticker": ticker,
        "Score": score,
        "Stage": stage,
        "Signals": ",".join(signals) if signals else "-",
        "Mom10": round(m["Mom10d"], 1),
        "Accel": round(m["Accel"], 1),
        "Vol": round(m["Volatility"], 1),
        "VCP": "V" if m["VCP"] else "",
        "RS": m["RS"],
        "Price": round(m["Price"], 2),
        "VolRatio5": round(vr, 2),
        "BreakoutProx": m["BreakoutProx"],
        "Contraction": m["Contraction"],
        "Ma20_Dist": m["Ma20_Dist"],
        "Slope": m["Slope"],
        "Curvature": m["Curvature"],
        "RiskLabel": get_risk_label(m["Curvature"]),
        "SectorTag": get_sector_tag(ticker),
        "SearchQuery": search["query"],
        "GoogleURL": search["google"],
        "RedditURL": search["reddit"],
        "TwitterURL": search["twitter"]
    }

def load_previous_data():
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    voodoo_file = f"{DATA_DIR}/voodoo_{yesterday}.csv"
    rocket_file = f"{DATA_DIR}/rocket_{yesterday}.csv"
    
    prev_voodoo = set()
    prev_rocket = set()
    
    if os.path.exists(voodoo_file):
        try:
            df = pd.read_csv(voodoo_file)
            prev_voodoo = set(df["Ticker"].tolist())
        except:
            pass
    
    if os.path.exists(rocket_file):
        try:
            df = pd.read_csv(rocket_file)
            prev_rocket = set(df["Ticker"].tolist())
        except:
            pass
    
    return prev_voodoo, prev_rocket

def calculate_conversion(prev_set, current_set):
    if not prev_set:
        return 0, []
    converted = [t for t in prev_set if t in current_set]
    rate = len(converted) / len(prev_set) * 100
    return rate, converted

# =========================================
# 回测模块
# =========================================
def backtest_ignition(forward_days=30):
    """回测历史 IGNITION 列表的 forward 收益"""
    print("\n" + "=" * 60)
    print("[BACKTEST] 历史点火信号回测")
    print("=" * 60)
    
    results = []
    
    for file in os.listdir(DATA_DIR):
        if not file.startswith("ignition_") or not file.endswith(".csv"):
            continue
        
        date_str = file.replace("ignition_", "").replace(".csv", "")
        try:
            scan_date = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue
        
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            ticker = row["Ticker"]
            score = row.get("Score", 0)
            mom10 = row.get("Mom10", 0)
            rs = row.get("RS", 0)
            curvature = row.get("Curvature", 0)
            
            end_date = scan_date + timedelta(days=forward_days)
            if end_date > datetime.now():
                continue
            
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=scan_date, end=end_date + timedelta(days=1))
                
                if len(hist) >= 2:
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    return_pct = (end_price - start_price) / start_price * 100
                    
                    results.append({
                        "Date": date_str,
                        "Ticker": ticker,
                        "Score": score,
                        "Mom10": mom10,
                        "RS": rs,
                        "Curvature": curvature,
                        "Forward_Return": round(return_pct, 2)
                    })
            except:
                pass
    
    if not results:
        print("   无有效回测数据（需要至少2天前的点火记录）")
        return
    
    df_results = pd.DataFrame(results)
    
    conditions = [
        ("RS=100", df_results["RS"] == 100),
        ("RS=100 & CV<2", (df_results["RS"] == 100) & (df_results["Curvature"] < 2)),
        ("RS=100 & CV<2 & M>30", (df_results["RS"] == 100) & (df_results["Curvature"] < 2) & (df_results["Mom10"] > 30)),
        ("Score≥15", df_results["Score"] >= 15),
        ("CV>3 (高风险)", df_results["Curvature"] > 3),
        ("CV<1.5 (干净突破)", df_results["Curvature"] < 1.5),
    ]
    
    print(f"\n   回测记录: {len(df_results)} 条")
    print(f"   统计周期: {forward_days} 天\n")
    
    for name, condition in conditions:
        subset = df_results[condition]
        if len(subset) > 0:
            win_rate = (subset["Forward_Return"] > 0).sum() / len(subset) * 100
            avg_return = subset["Forward_Return"].mean()
            print(f"   {name:35} | 数量:{len(subset):3} | 胜率:{win_rate:5.1f}% | 平均收益:{avg_return:6.2f}%")
    
    df_results.to_csv(f"{DATA_DIR}/backtest_results_{forward_days}d.csv", index=False)
    print(f"\n   [SAVE] 回测结果已保存至: {DATA_DIR}/backtest_results_{forward_days}d.csv")

# =========================================
# 主程序
# =========================================
def main():
    print("\n" + "=" * 50)
    print("   ROCKET DETECTOR v9.6 - MONSTER PIPELINE")
    print("   (RS≥55早期候选 + 流动性真空 + 早期钩子+去commodity)")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    prev_voodoo, prev_rocket = load_previous_data()
    
    print("\n[CONFIG] 筛选条件:")
    print(f"   MIN_PRICE:        ${MIN_PRICE}")
    print(f"   MIN_VOLUME:       {MIN_VOLUME/1e6:.0f}M")
    print(f"   MIN_DOLLAR_VOL:   ${MIN_DOLLAR_VOLUME/1e9:.0f}B")
    print(f"   ETF黑名单:        {len(ETF_BLACKLIST)} 只")
    print(f"   防御板块:          {len(DEFENSIVE_SECTORS)} 只 (分数打6折)")
    print(f"   VOODOO RS门槛:    55 (早期候选)")
    print(f"   MAX_WORKERS:      {MAX_WORKERS}")
    
    start = time.time()
    
    passed = pre_screen(STOCK_UNIVERSE)
    if not passed:
        print("[ERROR] 无股票通过预筛选")
        return
    
    spy_ret = get_spy_return()
    print(f"[SPY] 3月回报: {spy_ret:.1f}%")
    
    print(f"[ANALYZE] 分析 {len(passed)} 只股票...")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze, t, spy_ret): t for t in passed}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            if r:
                results.append(r)
            if (i + 1) % 50 == 0:
                print(f"   进度: {i + 1}/{len(passed)}", flush=True)
    
    if not results:
        print("[ERROR] 无有效分析结果")
        return
    
    df = pd.DataFrame(results).sort_values("Score", ascending=False)
    
    ignition = df[df["Signals"].str.contains("LAUNCH", na=False)]
    voodoo = df[(df["Signals"].str.contains("VOODOO", na=False)) & 
                (~df["Signals"].str.contains("LAUNCH", na=False))]
    rocket = df[df["Score"] >= 12]
    
    current_ignition_set = set(ignition["Ticker"].tolist())
    voodoo_conversion_rate, voodoo_converted = calculate_conversion(prev_voodoo, current_ignition_set)
    rocket_conversion_rate, rocket_converted = calculate_conversion(prev_rocket, current_ignition_set)
    
    elapsed = time.time() - start
    
    # 转化追踪报告
    print("\n" + "=" * 120)
    print("[TRACKING] 转化追踪报告")
    print("=" * 120)
    print(f"   昨日 VOODOO 数量: {len(prev_voodoo)}")
    print(f"   今日 IGNITION 数量: {len(ignition)}")
    print(f"   VOODOO → IGNITION 转化: {len(voodoo_converted)} 只 ({voodoo_conversion_rate:.1f}%)")
    if voodoo_converted:
        print(f"   转化股票: {', '.join(voodoo_converted)}")
    
    print(f"\n   昨日 ROCKET 数量: {len(prev_rocket)}")
    print(f"   ROCKET → IGNITION 转化: {len(rocket_converted)} 只 ({rocket_conversion_rate:.1f}%)")
    if rocket_converted:
        print(f"   转化股票: {', '.join(rocket_converted)}")
    
    # IGNITION 列表
    print("\n" + "=" * 120)
    print("[FIRE] IGNITION - 爆发点火 (放量突破 + 高动量)")
    print("=" * 120)
    if ignition.empty:
        print("  [WARN] 无点火信号")
    else:
        for _, r in ignition.head(20).iterrows():
            risk = f" {r['RiskLabel']}" if r['RiskLabel'] else ""
            defensive = f" {r['SectorTag']}" if r['SectorTag'] else ""
            print(f"{r['Stage']} {r['Ticker']:6} | S:{r['Score']:4} | M:{r['Mom10']:5.1f}% | "
                  f"VR:{r['VolRatio5']:3.1f} | RS:{r['RS']:4.0f} | CV:{r['Curvature']:.2f} | "
                  f"${r['Price']:.2f}{risk}{defensive}")
    
    # VOODOO 列表
    print("\n" + "=" * 120)
    print("[VOODOO_ENTRY] VOODOO - 潜伏蓄势 (缩量盘整 + VCP压缩)")
    print("=" * 120)
    if voodoo.empty:
        print("  [WARN] 无潜伏信号")
    else:
        for _, r in voodoo.head(20).iterrows():
            defensive = f" {r['SectorTag']}" if r['SectorTag'] else ""
            print(f"{r['Stage']} {r['Ticker']:6} | S:{r['Score']:4} | M:{r['Mom10']:5.1f}% | "
                  f"VR:{r['VolRatio5']:3.1f} | RS:{r['RS']:4.0f} | C:{r['Contraction']:.2f} | "
                  f"MA20:{r['Ma20_Dist']:.3f} | ${r['Price']:.2f}{defensive}")
    
    # 背离警告
    divergence_df = df[df["Signals"].str.contains("DIVERGENCE", na=False)]
    if not divergence_df.empty:
        print("\n" + "=" * 120)
        print("[WARNING] 量价背离信号 (高动量 + 缩量) - 建议手动复核")
        print("=" * 120)
        for _, r in divergence_df.head(10).iterrows():
            print(f"   {r['Ticker']:6} | M:{r['Mom10']:5.1f}% | VR:{r['VolRatio5']:3.1f} | RS:{r['RS']:4.0f} | ${r['Price']:.2f}")
    
    # 统计
    print("\n" + "=" * 120)
    print(f"[STATS] 股票池 {len(STOCK_UNIVERSE)} -> 预筛选 {len(passed)} -> 有效 {len(results)}")
    print(f"   IGNITION(点火): {len(ignition)} | VOODOO(潜伏): {len(voodoo)} | 高分(≥12): {len(rocket)}")
    print(f"   耗时: {elapsed:.1f}s")
    print("=" * 120)
    
    # 保存 CSV
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    if len(df) > 0:
        df.to_csv(f"{DATA_DIR}/rocket_results_full_{date_str}.csv", index=False, encoding='utf-8-sig')
        print(f"\n[SAVE] 完整数据已保存至: {DATA_DIR}/rocket_results_full_{date_str}.csv")
        
        if not ignition.empty:
            ignition.to_csv(f"{DATA_DIR}/ignition_{date_str}.csv", index=False, encoding='utf-8-sig')
            print(f"[SAVE] 点火列表已保存至: {DATA_DIR}/ignition_{date_str}.csv")
        
        if not voodoo.empty:
            voodoo.to_csv(f"{DATA_DIR}/voodoo_{date_str}.csv", index=False, encoding='utf-8-sig')
            print(f"[SAVE] 潜伏列表已保存至: {DATA_DIR}/voodoo_{date_str}.csv")
        
        if not rocket.empty:
            rocket.to_csv(f"{DATA_DIR}/rocket_{date_str}.csv", index=False, encoding='utf-8-sig')
            print(f"[SAVE] 高分列表已保存至: {DATA_DIR}/rocket_{date_str}.csv")
        
        # 保存转化追踪
        tracking_data = {
            "Date": date_str,
            "Prev_Voodoo_Count": len(prev_voodoo),
            "Current_Ignition_Count": len(ignition),
            "Voodoo_To_Ignition_Count": len(voodoo_converted),
            "Voodoo_Conversion_Rate": round(voodoo_conversion_rate, 2),
            "Voodoo_Converted_List": ",".join(voodoo_converted),
            "Prev_Rocket_Count": len(prev_rocket),
            "Rocket_To_Ignition_Count": len(rocket_converted),
            "Rocket_Conversion_Rate": round(rocket_conversion_rate, 2),
            "Rocket_Converted_List": ",".join(rocket_converted)
        }
        
        tracking_df = pd.DataFrame([tracking_data])
        tracking_file = f"{DATA_DIR}/conversion_tracking.csv"
        
        if os.path.exists(tracking_file):
            existing = pd.read_csv(tracking_file)
            tracking_df = pd.concat([existing, tracking_df], ignore_index=True)
        
        tracking_df.to_csv(tracking_file, index=False, encoding='utf-8-sig')
        print(f"[SAVE] 转化追踪已保存至: {tracking_file}")

if __name__ == "__main__":
    main()