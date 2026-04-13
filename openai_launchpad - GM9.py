"""
🚀 ROCKET LAUNCH DETECTOR v8.1 - EARLY LAUNCH EDITION
核心改进：移除Contraction，聚焦初期launch（突破+动量+放量+斜率）
"""

import pandas as pd
import numpy as np
import yfinance as yf
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
import time
from urllib.parse import quote
warnings.filterwarnings("ignore")

# =========================================
# 配置
# =========================================
yf.set_tz_cache_location("cache")

# 筛选条件
MIN_PRICE = 5.0
MIN_VOLUME = 5000000
MIN_DOLLAR_VOLUME = 1e9
MAX_WORKERS = 4

# ETF 黑名单（彻底排除）
ETF_BLACKLIST = set([
    "SPY", "QQQ", "IWM", "XLK", "XLE", "XLV", "XLF", "XLI", "XLB", "XLU",
    "SOXX", "SMH", "ARKK", "ARKG", "DIA", "VOO", "VTI", "BND", "TLT", "SHY",
    "IEF", "LQD", "HYG", "EWJ", "EFA", "EEM", "VNQ", "GLD", "SLV", "USO"
])

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
# 过滤 ETF
STOCK_UNIVERSE = [t for t in STOCK_UNIVERSE if t not in ETF_BLACKLIST]

# =========================================
# 工具函数
# =========================================
def safe_yf_download(ticker, period="6mo", retries=3):
    """带重试的数据获取"""
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
    """带重试的info获取"""
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

# =========================================
# 步骤1：基础筛选
# =========================================
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

# =========================================
# 步骤2：获取 SPY 基准
# =========================================
def get_spy_return():
    try:
        df = safe_yf_download("SPY", period="3mo")
        if df is not None and len(df) > 0:
            return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    except:
        pass
    return 0

# =========================================
# 步骤3：核心指标计算（曲率 + 斜率 + 动量）
# =========================================
def calc_curvature(close):
    """计算非线性曲率：短斜率 / 长斜率，检测向上拐头"""
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
    
    # 基础移动平均
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    
    # 动量
    mom5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    mom10 = float((close.iloc[-1] / close.iloc[-10] - 1) * 100)
    mom20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0
    accel = mom5 - mom20
    
    # RS Rating
    if len(close) >= 60:
        stock_ret = (close.iloc[-1] / close.iloc[-60] - 1) * 100
        rs = stock_ret - spy_ret
        rs_rating = min(100, max(0, 50 + rs * 2))
    else:
        rs_rating = 50
    
    # 波动率
    vol_hist = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    volatility = float(vol_hist.iloc[-1]) if not pd.isna(vol_hist.iloc[-1]) else 50
    
    # 量比
    avg_vol = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1
    avg_vol_5 = float(vol.rolling(5).mean().iloc[-1])
    volume_ratio_5 = float(vol.iloc[-1]) / avg_vol_5 if avg_vol_5 > 0 else 1
    
    # 趋势分
    trend = 0
    if price > float(ma20.iloc[-1]): trend += 1
    if float(ma20.iloc[-1]) > float(ma50.iloc[-1]): trend += 1
    
    # 价格位置
    high20 = float(close[-20:].max())
    near_high = (price / high20) > 0.95 if high20 > 0 else False
    
    # VCP 检测（保留但不再用于评分）
    vol10 = float(close.pct_change().rolling(10).std().iloc[-1])
    vol30_series = close.pct_change().rolling(30).std()
    vol30 = float(vol30_series.iloc[-20:].mean()) if len(vol30_series) >= 20 else vol10
    vcp = vol10 < vol30 * 0.8
    
    # 压缩度 (Contraction) - 保留但不用于评分
    vol5 = close.pct_change().rolling(5).std().iloc[-1]
    vol20 = close.pct_change().rolling(20).std().iloc[-1]
    if vol20 > 0 and not np.isnan(vol5) and not np.isnan(vol20):
        contraction = 1 - (vol5 / vol20)
        contraction = max(0, min(1, contraction))
    else:
        contraction = 0
    
    # 突破临界 (Breakout Proximity)
    high20_price = close.rolling(20).max().iloc[-1]
    breakout_proximity = price / high20_price if high20_price > 0 else 0
    breakout_proximity = max(0, min(1, breakout_proximity))
    
    # 趋势确认 (价格 > 50MA)
    ma50_price = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else 0
    trend_confirmed = price > ma50_price
    
    # 波动惩罚因子
    vol_penalty = min(volatility / 50, 1.5)
    
    # 趋势斜率
    if len(ma20) >= 10 and ma20.iloc[-10] > 0:
        slope = (ma20.iloc[-1] - ma20.iloc[-10]) / ma20.iloc[-10]
    else:
        slope = 0
    
    # 曲率检测
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
        "TrendConfirmed": trend_confirmed,
        "VolPenalty": round(vol_penalty, 2),
        "Slope": round(slope, 4),
        "Curvature": round(curvature, 2)
    }

# =========================================
# 评分函数（v8.1 - 聚焦初期launch）
# =========================================
def calc_score(m):
    score = 0
    signals = []
    
    # 1. 趋势分
    if m["TrendScore"] >= 2:
        score += 3
        signals.append("T")
    elif m["TrendScore"] == 1:
        score += 1
    
    # 2. 突破临界（核心）
    if m["BreakoutProx"] > 0.98:
        score += 4
        signals.append("P")
    elif m["BreakoutProx"] > 0.95:
        score += 2
        signals.append(f"p{m['BreakoutProx']:.2f}")
    elif m["BreakoutProx"] > 1.02:
        score -= 2
        signals.append("LATE")
    
    # 3. 趋势斜率（正斜率加分）
    if m["Slope"] > 0.03:
        score += 3
        signals.append(f"S{m['Slope']*100:.1f}")
    elif m["Slope"] > 0.01:
        score += 1
    elif m["Slope"] < 0:
        score -= 2
        signals.append("FLAT")
    
    # 4. 动量分（10日涨幅）- 核心
    mom = m["Mom10d"]
    if 5 <= mom < 15:
        score += 2
        signals.append(f"M{mom:.0f}")
    elif 15 <= mom < 25:
        score += 3
        signals.append(f"S{mom:.0f}")
    elif mom >= 25:
        score += 4
        signals.append(f"BOOM{mom:.0f}")
    elif mom < -5:
        score -= 2
        signals.append(f"W{mom:.0f}")
    
    # 5. 曲率检测（非线性加速）
    if 1.2 < m["Curvature"] < 3.0:
        score += 3
        signals.append("HOOK")
    
    # 6. 量比（放量加分）
    vr = m["VolumeRatio5"]
    if 1.2 <= vr < 2.0:
        score += 2
        signals.append(f"V{vr:.1f}")
    elif vr >= 2.0:
        score += 3
        signals.append(f"VV{vr:.1f}")
    
    # 7. 波动率（低波动加分）
    if m["Volatility"] < 40:
        score += 1
        signals.append(f"L{m['Volatility']:.0f}")
    elif m["Volatility"] > 80:
        score -= 1
    
    # 8. RS Rating
    if m["RS"] >= 70:
        score += 2
        signals.append(f"R{m['RS']:.0f}")
    
    return max(0, round(score, 1)), ",".join(signals) if signals else "-"

def stage(score):
    if score >= 12: return "[FIRE]"
    elif score >= 8: return "[ROCKET]"
    elif score >= 5: return "[SEED]"
    elif score >= 3: return "[CHART]"
    return "[WAIT]"

# =========================================
# 动态生成搜索关键词
# =========================================
def get_search_keywords(ticker, score, mom10, accel, vol, vol_ratio, contraction, breakout_prox, rs, slope, curvature):
    """根据股票特征动态生成搜索关键词"""
    
    keywords = [ticker, "stock"]
    
    # 1. 价格位置
    if breakout_prox > 0.98:
        keywords.append("breakout")
        keywords.append("near high")
    elif breakout_prox > 0.95:
        keywords.append("resistance")
    
    # 2. 量能特征
    if vol_ratio > 1.5:
        keywords.append("high volume")
    elif vol_ratio > 1.2:
        keywords.append("volume spike")
    
    # 3. 动量特征
    if mom10 > 15:
        keywords.append("momentum")
    elif mom10 < -5:
        keywords.append("pullback")
    
    # 4. 波动特征
    if vol < 30:
        keywords.append("low volatility")
    
    # 5. 相对强度
    if rs > 80:
        keywords.append("RS strong")
    
    # 6. 趋势斜率
    if slope > 0.03:
        keywords.append("uptrend")
    
    # 7. 曲率（钩子形态）
    if 1.2 < curvature < 3.0:
        keywords.append("inflection")
        keywords.append("hook")
    
    # 8. 评分等级
    if score >= 12:
        keywords.append("top pick")
    elif score >= 8:
        keywords.append("watchlist")
    
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

# =========================================
# 步骤4：分析函数（v8.1 - 极简过滤）
# =========================================
def analyze(ticker, spy_ret):
    df = safe_yf_download(ticker, period="6mo")
    if df is None or len(df) < 100:
        return None
    
    m = calc_metrics(df, spy_ret)
    
    # 只过滤极端量比
    vr = m["VolumeRatio5"]
    if vr < 0.3 or vr > 10:
        return None
    
    score, sig = calc_score(m)
    
    # 动态生成搜索关键词
    search = get_search_keywords(
        ticker, score, m["Mom10d"], m["Accel"], 
        m["Volatility"], m["VolumeRatio5"], 
        m["Contraction"], m["BreakoutProx"], m["RS"],
        m["Slope"], m["Curvature"]
    )
    
    return {
        "Ticker": ticker,
        "Score": score,
        "Stage": stage(score),
        "Signals": sig,
        "Mom10": round(m["Mom10d"], 1),
        "Accel": round(m["Accel"], 1),
        "Vol": round(m["Volatility"], 1),
        "VCP": "V" if m["VCP"] else "",
        "RS": m["RS"],
        "Price": round(m["Price"], 2),
        "VolRatio5": round(vr, 2),
        "BreakoutProx": m["BreakoutProx"],
        "Slope": m["Slope"],
        "Curvature": m["Curvature"],
        "TrendScore": m["TrendScore"],
        "SearchQuery": search["query"],
        "GoogleURL": search["google"],
        "RedditURL": search["reddit"],
        "TwitterURL": search["twitter"]
    }

def quality_filter(df):
    if df.empty:
        return df
    return df[
        (df["Accel"] > -15) & 
        (df["Vol"] < 100) & 
        (df["Mom10"] < 50)
    ]

def print_recommendations(high_score):
    print("\n[INSIGHT] 重点推荐解读:")
    print("-" * 60)
    
    for _, r in high_score.head(5).iterrows():
        signals = r["Signals"]
        ticker = r["Ticker"]
        price = r["Price"]
        score = r["Score"]
        
        good, warn = [], []
        if "P" in signals or "p" in signals: good.append("接近突破")
        if "BOOM" in signals: good.append("强势动量")
        if "HOOK" in signals: good.append("钩子形态(加速)")
        if "V" in signals or "VV" in signals: good.append("放量")
        if "L" in signals: good.append("低波动")
        if "T" in signals: good.append("趋势确认")
        if "R" in signals: good.append("RS强势")
        
        if "LATE" in signals: warn.append("突破过晚")
        if "FLAT" in signals: warn.append("趋势平缓")
        if "W" in signals: warn.append("负动量")
        
        print(f"\n{ticker} (${price:.2f}) - Score: {score}")
        print(f"  [GOOD] {', '.join(good) if good else '无明显优势'}")
        if warn:
            print(f"  [WARN] {', '.join(warn)}")
        print(f"  [SEARCH] {r['SearchQuery'][:80]}...")

# =========================================
# 主程序
# =========================================
def main():
    print("\n" + "=" * 50)
    print("   ROCKET LAUNCH DETECTOR v8.1 - EARLY LAUNCH")
    print("   (突破+动量+放量+斜率)")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[CONFIG] 筛选条件:")
    print(f"   MIN_PRICE:        ${MIN_PRICE}")
    print(f"   MIN_VOLUME:       {MIN_VOLUME/1e6:.0f}M")
    print(f"   MIN_DOLLAR_VOL:   ${MIN_DOLLAR_VOLUME/1e9:.0f}B")
    print(f"   ETF黑名单:        {len(ETF_BLACKLIST)} 只")
    print(f"   MAX_WORKERS:      {MAX_WORKERS}")
    
    start = time.time()
    
    # 预筛选
    passed = pre_screen(STOCK_UNIVERSE)
    if not passed:
        print("[ERROR] 无股票通过预筛选")
        return
    
    # 获取 SPY 基准
    spy_ret = get_spy_return()
    print(f"[SPY] 3月回报: {spy_ret:.1f}%")
    
    # Launch 分析
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
    df_filtered = quality_filter(df)
    high_score = df_filtered[df_filtered["Score"] >= 8]
    
    # 初期 LAUNCH 定义（刚起飞）
    if len(df_filtered) > 0:
        early_launch = df_filtered[
            (df_filtered["BreakoutProx"] > 0.95) &
            (df_filtered["Slope"] > 0.01) &
            (df_filtered["VolRatio5"] > 1.0) &
            (df_filtered["Mom10"] > 0)
        ]
    else:
        early_launch = pd.DataFrame()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 120)
    print("[TOP] CANDIDATES (Score >= 8)")
    print("=" * 120)
    
    if high_score.empty:
        print("  [WARN] 无 Score >= 8 的股票")
    else:
        for _, r in high_score.iterrows():
            print(f"{r['Stage']} {r['Ticker']:6} | S:{r['Score']:4} | M:{r['Mom10']:5.1f}% | A:{r['Accel']:5.1f} | "
                  f"V:{r['Vol']:4.1f}% | VR:{r['VolRatio5']:3.1f} | RS:{r['RS']:4.0f} | "
                  f"P:{r['BreakoutProx']:.2f} | SL:{r['Slope']*100:.1f}% | CV:{r['Curvature']:.2f} | ${r['Price']:.2f}")
        
        print_recommendations(high_score)
    
    # 打印初期 LAUNCH
    if not early_launch.empty:
        print("\n" + "=" * 120)
        print("[EARLY LAUNCH] 初期起飞 (突破>0.95 + 斜率>1% + 放量>1.0 + 正动量)")
        print("=" * 120)
        for _, r in early_launch.head(10).iterrows():
            print(f"{r['Stage']} {r['Ticker']:6} | S:{r['Score']:4} | M:{r['Mom10']:5.1f}% | "
                  f"P:{r['BreakoutProx']:.2f} | SL:{r['Slope']*100:.1f}% | VR:{r['VolRatio5']:.1f} | "
                  f"CV:{r['Curvature']:.2f} | ${r['Price']:.2f}")
    
    print("\n" + "=" * 120)
    print(f"[STATS] 股票池 {len(STOCK_UNIVERSE)} -> 预筛选 {len(passed)} -> Score>=8 {len(high_score)} | 耗时 {elapsed:.1f}s")
    print(f"   EARLY LAUNCH: {len(early_launch)} 只")
    print("=" * 120)
    
    if len(df_filtered) > 0:
        print("\n[LIST] 完整排名 (Top 20):")
        df_display = df_filtered[["Ticker", "Score", "Stage", "Mom10", "Accel", "Vol", "VolRatio5", 
                                   "RS", "BreakoutProx", "Slope", "Curvature", "Price"]]
        print(df_display.head(20).to_string(index=False))
    
    # ========== 保存结果到 CSV ==========
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    if len(df_filtered) > 0:
        full_filename = f"rocket_results_full_{date_str}.csv"
        df_filtered.to_csv(full_filename, index=False, encoding='utf-8-sig')
        print(f"\n[SAVE] 完整数据已保存至: {full_filename}")
        
        if len(high_score) > 0:
            filename = f"rocket_results_{date_str}.csv"
            df_to_save = high_score[[
                "Ticker", "Score", "Stage", "Mom10", "Accel", "Vol", 
                "VolRatio5", "RS", "BreakoutProx", "Slope", "Curvature", "Price",
                "SearchQuery", "GoogleURL", "RedditURL", "TwitterURL"
            ]]
            df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"[SAVE] 高分结果已保存至: {filename}")
        
        if not early_launch.empty:
            el_filename = f"early_launch_{date_str}.csv"
            early_launch.to_csv(el_filename, index=False, encoding='utf-8-sig')
            print(f"[SAVE] 初期Launch已保存至: {el_filename}")
    else:
        print("[WARN] 无任何数据可保存")
    # ====================================

if __name__ == "__main__":
    main()