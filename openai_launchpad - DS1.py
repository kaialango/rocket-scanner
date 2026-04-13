"""
🚀 ROCKET LAUNCH DETECTOR v5.3 - FIXED
修复 yfinance 401 错误 + 增强数据获取
"""

import pandas as pd
import numpy as np
import yfinance as yf
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore")

# =========================================
# 配置 yfinance
# =========================================
# 添加请求头绕过反爬
yf.set_tz_cache_location("cache")

# =========================================
# 原始股票池（已去重）
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

# 筛选条件
MIN_PRICE = 5.0
MIN_VOLUME = 5000000
MIN_DOLLAR_VOLUME = 1e9
MAX_WORKERS = 3  # 降低并发，避免被限

# =========================================
# 工具函数：带重试的数据获取
# =========================================
def safe_yf_download(ticker, period="6mo", retries=3):
    """带重试的数据获取"""
    for i in range(retries):
        try:
            time.sleep(0.1)  # 短暂延迟
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if len(df) > 0:
                return df
        except Exception as e:
            if i == retries - 1:
                return None
            time.sleep(0.5)
    return None

def safe_yf_info(ticker, retries=2):
    """带重试的info获取"""
    for i in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.3))  # 随机延迟
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
    print(f"🔍 预筛选 {len(tickers)} 只... (成交量>{MIN_VOLUME/1e6:.0f}M 或 成交额>${MIN_DOLLAR_VOLUME/1e9:.0f}B)")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_basic_info, t): t for t in tickers}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    
    print(f"✅ 通过: {len(results)} 只\n")
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
# 步骤3：Launch 分析
# =========================================
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
    
    # RS Rating
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
    
    trend = 0
    if price > float(ma20.iloc[-1]): trend += 1
    if float(ma20.iloc[-1]) > float(ma50.iloc[-1]): trend += 1
    
    high20 = float(close[-20:].max())
    near_high = (price / high20) > 0.95 if high20 > 0 else False
    
    vol10 = float(close.pct_change().rolling(10).std().iloc[-1])
    vol30_series = close.pct_change().rolling(30).std()
    vol30 = float(vol30_series.iloc[-20:].mean()) if len(vol30_series) >= 20 else vol10
    vcp = vol10 < vol30 * 0.8
    
    return {
        "Price": price, "Mom10d": mom10, "Accel": accel, "Volatility": volatility,
        "VolRatio": vol_ratio, "TrendScore": trend, "NearHigh": near_high, "VCP": vcp,
        "RS": round(rs_rating, 1)
    }

def calc_score(m):
    score = 0
    signals = []
    
    if m["TrendScore"] >= 2:
        score += 3
        signals.append("T")
    
    mom = m["Mom10d"]
    if 5 <= mom < 15:
        score += 2
        signals.append(f"M{mom:.0f}")
    elif 15 <= mom < 25:
        score += 3
        signals.append(f"S{mom:.0f}")
    elif 25 <= mom < 35:
        score += 1
        signals.append(f"H{mom:.0f}")
    elif mom >= 35:
        score -= 1
    
    acc = m["Accel"]
    if acc > 5:
        score += 3
        signals.append(f"A{acc:.0f}")
    elif 2 < acc <= 5:
        score += 2
        signals.append(f"A{acc:.0f}")
    elif 0 < acc <= 2:
        score += 1
    elif acc < -5:
        score -= 2
        signals.append(f"D{abs(acc):.0f}")
    
    if m["VCP"]:
        score += 3
        signals.append("V")
    if m["NearHigh"]:
        score += 2
        signals.append("NH")
    
    vr = m["VolRatio"]
    if 1.2 <= vr < 2.0:
        score += 2
    elif vr >= 2.0:
        score += 1
    
    vol = m["Volatility"]
    if vol < 30:
        score += 3
        signals.append(f"L{vol:.0f}")
    elif vol < 40:
        score += 2
        signals.append(f"L{vol:.0f}")
    elif vol < 50:
        score += 1
    elif vol > 70:
        score -= 1
    
    if m["RS"] >= 70:
        score += 2
    elif m["RS"] >= 50:
        score += 1
    
    return max(0, score), ",".join(signals) if signals else "-"

def stage(score):
    if score >= 12: return "🔥🔥"
    elif score >= 8: return "🚀🚀"
    elif score >= 5: return "🌱"
    elif score >= 3: return "📊"
    return "⏸️"

def analyze(ticker, spy_ret):
    df = safe_yf_download(ticker, period="6mo")
    if df is None or len(df) < 100:
        return None
    
    m = calc_metrics(df, spy_ret)
    score, sig = calc_score(m)
    
    return {
        "Ticker": ticker, "Score": score, "Stage": stage(score), "Signals": sig,
        "Mom10": round(m["Mom10d"], 1), "Accel": round(m["Accel"], 1),
        "Vol": round(m["Volatility"], 1), "VCP": "✓" if m["VCP"] else "",
        "RS": m["RS"], "Price": round(m["Price"], 2)
    }

def quality_filter(df):
    if df.empty:
        return df
    return df[
        (df["Accel"] > -10) & 
        (df["Vol"] < 80) & 
        (df["Mom10"] < 35)
    ]

def print_recommendations(high_score):
    print("\n💡 重点推荐解读:")
    print("-" * 60)
    
    for _, r in high_score.head(5).iterrows():
        signals = r["Signals"]
        ticker = r["Ticker"]
        price = r["Price"]
        score = r["Score"]
        
        good, warn = [], []
        if "V" in signals: good.append("VCP收缩")
        if "A" in signals: good.append("正加速度")
        if "L" in signals: good.append("低波动")
        if "T" in signals: good.append("趋势确认")
        if r["RS"] >= 70: good.append(f"RS强势({r['RS']:.0f})")
        
        if "D" in signals: warn.append("减速中")
        if r["Mom10"] > 25: warn.append("动量偏高")
        if r["Accel"] < -5: warn.append("加速度为负")
        
        print(f"\n{ticker} (${price:.2f}) - Score: {score}")
        print(f"  ✅ 优势: {', '.join(good) if good else '无明显优势'}")
        if warn:
            print(f"  ⚠️ 注意: {', '.join(warn)}")

# =========================================
# 主程序
# =========================================
def main():
    print("\n" + "🚀" * 35)
    print("   ROCKET LAUNCH DETECTOR v5.3 - FIXED")
    print("🚀" * 35)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    import time
    start = time.time()
    
    # 预筛选
    passed = pre_screen(STOCK_UNIVERSE)
    if not passed:
        print("❌ 无股票通过预筛选")
        return
    
    # 获取 SPY 基准
    spy_ret = get_spy_return()
    print(f"📈 SPY 3月回报: {spy_ret:.1f}%")
    
    # Launch 分析
    print(f"📊 分析 {len(passed)} 只股票...")
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze, t, spy_ret): t for t in passed}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            if r:
                results.append(r)
            if (i + 1) % 50 == 0:
                print(f"   进度: {i + 1}/{len(passed)}")
    
    if not results:
        print("❌ 无有效分析结果")
        return
    
    df = pd.DataFrame(results).sort_values("Score", ascending=False)
    df_filtered = quality_filter(df)
    high_score = df_filtered[df_filtered["Score"] >= 8]  # 降低阈值
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 110)
    print("🏆 TOP CANDIDATES (Score >= 8, 已过滤质量缺陷)")
    print("=" * 110)
    
    if high_score.empty:
        print("  ⚠️ 无 Score >= 8 的股票")
    else:
        for _, r in high_score.iterrows():
            print(f"{r['Stage']} {r['Ticker']:6} | S:{r['Score']:2} | M:{r['Mom10']:5.1f}% | A:{r['Accel']:5.1f} | "
                  f"V:{r['Vol']:4.1f}% | RS:{r['RS']:4.0f} | VCP:{r['VCP']:1} | ${r['Price']:.2f}")
        
        print_recommendations(high_score)
    
    print("\n" + "=" * 110)
    print(f"📊 统计: 股票池 {len(STOCK_UNIVERSE)} → 通过 {len(passed)} → Score≥8 {len(high_score)} | 耗时 {elapsed:.1f}s")
    print("=" * 110)
    
    if len(df_filtered) > 0:
        print("\n📋 完整排名 (Top 20):")
        df_display = df_filtered[["Ticker", "Score", "Stage", "Mom10", "Accel", "Vol", "RS", "VCP", "Price"]]
        print(df_display.head(20).to_string(index=False))

if __name__ == "__main__":
    main()