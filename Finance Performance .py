#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf 


# In[3]:


df= yf.download("AAPL", period="5y", interval="1d", auto_adjust=True, progress=False)
print(df.head())        


# In[4]:


df= yf.download("MSFT", period="5y", interval="1d", auto_adjust=True, progress=False)
print(df.head())


# In[5]:


df= yf.download("GOOG", period="5y", interval="1d", auto_adjust=True, progress=False)
print(df.head())


# In[7]:


df = yf.download("AAPL",start="2020-01-01", end="2025-08-13",interval="1d", auto_adjust=True, progress=False)
df.to_csv("AAPL_2020_to_today_daily.csv")


# In[8]:


df = yf.download("MSFT",
                 start="2020-01-01", end="2025-08-13",
                 interval="1d", auto_adjust=True, progress=False)
df.to_csv("MSFT_2020_to_today_daily.csv")


# In[9]:


df = yf.download("GOOG",
                 start="2020-01-01", end="2025-08-13",
                 interval="1d", auto_adjust=True, progress=False)
df.to_csv("GOOG_2020_to_today_daily.csv")


# In[10]:


import pandas as pd


# In[11]:


#define tickers
tickers= ['AAPL','MSFT','GOOG']


# In[12]:


#Pull financials
data={}
for ticker in tickers:
    stock= yf.Ticker(ticker)
    financials=stock.financials.T


# In[13]:


#Income statement
balance_sheet=stock.balance_sheet.T
cash_flow=stock.cashflow.T
data[ticker]={'financials': financials, 'balance_sheet':balance_sheet, 'cash_flow':cash_flow}
print(f"downloaded data for {ticker}")


# In[14]:


df_all= yf.download(
    tickers,
    period="5y",            # or: start="2020-01-01", end="2025-08-13"
    interval="1d",
    auto_adjust=True,
    progress=False,
    group_by="ticker"       # makes columns per ticker
)


# In[15]:


df_all.to_csv("prices_5y_daily_APPL_MSFT_GOOG.csv")


# In[16]:


for t in tickers:
    df_all[t].to_csv(f"{t}_5y_daily.csv")
    print(t, df_all[t].shape)


# In[17]:


df_all = yf.download(
    ["AAPL","MSFT","GOOG"],
    start="2020-01-01", end="2025-08-13",
    interval="1d", auto_adjust=True, progress=False, group_by="ticker"
)


# In[18]:


print(df_all.columns.levels)


# In[22]:


from pathlib import Path


# In[26]:


import os


# In[29]:


import os
import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOG"]   # add more if needed
os.makedirs("data", exist_ok=True)   # create once

data = {}

# Build the dict correctly (key by the SYMBOL STRING)
for t in tickers:
    tk = yf.Ticker(t)
    data[t] = {
        "financials": tk.financials.T,
        "balance_sheet": tk.balance_sheet.T,
        "cashflow": tk.cashflow.T,   # <-- NOT 'cash_flow'
    }

# Save everything that exists
for t, items in data.items():
    for name, df in items.items():
        if df is not None and not df.empty:
            df.to_csv(os.path.join("data", f"{t}_{name}.csv"))
        else:
            print(f"[WARN] {t} {name} empty — skipped")


# In[15]:


print(df.shape)        # rows, cols
print(df.index[:5])    # first dates
print(df.columns)      # columns
display(df.head())     # first 5
display(df.tail())     # last 5
display(df.sample(4)) # random 10
df.info()              # dtypes + non-null counts
df.describe()          # numeric summary


# In[16]:


import os
BASE = "Financial Performance"                                # change if you want
os.makedirs(BASE, exist_ok=True)
df.to_csv(os.path.join(BASE, "AAPL_5y_daily.csv"))


# In[22]:


import glob, os
print("Data folder:", os.path.abspath("Finance Performance"))
print("Files:", glob.glob("Finance Performance/*.csv"))


# In[8]:


import glob, os
print("Drives root listing:", glob.glob(r"D:\*"))   # confirm D: is real
print("Exists C src?", os.path.exists(r"C:\Users\visha\Finance Performance"))
print("Exists D dest?", os.path.exists(r"D:\Finance Performance"))


# In[9]:


import os, shutil

d_path = r"D:\Financial Performance"   # use the exact name shown in your listing
print("Exists on D:", os.path.exists(d_path))
print("Sample contents:", os.listdir(d_path)[:10])


# In[10]:


import os, glob
import pandas as pd
import numpy as np

# ====== 0) Set base folders ======
BASE = r"D:\Financial Performance"             # <-- change if needed
CLEAN = os.path.join(BASE, "clean")
os.makedirs(CLEAN, exist_ok=True)
assert os.path.isdir(BASE), f"Base folder not found: {BASE}"

# ====== helpers ======
def _coerce_numeric(col: pd.Series) -> pd.Series:
    s = col.astype(str)
    is_pct = s.str.contains("%", na=False).any()
    s = s.str.replace(",", "", regex=False)         .str.replace(r"[$€£₹]", "", regex=True)         .str.replace(r"[^\d\.\-eE+]", "", regex=True)
    out = pd.to_numeric(s, errors="coerce")
    return out / 100 if is_pct else out

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    # Date handling
    if "Date" in df.columns: df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # numerics
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # choose price column
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None: raise ValueError("No Adj Close/Close in prices")
    # returns
    df["ret_d"] = df[price_col].pct_change()
    # month-end return aligned to month end
    m_end = df[price_col].resample("M").last()
    df["ret_m"] = m_end.reindex(df.index).ffill().pct_change()
    return df

def clean_financials_df(df: pd.DataFrame) -> pd.DataFrame:
    # index are periods (after your earlier .T), try making them datetimes
    if df.index.name is None or df.index.name.lower() != "period":
        df.index.name = "Period"
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass
    df = df.sort_index()
    # coerce numerics
    for c in df.columns:
        df[c] = _coerce_numeric(df[c]) if df[c].dtype == object else pd.to_numeric(df[c], errors="coerce")
    # standard fields (only keep those present)
    rename = {
        "Total Revenue": "revenue",
        "Net Income": "net_income",
        "Gross Profit": "gross_profit",
        "Operating Income": "oper_income",
    }
    present = {k:v for k,v in rename.items() if k in df.columns}
    if present:
        df2 = df[list(present.keys())].rename(columns=present).copy()
    else:
        df2 = df.copy()
    # metrics
    if "revenue" in df2: df2["revenue_yoy"] = df2["revenue"].pct_change()
    if "net_income" in df2 and "revenue" in df2: df2["net_margin"] = df2["net_income"] / df2["revenue"]
    if "gross_profit" in df2 and "revenue" in df2: df2["gross_margin"] = df2["gross_profit"] / df2["revenue"]
    if "oper_income" in df2 and "revenue" in df2: df2["oper_margin"] = df2["oper_income"] / df2["revenue"]
    return df2

written = []

# ====== 1) Clean PRICE CSVs ======
price_files = glob.glob(os.path.join(BASE, "**", "*_daily.csv"), recursive=True)
for p in price_files:
    try:
        df = pd.read_csv(p, parse_dates=["Date"]) if "Date" in pd.read_csv(p, nrows=0).columns else pd.read_csv(p)
        df_clean = clean_price_df(df)
        out = os.path.join(CLEAN, os.path.splitext(os.path.basename(p))[0] + "_clean.csv")
        df_clean.to_csv(out)
        written.append(out)
    except Exception as e:
        print(f"[PRICE][SKIP] {p}: {e}")

# ====== 2) Clean FINANCIALS CSVs ======
fin_patterns = ["*_financials.csv", "*_balance_sheet.csv", "*_cashflow.csv", "*_cash_flow.csv"]
fin_files = []
for pat in fin_patterns:
    fin_files += glob.glob(os.path.join(BASE, "**", pat), recursive=True)

for p in sorted(set(fin_files)):
    try:
        df = pd.read_csv(p, index_col=0)
        df_clean = clean_financials_df(df)
        out = os.path.join(CLEAN, os.path.splitext(os.path.basename(p))[0] + "_clean.csv")
        df_clean.to_csv(out)
        written.append(out)
    except Exception as e:
        print(f"[FIN][SKIP] {p}: {e}")

print(f"\nWrote {len(written)} cleaned files to:\n{CLEAN}")
for w in written[:10]:
    print(" -", w)
if len(written) > 10:
    print(" ...")


# In[11]:


import pandas as pd
p = r"D:\Financial Performance\AAPL_5y_daily.csv"
hdr = pd.read_csv(p, nrows=5)
print("Columns:", list(hdr.columns))
print(hdr.head(2))


# In[15]:


import os, pandas as pd, numpy as np, yfinance as yf

# ---- configure ----
TICKERS = ["AAPL","MSFT","GOOG"]                       # add more
OUTDIR  = r"D:\Financial Performance\clean"            # where to save
os.makedirs(OUTDIR, exist_ok=True)

# column name fallbacks by metric (Yahoo labels vary)
COLS = {
    "revenue": ["Total Revenue","Revenue"],
    "net_income": ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares"],
    "gross_profit": ["Gross Profit"],
    "total_assets": ["Total Assets"],
    "shareholders_equity": ["Total Stockholder Equity","Total Stockholders Equity","Stockholders Equity","Total Equity"],
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "Net Cash Provided By Operating Activities",
    ],
}

def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _pick(series_frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in series_frame.columns:
            return series_frame[c]
    return pd.Series(index=series_frame.index, dtype="float64")
from typing import Optional
from IPython.display import display  # so display(...) works in Jupyter

def _latest_shares(tkr) -> Optional[float]:
    """Best-effort shares outstanding (last known)."""
    # try detailed history
    try:
        s = tkr.get_shares_full()
        if s is not None and not s.empty:
            return float(s.iloc[-1])
    except Exception:
        pass
    # fallback: fast_info
    try:
        val = getattr(tkr, "fast_info", {}).get("shares", None)
        return float(val) if val is not None else None
    except Exception:
        return None

rows = []
for t in TICKERS:
    tk = yf.Ticker(t)

    fin = tk.financials.T if tk.financials is not None else pd.DataFrame()
    bs  = tk.balance_sheet.T if tk.balance_sheet is not None else pd.DataFrame()
    cf  = tk.cashflow.T if tk.cashflow is not None else pd.DataFrame()

    # index = period, ensure datetime + sorted
    for df in (fin, bs, cf):
        if not df.empty:
            df.index = pd.to_datetime(df.index, errors="coerce")
    fin = _coerce_numeric_frame(fin.sort_index())
    bs  = _coerce_numeric_frame(bs.sort_index())
    cf  = _coerce_numeric_frame(cf.sort_index())

    # align on union of periods
    idx = fin.index.union(bs.index).union(cf.index)
    fin, bs, cf = fin.reindex(idx), bs.reindex(idx), cf.reindex(idx)

    metrics = pd.DataFrame(index=idx)
    metrics["ticker"] = t
    metrics["revenue"] = _pick(fin, COLS["revenue"])
    metrics["net_income"] = _pick(fin, COLS["net_income"])
    metrics["gross_profit"] = _pick(fin, COLS["gross_profit"])
    metrics["total_assets"] = _pick(bs, COLS["total_assets"])
    metrics["shareholders_equity"] = _pick(bs, COLS["shareholders_equity"])
    metrics["operating_cash_flow"] = _pick(cf, COLS["operating_cash_flow"])

    # optional: shares outstanding (point-in-time, will repeat across rows)
    shares = _latest_shares(tk)
    metrics["shares_outstanding"] = shares

    # optional: margins/growth
    if "revenue" in metrics:
        metrics["net_margin"] = metrics["net_income"] / metrics["revenue"]
        metrics["gross_margin"] = metrics["gross_profit"] / metrics["revenue"]
        metrics["revenue_yoy"] = metrics["revenue"].pct_change()
        metrics["net_income_yoy"] = metrics["net_income"].pct_change()
        metrics["cfo_yoy"] = metrics["operating_cash_flow"].pct_change()

    rows.append(metrics)

# tidy table
key_metrics = pd.concat(rows).reset_index().rename(columns={"index":"period"})
key_metrics = key_metrics.sort_values(["ticker","period"])

# save + preview
out_csv = os.path.join(OUTDIR, "key_metrics.csv")
key_metrics.to_csv(out_csv, index=False)
print("Wrote:", out_csv)
display(key_metrics.tail(10))


# In[4]:


print(type(df), df.shape)
print(df.columns.tolist())
df.head()


# In[12]:


import yfinance as yf, pandas as pd, numpy as np

def pick(df, *names):
    """Return the first matching column, or an empty Series if none exist."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(index=df.index, dtype="float64")

def metrics_for(ticker: str) -> pd.DataFrame:
    tk  = yf.Ticker(ticker)
    fin = tk.financials.T if tk.financials is not None else pd.DataFrame()     # income stmt
    bs  = tk.balance_sheet.T if tk.balance_sheet is not None else pd.DataFrame()# balance sheet

    # make the index time-like & align periods
    for d in (fin, bs):
        if not d.empty:
            d.index = pd.to_datetime(d.index, errors="coerce")
    idx = fin.index.union(bs.index).sort_values()
    fin, bs = fin.reindex(idx), bs.reindex(idx)

    m = pd.DataFrame(index=idx)
    m["Ticker"] = ticker
    m["Revenue"]              = pick(fin, "Total Revenue", "Revenue")
    m["Net Income"]           = pick(fin, "Net Income", "Net Income Common Stockholders",
                                     "Net Income Applicable To Common Shares",
                                     "Net Income From Continuing Operations")
    m["Gross Profit"]         = pick(fin, "Gross Profit")
    m["Total Assets"]         = pick(bs,  "Total Assets")
    m["Shareholders Equity"]  = pick(bs,  "Total Stockholder Equity", "Total Stockholders Equity",
                                     "Stockholders Equity", "Total Equity")
    m["Total Liabilities"]    = pick(bs,  "Total Liabilities", "Total Liabilities Net Minority Interest")

    # Ratios
    m["Net Profit Margin"] = m["Net Income"] / m["Revenue"]
    avg_eq = (m["Shareholders Equity"] + m["Shareholders Equity"].shift()) / 2
    m["ROE"] = m["Net Income"] / avg_eq.replace(0, np.nan)
    m["Debt to Equity"] = m["Total Liabilities"] / m["Shareholders Equity"].replace(0, np.nan)
    return m

# Example: multiple tickers
tickers = ["AAPL", "MSFT", "GOOG"]
key_metrics = (pd.concat([metrics_for(t) for t in tickers])
               .reset_index().rename(columns={"index":"Period"})
               .sort_values(["Ticker","Period"]))
key_metrics.tail(10)


# In[14]:


#One clear DataFrame for all table tickers  
import yfinance as yf, pandas as pd, numpy as np

def pick(df, *names):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(index=df.index, dtype="float64")

def metrics_for(ticker: str) -> pd.DataFrame:
    tk  = yf.Ticker(ticker)
    fin = tk.financials.T if tk.financials is not None else pd.DataFrame()     # income stmt (annual)
    bs  = tk.balance_sheet.T if tk.balance_sheet is not None else pd.DataFrame()# balance sheet (annual)

    for d in (fin, bs):
        if not d.empty: d.index = pd.to_datetime(d.index, errors="coerce")
    idx = fin.index.union(bs.index).sort_values()
    fin, bs = fin.reindex(idx), bs.reindex(idx)

    m = pd.DataFrame(index=idx)
    m["Ticker"] = ticker
    m["Revenue"]             = pick(fin, "Total Revenue","Revenue")
    m["Net Income"]          = pick(fin, "Net Income","Net Income Common Stockholders",
                                    "Net Income Applicable To Common Shares",
                                    "Net Income From Continuing Operations")
    m["Gross Profit"]        = pick(fin, "Gross Profit")
    m["Total Assets"]        = pick(bs,  "Total Assets")
    m["Shareholders Equity"] = pick(bs,  "Total Stockholder Equity","Total Stockholders Equity",
                                    "Stockholders Equity","Total Equity")
    m["Total Liabilities"]   = pick(bs,  "Total Liabilities","Total Liabilities Net Minority Interest")

    # Ratios
    m["Net Profit Margin"] = m["Net Income"] / m["Revenue"]
    avg_eq = (m["Shareholders Equity"] + m["Shareholders Equity"].shift()) / 2
    m["ROE"] = m["Net Income"] / avg_eq.replace(0, np.nan)
    m["Debt to Equity"] = m["Total Liabilities"] / m["Shareholders Equity"].replace(0, np.nan)
    return m

tickers = ["AAPL","MSFT","GOOG"]
key_metrics = (pd.concat([metrics_for(t) for t in tickers])
               .reset_index()
               .rename(columns={"index":"Period"})
               .sort_values(["Ticker","Period"]))
key_metrics


# In[15]:


key_metrics.to_csv(r"D:\Financial Performance\clean\key_metrics.csv", index=False)


# In[17]:


#Visualizing Trends 
#Line Chart     for revenue and Netincome over time  
#Bar chart      Profit margins 
#Table          All ratios side-by-side for 3 companies 
# run using plot (matplotlib or seaborn)


# In[19]:


# Visualizing Trends (matplotlib only)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Get the data (from variable 'key_metrics' if present; else from CSV)
try:
    km
except NameError:
    csv_path = r"D:\Financial Performance\clean\key_metrics.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("key_metrics not in memory and CSV not found at:\n" + csv_path)
    km = pd.read_csv(csv_path, parse_dates=["Period"])

# Basic hygiene
km["Period"] = pd.to_datetime(km["Period"], errors="coerce")
km = km.dropna(subset=["Period"]).sort_values(["Ticker","Period"])

# Ensure required columns exist (compute if missing)
if "Net Profit Margin" not in km.columns and {"Net Income","Revenue"}.issubset(km.columns):
    km["Net Profit Margin"] = km["Net Income"] / km["Revenue"]
if "ROE" not in km.columns and {"Net Income","Shareholders Equity"}.issubset(km.columns):
    avg_eq = (km["Shareholders Equity"] + km["Shareholders Equity"].shift()) / 2
    km["ROE"] = km["Net Income"] / avg_eq.replace(0, np.nan)
# Debt to Equity only if we have liabilities
if "Debt to Equity" not in km.columns and {"Total Liabilities","Shareholders Equity"}.issubset(km.columns):
    km["Debt to Equity"] = km["Total Liabilities"] / km["Shareholders Equity"].replace(0, np.nan)

# — Line chart: Revenue & Net Income over time (one chart, multiple lines)
fig, ax = plt.subplots()
for t in km["Ticker"].unique():
    kmt = km[km["Ticker"] == t].set_index("Period")
    if "Revenue" in kmt and "Net Income" in kmt:
        ax.plot(kmt.index, kmt["Revenue"].values, label=f"{t} Revenue")
        ax.plot(kmt.index, kmt["Net Income"].values, linestyle="--", label=f"{t} Net Income")
ax.set_title("Revenue & Net Income over Time")
ax.set_xlabel("Period")
ax.set_ylabel("USD")
ax.legend()
plt.show()

# — Bar chart: latest Net Profit Margin by ticker
latest = km.sort_values("Period").groupby("Ticker", as_index=False).tail(1)
if "Net Profit Margin" in latest:
    fig, ax = plt.subplots()
    # If margin is fraction, convert to %
    vals = latest["Net Profit Margin"].values * (100.0 if latest["Net Profit Margin"].max() < 1.0 else 1.0)
    ax.bar(latest["Ticker"], vals)
    ax.set_title("Latest Net Profit Margin")
    ax.set_ylabel("Percent" if vals.max() <= 100 else "Ratio")
    plt.show()

# — Table: all ratios side-by-side for the 3 companies (latest year)
cols = [c for c in ["Net Profit Margin","ROE","Debt to Equity"] if c in latest.columns]
table = latest[["Ticker","Period"] + cols].reset_index(drop=True)
print(table)


# # Insights
# -MSFT Net margin 36.1% > GOOG 28.6% > APPL 24.0%
# 
# -MSFT: Best profitability (36%)with strong ROE ~33% and high quality earnings, balanced sheet.
# 
# -ROE reality check: AAPL 157% is inflated by a tiny equity base from massive buybacks and not economically comparable; On a like-for-like basis, MSFT ~33% ≈ GOOG ~33%.
# 
# -GOOG:Solid margins, ROE ~33%, and lowest levearge (~0.38).

# In[ ]:




