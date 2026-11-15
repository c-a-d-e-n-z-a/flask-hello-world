import base64
import gc
import html
import json
import os
import pickle
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET


#import requests
from curl_cffi import requests
from flask import Flask, render_template, render_template_string, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line
try:
  import talib
except ImportError:
  talib = None





################################################################################################################################################################
#--------------------------------------------------------------------------------
# Telegarm info
bot_token = os.environ.get('BOT_TOKEN')
chat_id = os.environ.get('CHAT_ID')

# Json urls & API endpoints
url_git_json = os.environ.get('URL_GIT_JSON')
url_git_json_review = os.environ.get('URL_GIT_JSON_REVIEW')

token_git_json = os.environ.get('TOKEN_GIT_JSON')
content_git_json = os.environ.get('CONTENT_GIT_JSON')
content_git_json_review = os.environ.get('CONTENT_GIT_JSON_REVIEW')
mdj_url = os.environ.get('MDJ_URL')

# API KEY
gemini_api_key =  os.environ.get('API_KEY')

# Run environment
run_env =  os.environ.get('RUN_ENV')

# Query internal in minutes
query_interval = 3

#--------------------------------------------------------------------------------
# Other default constants
# Example: ['2454.TW', 878, 1200, 595.0, 640.78, 763.72, 849.66, 642.42, 766.95, 852.07, '2454.TW (聯發科) [-4 -0.67%]: 595.00 < 878']
IDX_T = 0  # ticker
IDX_F = 1  # price floor
IDX_C = 2  # price ceiling
IDX_P = 3  # price saved
IDX_10MA = 4  # 10MA today
IDX_20MA = 5  # 20MA today
IDX_60MA = 6  # 60MA today
IDX_200MA = 7  # 200MA today
IDX_10MA_1 = 8  # 10MA yesterday
IDX_20MA_1 = 9  # 20MA yesterday
IDX_60MA_1 = 10  # 60MA yesterday
IDX_200MA_1 = 11  # 200MA yesterday
IDX_STD200 = 12  # 200-day standard deviation
IDX_MSG = 13  # message saved, not check duplicate message, obsolete now

DELTA_U = 0.01618  # delta up
DELTA_D = -0.01618  # delta down
DELTA_A = 0.00809  # delta abs
DELTA_C_U = 0.01618  # delta up for crypto
DELTA_C_D = -0.01618  # delta down for crypto
DELTA_C_A = 0.00809  # delta abs for crypto
DELTA_I_U = 0.00618  # delta up for index
DELTA_I_D = -0.00618  # delta down for index
DELTA_I_A = 0.00382  # delta abs for index

#--------------------------------------------------------------------------------
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
headers = {"User-Agent": user_agent}

YAHOO_CRUMB = "dx7e5yMCafJ"  # stale fallback; fetch_yahoo_crumb() refreshes this at session creation

_yahoo_crumb = None  # dynamically fetched crumb, shared by all Yahoo chart requests

def fetch_yahoo_crumb(session, try_count=3):
  """Fetch a fresh Yahoo crumb (crumbs are session-scoped and rotate).
  Warms up the cookie session then reads the crumb embedded in the page HTML."""
  for _ in range(try_count):
    try:
      r = session.get("https://fc.yahoo.com", headers=headers, timeout=5, verify=False)
      r2 = session.get("https://finance.yahoo.com", headers=headers, timeout=5, verify=False)
      m = re.search(r'"crumb":"([^"]+)"', r2.text)
      if m:
        return m.group(1)
    except Exception as e:
      print(f"Error fetching Yahoo crumb: {e}", file=sys.stdout)
  return None

#--------------------------------------------------------------------------------
p_u = '\U0001F534'  # price mark 🔴
p_d = '\U0001F7E2'  # price mark 🟢
a_u = '\U00002191'  # arrow mark ↑
a_d = '\U00002193'  # arrow mark ↓
l_c_u = '\U0000274C'  # MA cross upwards mark ❌
l_c_d = '\U0000274E'  # MA corss downwards mark ❎


################################################################################################################################################################
def calc_macd_hist_r_fall(close):
  """
  Calculate MACD Hist (R) fall: fast=50, slow=120, signal=30 (matches StockReview report.py)
  Returns 1 if hist[-1] < hist[-2], else 0.
  """
  if not close or len(close) < 150:
    return 0

  try:
    close_arr = np.array(close, dtype=float)
    if talib is not None:
      _, _, hist = talib.MACD(close_arr, fastperiod=50, slowperiod=120, signalperiod=30)
    else:
      # Exact NumPy calculation identical to TA-Lib MACD (standard SMA initialization)
      def _ema(arr, period):
        n = len(arr)
        if n < period:
          return np.full(n, np.nan)
        res = np.full(n, np.nan)
        res[period - 1] = np.mean(arr[:period])
        alpha = 2.0 / (period + 1.0)
        for i in range(period, n):
          res[i] = arr[i] * alpha + res[i - 1] * (1.0 - alpha)
        return res

      fast_ema = _ema(close_arr, 50)
      slow_ema = _ema(close_arr, 120)
      macd = fast_ema - slow_ema

      valid_start = 120 - 1
      sig_valid = _ema(macd[valid_start:], 30)
      sig = np.full_like(macd, np.nan)
      sig[valid_start:] = sig_valid
      hist = macd - sig

    if len(hist) >= 2 and not np.isnan(hist[-1]) and not np.isnan(hist[-2]):
      return 1 if hist[-1] < hist[-2] else 0
    return 0
  except Exception as e:
    print(f"Error calculating MACD Hist (R): {e}", file=sys.stdout)
    return 0




################################################################################################################################################################
def calc_kalman_filter_signal(close, atr_val=None, r=10.0, q1=0.01, q2=0.01):
  """
  Calculate scale-invariant log-price Kalman Filter velocity, zero-crossing, and trend exhaustion:
  Matches StockReview report.py stock_check_critical_points_after_resample()
  """
  if not close or len(close) < 4:
    return None

  try:
    prices = np.asarray(close, dtype=np.float64)
    n = len(prices)
    log_scale = 100.0

    first_valid_idx = -1
    for idx in range(n):
      if not np.isnan(prices[idx]) and prices[idx] > 0:
        first_valid_idx = idx
        break

    if first_valid_idx == -1 or n - first_valid_idx < 4:
      return None

    p11, p12, p21, p22 = 1.0, 0.0, 0.0, 1.0
    curr_x = np.log(prices[first_valid_idx]) * log_scale
    curr_v = 0.0
    v_price_hist = [0.0]

    for i in range(first_valid_idx + 1, n):
      c = prices[i]
      if np.isnan(c) or c <= 0:
        v_price_hist.append(v_price_hist[-1])
        continue

      y_obs = np.log(c) * log_scale
      x_pred = curr_x + curr_v
      v_pred = curr_v
      p11_pred = p11 + p12 + p21 + p22 + q1
      p12_pred = p12 + p22
      p21_pred = p21 + p22
      p22_pred = p22 + q2

      y_res = y_obs - x_pred
      s_kf = max(p11_pred + r, 1e-10)
      k1 = p11_pred / s_kf
      k2 = p21_pred / s_kf

      curr_x = x_pred + k1 * y_res
      curr_v = v_pred + k2 * y_res
      v_price_hist.append(curr_v * (c / log_scale))

      ik1 = 1.0 - k1
      new_p11 = ik1 * ik1 * p11_pred + k1 * k1 * r
      new_p12 = ik1 * (p12_pred - k2 * p11_pred) + k1 * k2 * r
      new_p21 = new_p12
      new_p22 = (k2 * k2 * p11_pred - k2 * p21_pred - k2 * p12_pred +
                 p22_pred + k2 * k2 * r)

      p11 = max(new_p11, 1e-9)
      p12 = new_p12
      p21 = new_p21
      p22 = max(new_p22, 1e-9)

    if len(v_price_hist) < 3:
      return None

    v_today = v_price_hist[-1]
    v_prev = v_price_hist[-2]
    v_prev2 = v_price_hist[-3]
    kf_tags = []

    if not (np.isnan(v_today) or np.isnan(v_prev)):
      if (v_prev < 0.0) and (v_today >= 0.0):
        kf_tags.append('[KF] 多空轉換(空轉多)')
      elif (v_prev >= 0.0) and (v_today < 0.0):
        kf_tags.append('[KF] 多空轉換(多轉空)')

      if atr_val is not None and not np.isnan(atr_val) and atr_val > 0 and not np.isnan(v_prev2):
        acc_today = v_today - v_prev
        acc_prev = v_prev - v_prev2
        norm_v = abs(v_today) / atr_val
        norm_acc = abs(acc_today) / atr_val
        if norm_v >= 0.25 and norm_acc <= 0.04:
          if v_today > 0.0 and acc_prev > 0.0 and acc_today <= 0.0:
            kf_tags.append('[KF] 賣點-趨勢力竭')
          elif v_today < 0.0 and acc_prev < 0.0 and acc_today >= 0.0:
            kf_tags.append('[KF] 買點-趨勢力竭')

    return ' '.join(kf_tags) if kf_tags else None
  except Exception as e:
    print(f"Error calculating Kalman Filter signal: {e}", file=sys.stdout)
    return None




################################################################################################################################################################
def ma_calculation(ticker, session, use_adj=True, return_macd=False, return_indicators=False):

  today = date.today()
  startDate = today - timedelta(days=365)
  endDate = today

  startDate_epoch = int(
      datetime.combine(startDate,
                       datetime.now().time()).timestamp())
  endDate_epoch = int(
      datetime.combine(endDate,
                       datetime.now().time()).timestamp())

  crumb = _yahoo_crumb if _yahoo_crumb else YAHOO_CRUMB
  url_history = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker[0]}?period1={startDate_epoch}&period2={endDate_epoch}&interval=1d&events=history&includeAdjustedClose=true&events=div%2Csplits&crumb={crumb}"
  #print('  url=' + url_history)

  r = session.get(url_history, headers=headers, timeout=10, verify=False)

  if r.status_code == 200:
    r.encoding = 'utf-8'
    json_history = r.json()

    result = json_history.get("chart", {}).get("result")
    if not result:
      print(f'{ticker[0]}: No chart result in response', file=sys.stdout)
      res = [None, None, None, None, None, None, None, None, None, None]
      if return_indicators:
        return (res, 0, None)
      return (res, 0) if return_macd else res

    close_raw = result[0]["indicators"]["quote"][0]["close"]
    if use_adj == True:
      close = result[0]["indicators"]["adjclose"][0]["adjclose"]
      # Fallback to close if adjclose has more Nones
      if None in close and close.count(None) > close_raw.count(None):
        print(f'{ticker[0]}: adjclose has {close.count(None)} None(s), falling back to close ({close_raw.count(None)})', file=sys.stdout)
        close = close_raw
    else:
      close = close_raw

    # Forward-fill None gaps (Yahoo API occasionally returns sparse data)
    none_count = close.count(None)
    if none_count > 0:
      print(f'{ticker[0]}: {none_count} None(s) in price list, forward-filling', file=sys.stdout)
      for k in range(1, len(close)):
        if close[k] is None:
          close[k] = close[k - 1]
      # Remove leading Nones (no previous value to fill from)
      while close and close[0] is None:
        close.pop(0)
      if len(close) == 0:
        res = [None, None, None, None, None, None, None, None, None, None]
        if return_indicators:
          return (res, 0, None)
        return (res, 0) if return_macd else res

    # Guard: no None values but empty/all-invalid data
    if len(close) == 0:
      res = [None, None, None, None, None, None, None, None, None, None]
      if return_indicators:
        return (res, 0, None)
      return (res, 0) if return_macd else res

    n = len(close)
    precision = 4 if close[-1] < 1 else 2

    # Today (gradual: only calculate if enough data)
    ma10  = round(sum(close[-10:]) / 10,   precision) if n >= 10  else None
    ma20  = round(sum(close[-20:]) / 20,   precision) if n >= 20  else None
    ma60  = round(sum(close[-60:]) / 60,   precision) if n >= 60  else None
    ma200 = round(sum(close[-200:]) / 200, precision) if n >= 200 else None

    # 200-day standard deviation
    if n >= 200:
      mean200 = sum(close[-200:]) / 200
      std200 = round((sum((x - mean200) ** 2 for x in close[-200:]) / 200) ** 0.5, precision)
    else:
      std200 = None

    # Yesterday (need one extra data point)
    ma10_1  = round(sum(close[-11:-1]) / 10,   precision) if n >= 11  else None
    ma20_1  = round(sum(close[-21:-1]) / 20,   precision) if n >= 21  else None
    ma60_1  = round(sum(close[-61:-1]) / 60,   precision) if n >= 61  else None
    ma200_1 = round(sum(close[-201:-1]) / 200, precision) if n >= 201 else None

    if n < 200:
      print(f'{ticker[0]}: Only {n} data points, MA200 unavailable', file=sys.stdout)

    # 順便從同一個 chart JSON 提取最近 30 筆 OHLCV 計算輕量化微觀結構靜態參考位 (零額外網路 I/O)
    atr_for_kf = None
    try:
      q_ind = (result[0]["indicators"].get("quote") or [{}])[0]
      h_raw = q_ind.get("high") or []
      l_raw = q_ind.get("low") or []
      c_raw_q = q_ind.get("close") or []
      v_raw = q_ind.get("volume") or [0] * len(c_raw_q)
      bars = [
        (float(h), float(l), float(c), float(v if v is not None else 0))
        for h, l, c, v in zip(h_raw[-30:], l_raw[-30:], c_raw_q[-30:], v_raw[-30:])
        if h is not None and l is not None and c is not None
      ]
      def _build_snap(b):
        if len(b) < 21:
          return None
        prev_h, prev_l, prev_c, _ = b[-1]
        trs = [
          max(b[k][0] - b[k][1], abs(b[k][0] - b[k - 1][2]), abs(b[k][1] - b[k - 1][2]))
          for k in range(-20, 0)
        ]
        atr20 = sum(trs) / 20.0
        rng_prev = prev_h - prev_l
        min_rng_7 = min(x[0] - x[1] for x in b[-7:])
        was_nr7 = (rng_prev <= min_rng_7) and (atr20 > 0 and (rng_prev / atr20) < 0.65)
        was_ib = (prev_h < b[-2][0]) and (prev_l > b[-2][1])
        nr7_h = (b[-2][0] if was_ib else prev_h) if (was_nr7 or was_ib) else None
        nr7_l = (b[-2][1] if was_ib else prev_l) if (was_nr7 or was_ib) else None
        return {
          "prev_h": prev_h,
          "prev_l": prev_l,
          "prev_c": prev_c,
          "atr20": atr20,
          "nr7_ref_h": nr7_h,
          "nr7_ref_l": nr7_l,
          "swing_low_20": min(x[1] for x in b[-20:]),
          "swing_high_20": max(x[0] for x in b[-20:]),
        }
      if len(bars) >= 22:
        snap_norm = _build_snap(bars[:-1])
        snap_pre = _build_snap(bars)
        atr_for_kf = snap_pre["atr20"] if snap_pre else None
        # 口袋支點 (Pocket Pivot): 最新一根收漲、站上 MA60 且成交量大於過去 10 日最大下跌量
        vcp_pivot = False
        if len(bars) >= 12 and ma60 is not None:
          c_last, v_last = bars[-1][2], bars[-1][3]
          c_prev_bar = bars[-2][2]
          if c_last > c_prev_bar and c_last >= ma60 and v_last > 0:
            down_vols = [bars[k][3] for k in range(-11, -1) if bars[k][2] < bars[k - 1][2]]
            max_dv = max(down_vols) if down_vols else 0
            if max_dv > 0 and v_last > max_dv:
              vcp_pivot = True
        q_entry = {"normal": snap_norm, "preopen": snap_pre, "vcp_pivot": vcp_pivot}
        _quant_ref[ticker[0]] = q_entry
        _quant_ref[ticker[0].upper()] = q_entry
    except Exception as e:
      print(f"Error building quant_ref for {ticker[0]}: {e}", file=sys.stdout)

    macd_fall = calc_macd_hist_r_fall(close) if (return_macd or return_indicators) else 0
    kf_sig = calc_kalman_filter_signal(close, atr_val=atr_for_kf) if return_indicators else None
    res = [None, ma10, ma20, ma60, ma200, ma10_1, ma20_1, ma60_1, ma200_1, std200]
    if return_indicators:
      return (res, macd_fall, kf_sig)
    return (res, macd_fall) if return_macd else res

  else:

    print(f'{ticker[0]}:\n  Error in price list: {url_history}',
          file=sys.stdout)
    res = [None, None, None, None, None, None, None, None, None, None]
    if return_indicators:
      return (res, 0, None)
    return (res, 0) if return_macd else res




################################################################################################################################################################
def get_fitx_histock(session):
  # Replaced by WTX& (台指期) via tw.stock Yahoo StockServices.stockList, see tickers_url quote()

  """
  r = session.get('https://histock.tw/stock/module/function.aspx?m=stocktop2017&no=FITX')
  
  quote = '台指期 []'
  
  if r.status_code == 200:
    r.encoding = 'utf-8'
    resp = r.text or ''
    items = resp.split('</span>')

    if len(items) >= 7:
      values = []
      for item in items[0:-1]:
        b = item.rfind('>')
        if b > 0:
          #print(item[b+1:])
          values.append(item[b+1:].strip())

      quote = f'台指期 [{values[1]} {values[2]}]: {values[0]} ({values[6]})'

  print(quote)
  return quote
  """




################################################################################################################################################################
def get_taiex_sentiment(session=None):
  """抓取加權指數、台指期即時價差、櫃買OTC市場寬度、類股資金流向與多空情緒量化模型 (與 YfinanceAnalysis 完全同步；週六日不執行)"""
  if datetime.now(ZoneInfo("Asia/Taipei")).weekday() >= 5:
    return None
  req_sess = session or requests.Session(impersonate="chrome")
  try:
    # 1. 加權指數即時行情 (TSE,1)
    url_tse = f"{mdj_url.rstrip('/')}/Z/js/SKV/TSE,1/SKVgetv1.djxml?x=1&a=TSE,1&b=0&c=0&d=0"
    resp_tse = req_sess.get(url_tse, timeout=5)
    if resp_tse.status_code != 200 or not resp_tse.content:
      print(f"[TAIEX Sentiment] HTTP error: {resp_tse.status_code}")
      return None

    root_tse = ET.fromstring(resp_tse.content)
    data = root_tse.find('Data')
    if data is None:
      print("[TAIEX Sentiment] No Data element in XML")
      return None

    p = float(data.get('P', 0)) / 100.0
    pc = float(data.get('PC', 0)) / 100.0
    chg = p - pc
    chg_pct = (chg / pc) * 100.0 if pc else 0.0
    h = float(data.get('H', 0)) / 100.0
    l = float(data.get('L', 0)) / 100.0
    clv = (p - l) / (h - l) if (h - l) > 0 else 0.5
    day_range_pct = ((h - l) / pc) * 100.0 if pc > 0 and h >= l else 0.0

    up_cnt = int(data.get('UpS', 0))
    down_cnt = int(data.get('DownS', 0))
    upl_cnt = int(data.get('UpLS', 0))
    downl_cnt = int(data.get('DownLS', 0))
    total_active = up_cnt + down_cnt
    up_ratio = (up_cnt / total_active * 100.0) if total_active else 50.0
    ad_ratio = (up_cnt / down_cnt) if down_cnt > 0 else 99.0

    tbs = int(data.get('TBS', 0))
    tas = int(data.get('TAS', 0))
    tbc = int(data.get('TBC', 1))
    tac = int(data.get('TAC', 1))

    ba_ratio = (tbs / tas) if tas > 0 else 1.0
    avg_bid = tbs / tbc if tbc > 0 else 0.0
    avg_ask = tas / tac if tac > 0 else 0.0
    large_order_ratio = (avg_bid / avg_ask) if avg_ask > 0 else 1.0

    # 2. 台指期近月即時行情與相對基差百分比 (TX,3)
    tx_p = 0.0
    tx_chg = 0.0
    tx_chg_pct = 0.0
    tx_vol = 0
    basis = 0.0
    basis_pct = 0.0
    basis_status = "中性價差"
    try:
      url_tx = f"{mdj_url.rstrip('/')}/Z/js/SKV/TX,3/SKVgetv1.djxml?x=1&a=TX,3&b=0&c=0&d=0"
      resp_tx = req_sess.get(url_tx, timeout=3)
      if resp_tx.status_code == 200 and resp_tx.content:
        root_tx = ET.fromstring(resp_tx.content)
        data_tx = root_tx.find('Data')
        if data_tx is not None:
          tx_p = float(data_tx.get('P', 0)) / 100.0
          tx_pc = float(data_tx.get('PC', 0)) / 100.0
          tx_chg = tx_p - tx_pc
          tx_chg_pct = (tx_chg / tx_pc) * 100.0 if tx_pc else 0.0
          tx_vol = int(data_tx.get('V', 0))
          if tx_p > 0 and p > 0:
            basis = round(tx_p - p, 2)
            basis_pct = round((basis / p) * 100.0, 3)
            if basis_pct >= 0.25:
              basis_status = f"大額正價差 (+{basis:.1f}點 / +{basis_pct:.2f}%) 軋空推升"
            elif basis_pct <= -0.30:
              basis_status = f"深度逆價差 ({basis:.1f}點 / {basis_pct:.2f}%) 避險貼水"
            else:
              basis_status = f"{basis:+.1f}點 ({basis_pct:+.2f}%)"
    except Exception:
      pass

    # 3. 櫃買指數即時市場寬度 (OTC,1)
    otc_p = 0.0
    otc_chg = 0.0
    otc_chg_pct = 0.0
    otc_up = 0
    otc_down = 0
    otc_upl = 0
    otc_downl = 0
    otc_ad = 1.0
    data_otc = None
    try:
      url_otc = f"{mdj_url.rstrip('/')}/Z/js/SKV/OTC,1/SKVgetv1.djxml?x=1&a=OTC,1&b=0&c=0&d=0"
      resp_otc = req_sess.get(url_otc, timeout=3)
      if resp_otc.status_code == 200 and resp_otc.content:
        root_otc = ET.fromstring(resp_otc.content)
        data_otc = root_otc.find('Data')
        if data_otc is not None:
          otc_p = float(data_otc.get('P', 0)) / 100.0
          otc_pc = float(data_otc.get('PC', 0)) / 100.0
          otc_chg = otc_p - otc_pc
          otc_chg_pct = (otc_chg / otc_pc) * 100.0 if otc_pc else 0.0
          otc_up = int(data_otc.get('UpS', 0))
          otc_down = int(data_otc.get('DownS', 0))
          otc_upl = int(data_otc.get('UpLS', 0))
          otc_downl = int(data_otc.get('DownLS', 0))
          otc_ad = round(otc_up / otc_down, 2) if otc_down > 0 else 99.0
    except Exception:
      pass

    # 4. 類股即時資金流向 (Sector flows - 免額外套件正則與基礎解析)
    elec_pct = 0.0
    elec_status = "常態輪動"
    top_sectors_str = ""
    try:
      url_sec = f"{mdj_url.rstrip('/')}/z/zba.djhtm?ASPID=sysjust"
      resp_sec = req_sess.get(url_sec, timeout=3)
      if resp_sec.status_code == 200 and resp_sec.content:
        html_sec = resp_sec.content.decode('cp950', errors='ignore')
        cut_idx = html_sec.find('上櫃資金流向表')
        if cut_idx < 0:
          cut_idx = html_sec.find('資金流向明細圖')
        tse_html = html_sec[:cut_idx] if cut_idx > 0 else html_sec
        cells = [re.sub(r'<[^>]+>', '', m).strip() for m in re.findall(r'<td[^>]*>(.*?)</td>', tse_html, re.S | re.I)]
        cells = [c for c in cells if c]
        tse_list = []
        for i in range(len(cells) - 1):
          if not cells[i].endswith('%') and cells[i + 1].endswith('%') and len(cells[i]) <= 10:
            try:
              tse_list.append((cells[i], float(cells[i + 1].replace('%', ''))))
            except Exception:
              pass
        tse_list.sort(key=lambda x: x[1], reverse=True)
        elec_pct = next((x[1] for x in tse_list if x[0] == '電子'), 0.0)
        if elec_pct >= 70.0:
          elec_status = f"🔥 電子吸金攻擊盤 ({elec_pct:.1f}%)"
        elif 0 < elec_pct <= 55.0:
          elec_status = f"⚠️ 電子資金外溢 ({elec_pct:.1f}%)"
        else:
          elec_status = f"⚪ 電子常態輪動 ({elec_pct:.1f}%)"
        top_list = [f"{x[0]} {x[1]:.1f}%" for x in tse_list if x[0] != '電子'][:3]
        top_sectors_str = " | ".join(top_list)
    except Exception:
      pass

    # 5. 全市場總寬度 (TSE + OTC)
    total_up = up_cnt + otc_up
    total_down = down_cnt + otc_down
    total_upl = upl_cnt + otc_upl
    total_downl = downl_cnt + otc_downl
    total_ad = round(total_up / total_down, 2) if total_down > 0 else 99.0
    total_up_ratio = round(total_up / (total_up + total_down) * 100.0, 1) if (total_up + total_down) else 50.0

    # 6. 即時背離與動能訊號 (含濾噪緩衝區 Deadband 與台股微觀大單邏輯)
    divergence_flags = []
    if chg_pct >= 0.30 and (total_up_ratio < 42.0 or total_ad <= 0.75):
      divergence_flags.append(f"⚠️ 【拉積盤/指跌背離】加權上漲 {chg_pct:+.2f}% 但全市場上漲僅 {total_up_ratio:.1f}% (AD={total_ad:.2f})，主力拉權值掩護中小型股出貨！")
    elif chg_pct <= -0.20 and (total_up_ratio >= 55.0 or total_ad >= 1.20 or otc_chg_pct >= 0.25):
      divergence_flags.append(f"💎 【壓盤洗籌/指漲背離】加權下跌 {chg_pct:+.2f}% 但全市場上漲達 {total_up_ratio:.1f}% (AD={total_ad:.2f})，市場體質強韌，主力暗中吃貨！")

    if chg_pct >= 0.25 and otc_chg_pct <= -0.30 and otc_ad <= 0.75:
      divergence_flags.append(f"⚠️ 【拉大出小/中小型失血】加權漲 {chg_pct:+.2f}% 但櫃買跌 {otc_chg_pct:+.2f}% (OTC AD={otc_ad:.2f})，資金抽離中小型股！")
    elif otc_chg_pct > (chg_pct + 0.50) and otc_upl >= 15:
      divergence_flags.append(f"🚀 【內資狂熱/中小型衝刺】櫃買動能 ({otc_chg_pct:+.2f}%) 顯著強於大盤，上櫃漲停達 {otc_upl} 檔，內資主力積極主導！")

    if basis_pct >= 0.25:
      divergence_flags.append(f"🔴 【期現大額正價差】台指期大幅溢價 +{basis:.1f} 點 ({basis_pct:+.2f}%)，期貨大戶極度偏多軋空！")
    elif basis_pct <= -0.30:
      divergence_flags.append(f"🟢 【期現深度逆價差】台指期逆價差達 {basis:.1f} 點 ({basis_pct:+.2f}%)，期貨避險賣壓或除息貼水沉重！")

    if chg_pct <= -0.20 and large_order_ratio >= 1.30:
      divergence_flags.append(f"💎 【大戶逆勢掛大單低接】指數回檔但主力大單比高達 {large_order_ratio:.2f} (均買 {avg_bid:.1f}張 vs 均賣 {avg_ask:.1f}張)，下檔大戶承接強勁！")
    elif chg_pct >= 0.30 and large_order_ratio <= 0.75:
      divergence_flags.append(f"⚠️ 【大戶逢高拆單出貨】指數上漲但主力大單比降至 {large_order_ratio:.2f} (均買 {avg_bid:.1f}張 vs 均賣 {avg_ask:.1f}張)，提防大戶倒貨散戶追高！")

    if 0 < elec_pct <= 55.0:
      sec_hint = f" ({top_sectors_str})" if top_sectors_str else ""
      divergence_flags.append(f"⚠️ 【電子資金外溢】電子成交比重降至 {elec_pct:.1f}% (<55%)，資金轉向非電族群{sec_hint}，電子易盤整！")

    if chg_pct > 0.80 and day_range_pct >= 0.80 and clv < 0.35:
      divergence_flags.append(f"🚨 【高檔引爆獲利賣壓】日內振幅達 {day_range_pct:.2f}% 且收近當日最低 (CLV={clv:.2f})，留長上影線！")
    elif chg_pct < -0.80 and day_range_pct >= 0.80 and clv > 0.65:
      divergence_flags.append(f"🔴 【低檔強力承接反彈】日內振幅達 {day_range_pct:.2f}% 且急拉收近當日最高 (CLV={clv:.2f})，留長下影線！")

    # 7. 情緒評分模型 (綜合全市場寬度 40%、漲跌停投機度 25%、大單+委買賣訂單流 20%、日內收盤位置 CLV 15%)
    score_breadth = min(max(total_up_ratio, 0.0), 100.0)
    score_speculation = min(max(50.0 + (total_upl - total_downl) * 1.5, 0.0), 100.0)
    score_orderflow_ba = min(max(50.0 + (ba_ratio - 1.0) * 100.0, 0.0), 100.0)
    score_orderflow_lo = min(max(50.0 + (large_order_ratio - 1.0) * 80.0, 0.0), 100.0)
    score_orderflow = score_orderflow_lo * 0.60 + score_orderflow_ba * 0.40
    score_clv = clv * 100.0

    sentiment_score = round(
      score_breadth * 0.40 +
      score_speculation * 0.25 +
      score_orderflow * 0.20 +
      score_clv * 0.15,
      1
    )

    if sentiment_score >= 80:
      regime = "🔥 極度樂觀 (Extreme Greed)"
      action = "全市場普漲追價熱絡，漲停家數激增。慎防拉高主力獲利了結，嚴禁追高破線股！"
    elif sentiment_score >= 60:
      regime = "🔴 偏多主導 (Bullish / Risk-On)"
      action = "多方佔據優勢，市場廣度健康，個股普遍具備延續力，可順勢操作強勢族群。"
    elif sentiment_score >= 40:
      regime = "⚪ 中性拉鋸 (Neutral / Diverged)"
      action = "多空勢均力敵或結構分化，缺乏明確族群共識，建議縮小部位、高出低進。"
    elif sentiment_score >= 20:
      regime = "🟢 偏空防守 (Bearish / Risk-Off)"
      action = "賣壓沉重，下跌家數佔比高，資金傾向防守避險，嚴格控管曝險降低持股。"
    else:
      regime = "❄️ 極度恐慌 (Extreme Fear)"
      action = "市場出現踩踏拋售與流動性恐慌，跌停家數偏高。短線不宜盲目殺低，準備留意超跌黃金買點！"

    tse_tv = round(float(data.get('TV', 0)) / 10000.0, 2)
    otc_tv = round(float(data_otc.get('TV', 0)) / 10000.0, 2) if data_otc is not None else 0.0
    total_tv = round(tse_tv + otc_tv, 2)

    result = {
      'symbol': '加權指數',
      'price': p,
      'change': round(chg, 2),
      'change_pct': round(chg_pct, 2),
      'turnover_billion': tse_tv,
      'turnover': {
        'tse_billion': tse_tv,
        'otc_billion': otc_tv,
        'total_billion': total_tv,
      },
      'sentiment_score': sentiment_score,
      'regime': regime,
      'action_guidance': action,
      'breadth': {
        'up_count': up_cnt,
        'down_count': down_cnt,
        'up_ratio_pct': round(up_ratio, 1),
        'ad_ratio': round(ad_ratio, 2),
        'limit_up': upl_cnt,
        'limit_down': downl_cnt,
      },
      'futures': {
        'symbol': '台指期近月',
        'price': tx_p,
        'change': round(tx_chg, 2),
        'change_pct': round(tx_chg_pct, 2),
        'volume': tx_vol,
        'basis': basis,
        'basis_pct': basis_pct,
        'basis_status': basis_status,
      },
      'otc': {
        'symbol': '櫃買指數',
        'price': otc_p,
        'change': round(otc_chg, 2),
        'change_pct': round(otc_chg_pct, 2),
        'up_count': otc_up,
        'down_count': otc_down,
        'limit_up': otc_upl,
        'limit_down': otc_downl,
        'ad_ratio': otc_ad,
      },
      'total_breadth': {
        'total_up': total_up,
        'total_down': total_down,
        'total_upl': total_upl,
        'total_downl': total_downl,
        'total_ad': total_ad,
        'total_up_ratio': total_up_ratio,
      },
      'sector_flows': {
        'elec_pct': round(elec_pct, 1),
        'elec_status': elec_status,
        'top_sectors_str': top_sectors_str,
      },
      'order_flow': {
        'bid_volume_lots': tbs,
        'ask_volume_lots': tas,
        'net_bid_lots': tbs - tas,
        'bid_ask_ratio': round(ba_ratio, 2),
        'avg_bid_lot': round(avg_bid, 2),
        'avg_ask_lot': round(avg_ask, 2),
        'large_order_ratio': round(large_order_ratio, 2),
      },
      'up_count': up_cnt,
      'limit_up_count': upl_cnt,
      'down_count': down_cnt,
      'limit_down_count': downl_cnt,
      'bid_volume_lots': tbs,
      'ask_volume_lots': tas,
      'net_bid_lots': tbs - tas,
      'divergence_alerts': divergence_flags,
      'time': f"{data.get('Date')} {data.get('T')[:2]}:{data.get('T')[2:4]}:{data.get('T')[4:]}"
    }

    chg_icon = "🔴 ▲" if chg > 0 else ("🟢 ▼" if chg < 0 else "⚪")
    otc_icon = "🔴 ▲" if otc_chg > 0 else ("🟢 ▼" if otc_chg < 0 else "⚪")
    tg_lines = [
      f"🏛️ <b>【大盤即時情緒與全市場寬度】</b>",
      f"• <b>加權指數</b>: <code>{p:,.2f}</code> {chg_icon} <b>{chg:+,.2f} ({chg_pct:+.2f}%)</b> | 成交 <code>{tse_tv:.0f}億</code>",
    ]
    if otc_p > 0:
      tg_lines.append(f"• <b>櫃買指數</b>: <code>{otc_p:,.2f}</code> {otc_icon} <b>{otc_chg:+,.2f} ({otc_chg_pct:+.2f}%)</b> | AD <code>{otc_ad:.2f}</code>")
    if tx_p > 0:
      tg_lines.append(f"• <b>台指期近月</b>: <code>{tx_p:,.0f}</code> (價差: <b>{basis_status}</b>)")
    tg_lines.extend([
      f"• <b>情緒指數</b>: <b>{sentiment_score}</b> | <b>{regime}</b>",
      f"• <b>全市場寬度</b>: 🔴 漲 <b>{total_up}</b> / 🟢 跌 <b>{total_down}</b> (總騰落比: <code>{total_ad:.2f}</code>)",
      f"• <b>全市場漲跌停</b>: 漲停 <b>{total_upl}</b> (上市{upl_cnt}/上櫃{otc_upl}) | 跌停 <b>{total_downl}</b>",
      f"• <b>主力大單</b>: 均買 <code>{avg_bid:.2f}張</code> vs 均賣 <code>{avg_ask:.2f}張</code> (主力比: <code>{large_order_ratio:.2f}</code>)",
    ])
    if elec_pct > 0:
      sec_tail = f" | 熱門: {top_sectors_str}" if top_sectors_str else ""
      tg_lines.append(f"• <b>類股資金</b>: {elec_status}{sec_tail}")
    if divergence_flags:
      for a in divergence_flags:
        tg_lines.append(f"• <b>警報提示</b>: <b>{a}</b>")
    if action:
      tg_lines.append(f"• <b>策略指引</b>: {action}")
    telegram_msg = "\n".join(tg_lines)

    result['telegram_msg'] = telegram_msg

    print(f"\n[TAIEX Sentiment] {result['symbol']}: {result['price']:,.2f} {chg_icon} ({result['change']:+.2f}, {result['change_pct']:+.2f}%)")
    print(f"  Score: {result['sentiment_score']} | {result['regime']}")
    print(f"  Total Breadth: 🔴 漲 {total_up} / 🟢 跌 {total_down} (AD: {total_ad}) | 漲停: {total_upl} / 跌停: {total_downl}")
    print(f"  OrderFlow: Net Bid {result['order_flow']['net_bid_lots']:,} lots (B/A: {result['order_flow']['bid_ask_ratio']}) | Large Order Ratio: {result['order_flow']['large_order_ratio']}")
    for alert in divergence_flags:
      print(f"  {alert}")

    return result

  except Exception as e:
    print(f"[TAIEX Sentiment] Error fetching market breadth: {e}")
    return None




_fire_timestamp = [30, 450, 810, 1290]  # persists across requests
_macd_w_is_fall = {}  # persists across requests
_kf_signals = {}  # persists across requests
_quant_ref = {}  # persists across requests
_taiex_sentiment = {}  # persists across requests

################################################################################################################################################################
app = Flask(__name__)




################################################################################################################################################################
@app.route('/')
def index():
  return 'index'
 



################################################################################################################################################################
@app.route('/fire/')
def fire():

  global _fire_timestamp, _macd_w_is_fall, _kf_signals, _quant_ref, _yahoo_crumb, _taiex_sentiment

  gc.collect()

  #--------------------------------------------------------------------------------
  portfolio_cnt = 0
  portfolio_reload = 24
  leisure_time = False
  macd_w_is_fall = dict(_macd_w_is_fall)
  kf_signals = dict(_kf_signals)
  quant_ref = dict(_quant_ref)
  taiex_sentiment = dict(_taiex_sentiment)

  timestamp = _fire_timestamp  # use persisted value; updated when JSON loads

  portfolio = [["2454.TW", 820, 1200], ["2330.TW", 1000, 1200]]

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  json_file_path = os.path.join(BASE_DIR, "data.json")
  pcnt_file_path = os.path.join(BASE_DIR, "pcnt.pkl")

  def send_telegram_html(msg, session=None):
    if not bot_token or not chat_id:
      return
    url_tg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
      "chat_id": chat_id,
      "text": msg,
      "parse_mode": "HTML",
      "disable_web_page_preview": False
    }
    try:
      s = session if session else requests
      resp = s.post(url_tg, json=payload, timeout=5)
      if resp.status_code != 200:
        payload["parse_mode"] = None
        s.post(url_tg, json=payload, timeout=5)
    except Exception as e:
      print(f"[TG Error] {e}")

  #--------------------------------------------------------------------------------
  #now = datetime.now()
  #mins = now.hour * 60 + now.minute
  #day = datetime.today().weekday()
  now = datetime.now(ZoneInfo("Asia/Taipei"))
  mins = now.hour * 60 + now.minute
  day = now.weekday()

  if os.path.exists(pcnt_file_path):
    with open(pcnt_file_path, "rb") as f:
      portfolio_cnt = pickle.load(f)
      print(f"Loaded data - counter:\n {portfolio_cnt}")
      if not isinstance(portfolio_cnt, int):
        portfolio_cnt = 0
      if portfolio_cnt > 360:
        portfolio_cnt = 0

  if os.path.exists(json_file_path):
    with open(json_file_path, "r") as f:
      portfolio = json.load(f)
      print(f"Loaded data - portfolio:\n {portfolio}")
      if len(portfolio) == 0:
        portfolio_cnt = 0
        portfolio.clear()

  # Return immediately in sleep duration
  if (mins > (timestamp[0] + query_interval)) and (mins < (timestamp[1] - query_interval)):  # 00:30 (30) ~ 07:30 (450)
    return f'portfolio_cnt = {portfolio_cnt} (sleep)'

  # For leisure hours, reduce report frequency (weekend, 13:30 (810) ~ 21:30 (1290))
  if (day > 4) or ((mins > timestamp[2] + query_interval*3) and (mins < timestamp[3] - query_interval*3)):
    leisure_time = True
    portfolio_reload = 240 // query_interval  # 4H
  else:
    leisure_time = False
    portfolio_reload = 120 // query_interval  # 2H

  # Reset portolio to trigee full report, like @13:33 and @21:27 (TW stock close, and before US open), and always reset @sleep end (7:30 = timestamp[1])
  if (day < 5 and ((mins == timestamp[2] + query_interval) or (mins == timestamp[3] - query_interval))) or (mins == timestamp[1]):
    portfolio_cnt = 0
    portfolio.clear()
    print("\nReset portfolio - 1")

  if portfolio_cnt % portfolio_reload == 0:
    portfolio_cnt = 0

  
  #--------------------------------------------------------------------------------
  timestamp_msg = datetime.strftime(now, '%H:%M:%S')
  msg_toast = []
  msg_toast.append(timestamp_msg + f' ({day+1}) - {run_env} (cnt={portfolio_cnt})')
  print(msg_toast[0])

  #--------------------------------------------------------------------------------
  session = requests.Session(impersonate="chrome")
  
  # Refresh Yahoo crumb every full reload (crumbs are session-scoped and expire)
  if portfolio_cnt == 0:
    _yahoo_crumb = fetch_yahoo_crumb(session)
    if _yahoo_crumb:
      print(f'Fetched Yahoo crumb: {_yahoo_crumb}')
    else:
      print(f'Failed to fetch Yahoo crumb, falling back to hardcoded one', file=sys.stdout)

  # Reload portfolio every 60 runs
  if portfolio_cnt == 0:

    reset_portfolio = False  # default when GitHub JSON fetch fails (avoids UnboundLocalError below)
    r = session.get(url_git_json, headers=headers, timeout=5)
    if r.status_code == 200:
      r.encoding = 'utf-8'
      json_git = r.json()

      # .get() with default to survive older-format portfolio.json
      timestamp = json_git.get("timestamp", [30, 450, 810, 1290])
      _fire_timestamp = timestamp
      print(f'\nLoad timestamp: {timestamp}')

      remote_portfolio = json_git["portfolio"]
      if len(portfolio) != len(remote_portfolio):
        print(f'[POR] LEN: {len(portfolio)} != {len(remote_portfolio)}')
        reset_portfolio = True

      # bounds-checked element-by-element comparison (covers shorter lists)
      if not reset_portfolio:
        for i in range(len(portfolio)):
          for j in range(3):
            if portfolio[i][j] != remote_portfolio[i][j]:
              print(f'[POR] ELM: {portfolio[i][j]} != {remote_portfolio[i][j]}')
              reset_portfolio = True
              break
          if reset_portfolio:
            break

      if reset_portfolio == True:
        portfolio.clear()
        print("\nReset portfolio - 2")
        portfolio = remote_portfolio
        print('--------------------------------------------------------------------------------')
        print(portfolio)
        print('--------------------------------------------------------------------------------')

    macd_w_is_fall = {}
    kf_signals = {}
    for p in portfolio:
      if '&' in p[IDX_T]:          # Futures (e.g. WTX&) have no query1 daily history
        if reset_portfolio == True:
          p.extend([None]*9)
        print(f'{p[IDX_T]}: skip MA calculation (futures)', file=sys.stdout)
        continue
      ma, is_fall, kf_sig = ma_calculation(p, session, return_indicators=True)
      #ma = ema_calculation(p, session)
      macd_w_is_fall[p[IDX_T]] = is_fall
      macd_w_is_fall[p[IDX_T].upper()] = is_fall
      if kf_sig:
        kf_signals[p[IDX_T]] = kf_sig
        kf_signals[p[IDX_T].upper()] = kf_sig
      if reset_portfolio == True:
        p.extend(ma)
      else:
        p[IDX_10MA:IDX_10MA+9] = ma[1:]

      print(p)

    print("\nMACD Hist (W) fall check (calculated)")
    print(macd_w_is_fall)
    print("\nKalman Filter zero-crossing check (calculated)")
    print(kf_signals)

    _macd_w_is_fall = dict(macd_w_is_fall)
    _kf_signals = dict(kf_signals)
    quant_ref = dict(_quant_ref)

    if day < 5:
      taiex_sentiment = get_taiex_sentiment(session)
      if taiex_sentiment:
        _taiex_sentiment = taiex_sentiment
        if taiex_sentiment.get('telegram_msg'):
          msg_toast.append(taiex_sentiment['telegram_msg'])

    #msg_toast.append(get_fitx_histock(session))     # Replaced by WTX& (台指期) in tw.stock Yahoo portfolio


  #--------------------------------------------------------------------------------
  # Start get stock quotes
  #chunk_len = len(portfolio)   # Set chunk length = portfolio length means only 1 package
  chunk_len = 10  # Set chunk length = portfolio length means only 1 package

  sdp_base_tw = sdp_base_us = sdp_base = None

  for c in range(0, len(portfolio), chunk_len):

    # Only can query 3 tickers at a time.
    chunk = portfolio[c:c + chunk_len]
    tickers = [p[IDX_T] for p in chunk]
    # Encode each ticker (& -> %26 for e.g. WTX&, ^ -> %5E); ',' separator stays raw
    tickers_url = ','.join(quote(t, safe='') for t in tickers)

    url = 'https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.stockList;symbols=' + tickers_url
    r = session.get(url, headers=headers, timeout=5)

    if r.status_code == 200:
      r.encoding = 'utf-8'
      yahoo_portfolio = r.json()
      ticker_map = {portfolio[c + i][IDX_T]: c + i for i in range(len(chunk))}

      for x, jp in enumerate(yahoo_portfolio):
        s = jp.get('symbol', '')
        symbol_name = jp.get('symbolName') or s
        sn = str(symbol_name).split(' ')[0] if symbol_name else s

        if 'change' not in jp:
          continue

        sd = jp['change'].get('raw', 0)
        sdp = jp.get('changePercent', '-')

        # Add color symbol as prefix
        if str(sd).startswith('-'):
          sc = p_d
        else:
          sc = p_u

        if s == "^TWII":
          sdp_base_tw = sdp
        if s == "^GSPC":
          sdp_base_us = sdp

        # O(1) Hash map ticker lookup
        if s not in ticker_map:
          continue
        port_idx = ticker_map[s]

        if jp.get('price', {}).get('raw') == '-':
          price = float(jp.get('regularMarketPreviousClose', {}).get('raw', 0))
        else:
          price = float(jp.get('price', {}).get('raw', 0))

        try:
          price_1 = float(jp.get('regularMarketPreviousClose', {}).get('raw', price))
        except (ValueError, TypeError):
          price_1 = price

        try:
          day_open = float(jp.get('regularMarketOpen', {}).get('raw', price))
        except (ValueError, TypeError):
          day_open = price
        try:
          day_high = float(jp.get('regularMarketDayHigh', {}).get('raw', price))
        except (ValueError, TypeError):
          day_high = price
        try:
          day_low = float(jp.get('regularMarketDayLow', {}).get('raw', price))
        except (ValueError, TypeError):
          day_low = price

        sdp_base = sdp_base_tw if ".TW" in s else sdp_base_us
        if sdp_base is None:
          sdp_base = sdp

        # ^TWII & ^GSPC index comparison
        sdp_radio = '?'
        if sdp_base not in (None, '-') and sdp not in (None, '-'):
          try:
            base_val = float(str(sdp_base).strip('%'))
            curr_val = float(str(sdp).strip('%'))
            if base_val != 0:
              sdp_ratio_float = curr_val / base_val
              sdp_radio = f"{sdp_ratio_float:.1f}x"
            else:
              print(f"{s} SDP: {sdp_base} {sdp} {sdp_base_tw} {sdp_base_us}", file=sys.stdout)
          except (ValueError, TypeError, ZeroDivisionError):
            sdp_radio = '?'

        # 200MA diff (z-score)
        if portfolio[port_idx][IDX_200MA] is not None and portfolio[port_idx][IDX_STD200] is not None and portfolio[port_idx][IDX_STD200] != 0:
          z_score = (price - portfolio[port_idx][IDX_200MA]) / portfolio[port_idx][IDX_STD200]
          sdp_radio += f" {z_score:+.1f}σ"

        msg = ''

        s_dot = s.find('.')
        #if s_dot > -1:
        #  s = s[:s_dot]

        precision = 4 if price < 1 else 2
        sd_str = f"{sd:+.{precision}f}" if isinstance(sd, (int, float)) else str(sd)

        safe_s = html.escape(s)
        safe_sn = html.escape(sn)
        safe_sdp_radio = html.escape(sdp_radio)
        base_hdr = f"<b>{sc} {safe_s} ({safe_sn})</b> [<b>{sd_str} {sdp}</b> {safe_sdp_radio}]"

        if portfolio[port_idx][IDX_P] is not None:  # Already has history record
          delta = (price - portfolio[port_idx][IDX_P]) / portfolio[port_idx][IDX_P]

          if delta >= 0:
            sc = sc + a_u
          else:
            sc = sc + a_d
          base_hdr = f"<b>{sc} {safe_s} ({safe_sn})</b> [<b>{sd_str} {sdp}</b> {safe_sdp_radio}]"

          update_flag = False

          # Set criteria
          if s.find("-USD") > -1:  # Crypto delta
            delta_u = DELTA_C_U
            delta_d = DELTA_C_D
            delta_a = DELTA_C_A
          else:
            delta_u = DELTA_U
            delta_d = DELTA_D
            delta_a = DELTA_A

          if s in [
              "^TWII", "^TWOII", "^GSPC", "^NDX", "^RUT", "^N225", "^KS11", "VOO", "ES=F", "NQ=F",
              "QQQ", "000300.SS", "WTX&"
          ]:
            delta_u = DELTA_I_U
            delta_d = DELTA_I_D
            delta_a = DELTA_I_A

          delta_pts = price - portfolio[port_idx][IDX_P]

          # Judge criteria
          if delta > delta_u:
            msg = f"{base_hdr}: <code>{price:.{precision}f}</code> 🔥 <b>{delta_pts:+.{precision}f} +{delta*100:.{precision}f}% ▲</b>"  # Check quick +1.618% price change
            update_flag = True

          if delta < delta_d:
            msg = f"{base_hdr}: <code>{price:.{precision}f}</code> ❄️ <b>{delta_pts:+.{precision}f} {delta*100:.{precision}f}% ▼</b>"  # Check quick -1.618% price change
            update_flag = True

          # Skip small price variation (0.618%)
          if abs(delta) > delta_a:  # Smooth report, only report when variation > 0.618%
            if price < portfolio[port_idx][IDX_F]:  # Check low price
              msg = f"{base_hdr}: <code>{price:.{precision}f}</code> ({delta_pts:+.{precision}f} {delta*100:+.2f}%) ⚠️ <b>&lt; {portfolio[port_idx][IDX_F]}</b>"
              update_flag = True

            if price > portfolio[port_idx][IDX_C]:  # Check high price
              msg = f"{base_hdr}: <code>{price:.{precision}f}</code> ({delta_pts:+.{precision}f} {delta*100:+.2f}%) 🚀 <b>&gt; {portfolio[port_idx][IDX_C]}</b>"
              update_flag = True

          if update_flag == True:
            portfolio[port_idx][IDX_P] = price  # To save curent price

        else:  # 1st time get price and msg
          if price < portfolio[port_idx][IDX_F]:  # Check low price
            msg = f"{base_hdr}: <code>{price:.{precision}f}</code> ⚠️ <b>&lt; {portfolio[port_idx][IDX_F]}</b>"

          elif price > portfolio[port_idx][IDX_C]:  # Check high price
            msg = f"{base_hdr}: <code>{price:.{precision}f}</code> 🚀 <b>&gt; {portfolio[port_idx][IDX_C]}</b>"

          else:
            msg = f"{base_hdr}: <code>{price:.{precision}f}</code>"

          portfolio[port_idx][IDX_P] = price  # To append curent price (new list item)

        if msg != '':
            # Check if SMA cross (today vs. yesterday)
          ma10 = portfolio[port_idx][IDX_10MA]
          ma20 = portfolio[port_idx][IDX_20MA]
          ma60 = portfolio[port_idx][IDX_60MA]
          ma200 = portfolio[port_idx][IDX_200MA]
          ma10_1 = portfolio[port_idx][IDX_10MA_1]
          ma20_1 = portfolio[port_idx][IDX_20MA_1]
          ma60_1 = portfolio[port_idx][IDX_60MA_1]
          ma200_1 = portfolio[port_idx][IDX_200MA_1]

          price_low = portfolio[port_idx][IDX_F]
          price_high = portfolio[port_idx][IDX_C]

          # Floor/Ceilng cross
          if price_low is not None:
            if (price > price_low) and (price_1 <= price_low):
              msg += f' {l_c_u}<b>L={price_low:.{precision}f}</b>'
            if (price < price_low) and (price_1 >= price_low):
              msg += f' {l_c_d}<b>L={price_low:.{precision}f}</b>'
          if price_high is not None:
            if (price > price_high) and (price_1 <= price_high):
              msg += f' {l_c_u}<b>H={price_high:.{precision}f}</b>'
            if (price < price_high) and (price_1 >= price_high):
              msg += f' {l_c_d}<b>H={price_high:.{precision}f}</b>'

          # SMA trend
          if ma10_1 is not None:
            if (price > ma10) and (price_1 <= ma10_1):
              msg += f' {l_c_u}<b>MA10={ma10:.{precision}f}</b>'
            if (price < ma10) and (price_1 >= ma10_1):
              msg += f' {l_c_d}<b>MA10={ma10:.{precision}f}</b>'

              if portfolio[port_idx][IDX_T] in macd_w_is_fall:
                if macd_w_is_fall[portfolio[port_idx][IDX_T]] == True:  # SMA10 cross and weekly MACD is fall
                  msg += ' <b>[MACD Fall]</b>'

          if ma20_1 is not None:
            if (price > ma20) and (price_1 <= ma20_1):
              msg += f' {l_c_u}<b>MA20={ma20:.{precision}f}</b>'
            if (price < ma20) and (price_1 >= ma20_1):
              msg += f' {l_c_d}<b>MA20={ma20:.{precision}f}</b>'
              if ma10 > ma20:
                msg += ' <b>[JUMP Fall]</b>'

          if ma60_1 is not None:
            if (price > ma60) and (price_1 <= ma60_1):
              msg += f' {l_c_u}<b>MA60={ma60:.{precision}f}</b>'
            if (price < ma60) and (price_1 >= ma60_1):
              msg += f' {l_c_d}<b>MA60={ma60:.{precision}f}</b>'

          if ma200_1 is not None:
            if (price > ma200) and (price_1 <= ma200_1):
              msg += f' {l_c_u}<b>MA200={ma200:.{precision}f}</b>'
            if (price < ma200) and (price_1 >= ma200_1):
              msg += f' {l_c_d}<b>MA200={ma200:.{precision}f}</b>'

          # SMA cross
          if (ma60_1 is not None) and (ma10_1 is not None):
            if (ma10 > ma60) and (ma10_1 <= ma60_1):
              msg += f' {l_c_u}<b>MA1060={ma10:.{precision}f},{ma60:.{precision}f}</b>'
            if (ma10 < ma60) and (ma10_1 >= ma60_1):
              msg += f' {l_c_d}<b>MA1060={ma10:.{precision}f},{ma60:.{precision}f}</b>'
          if (ma20_1 is not None) and (ma10_1 is not None):
            if (ma10 > ma20) and (ma10_1 <= ma20_1):
              msg += f' {l_c_u}<b>MA1020={ma10:.{precision}f},{ma20:.{precision}f}</b>'
            if (ma10 < ma20) and (ma10_1 >= ma20_1):
              msg += f' {l_c_d}<b>MA1020={ma10:.{precision}f},{ma20:.{precision}f}</b>'
          if (ma60_1 is not None) and (ma20_1 is not None):
            if (ma20 > ma60) and (ma20_1 <= ma60_1):
              msg += f' {l_c_u}<b>MA2060={ma20:.{precision}f},{ma60:.{precision}f}</b>'
            if (ma20 < ma60) and (ma20_1 >= ma60_1):
              msg += f' {l_c_d}<b>MA2060={ma20:.{precision}f},{ma60:.{precision}f}</b>'

          # Kalman Filter Zero-crossing & Trend Exhaustion
          ticker_kf = portfolio[port_idx][IDX_T]
          if ticker_kf in kf_signals and kf_signals[ticker_kf]:
            msg += f' <b>{kf_signals[ticker_kf]}</b>'

          # 輕量化華爾街微觀結構附掛檢查 (模式 A：僅於 msg 已觸發時附掛，O(1) 運算)
          q_pair = quant_ref.get(ticker_kf)
          if q_pair:
            q_norm = q_pair.get("normal")
            q_pre = q_pair.get("preopen")
            if q_norm and q_pre:
              q = q_pre if abs(q_pre["prev_c"] - price_1) < abs(q_norm["prev_c"] - price_1) else q_norm
            else:
              q = q_norm or q_pre

            if q:
              o_now = day_open
              h_now = max(day_high, price)
              l_now = min(day_low, price)
              rng_now = h_now - l_now
              clv_now = (price - l_now) / rng_now if rng_now > 1e-6 else 0.5
              atr20 = q["atr20"]

              # (1) [NR7-BRK] 昨日 NR7/孕線極度壓縮，今日突破/破底
              if q["nr7_ref_h"] is not None and rng_now > 0:
                if price > q["nr7_ref_h"] and clv_now >= 0.70:
                  msg += ' <b>[↗NR7突破]</b>'
                elif price < q["nr7_ref_l"] and clv_now <= 0.30:
                  msg += ' <b>[↘NR7破底]</b>'

              # (2) [SWEEP-2B] 20日區間流動性掃蕩 (破底翻 / 假突破)
              if l_now < q["swing_low_20"] and price > q["swing_low_20"] and clv_now >= 0.65 and price > o_now:
                msg += ' <b>[↗2B破底翻]</b>'
              elif h_now > q["swing_high_20"] and price < q["swing_high_20"] and clv_now <= 0.35 and price < o_now:
                msg += ' <b>[↘2B假突破]</b>'

              # (3) [GAP-GO] 跳空不補強勢動能
              if atr20 > 0:
                if (o_now - q["prev_h"]) >= 0.3 * atr20 and l_now >= q["prev_h"] and price > o_now and clv_now >= 0.65:
                  msg += ' <b>[↗跳空不補]</b>'
                elif (q["prev_l"] - o_now) >= 0.3 * atr20 and h_now <= q["prev_l"] and price < o_now and clv_now <= 0.35:
                  msg += ' <b>[↘跳空破位]</b>'

              # (4) [ATR-Z] 極端波動率乖離耗竭
              if ma20 is not None and atr20 > 0:
                atr_z = (price - ma20) / atr20
                if atr_z <= -2.5 and price >= o_now and clv_now >= 0.50:
                  msg += f' <b>[↗超跌反彈(Z={atr_z:.1f})]</b>'
                elif atr_z >= 3.0 and price <= o_now and clv_now <= 0.50:
                  msg += f' <b>[↘過熱鈍化(Z={atr_z:.1f})]</b>'

            # (5) [VCP] 口袋支點 (僅在整點/收盤全報表 portfolio_cnt == 0 時附掛)
            if portfolio_cnt == 0 and q_pair.get("vcp_pivot"):
              msg += ' <b>[VCP口袋支點]</b>'

          # Add chart
          if (l_c_u in msg) or (l_c_d in msg):
            ticker = portfolio[port_idx][IDX_T]
            url_chart = ''
            ts = int(time.time())
            if '^' in ticker or '-' in ticker or '=' in ticker or '.S' in ticker or '.HK' in ticker or '&' in ticker:  # Skip index/futures
              pass
            elif '.TW' in ticker:  # TW
              t = ticker[:ticker.index('.')]
              url_chart = f'https://stock.wearn.com/finance_chart.asp?stockid={t}&timeblock=365&sma1=10&sma2=20&sma3=60&volume=1&_t={ts}'
            else:  # US
              t = ticker
              url_chart = f'https://charts2.finviz.com/chart.ashx?t={t}&ta=1&ty=c&p=d&s=l&_t={ts}'  # technical chart

            if url_chart != '':
              msg += f'\n{url_chart}\n'

          msg_toast.append(msg)

    else:
      print(f"\nRead error: {r.status_code}")

    # Avoid server block
    time.sleep(1)

  #--------------------------------------------------------------------------------
  portfolio_cnt += 1

  with open(json_file_path, "w") as f:
    json.dump(portfolio, f, indent=2)

  with open(pcnt_file_path, "wb") as f:
    pickle.dump(portfolio_cnt, f)

  gc.collect()

  #--------------------------------------------------------------------------------
  if len(msg_toast) > 1:

    print('\n--------------------------------------------------------------------------------')
    #print('\n'.join(msg_toast))
    s_len = 5
    for s in range(0, len(msg_toast), s_len):  # Send in compact batches
      msg_segment = msg_toast[s:s + s_len]
      batch_msg = '\n'.join(msg_segment).replace('%0A', '\n')
      send_telegram_html(batch_msg, session=session)

    return ('<br>'.join(msg_toast)).replace('%0A', '<br>')

  else:
    return msg_toast[0]



################################################################################################################################################################
################################################################################################################################################################
#from pyecharts import options as opts
#from pyecharts.charts import Line
#import pandas as pd
#import numpy as np
#from flask import request



################################################################################################################################################################
def get_stock_data(ticker, start_date, end_date, session, crumb=None):

  # Use dynamically fetched crumb if available; fall back to hardcoded
  crumb = _yahoo_crumb if _yahoo_crumb else YAHOO_CRUMB

  start_epoch = int(datetime.combine(start_date, datetime.min.time()).timestamp())
  end_epoch = int(datetime.combine(end_date, datetime.min.time()).timestamp())
  url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=history&includeAdjustedClose=true&events=div%2Csplits&crumb={crumb}"
  
  headers = {'user-agent': 'Mozilla/5.0'}
  r = session.get(url, headers=headers, timeout=5)
  r.raise_for_status()
  data = r.json()
  result = data["chart"]["result"][0]
  quote = result["indicators"]["quote"][0]
  adjclose = result["indicators"]["adjclose"][0]["adjclose"]
  df = pd.DataFrame({
    "Date": pd.to_datetime(result["timestamp"], unit='s'),
    "Open": quote["open"],
    "High": quote["high"],
    "Low": quote["low"],
    "Close": quote["close"],
    "Adj Close": adjclose,
    "Volume": quote["volume"]
  }).set_index("Date")
  df.name = ticker
  return df




################################################################################################################################################################
def calculate_variation(df):
  
  df['Adj Close Var'] = (df['Adj Close'] / df['Adj Close'].iloc[0]) * 100
  return df




################################################################################################################################################################
def align_dataframes(dfs):
  
  min_len = min(len(df) for df in dfs)
  base_index = min(range(len(dfs)), key=lambda i: len(dfs[i]))
  base_dates = dfs[base_index].index
  for i, df in enumerate(dfs):
    if len(df) != min_len:
      dfs[i] = df.reindex(base_dates, method='ffill')
      dfs[i].name = df.name
  return dfs, base_index




################################################################################################################################################################
def compute_beta(df1, df2):
  m = df1['Adj Close'].pct_change().dropna()
  t = df2['Adj Close'].pct_change().dropna()
  min_len = min(len(m), len(t))
  m, t = m[-min_len:], t[-min_len:]
  cov = np.cov(m, t, ddof=1)[0][1]
  var = np.var(m, ddof=1)
  return cov / var if var != 0 else np.nan




################################################################################################################################################################
@app.route('/compare', methods=['GET', 'POST'])
def compare():
  
  if request.method == 'POST':
    tickers = request.form.get('tickers')
    days = int(request.form.get('days', 1800))
  else:
    tickers = request.args.get('tickers')
    days = int(request.args.get('days', 1800))
    
  if not tickers:
    return "Please provide tickers parameter, e.g. ?tickers=AAPL,MSFT", 400
  tickers = [t.strip() for t in tickers.replace(' ', ',').split(',') if t.strip()]
  if len(tickers) < 1:
    return "Please provide at least one ticker.", 400

  today = date.today()
  start_date = today - timedelta(days=days)
  session = requests.Session(impersonate="chrome")
 
  stock_dfs = []
  errors = []
  for ticker in tickers:
    try:
      df = get_stock_data(ticker, start_date, today, session)
      df = calculate_variation(df)
      stock_dfs.append(df)
    except Exception as e:
      errors.append(f"{ticker}: {e}")

  if not stock_dfs:
    return "No data fetched.<br>" + "<br>".join(errors), 500

  stock_dfs, base_idx = align_dataframes(stock_dfs)

  # Calculate beta
  beta_dict = {}
  base_df = stock_dfs[0]
  for i in range(1, len(stock_dfs)):
    beta_value = compute_beta(base_df, stock_dfs[i])
    beta_dict[stock_dfs[i].name] = beta_value

  # Beta string
  beta_str = "\n".join([f"{name} / {base_df.name}: β={beta_value:.2f}" for name, beta_value in beta_dict.items()])

  # Stats string
  stats_string = ""
  for df in stock_dfs:
    stats_string += f'{df.name}: δ={df["Adj Close Var"].iloc[-1] - df["Adj Close Var"].iloc[0]:5.2f}%, σ={df["Adj Close Var"].std():5.2f}%\n'
  
  # Plot
  line = Line(init_opts=opts.InitOpts(page_title=" vs ".join(tickers), height='900px', width='1880px'))
  dates = stock_dfs[base_idx].index.strftime('%Y%m%d').tolist()
  line.add_xaxis(xaxis_data=dates)
  for df in stock_dfs:
    line.add_yaxis(
      series_name=df.name,
      y_axis=df["Adj Close Var"].map('{:.2f}'.format).tolist(),
      is_smooth=False,
      is_symbol_show=False,
      is_hover_animation=False,
      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.9)
    )
  
  line.set_global_opts(
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10)),
    yaxis_opts=opts.AxisOpts(is_scale=False, splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.5))),
    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", textstyle_opts=opts.TextStyleOpts(font_size=12)),
    legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(font_size=12)),
    datazoom_opts=[
      opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0], range_start=0, range_end=100, is_realtime=False),
      opts.DataZoomOpts(is_show=True, xaxis_index=[0], type_="slider", pos_top="98%", range_start=0, range_end=100, is_realtime=False),
    ],
    title_opts=opts.TitleOpts(
      title=stats_string,
      subtitle=beta_str,  # beta in subtitle
      pos_left='10%',
      pos_top='10%',
      title_textstyle_opts=opts.TextStyleOpts(font_size=12),
      subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12)
    ),
    toolbox_opts=opts.ToolboxOpts(is_show=True, feature={"dataZoom": {"yAxisIndex": "none"}, "restore": {}, "saveAsImage": {}}),
  )

  # Return HTML
  return line.render_embed()




################################################################################################################################################################
@app.route('/performance/', methods=['GET'])
def performance_diff():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Stock Compare</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: #f8f9fa;
            }
            .container {
                max-width: 600px;
                margin-top: 80px;
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 2px 16px rgba(0,0,0,0.08);
                padding: 32px 32px 24px 32px;
            }
            .form-label {
                font-weight: 500;
            }
            .btn-primary {
                width: 100%;
                font-size: 1.1rem;
                padding: 10px;
            }
            h2 {
                text-align: center;
                margin-bottom: 32px;
                font-weight: 700;
                color: #2c3e50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Stock Performance Comparison</h2>
            <form action="/compare" method="post">
                <div class="mb-3">
                    <label for="tickers" class="form-label">Tickers (comma separated):</label>
                    <input type="text" class="form-control" id="tickers" name="tickers" value="^GSPC,AAPL" required>
                </div>
                <div class="mb-3">
                    <label for="days" class="form-label">Days:</label>
                    <input type="number" class="form-control" id="days" name="days" value="1800" min="1" required>
                </div>
                <button type="submit" class="btn btn-primary">Compare</button>
            </form>
            <div class="text-center mt-4" style="color:#888;font-size:0.95em;">
                Example: <code>^GSPC,AAPL,MSFT,GOOG</code> &nbsp; | &nbsp; Days: <code>3650</code><br>
                Following tickers can be put as 1st position for beta calculation.<br>
                <code>^GSPC=S&P 500, ^IXIC=NASDAQ, ^DJI=Dow Jones, ^TWII=TAIEX</code>
                
            </div>
        </div>
    </body>
    </html>
    '''




################################################################################################################################################################
################################################################################################################################################################
def get_expirations(ticker):
    stock = yf.Ticker(ticker)
    return stock.options




################################################################################################################################################################
def get_option_chain(ticker, expiration):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
  
    del stock
    gc.collect()
  
    return chain.calls, chain.puts




################################################################################################################################################################
def compute_pain_table(calls, puts):
    """Compute per-strike total option-seller loss for every expiry strike at once.
    Shared by calculate_max_pain and build_chart_option so the loss table is computed only once."""
    strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    call_strikes = calls['strike'].to_numpy()
    call_oi = calls['openInterest'].to_numpy()
    put_strikes = puts['strike'].to_numpy()
    put_oi = puts['openInterest'].to_numpy()

    call_losses = []
    put_losses = []
    for expiry_price in strikes:
        # Call seller loss: ITM strikes below expiry price
        call_mask = call_strikes < expiry_price
        call_loss = float(((expiry_price - call_strikes[call_mask]) * call_oi[call_mask]).sum())
        # Put seller loss: ITM strikes above expiry price
        put_mask = put_strikes > expiry_price
        put_loss = float(((put_strikes[put_mask] - expiry_price) * put_oi[put_mask]).sum())
        call_losses.append(call_loss)
        put_losses.append(put_loss)

    return strikes, call_losses, put_losses


def calculate_max_pain(calls, puts):
    strikes, call_losses, put_losses = compute_pain_table(calls, puts)
    total_losses = [c + p for c, p in zip(call_losses, put_losses)]
    pain = dict(zip(strikes, total_losses))
    return min(pain, key=pain.get) if pain else None




################################################################################################################################################################
def build_chart_option(calls, puts, ticker, max_pain, underlying_price):

    df_calls = calls[['strike', 'openInterest']].dropna()
    df_puts = puts[['strike', 'openInterest']].dropna()

    # 計算選擇權賣方的總損失
    strikes = sorted(set(df_calls['strike']).union(set(df_puts['strike'])))
    strike_labels = [str(s) for s in strikes]

    # Reuse the shared loss table (same computation calculate_max_pain just did)
    _, call_losses, put_losses = compute_pain_table(df_calls, df_puts)

    """
    mark_line = {
        "symbol": ["none", "none"],
        "label": {"formatter": "{b}: {c}", "position": "insideMiddle"},
        "lineStyle": {"type": "dashed"},
        "data": [
            {"xAxis": str(max_pain), "name": "Max Pain", "lineStyle": {"color": "blue"}},
            {"xAxis": str(round(underlying_price, 2)), "name": "Underlying", "lineStyle": {"color": "orange"}}
        ]
    }
    """
    mark_line = {
        "symbol": ["none", "none"],
        "label": {"formatter": "{b}: {c}", "position": "insideMiddle"},
        "lineStyle": {"type": "dashed"},
        "data": []
    }

    # 保證轉成字串
    if max_pain is not None:
        mark_line["data"].append({
            "xAxis": str(max_pain),
            "name": "Max Pain",
            "lineStyle": {"color": "blue"}
        })

    if underlying_price is not None:
        # 找到離 underlying_price 最近的 strike（讓 x 軸可以對得上）
        closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        mark_line["data"].append({
            "xAxis": str(closest_strike),
            "name": "Underlying",
            "lineStyle": {"color": "orange"}
        })

    # chart1：選擇權賣方的總損失
    chart1 = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Call Loss", "Put Loss"]},
        "xAxis": {
            "type": "category",
            "data": strike_labels,
            "name": "履約價",
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Total Loss ($)",
            "min": "dataMin",
            "max": "dataMax"
        },
        "series": [
            {"name": "Call Loss", "type": "bar", "data": call_losses, "itemStyle": {"color": "#d62728"}, "markLine": mark_line},
            {"name": "Put Loss", "type": "bar", "data": put_losses, "itemStyle": {"color": "#2ca02c"}},
        ]
    }

    # chart2：未平倉合約數 (維持不變)
    call_oi = [int(df_calls.set_index('strike').openInterest.get(s, 0)) for s in strikes]
    put_oi = [-int(df_puts.set_index('strike').openInterest.get(s, 0)) for s in strikes]

    chart2 = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Call OI", "Put OI"]},
        "xAxis": {
            "type": "category",
            "data": strike_labels,
            "name": "履約價",
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Open Interest",
            "min": "dataMin",
            "max": "dataMax"
        },
        "series": [
            {"name": "Call OI", "type": "bar", "stack": "x", "data": call_oi, "itemStyle": {"color": "#d62728"}, "markLine": mark_line},
            {"name": "Put OI", "type": "bar", "stack": "x", "data": put_oi, "itemStyle": {"color": "#2ca02c"}},
        ]
    }

    del call_losses, put_losses, call_oi, put_oi
    gc.collect()

    return json.dumps({"chart1": chart1, "chart2": chart2})




################################################################################################################################################################
@app.route('/maxpain/', methods=['GET', 'POST'])
def maxpain():
    ticker = ""
    expirations = []
    selected_exp = ""
    max_pain = None
    chart = None
    error = None
    underlying_price = None

    if request.method == 'POST':
        action = request.form.get('action')
        ticker = request.form.get('ticker', '').upper()
        selected_exp = request.form.get('expiration')

        try:
            if action == 'get_expirations':
                expirations = get_expirations(ticker)

            elif action == 'get_chart':
                expirations = get_expirations(ticker)               
                if not selected_exp:
                    raise ValueError("請選擇到期日")
                calls, puts = get_option_chain(ticker, selected_exp)
                max_pain = calculate_max_pain(calls, puts)
                
                hist = yf.Ticker(ticker).history(period="1d")
                if hist.empty:
                    underlying_price = 0
                else:
                    underlying_price = hist['Close'][-1]
                  
                del hist
                gc.collect()
              
                chart = build_chart_option(calls, puts, ticker, max_pain, underlying_price)

        except Exception as e:
            error = str(e)

    print(f"max_pain_price = {max_pain}, underlying_price = {underlying_price}")
  
    return render_template(
        'maxpain.html',
        ticker=ticker,
        expirations=expirations,
        selected_exp=selected_exp,
        max_pain=max_pain,
        underlying_price=underlying_price,
        chart=chart,
        error=error
    )




JSON_EDITOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GitHub Portfolio Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.75);
            --border-card: rgba(255, 255, 255, 0.1);
            --accent-blue: #3b82f6;
            --accent-blue-hover: #2563eb;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        body { background-color: var(--bg-main); color: var(--text-main); min-height: 100vh; padding: 2rem 1.5rem; }
        
        .container { max-width: 1250px; margin: 0 auto; }
        
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
        .logo-area { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { font-size: 1.75rem; }
        h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        .token-bar { display: flex; align-items: center; gap: 0.5rem; background: rgba(30, 41, 59, 0.6); padding: 0.5rem 1rem; border-radius: 0.75rem; border: 1px solid var(--border-card); }
        .token-bar label { font-size: 0.8rem; color: var(--text-muted); font-weight: 600; white-space: nowrap; }
        .token-bar input { background: #0f172a; border: 1px solid var(--border-card); color: #fff; padding: 0.4rem 0.7rem; border-radius: 0.5rem; font-size: 0.85rem; width: 220px; outline: none; }

        .tabs { display: flex; gap: 1rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem; flex-wrap: wrap; }
        .tab-btn { background: rgba(30, 41, 59, 0.5); border: 1px solid var(--border-card); color: var(--text-muted); padding: 0.75rem 1.4rem; border-radius: 0.75rem; cursor: pointer; font-weight: 600; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.5rem; }
        .tab-btn:hover { background: rgba(51, 65, 85, 0.7); color: #fff; }
        .tab-btn.active { background: var(--accent-blue); border-color: var(--accent-blue); color: #fff; box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35); }

        .card { background: var(--bg-card); backdrop-filter: blur(12px); border: 1px solid var(--border-card); border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25); }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem; }
        .card-title { font-size: 1.15rem; font-weight: 600; color: #f1f5f9; display: flex; align-items: center; gap: 0.5rem; }

        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .stat-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border-card); border-radius: 0.75rem; padding: 1rem; text-align: center; }
        .stat-val { font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }
        .stat-label { font-size: 0.775rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }

        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .form-group { margin-bottom: 0.75rem; }
        label { display: block; font-size: 0.825rem; color: var(--text-muted); margin-bottom: 0.35rem; font-weight: 500; }
        input, select, textarea { width: 100%; background: #0f172a; border: 1px solid var(--border-card); color: #fff; padding: 0.6rem 0.8rem; border-radius: 0.5rem; font-size: 0.9rem; outline: none; transition: border 0.2s; }
        input:focus, textarea:focus { border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
        textarea { min-height: 90px; font-family: monospace; line-height: 1.4; }

        .btn-toolbar { display: flex; gap: 0.75rem; margin-top: 1rem; justify-content: space-between; align-items: center; flex-wrap: wrap; }
        .btn { padding: 0.65rem 1.25rem; border-radius: 0.6rem; border: none; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; transition: all 0.2s; }
        .btn-primary { background: var(--accent-blue); color: #fff; }
        .btn-primary:hover { background: var(--accent-blue-hover); transform: translateY(-1px); }
        .btn-success { background: var(--accent-green); color: #fff; font-size: 1rem; padding: 0.75rem 1.75rem; }
        .btn-success:hover { background: #059669; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); }
        .btn-danger { background: var(--accent-red); color: #fff; padding: 0.35rem 0.65rem; font-size: 0.8rem; border-radius: 0.4rem; }
        .btn-danger:hover { background: #dc2626; }
        .btn-secondary { background: #334155; color: #e2e8f0; }
        .btn-secondary:hover { background: #475569; }
        .btn-sm { padding: 0.35rem 0.65rem; font-size: 0.8rem; border-radius: 0.4rem; }

        .filter-btn-group { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .filter-btn { background: rgba(30, 41, 59, 0.6); border: 1px solid var(--border-card); color: var(--text-muted); padding: 0.4rem 0.8rem; border-radius: 0.5rem; font-size: 0.8rem; font-weight: 600; cursor: pointer; }
        .filter-btn.active { background: #334155; color: #fff; border-color: var(--accent-blue); }

        .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700; }
        .badge-both { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-main { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-review { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-diff { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

        .search-box { width: 260px; }

        table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
        th, td { text-align: left; padding: 0.65rem 0.75rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 0.875rem; }
        th { font-size: 0.775rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; background: rgba(15, 23, 42, 0.4); }
        td input { padding: 0.4rem 0.6rem; font-size: 0.85rem; }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }

        .actions-cell { display: flex; gap: 0.4rem; align-items: center; }

        .toast { position: fixed; top: 1.5rem; right: 1.5rem; padding: 1rem 1.5rem; border-radius: 0.75rem; color: #fff; font-weight: 600; display: none; z-index: 1000; box-shadow: 0 10px 25px rgba(0,0,0,0.5); backdrop-filter: blur(8px); }
        .toast-success { background: rgba(16, 185, 129, 0.9); border: 1px solid #10b981; }
        .toast-error { background: rgba(239, 68, 68, 0.9); border: 1px solid #ef4444; }

        .spinner { border: 3px solid rgba(255,255,255,0.3); border-top: 3px solid #fff; border-radius: 50%; width: 18px; height: 18px; animation: spin 0.8s linear infinite; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .author-tag { display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border-card); border-radius: 0.5rem; padding: 0.35rem 0.5rem; }
        .author-tag input { width: 120px; padding: 0.25rem 0.4rem; font-size: 0.85rem; }
        .author-tag .btn-danger { padding: 0.2rem 0.4rem; font-size: 0.7rem; line-height: 1; }
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="logo-area">
            <span class="logo-icon">📊</span>
            <h1>GitHub Portfolio Studio</h1>
        </div>
        <div class="token-bar">
            <label for="patToken">GitHub PAT Token:</label>
            <input type="password" id="patToken" placeholder="Default: TOKEN_GIT_JSON" autocomplete="off">
            <button class="btn btn-secondary btn-sm" onclick="toggleTokenVis()">👁️</button>
        </div>
    </header>

    <div class="tabs">
        <button class="tab-btn active" id="tab-portfolio" onclick="switchTab('portfolio')">📁 Main Portfolio (portfolio.json)</button>
        <button class="tab-btn" id="tab-review" onclick="switchTab('review')">📁 Review Portfolio (portfolio_review.json)</button>
        <button class="tab-btn" id="tab-compare" onclick="switchTab('compare')">⚖️ Compare Portfolios</button>
    </div>

    <!-- TAB 1: portfolio.json -->
    <div id="view-portfolio">
        <div class="card">
            <div class="card-header">
                <div class="card-title">⏰ Sleep Schedule Configuration</div>
                <span id="time-readable" style="font-size: 0.85rem; color: var(--accent-green); font-weight: 600;"></span>
            </div>
            <div class="form-grid">
                <div class="form-group">
                    <label>Sleep Start (Mins from Midnight):</label>
                    <input type="number" id="ts-0" placeholder="e.g. 30 (00:30)" oninput="updateTimeReadable()">
                </div>
                <div class="form-group">
                    <label>Sleep End (Mins from Midnight):</label>
                    <input type="number" id="ts-1" placeholder="e.g. 450 (07:30)" oninput="updateTimeReadable()">
                </div>
                <div class="form-group">
                    <label>Leisure Start (Mins from Midnight):</label>
                    <input type="number" id="ts-2" placeholder="e.g. 810 (13:30)" oninput="updateTimeReadable()">
                </div>
                <div class="form-group">
                    <label>Leisure End (Mins from Midnight):</label>
                    <input type="number" id="ts-3" placeholder="e.g. 1290 (21:30)" oninput="updateTimeReadable()">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">👤 Author List (PTT)</div>
                <span id="author-count" style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;"></span>
            </div>
            <div id="author-container" style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;"></div>
            <div style="margin-top: 0.5rem;">
                <button class="btn btn-secondary btn-sm" onclick="addAuthor()">➕ Add Author</button>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">📈 Main Portfolio Tickers</div>
                <input type="text" class="search-box" id="search-p1" placeholder="🔍 Search ticker..." oninput="filterTable('table-p1', this.value)">
            </div>
            <table id="table-p1">
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th>Ticker Symbol</th>
                        <th>Price Floor ($)</th>
                        <th>Price Ceiling ($)</th>
                        <th style="width: 130px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="body-p1"></tbody>
            </table>
            <div class="btn-toolbar">
                <button class="btn btn-secondary btn-sm" onclick="addRowP1()">➕ Add Symbol</button>
                <button class="btn btn-success" onclick="saveData()">
                    <div class="spinner" id="spin-p1"></div>
                    💾 Save Main Portfolio to GitHub
                </button>
            </div>
        </div>
    </div>

    <!-- TAB 2: portfolio_review.json -->
    <div id="view-review" style="display: none;">
        <!-- Stock Section -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">📉 Stock Watchlist & Indicators</div>
                <input type="text" class="search-box" id="search-p2-stock" placeholder="🔍 Search ticker or description..." oninput="filterTable('table-p2-stock', this.value)">
            </div>
            <table id="table-p2-stock">
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>Ticker Symbol</th>
                        <th>Description</th>
                        <th>Price Floor ($)</th>
                        <th>Price Ceiling ($)</th>
                        <th style="width: 130px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="body-p2-stock"></tbody>
            </table>
            <div style="margin-top: 0.75rem;">
                <button class="btn btn-secondary btn-sm" onclick="addRowP2Stock()">➕ Add Stock</button>
            </div>
        </div>

        <!-- Currency Section -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">💱 Currency Watchlist</div>
            </div>
            <table id="table-p2-currency">
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>Ticker Symbol</th>
                        <th>Description</th>
                        <th>Price Floor ($)</th>
                        <th>Price Ceiling ($)</th>
                        <th>Data Source</th>
                        <th style="width: 90px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="body-p2-currency"></tbody>
            </table>
            <div style="margin-top: 0.75rem;">
                <button class="btn btn-secondary btn-sm" onclick="addRowP2Currency()">➕ Add Currency</button>
            </div>
        </div>

        <!-- Indicator Rules & MacroMicro Section -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">⚙️ Technical Indicator Thresholds & Macro Indicators</div>
            </div>
            <div class="form-grid" style="margin-bottom: 1.25rem;">
                <div class="form-group"><label>KD High Zone (KdHigh):</label><input type="number" id="rule-kd-high"></div>
                <div class="form-group"><label>KD Low Zone (KdLow):</label><input type="number" id="rule-kd-low"></div>
                <div class="form-group"><label>CCI High Zone (CciHigh):</label><input type="number" id="rule-cci-high"></div>
                <div class="form-group"><label>CCI Low Zone (CciLow):</label><input type="number" id="rule-cci-low"></div>
                <div class="form-group"><label>RSI High Zone (RsiHigh):</label><input type="number" id="rule-rsi-high"></div>
                <div class="form-group"><label>RSI Low Zone (RsiLow):</label><input type="number" id="rule-rsi-low"></div>
                <div class="form-group"><label>Display Days (DisplayPeriod):</label><input type="number" id="rule-display-period"></div>
            </div>

            <div class="form-group">
                <label>MacroMicro Indicator ID List (macrom!cro - Comma-Separated):</label>
                <textarea id="macro-ids" placeholder="3,40,44,54530..."></textarea>
            </div>

            <div class="btn-toolbar" style="margin-top: 1.5rem;">
                <div></div>
                <button class="btn btn-success" onclick="saveData()">
                    <div class="spinner" id="spin-p2"></div>
                    💾 Save Review Portfolio to GitHub
                </button>
            </div>
        </div>
    </div>

    <!-- TAB 3: Compare Portfolios -->
    <div id="view-compare" style="display: none;">
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Main Count</div>
                <div class="stat-val" id="stat-main-cnt" style="color: #60a5fa;">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Review Count</div>
                <div class="stat-val" id="stat-review-cnt" style="color: #c084fc;">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Matched In Both</div>
                <div class="stat-val" id="stat-both-cnt" style="color: #34d399;">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Main Only</div>
                <div class="stat-val" id="stat-main-only-cnt" style="color: #38bdf8;">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Review Only</div>
                <div class="stat-val" id="stat-review-only-cnt" style="color: #a78bfa;">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Mismatches</div>
                <div class="stat-val" id="stat-diff-cnt" style="color: #fbbf24;">0</div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">⚖️ Portfolio Comparison Matrix</div>
                <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
                    <div class="filter-btn-group">
                        <button class="filter-btn active" id="fbtn-all" onclick="filterCompare('all')">All Symbols</button>
                        <button class="filter-btn" id="fbtn-both" onclick="filterCompare('both')">In Both (🟢)</button>
                        <button class="filter-btn" id="fbtn-main" onclick="filterCompare('main')">Main Only (🔵)</button>
                        <button class="filter-btn" id="fbtn-review" onclick="filterCompare('review')">Review Only (🟣)</button>
                        <button class="filter-btn" id="fbtn-diff" onclick="filterCompare('diff')">Mismatches (⚠️)</button>
                    </div>
                    <input type="text" class="search-box" id="search-compare" placeholder="🔍 Search ticker or description..." oninput="filterTable('table-compare', this.value)">
                </div>
            </div>

            <table id="table-compare">
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th style="width: 110px;">Status</th>
                        <th>Ticker Symbol</th>
                        <th>Description</th>
                        <th>Main (Floor / Ceiling)</th>
                        <th>Review (Floor / Ceiling)</th>
                        <th>Comparison Result</th>
                    </tr>
                </thead>
                <tbody id="body-compare"></tbody>
            </table>
        </div>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
    let activeTab = 'portfolio';
    let dataP1 = null;
    let dataP2 = null;
    let compareFilter = 'all';

    document.addEventListener('DOMContentLoaded', () => {
        const savedToken = localStorage.getItem('gh_pat_token');
        if (savedToken) {
            document.getElementById('patToken').value = savedToken;
        }
        document.getElementById('patToken').addEventListener('input', (e) => {
            localStorage.setItem('gh_pat_token', e.target.value);
        });
        loadData(activeTab);
    });

    function toggleTokenVis() {
        const input = document.getElementById('patToken');
        input.type = input.type === 'password' ? 'text' : 'password';
    }

    // HTML-escape values interpolated into innerHTML (input values, cell text)
    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                         .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function showToast(msg, isSuccess = true) {
        const t = document.getElementById('toast');
        t.innerText = msg;
        t.className = 'toast ' + (isSuccess ? 'toast-success' : 'toast-error');
        t.style.display = 'block';
        setTimeout(() => { t.style.display = 'none'; }, 4000);
    }

    function switchTab(tab) {
        activeTab = tab;
        document.getElementById('tab-portfolio').className = 'tab-btn' + (tab === 'portfolio' ? ' active' : '');
        document.getElementById('tab-review').className = 'tab-btn' + (tab === 'review' ? ' active' : '');
        document.getElementById('tab-compare').className = 'tab-btn' + (tab === 'compare' ? ' active' : '');
        
        document.getElementById('view-portfolio').style.display = tab === 'portfolio' ? 'block' : 'none';
        document.getElementById('view-review').style.display = tab === 'review' ? 'block' : 'none';
        document.getElementById('view-compare').style.display = tab === 'compare' ? 'block' : 'none';
        
        if (tab === 'compare') {
            loadBothForCompare();
        } else {
            loadData(tab);
        }
    }

    function loadData(tab) {
        fetch(`/json/?action=load&target=${tab}`)
            .then(res => res.json())
            .then(res => {
                if (!res.success) {
                    showToast('Load Failed: ' + res.error, false);
                    return;
                }
                if (tab === 'portfolio') {
                    dataP1 = res.data;
                    renderP1();
                } else if (tab === 'review') {
                    dataP2 = res.data;
                    renderP2();
                }
            })
            .catch(err => showToast('Network Error: ' + err, false));
    }

    function loadBothForCompare() {
        Promise.all([
            fetch('/json/?action=load&target=portfolio').then(r => r.json()),
            fetch('/json/?action=load&target=review').then(r => r.json())
        ])
        .then(([r1, r2]) => {
            if (r1.success) dataP1 = r1.data;
            if (r2.success) dataP2 = r2.data;
            renderCompare();
        })
        .catch(err => showToast('Failed to load comparison data: ' + err, false));
    }

    function updateTimeReadable() {
        const ts0 = parseInt(document.getElementById('ts-0').value) || 0;
        const ts1 = parseInt(document.getElementById('ts-1').value) || 0;
        const ts2 = parseInt(document.getElementById('ts-2').value) || 0;
        const ts3 = parseInt(document.getElementById('ts-3').value) || 0;
        const formatMin = (m) => {
            const hrs = String(Math.floor(m / 60)).padStart(2, '0');
            const mins = String(m % 60).padStart(2, '0');
            return `${hrs}:${mins}`;
        };
        document.getElementById('time-readable').innerText = `Sleep: ${formatMin(ts0)} ~ ${formatMin(ts1)} | Leisure: ${formatMin(ts2)} ~ ${formatMin(ts3)}`;
    }

    // --- TAB 1 Rendering ---
    function renderP1() {
        if (!dataP1) return;
        const ts = dataP1.timestamp || [30, 450, 810, 1290];
        document.getElementById('ts-0').value = ts[0] ?? 30;
        document.getElementById('ts-1').value = ts[1] ?? 450;
        document.getElementById('ts-2').value = ts[2] ?? 810;
        document.getElementById('ts-3').value = ts[3] ?? 1290;
        updateTimeReadable();

        const tbody = document.getElementById('body-p1');
        tbody.innerHTML = '';
        (dataP1.portfolio || []).forEach((row, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color: var(--text-muted); font-weight: 600;">${i + 1}</td>
                <td><input type="text" value="${esc(row[0] || '')}" onchange="dataP1.portfolio[${i}][0] = this.value"></td>
                <td><input type="number" step="any" value="${esc(row[1] ?? '')}" onchange="dataP1.portfolio[${i}][1] = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${esc(row[2] ?? '')}" onchange="dataP1.portfolio[${i}][2] = parseFloat(this.value) || 0"></td>
                <td class="actions-cell">
                    <button class="btn btn-secondary btn-sm" onclick="moveP1(${i}, -1)">⬆️</button>
                    <button class="btn btn-secondary btn-sm" onclick="moveP1(${i}, 1)">⬇️</button>
                    <button class="btn btn-danger" onclick="delP1(${i})">🗑️</button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        renderAuthors();
    }

    function addRowP1() {
        if (!dataP1) dataP1 = { timestamp: [30, 450, 810, 1290], portfolio: [] };
        dataP1.portfolio.push(["NEW.TW", 100, 200]);
        renderP1();
    }

    function moveP1(idx, dir) {
        const target = idx + dir;
        if (target < 0 || target >= dataP1.portfolio.length) return;
        const temp = dataP1.portfolio[idx];
        dataP1.portfolio[idx] = dataP1.portfolio[target];
        dataP1.portfolio[target] = temp;
        renderP1();
    }

    function delP1(idx) {
        dataP1.portfolio.splice(idx, 1);
        renderP1();
    }

    // --- Author List Rendering ---
    function renderAuthors() {
        if (!dataP1) return;
        if (!dataP1.author) dataP1.author = [];
        const container = document.getElementById('author-container');
        container.innerHTML = '';
        dataP1.author.forEach((name, i) => {
            const tag = document.createElement('div');
            tag.className = 'author-tag';
            tag.innerHTML = `
                <input type="text" value="${esc(name || '')}" onchange="dataP1.author[${i}] = this.value">
                <button class="btn btn-danger" onclick="delAuthor(${i})">✕</button>
            `;
            container.appendChild(tag);
        });
        document.getElementById('author-count').innerText = `${dataP1.author.length} author(s)`;
    }

    function addAuthor() {
        if (!dataP1) dataP1 = { timestamp: [30, 450, 810, 1290], portfolio: [], author: [] };
        if (!dataP1.author) dataP1.author = [];
        dataP1.author.push('NewAuthor');
        renderAuthors();
    }

    function delAuthor(idx) {
        dataP1.author.splice(idx, 1);
        renderAuthors();
    }

    // --- TAB 2 Rendering ---
    function renderP2() {
        if (!dataP2) return;

        // Stocks
        const tbodyStock = document.getElementById('body-p2-stock');
        tbodyStock.innerHTML = '';
        (dataP2.stock || []).forEach((item, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color: var(--text-muted); font-weight: 600;">${i + 1}</td>
                <td><input type="text" value="${esc(item.ticker || '')}" onchange="dataP2.stock[${i}].ticker = this.value"></td>
                <td><input type="text" value="${esc(item.description || '')}" onchange="dataP2.stock[${i}].description = this.value"></td>
                <td><input type="number" step="any" value="${esc(item.priceFloor ?? '')}" onchange="dataP2.stock[${i}].priceFloor = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${esc(item.priceCeiling ?? '')}" onchange="dataP2.stock[${i}].priceCeiling = parseFloat(this.value) || 0"></td>
                <td class="actions-cell">
                    <button class="btn btn-secondary btn-sm" onclick="moveP2Stock(${i}, -1)">⬆️</button>
                    <button class="btn btn-secondary btn-sm" onclick="moveP2Stock(${i}, 1)">⬇️</button>
                    <button class="btn btn-danger" onclick="delP2Stock(${i})">🗑️</button>
                </td>
            `;
            tbodyStock.appendChild(tr);
        });

        // Currencies
        const tbodyCurr = document.getElementById('body-p2-currency');
        tbodyCurr.innerHTML = '';
        (dataP2.currency || []).forEach((item, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color: var(--text-muted); font-weight: 600;">${i + 1}</td>
                <td><input type="text" value="${esc(item.ticker || '')}" onchange="dataP2.currency[${i}].ticker = this.value"></td>
                <td><input type="text" value="${esc(item.description || '')}" onchange="dataP2.currency[${i}].description = this.value"></td>
                <td><input type="number" step="any" value="${esc(item.priceFloor ?? '')}" onchange="dataP2.currency[${i}].priceFloor = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${esc(item.priceCeiling ?? '')}" onchange="dataP2.currency[${i}].priceCeiling = parseFloat(this.value) || 0"></td>
                <td><input type="text" value="${esc(item.source || '')}" onchange="dataP2.currency[${i}].source = this.value"></td>
                <td class="actions-cell">
                    <button class="btn btn-danger" onclick="delP2Currency(${i})">🗑️</button>
                </td>
            `;
            tbodyCurr.appendChild(tr);
        });

        // Rules & Macro
        const rules = dataP2.stockCheckingRule || {};
        document.getElementById('rule-kd-high').value = rules.KdHigh ?? 80;
        document.getElementById('rule-kd-low').value = rules.KdLow ?? 20;
        document.getElementById('rule-cci-high').value = rules.CciHigh ?? 100;
        document.getElementById('rule-cci-low').value = rules.CciLow ?? -100;
        document.getElementById('rule-rsi-high').value = rules.RsiHigh ?? 70;
        document.getElementById('rule-rsi-low').value = rules.RsiLow ?? 30;
        document.getElementById('rule-display-period').value = rules.DisplayPeriod ?? 720;

        document.getElementById('macro-ids').value = (dataP2 && dataP2['macrom!cro'] != null) ? dataP2['macrom!cro'] : '';
    }

    function addRowP2Stock() {
        if (!dataP2) dataP2 = {};
        if (!dataP2.stock) dataP2.stock = [];
        dataP2.stock.push({ ticker: "NEW", description: "Description", priceFloor: 100, priceCeiling: 200 });
        renderP2();
    }

    function moveP2Stock(idx, dir) {
        const target = idx + dir;
        if (target < 0 || target >= dataP2.stock.length) return;
        const temp = dataP2.stock[idx];
        dataP2.stock[idx] = dataP2.stock[target];
        dataP2.stock[target] = temp;
        renderP2();
    }

    function delP2Stock(idx) {
        dataP2.stock.splice(idx, 1);
        renderP2();
    }

    function addRowP2Currency() {
        if (!dataP2) dataP2 = {};
        if (!dataP2.currency) dataP2.currency = [];
        dataP2.currency.push({ ticker: "USD", description: "US Dollar", priceFloor: 28, priceCeiling: 32, source: "HNCB" });
        renderP2();
    }

    function delP2Currency(idx) {
        dataP2.currency.splice(idx, 1);
        renderP2();
    }

    // --- TAB 3: Comparison Rendering ---
    function renderCompare() {
        const mainMap = new Map();
        if (dataP1 && dataP1.portfolio) {
            dataP1.portfolio.forEach(item => {
                if (item && item[0]) mainMap.set(item[0], { floor: item[1], ceiling: item[2] });
            });
        }

        const reviewMap = new Map();
        if (dataP2 && dataP2.stock) {
            dataP2.stock.forEach(item => {
                if (item && item.ticker) reviewMap.set(item.ticker, { desc: item.description || '', floor: item.priceFloor, ceiling: item.priceCeiling });
            });
        }

        const allTickers = new Set([...mainMap.keys(), ...reviewMap.keys()]);
        
        let cntMain = mainMap.size;
        let cntReview = reviewMap.size;
        let cntBoth = 0;
        let cntMainOnly = 0;
        let cntReviewOnly = 0;
        let cntDiff = 0;

        const compareList = [];

        allTickers.forEach(ticker => {
            const inMain = mainMap.has(ticker);
            const inReview = reviewMap.has(ticker);
            
            const mainData = mainMap.get(ticker) || null;
            const reviewData = reviewMap.get(ticker) || null;

            let status = 'both';
            let resultText = 'Match ✅';
            let isDiff = false;

            if (inMain && inReview) {
                cntBoth++;
                const floorMatch = (mainData.floor === reviewData.floor);
                const ceilingMatch = (mainData.ceiling === reviewData.ceiling);
                if (!floorMatch || !ceilingMatch) {
                    isDiff = true;
                    cntDiff++;
                    const diffParts = [];
                    if (!floorMatch) diffParts.push(`Floor: $${mainData.floor} vs $${reviewData.floor}`);
                    if (!ceilingMatch) diffParts.push(`Ceiling: $${mainData.ceiling} vs $${reviewData.ceiling}`);
                    resultText = `Mismatch ⚠️ (${diffParts.join(', ')})`;
                }
            } else if (inMain) {
                status = 'main';
                cntMainOnly++;
                resultText = 'Main Only 🔵';
            } else {
                status = 'review';
                cntReviewOnly++;
                resultText = 'Review Only 🟣';
            }

            compareList.push({
                ticker,
                status,
                isDiff,
                desc: reviewData ? reviewData.desc : '',
                mainFloor: mainData ? mainData.floor : '-',
                mainCeiling: mainData ? mainData.ceiling : '-',
                reviewFloor: reviewData ? reviewData.floor : '-',
                reviewCeiling: reviewData ? reviewData.ceiling : '-',
                resultText
            });
        });

        // Update stats
        document.getElementById('stat-main-cnt').innerText = cntMain;
        document.getElementById('stat-review-cnt').innerText = cntReview;
        document.getElementById('stat-both-cnt').innerText = cntBoth;
        document.getElementById('stat-main-only-cnt').innerText = cntMainOnly;
        document.getElementById('stat-review-only-cnt').innerText = cntReviewOnly;
        document.getElementById('stat-diff-cnt').innerText = cntDiff;

        // Render table
        const tbody = document.getElementById('body-compare');
        tbody.innerHTML = '';

        compareList.forEach((item, i) => {
            let badgeHtml = '';
            if (item.status === 'both') {
                badgeHtml = item.isDiff ? '<span class="badge badge-diff">Both (Mismatch)</span>' : '<span class="badge badge-both">Both</span>';
            } else if (item.status === 'main') {
                badgeHtml = '<span class="badge badge-main">Main Only</span>';
            } else {
                badgeHtml = '<span class="badge badge-review">Review Only</span>';
            }

            const tr = document.createElement('tr');
            tr.setAttribute('data-status', item.status);
            tr.setAttribute('data-diff', item.isDiff ? 'true' : 'false');
            tr.innerHTML = `
                <td style="color: var(--text-muted); font-weight: 600;">${i + 1}</td>
                <td>${badgeHtml}</td>
                <td style="font-weight: 700; color: #fff;">${esc(item.ticker)}</td>
                <td style="color: var(--text-muted);">${esc(item.desc || '-')}</td>
                <td><code>$${item.mainFloor} / $${item.mainCeiling}</code></td>
                <td><code>$${item.reviewFloor} / $${item.reviewCeiling}</code></td>
                <td style="${item.isDiff ? 'color: var(--accent-amber); font-weight: 600;' : 'color: var(--text-muted);'}">${item.resultText}</td>
            `;
            tbody.appendChild(tr);
        });

        filterCompare(compareFilter);
    }

    function filterCompare(filterType) {
        compareFilter = filterType;
        ['all', 'both', 'main', 'review', 'diff'].forEach(f => {
            const btn = document.getElementById(`fbtn-${f}`);
            if (btn) btn.className = 'filter-btn' + (f === filterType ? ' active' : '');
        });

        const rows = document.querySelectorAll('#table-compare tbody tr');
        rows.forEach(tr => {
            const status = tr.getAttribute('data-status');
            const isDiff = tr.getAttribute('data-diff') === 'true';

            let show = false;
            if (filterType === 'all') show = true;
            else if (filterType === 'both') show = (status === 'both');
            else if (filterType === 'main') show = (status === 'main');
            else if (filterType === 'review') show = (status === 'review');
            else if (filterType === 'diff') show = isDiff;

            tr.style.display = show ? '' : 'none';
        });
    }

    function filterTable(tableId, query) {
        const q = query.toLowerCase();
        const rows = document.querySelectorAll(`#${tableId} tbody tr`);
        rows.forEach(tr => {
            const text = tr.innerText.toLowerCase();
            tr.style.display = text.includes(q) ? '' : 'none';
        });
    }

    // --- Save Handler ---
    function saveData() {
        const spin = document.getElementById(activeTab === 'portfolio' ? 'spin-p1' : 'spin-p2');
        if (spin) spin.style.display = 'inline-block';

        let finalData = null;

        if (activeTab === 'portfolio') {
            const ts0 = parseInt(document.getElementById('ts-0').value) || 0;
            const ts1 = parseInt(document.getElementById('ts-1').value) || 0;
            const ts2 = parseInt(document.getElementById('ts-2').value) || 0;
            const ts3 = parseInt(document.getElementById('ts-3').value) || 0;
            dataP1.timestamp = [ts0, ts1, ts2, ts3];
            finalData = dataP1;
        } else {
            dataP2.stockCheckingRule = {
                KdHigh: parseFloat(document.getElementById('rule-kd-high').value) || 80,
                KdLow: parseFloat(document.getElementById('rule-kd-low').value) || 20,
                CciHigh: parseFloat(document.getElementById('rule-cci-high').value) || 100,
                CciLow: parseFloat(document.getElementById('rule-cci-low').value) || -100,
                RsiHigh: parseFloat(document.getElementById('rule-rsi-high').value) || 70,
                RsiLow: parseFloat(document.getElementById('rule-rsi-low').value) || 30,
                DisplayPeriod: parseInt(document.getElementById('rule-display-period').value) || 720
            };
            dataP2['macrom!cro'] = document.getElementById('macro-ids').value.trim();
            finalData = dataP2;
        }

        const payload = {
            target: activeTab,
            data: finalData,
            token: document.getElementById('patToken').value.trim()
        };

        fetch('/json/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(res => {
            if (spin) spin.style.display = 'none';
            if (res.success) {
                showToast(res.message, true);
            } else {
                showToast('Save Failed: ' + res.error, false);
            }
        })
        .catch(err => {
            if (spin) spin.style.display = 'none';
            showToast('Network Request Error: ' + err, false);
        });
    }
</script>
</body>
</html>"""


@app.route('/json/debug/')
def json_debug():
  """Debug endpoint: test fetching GitHub JSON with both urllib and curl_cffi."""
  import urllib.request
  results = {}

  for label, url in [('portfolio', url_git_json), ('review', url_git_json_review)]:
    # Test urllib
    try:
      req = urllib.request.Request(url, headers={'User-Agent': user_agent})
      with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode('utf-8')
        data = json.loads(body)
        results[f'{label}_urllib'] = {'ok': True, 'keys': list(data.keys()) if isinstance(data, dict) else f'type={type(data).__name__}, len={len(data)}'}
    except Exception as e:
      results[f'{label}_urllib'] = {'ok': False, 'error': str(e), 'trace': traceback.format_exc()[-300:]}

    # Test curl_cffi
    try:
      r = requests.get(url, headers={'User-Agent': user_agent}, timeout=10)
      data = r.json()
      results[f'{label}_curl_cffi'] = {'ok': True, 'status': r.status_code, 'keys': list(data.keys()) if isinstance(data, dict) else f'type={type(data).__name__}, len={len(data)}'}
    except Exception as e:
      results[f'{label}_curl_cffi'] = {'ok': False, 'error': str(e), 'trace': traceback.format_exc()[-300:]}

  results['env'] = {
    'URL_GIT_JSON': url_git_json[:80] if url_git_json else None,
    'URL_GIT_JSON_REVIEW': url_git_json_review[:80] if url_git_json_review else None,
    'CONTENT_GIT_JSON': content_git_json[:80] if content_git_json else None,
    'CONTENT_GIT_JSON_REVIEW': content_git_json_review[:80] if content_git_json_review else None,
  }
  return jsonify(results)


@app.route('/json/', methods=['GET', 'POST'])
def json_modifier():
  if request.method == 'GET' and request.args.get('action') == 'load':
    target = request.args.get('target', 'portfolio')
    url_target = url_git_json_review if target == 'review' else url_git_json
    if not url_target:
      return jsonify({'success': False, 'error': f'URL not configured: {"URL_GIT_JSON_REVIEW" if target == "review" else "URL_GIT_JSON"} is not set'})
    try:
      import urllib.request
      req = urllib.request.Request(url_target, headers={'User-Agent': user_agent})
      with urllib.request.urlopen(req, timeout=10) as resp:
        raw_data = json.loads(resp.read().decode('utf-8'))
      # Handle GitHub API response (base64 encoded content)
      if isinstance(raw_data, dict) and 'content' in raw_data and raw_data.get('encoding') == 'base64':
        decoded_bytes = base64.b64decode(raw_data['content'])
        raw_data = json.loads(decoded_bytes.decode('utf-8'))
      return jsonify({'success': True, 'data': raw_data})
    except Exception as e:
      tb_str = traceback.format_exc()
      print(f"[JSON Load Error] url={url_target}\n{tb_str}")
      return jsonify({'success': False, 'error': str(e)})

  if request.method == 'POST':
    try:
      payload_req = request.get_json() or {}
      target = payload_req.get('target', 'portfolio')
      data_content = payload_req.get('data')
      token_input = payload_req.get('token', '').strip()

      active_token = token_input if token_input else token_git_json
      if not active_token:
        return jsonify({'success': False, 'error': 'Missing GitHub PAT Token. Please enter token or set TOKEN_GIT_JSON.'})

      write_endpoint = content_git_json_review if target == 'review' else content_git_json
      headers_gh = {
        'Authorization': f'Bearer {active_token}',
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'StockReview-App'
      }

      # 1. Fetch file SHA from GitHub API
      r_sha = requests.get(write_endpoint, headers=headers_gh, timeout=8)
      sha = None
      if r_sha.status_code == 200:
        sha = r_sha.json().get('sha')
      elif r_sha.status_code == 401:
        return jsonify({'success': False, 'error': 'GitHub Token Authentication Failed (401 Unauthorized)'})
      elif r_sha.status_code == 404:
        return jsonify({'success': False, 'error': f'GitHub API File Not Found (404 Not Found): {write_endpoint}'})

      # 2. Convert updated JSON to Base64
      json_bytes = json.dumps(data_content, indent=2, ensure_ascii=False).encode('utf-8')
      b64_content = base64.b64encode(json_bytes).decode('utf-8')

      # 3. Commit PUT request to GitHub API
      target_filename = 'portfolio_review.json' if target == 'review' else 'portfolio.json'
      put_body = {
        'message': f'Update {target_filename} via Web Studio',
        'content': b64_content
      }
      if sha:
        put_body['sha'] = sha

      r_put = requests.request('PUT', write_endpoint, headers=headers_gh, json=put_body, timeout=10)

      if r_put.status_code in (200, 201):
        return jsonify({'success': True, 'message': f'Successfully updated {target_filename} on GitHub!'})
      else:
        return jsonify({'success': False, 'error': f'GitHub Commit Failed (HTTP {r_put.status_code}): {r_put.text}'})

    except Exception as e:
      return jsonify({'success': False, 'error': str(e)})

  return render_template_string(JSON_EDITOR_HTML)




################################################################################################################################################################
################################################################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)
