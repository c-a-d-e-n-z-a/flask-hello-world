import base64
import gc
import html
import json
import os
import pickle
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


#import requests
from curl_cffi import requests
from flask import Flask, render_template, render_template_string, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line




################################################################################################################################################################
#--------------------------------------------------------------------------------
# Telegarm info
bot_token = os.environ.get('BOT_TOKEN')
chat_id = os.environ.get('CHAT_ID')

# Json urls & API endpoints
url_git_json = os.environ.get('URL_GIT_JSON')
url_git_json_review = os.environ.get('URL_GIT_JSON_REVIEW')
macd_git_json = os.environ.get('MACD_GIT_JSON')

token_git_json = os.environ.get('TOKEN_GIT_JSON')
content_git_json = os.environ.get('CONTENT_GIT_JSON')
content_git_json_review = os.environ.get('CONTENT_GIT_JSON_REVIEW')

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

YAHOO_CRUMB = "dx7e5yMCafJ"

#--------------------------------------------------------------------------------
p_u = '\U0001F534'  # price mark 🔴
p_d = '\U0001F7E2'  # price mark 🟢
a_u = '\U00002191'  # arrow mark ↑
a_d = '\U00002193'  # arrow mark ↓
l_c_u = '\U0000274C'  # MA cross upwards mark ❌
l_c_d = '\U0000274E'  # MA corss downwards mark ❎


################################################################################################################################################################
def ma_calculation(ticker, session, use_adj=True):

  today = date.today()
  startDate = today - timedelta(days=365)
  endDate = today

  startDate_epoch = int(
      datetime.combine(startDate,
                       datetime.now().time()).timestamp())
  endDate_epoch = int(
      datetime.combine(endDate,
                       datetime.now().time()).timestamp())

  crumb = YAHOO_CRUMB
  url_history = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker[0]}?period1={startDate_epoch}&period2={endDate_epoch}&interval=1d&events=history&includeAdjustedClose=true&events=div%2Csplits&crumb={crumb}"
  #print('  url=' + url_history)

  r = session.get(url_history, headers=headers, timeout=10, verify=False)

  if r.status_code == 200:
    r.encoding = 'utf-8'
    json_history = r.json()

    close_raw = json_history["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    if use_adj == True:
      close = json_history["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
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
        return [None, None, None, None, None, None, None, None, None, None]

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

    return [None, ma10, ma20, ma60, ma200, ma10_1, ma20_1, ma60_1, ma200_1, std200]

  else:

    print(f'{ticker[0]}:\n  Error in price list: {url_history}',
          file=sys.stdout)
    return [None, None, None, None, None, None, None, None, None]




################################################################################################################################################################
def get_fitx_histock(session):

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




_fire_timestamp = [30, 450, 810, 1290]  # persists across requests

################################################################################################################################################################
app = Flask(__name__)




################################################################################################################################################################
@app.route('/')
def index():
  return 'index'
 



################################################################################################################################################################
@app.route('/fire/')
def fire():

  global _fire_timestamp

  gc.collect()

  #--------------------------------------------------------------------------------
  portfolio_cnt = 0
  portfolio_reload = 24
  leisure_time = False
  macd_w_is_fall = {}

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

  # Reset portolio to trigee full report, like @13:33 and @21:27 (TW stock close, and before US open), and always reset @7:30
  if (day < 5 and ((mins == timestamp[2] + query_interval) or (mins == timestamp[3] - query_interval))) or (mins == 450):
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

  # Reload portfolio every 60 runs
  if portfolio_cnt == 0:

    r = session.get(url_git_json, headers=headers, timeout=5)
    if r.status_code == 200:
      r.encoding = 'utf-8'
      json_git = r.json()

      timestamp = json_git["timestamp"]
      _fire_timestamp = timestamp
      print(f'\nLoad timestamp: {timestamp}')

      reset_portfolio = False

      if len(portfolio) != len(json_git["portfolio"]):
        print(f'[POR] LEN: {len(portfolio)} != {len(json_git["portfolio"])}')
        reset_portfolio = True

      for i in range(len(portfolio)):
        for j in range(3):
          if portfolio[i][j] != json_git["portfolio"][i][j]:
            print(f'[POR] ELM: {portfolio[i][j]} != {json_git["portfolio"][i][j]}')
            reset_portfolio = True
            break
        else:  # only execute when it's no break in the inner loop
          continue
        break

      if reset_portfolio == True:
        portfolio.clear()
        print("\nReset portfolio - 2")
        portfolio = json_git["portfolio"]
        print('--------------------------------------------------------------------------------')
        print(portfolio)
        print('--------------------------------------------------------------------------------')

    for p in portfolio:
      ma = ma_calculation(p, session)
      #ma = ema_calculation(p, session)
      if reset_portfolio == True:
        p.extend(ma)
      else:
        p[IDX_10MA:IDX_10MA+9] = ma[1:]

      print(p)

    r = session.get(macd_git_json, headers=headers, timeout=5)
    if r.status_code == 200:
      r.encoding = 'utf-8'
      macd_w_is_fall = r.json()
      print("\nMACD Hist (W) fall check")
      print(macd_w_is_fall)

    msg_toast.append(get_fitx_histock(session))


  #--------------------------------------------------------------------------------
  # Start get stock quotes
  #chunk_len = len(portfolio)   # Set chunk length = portfolio length means only 1 package
  chunk_len = 10  # Set chunk length = portfolio length means only 1 package

  sdp_base_tw = sdp_base_us = sdp_base = None

  for c in range(0, len(portfolio), chunk_len):

    # Only can query 3 tickers at a time.
    chunk = portfolio[c:c + chunk_len]
    tickers = [p[IDX_T] for p in chunk]
    tickers_url = ','.join(tickers)

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
              "^TWII", "^TWOII", "^GSPC", "^RUT", "^N225", "^KS11", "VOO",
              "QQQ", "000300.SS"
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
          msg_updated = True  # Remove stored msg to save the heap size

          if msg_updated == True:
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

            # Add chart
            if (l_c_u in msg) or (l_c_d in msg):
              ticker = portfolio[port_idx][IDX_T]
              url_chart = ''
              ts = int(time.time())
              if '^' in ticker or '-' in ticker or '=' in ticker or '.S' in ticker or '.HK' in ticker:  # Skip index
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
def get_stock_data(ticker, start_date, end_date, session, crumb=YAHOO_CRUMB):

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
  cov = np.cov(m, t)[0][1]
  var = np.var(m)
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
def calculate_max_pain(calls, puts):
    all_strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    pain = {}

    for strike in all_strikes:
        total_loss = 0
        for _, row in calls.iterrows():
            if row['strike'] < strike:
                loss = row['openInterest'] * (strike - row['strike'])
                total_loss += loss
        for _, row in puts.iterrows():
            if row['strike'] > strike:
                loss = row['openInterest'] * (row['strike'] - strike)
                total_loss += loss
        pain[strike] = total_loss

    del all_strikes
    gc.collect()

    return min(pain, key=pain.get)




################################################################################################################################################################
def build_chart_option(calls, puts, ticker, max_pain, underlying_price):

    df_calls = calls[['strike', 'openInterest']].dropna()
    df_puts = puts[['strike', 'openInterest']].dropna()

    # 計算選擇權賣方的總損失
    strikes = sorted(set(df_calls['strike']).union(set(df_puts['strike'])))
    strike_labels = [str(s) for s in strikes]

    call_losses = []
    put_losses = []
    for expiry_price in strikes:
        # 看漲選擇權損失：價內 (strike < expiry_price)
        call_loss = df_calls[df_calls['strike'] < expiry_price].apply(
            lambda r: (expiry_price - r['strike']) * r['openInterest'], axis=1).sum()
        # 看跌選擇權損失：價內 (strike > expiry_price)
        put_loss = df_puts[df_puts['strike'] > expiry_price].apply(
            lambda r: (r['strike'] - expiry_price) * r['openInterest'], axis=1).sum()
        call_losses.append(float(call_loss))
        put_losses.append(float(put_loss))

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
                <td><input type="text" value="${row[0] || ''}" onchange="dataP1.portfolio[${i}][0] = this.value"></td>
                <td><input type="number" step="any" value="${row[1] ?? ''}" onchange="dataP1.portfolio[${i}][1] = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${row[2] ?? ''}" onchange="dataP1.portfolio[${i}][2] = parseFloat(this.value) || 0"></td>
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
                <input type="text" value="${name || ''}" onchange="dataP1.author[${i}] = this.value">
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
                <td><input type="text" value="${item.ticker || ''}" onchange="dataP2.stock[${i}].ticker = this.value"></td>
                <td><input type="text" value="${item.description || ''}" onchange="dataP2.stock[${i}].description = this.value"></td>
                <td><input type="number" step="any" value="${item.priceFloor ?? ''}" onchange="dataP2.stock[${i}].priceFloor = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${item.priceCeiling ?? ''}" onchange="dataP2.stock[${i}].priceCeiling = parseFloat(this.value) || 0"></td>
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
                <td><input type="text" value="${item.ticker || ''}" onchange="dataP2.currency[${i}].ticker = this.value"></td>
                <td><input type="text" value="${item.description || ''}" onchange="dataP2.currency[${i}].description = this.value"></td>
                <td><input type="number" step="any" value="${item.priceFloor ?? ''}" onchange="dataP2.currency[${i}].priceFloor = parseFloat(this.value) || 0"></td>
                <td><input type="number" step="any" value="${item.priceCeiling ?? ''}" onchange="dataP2.currency[${i}].priceCeiling = parseFloat(this.value) || 0"></td>
                <td><input type="text" value="${item.source || ''}" onchange="dataP2.currency[${i}].source = this.value"></td>
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
                <td style="font-weight: 700; color: #fff;">${item.ticker}</td>
                <td style="color: var(--text-muted);">${item.desc || '-'}</td>
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
