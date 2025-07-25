import gc
import json
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

#import requests
from curl_cffi import requests
from flask import Flask, render_template, request
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

# Json url
url_git_json = os.environ.get('URL_GIT_JSON')
macd_git_json = os.environ.get('MACD_GIT_JSON')

# API KEY
gemini_api_key =  os.environ.get('API_KEY')

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
IDX_MSG = 12  # message saved, not check duplicate message, obsolete now

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

  crumb = "dx7e5yMCafJ"
  url_history = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker[0]}?period1={startDate_epoch}&period2={endDate_epoch}&interval=1d&events=history&includeAdjustedClose=true&events=div%2Csplits&crumb={crumb}"
  #print('  url=' + url_history)

  r = session.get(url_history, headers=headers, timeout=10, verify=False)

  if r.status_code == 200:
    r.encoding = 'utf-8'
    json_history = r.json()

    if use_adj == True:
      close = json_history["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
    else:
      close = json_history["chart"]["result"][0]["indicators"]["quote"][0]["close"]

    # Avoid nonetype calcuation
    if None in close:
      print(f'{ticker[0]}:\n  None in price list: {url_history}',
            file=sys.stdout)
      return [None, None, None, None, None, None, None, None, None]

    precision = 4 if close[-1] < 1 else 2

    # Today
    ma10 = round(sum(close[-10:]) / 10, precision)
    ma20 = round(sum(close[-20:]) / 20, precision)
    ma60 = round(sum(close[-60:]) / 60, precision)
    ma200 = round(sum(close[-200:]) / 200, precision)

    # Yesterday
    ma10_1 = round(sum(close[-11:-1]) / 10, precision)
    ma20_1 = round(sum(close[-21:-1]) / 20, precision)
    ma60_1 = round(sum(close[-61:-1]) / 60, precision)
    ma200_1 = round(sum(close[-201:-1]) / 200, precision)

    return [None, ma10, ma20, ma60, ma200, ma10_1, ma20_1, ma60_1, ma200_1]

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
    resp = r.text
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




################################################################################################################################################################
app = Flask(__name__)




################################################################################################################################################################
@app.route('/')
def index():
  return 'index'
 



################################################################################################################################################################
@app.route('/fire/')
def fire():

  gc.collect()
  
  #--------------------------------------------------------------------------------
  portfolio_cnt = 0
  portfolio_reload = 24
  leisure_time = False
  macd_w_is_fall = {}

  timestamp = [30, 450, 810, 1290]  # 00:30, 07:30, 13:30, 21:30

  portfolio = [["2454.TW", 820, 1200], ["2330.TW", 1000, 1200]]

  url_tg_prefix = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text='
  url_tg_getUpdate = f'https://api.telegram.org/bot{bot_token}/getupdates?offset=-1'
  json_file_path = "data.json"
  pcnt_file_path = "pcnt.pkl"

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
    portfolio_reload = 240 / query_interval  # 4H
  else:
    leisure_time = False
    portfolio_reload = 120 / query_interval  # 2H

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
  msg_toast.append(timestamp_msg + f' ({day+1}) - Render (cnt={portfolio_cnt})')
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
        p[IDX_10MA:-1] = ma[1:]

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

      for x, jp in enumerate(yahoo_portfolio):
        s = yahoo_portfolio[x]['symbol']
        sn = (yahoo_portfolio[x]['symbolName'].split(' '))[0]

        if 'change' not in yahoo_portfolio[x]:
          continue

        sd = yahoo_portfolio[x]['change']['raw']
        sdp = yahoo_portfolio[x]['changePercent']

        # Add color symbole as prefix
        if sd[0] == '-':
          sc = p_d
        else:
          sc = p_u

        if s == "^TWII":
          sdp_base_tw = sdp
        if s == "^GSPC":
          sdp_base_us = sdp

        # Symbols could be received in out-of-order, need match index
        for i in range(0, len(tickers)):
          #print(f'{i} {c} {i+c} {x} {s} {portfolio[i+c][0]}')

          if s != portfolio[i + c][IDX_T]:
            continue  # Check symobol name match

          if yahoo_portfolio[x]['price']['raw'] == '-':
            price = float(
                yahoo_portfolio[x]['regularMarketPreviousClose']['raw'])
          else:
            price = float(yahoo_portfolio[x]['price']['raw'])

          try:
            price_1 = float(yahoo_portfolio[x]['regularMarketPreviousClose']['raw'])
          except ValueError:
            price_1 = price

          #print(f"{yahoo_portfolio[x]['symbol']}: {price}")

          sdp_base = None
          if ".TW" in s:
            sdp_base = sdp_base_tw
          else:
            sdp_base = sdp_base_us

          if sdp_base == None:
            sdp_base = sdp

          # ^TWII & ^GSPC index comparison
          sdp_radio = '?'
          if (sdp_base != '-') and (sdp != '-'):
            if float(sdp_base.strip('%')) != 0:
              sdp_ratio_float = float(sdp.strip('%')) / float(
                  sdp_base.strip('%'))
              sdp_radio = f"{sdp_ratio_float:.1f}x"
            else:
              print(f"{s} SDP: {sdp_base} {sdp} {sdp_base_tw} {sdp_base_us}",
                    file=sys.stdout)

          # 200MA diff
          if portfolio[i + c][IDX_200MA] != None:
            sdp_radio += f" {((price-portfolio[i+c][IDX_200MA])/portfolio[i+c][IDX_200MA])*100  :.1f}%"

          msg = ''

          s_dot = s.find('.')
          #if s_dot > -1:
          #  s = s[:s_dot]

          precision = 4 if price < 1 else 2

          if portfolio[i + c][IDX_P] != None:  # Already has history record

            delta = (price - portfolio[i + c][IDX_P]) / portfolio[i + c][IDX_P]

            if delta >= 0:
              sc = sc + a_u
            else:
              sc = sc + a_d

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

            # Judge criteria
            if delta > delta_u:
              msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} {delta*100:.{precision}f}% ▲"  # Check quick +1.618% price change
              update_flag = True

            if delta < delta_d:
              msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} {delta*100:.{precision}f}% ▼"  # Check quick -1.618% price change
              update_flag = True

            # Skip small price variation (0.618%)
            if abs(
                delta
            ) > delta_a:  # Smooth report, only report when variation > 0.618%
              if price < portfolio[i + c][IDX_F]:  # Check low price
                msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} {delta*100:.{precision}f}% < {portfolio[i+c][IDX_F]}"
                update_flag = True

              if price > portfolio[i + c][IDX_C]:  # Check high price
                msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} {delta*100:.{precision}f}% > {portfolio[i+c][IDX_C]}"
                update_flag = True

            if update_flag == True:
              portfolio[i + c][IDX_P] = price  # To save curent price
            """
            if msg != "":
              print(f"{yahoo_portfolio[x]['symbol']}: {price} -- {delta:.5f} {update_flag} {msg}")
            """

          else:  # 1st time get price and msg

            if price < portfolio[i + c][IDX_F]:  # Check low price
              msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} < {portfolio[i+c][IDX_F]}"

            elif price > portfolio[i + c][IDX_C]:  # Check high price
              msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f} > {portfolio[i+c][IDX_C]}"

            else:
              msg = f"{sc}{s} ({sn}) [{sd} {sdp} {sdp_radio}]: {price:.{precision}f}"

            portfolio[
                i + c][IDX_P] = price  # To append curent price (new list item)


          if msg != '':
            msg_updated = True  # Remove stored msg to save the heap size

            if msg_updated == True:

              # Check if SMA cross (today vs. yesterday)
              ma10 = portfolio[i + c][IDX_10MA]
              ma20 = portfolio[i + c][IDX_20MA]
              ma60 = portfolio[i + c][IDX_60MA]
              ma200 = portfolio[i + c][IDX_200MA]
              ma10_1 = portfolio[i + c][IDX_10MA_1]
              ma20_1 = portfolio[i + c][IDX_20MA_1]
              ma60_1 = portfolio[i + c][IDX_60MA_1]
              ma200_1 = portfolio[i + c][IDX_200MA_1]

              price_low = portfolio[i + c][IDX_F]
              price_high = portfolio[i + c][IDX_C]

              # Floor/Ceilng cross
              if price_low != None:
                if (price > price_low) and (price_1 <= price_low):
                  msg += f'{l_c_u}L={price_low:.{precision}f}'
                if (price < price_low) and (price_1 >= price_low):
                  msg += f'{l_c_d}L={price_low:.{precision}f}'
              if price_high != None:
                if (price > price_high) and (price_1 <= price_high):
                  msg += f'{l_c_u}H={price_high:.{precision}f}'
                if (price < price_high) and (price_1 >= price_high):
                  msg += f'{l_c_d}H={price_high:.{precision}f}'

              # SMA trend
              if ma10_1 != None:
                if (price > ma10) and (price_1 <= ma10_1):
                  msg += f'{l_c_u}MA10={ma10:.{precision}f}'
                if (price < ma10) and (price_1 >= ma10_1):
                  msg += f'{l_c_d}MA10={ma10:.{precision}f}'
                  msg = msg[0] + msg

                  if portfolio[i + c][IDX_T] in macd_w_is_fall:
                    if macd_w_is_fall[portfolio[i + c][
                        IDX_T]] == True:  # SMA10 cross and weekly MACD is fall
                      msg = msg[0] + msg + ' MACD Fall'

              if ma20_1 != None:
                if (price > ma20) and (price_1 <= ma20_1):
                  msg += f'{l_c_u}MA20={ma20:.{precision}f}'
                if (price < ma20) and (price_1 >= ma20_1):
                  msg += f'{l_c_d}MA20={ma20:.{precision}f}'
                  if ma10 > ma20:
                    msg = msg[0] + msg + ' JUMP Fall'

              if ma60_1 != None:
                if (price > ma60) and (price_1 <= ma60_1):
                  msg += f'{l_c_u}MA60={ma60:.{precision}f}'
                if (price < ma60) and (price_1 >= ma60_1):
                  msg += f'{l_c_d}MA60={ma60:.{precision}f}'

              if (ma200_1 != None):
                if (price > ma200) and (price_1 <= ma200_1):
                  msg += f'{l_c_u}MA200={ma200:.{precision}f}'
                if (price < ma200) and (price_1 >= ma200_1):
                  msg += f'{l_c_d}MA200={ma200:.{precision}f}'

              # SMA cross
              if (ma60_1 != None) and (ma10_1 != None):
                if (ma10 > ma60) and (ma10_1 <= ma60_1):
                  msg += f'{l_c_u}MA1060={ma10:.{precision}f}, {ma60:.{precision}f}'
                if (ma10 < ma60) and (ma10_1 >= ma60_1):
                  msg += f'{l_c_d}MA1060={ma10:.{precision}f}, {ma60:.{precision}f}'
              if (ma20_1 != None) and (ma10_1 != None):
                if (ma10 > ma20) and (ma10_1 <= ma20_1):
                  msg += f'{l_c_u}MA1020={ma10:.{precision}f}, {ma20:.{precision}f}'
                if (ma10 < ma20) and (ma10_1 >= ma20_1):
                  msg += f'{l_c_d}MA1020={ma10:.{precision}f}, {ma20:.{precision}f}'
              if (ma60_1 != None) and (ma20_1 != None):
                if (ma20 > ma60) and (ma20_1 <= ma60_1):
                  msg += f'{l_c_u}MA2060={ma20:.{precision}f}, {ma60:.{precision}f}'
                if (ma20 < ma60) and (ma20_1 >= ma60_1):
                  msg += f'{l_c_d}MA2060={ma20:.{precision}f}, {ma60:.{precision}f}'

              # Add chart
              if (l_c_u in msg) or (l_c_d in msg):
                ticker = portfolio[i + c][IDX_T]
                url_chart = ''
                if '^' in ticker or '-' in ticker or '=' in ticker or '.S' in ticker or '.HK' in ticker:  # Skip index
                  pass
                elif '.TW' in ticker:  # TW
                  t = ticker[:ticker.index('.')]
                  #url_chart = f'https://goodinfo.tw/StockInfo/image/StockPrice/PRICE_DATE_{t}.gif'
                  url_chart = f'https://stock.wearn.com/finance_chart.asp?stockid={t}&timeblock=365&sma1=10&sma2=20&sma3=60&volume=1'
                else:  # US
                  t = ticker
                  url_chart = f'https://charts2.finviz.com/chart.ashx?t={t}&ta=1&ty=c&p=d&s=l'  # technical chart

                if url_chart != '':
                  msg += f'%0A{url_chart}%0A'

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
    s_len = 10
    for s in range(0, len(msg_toast),
                   s_len):  # if send all in one batch, TG will fail
      msg_segment = msg_toast[s:s + s_len]

      url_tg = url_tg_prefix + ('%0A'.join(msg_segment)).replace(
          '&', '%26')  # Handle '&' in url
      r = session.get(url_tg, headers=headers, timeout=2)

    return ('<br>'.join(msg_toast)).replace('%0A', '<br>')

  else:
    return msg_toast[0]

  gc.collect()



################################################################################################################################################################
################################################################################################################################################################
#from pyecharts import options as opts
#from pyecharts.charts import Line
#import pandas as pd
#import numpy as np
#from flask import request



################################################################################################################################################################
def get_stock_data(ticker, start_date, end_date, session, crumb="F7GXvns0Eji"):

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




################################################################################################################################################################
################################################################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)
