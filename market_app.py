import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET

# --- Page Configuration ---
st.set_page_config(page_title="Indian Market Sniper", layout="wide")
st.title("🇮🇳 ProTrade Sniper: AI-Powered Analyzer")

# --- DATASET: FULL NIFTY 100 LIST ---
STOCK_DICT = {
    # --- INDICES ---
    "Nifty 50 Index": "^NSEI",
    "Nifty Bank Index": "^NSEBANK",
    "Nifty 100 Index": "^CNX100",
    "BSE SENSEX": "^BSESN",
    
    # --- TOP GIANTS ---
    "Reliance Industries": "RELIANCE",
    "TCS": "TCS",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "Infosys": "INFY",
    "Bharti Airtel": "BHARTIARTL",
    "State Bank of India (SBI)": "SBIN",
    "ITC Ltd": "ITC",
    "Larsen & Toubro (L&T)": "LT",
    "Hindustan Unilever": "HINDUNILVR",
    
    # --- AUTOMOBILE ---
    "Tata Motors": "TATAMOTORS",
    "Mahindra & Mahindra": "M&M",
    "Maruti Suzuki": "MARUTI",
    "Bajaj Auto": "BAJAJ-AUTO",
    "Eicher Motors": "EICHERMOT",
    "Hero MotoCorp": "HEROMOTOCO",
    "TVS Motor": "TVSMOTOR",
    "Samvardhana Motherson": "MOTHERSON",
    
    # --- BANKING & FINANCE ---
    "Axis Bank": "AXISBANK",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Bajaj Finance": "BAJFINANCE",
    "Bajaj Finserv": "BAJAJFINSV",
    "IndusInd Bank": "INDUSINDBK",
    "Bank of Baroda": "BANKBARODA",
    "Punjab National Bank": "PNB",
    "Jio Financial Services": "JIOFIN",
    "Power Finance Corp (PFC)": "PFC",
    "REC Ltd": "RECLTD",
    "Shriram Finance": "SHRIRAMFIN",
    "Cholamandalam Invest": "CHOLAFIN",
    "HDFC Life Insurance": "HDFCLIFE",
    "SBI Life Insurance": "SBILIFE",
    "ICICI Prudential": "ICICIPRULI",
    "ICICI Lombard": "ICICIGI",
    "SBI Cards": "SBICARD",
    "Muthoot Finance": "MUTHOOTFIN",
    
    # --- IT & TECH ---
    "HCL Technologies": "HCLTECH",
    "Wipro": "WIPRO",
    "Tech Mahindra": "TECHM",
    "LTIMindtree": "LTIM",
    "Oracle Financial (OFSS)": "OFSS",
    "Persistent Systems": "PERSISTENT",
    
    # --- ENERGY, OIL & POWER ---
    "NTPC": "NTPC",
    "ONGC": "ONGC",
    "Power Grid Corp": "POWERGRID",
    "Coal India": "COALINDIA",
    "Adani Green Energy": "ADANIGREEN",
    "Adani Power": "ADANIPOWER",
    "Tata Power": "TATAPOWER",
    "BPCL": "BPCL",
    "IOC": "IOC",
    "GAIL": "GAIL",
    "NHPC": "NHPC",
    "JSW Energy": "JSWENERGY",
    
    # --- METALS & MINING ---
    "Tata Steel": "TATASTEEL",
    "JSW Steel": "JSWSTEEL",
    "Hindalco": "HINDALCO",
    "Vedanta": "VEDL",
    "Jindal Steel (JSL)": "JINDALSTEL",
    "NMDC": "NMDC",
    
    # --- CONSUMER GOODS (FMCG) ---
    "Nestle India": "NESTLEIND",
    "Titan Company": "TITAN",
    "Asian Paints": "ASIANPAINT",
    "Britannia": "BRITANNIA",
    "Godrej Consumer": "GODREJCP",
    "Dabur": "DABUR",
    "Colgate Palmolive": "COLPAL",
    "Tata Consumer": "TATACONSUM",
    "Varun Beverages": "VBL",
    "United Spirits (McDowell)": "MCDOWELL-N",
    "Berger Paints": "BERGERPAINT",
    
    # --- PHARMA & HEALTHCARE ---
    "Sun Pharma": "SUNPHARMA",
    "Cipla": "CIPLA",
    "Dr Reddys Labs": "DRREDDY",
    "Apollo Hospitals": "APOLLOHOSP",
    "Divis Labs": "DIVISLAB",
    "Mankind Pharma": "MANKIND",
    "Torrent Pharma": "TORNTPHARM",
    "Lupin": "LUPIN",
    "Max Healthcare": "MAXHEALTH",
    "Zydus Lifesciences": "ZYDUSLIFE",
    
    # --- INFRA, REAL ESTATE & CEMENT ---
    "UltraTech Cement": "ULTRACEMCO",
    "Ambuja Cements": "AMBUJACEM",
    "Grasim Industries": "GRASIM",
    "Shree Cement": "SHREECEM",
    "ACC": "ACC",
    "DLF": "DLF",
    "Macrotech (Lodha)": "LODHA",
    "Godrej Properties": "GODREJPROP",
    
    # --- ADANI GROUP (Others) ---
    "Adani Enterprises": "ADANIENT",
    "Adani Ports": "ADANIPORTS",
    "Adani Energy Solutions": "ADANIENSOL",
    "Adani Total Gas": "ATGL",
    
    # --- OTHERS (Defense, Transport, Services) ---
    "HAL (Hindustan Aeronautics)": "HAL",
    "BEL (Bharat Electronics)": "BEL",
    "Siemens": "SIEMENS",
    "ABB India": "ABB",
    "InterGlobe Aviation (Indigo)": "INDIGO",
    "Zomato": "ZOMATO",
    "Trent": "TRENT",
    "Havells India": "HAVELLS",
    "Pidilite Industries": "PIDILITIND",
    "Info Edge (Naukri)": "NAUKRI",
    "SRF Ltd": "SRF",
    "PI Industries": "PIIND",
    "Bosch": "BOSCHLTD",
    "Container Corp (CONCOR)": "CONCOR"
}

# --- Sidebar ---
st.sidebar.header("1. Select Stock")
exchange = st.sidebar.radio("Exchange:", ["NSE", "BSE"], horizontal=True)
suffix = ".NS" if exchange == "NSE" else ".BO"

search_mode = st.sidebar.selectbox(
    "Stock Name:", 
    ["Type Manually"] + list(STOCK_DICT.keys())
)

if search_mode == "Type Manually":
    raw = st.sidebar.text_input("Type Symbol:", "RELIANCE").upper()
    final_ticker = raw if (raw.endswith(".NS") or raw.endswith(".BO") or raw.startswith("^")) else f"{raw}{suffix}"
    news_query = raw
else:
    news_query = search_mode
    sym = STOCK_DICT[search_mode]
    final_ticker = sym if sym.startswith("^") else f"{sym}{suffix}"

# --- CHANGE: Analysis Button Moved Here ---
run_analysis = st.sidebar.button("Run Sniper Analysis", use_container_width=True)
st.sidebar.markdown("---")

# --- CHANGE: Market Screener Section ---
st.sidebar.header("2. Nifty 100 Scanner")
st.sidebar.caption("Scan for top BUY signals")

# Confirmation Checkbox
enable_scan = st.sidebar.checkbox("Enable Scan")

# Scan Button (Controlled by Checkbox)
if enable_scan:
    run_scan = st.sidebar.button("Scan Market Now", use_container_width=True)
else:
    run_scan = False
    st.sidebar.button("Scan Market Now", disabled=True, use_container_width=True)

period = "1y" 

# --- Logic: Fetch Data (CACHED) ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to speed up app
def get_data(t, p):
    try:
        d = yf.download(t, period=p, progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.reset_index(inplace=True)
        for c in ['Open','High','Low','Close','Volume']: 
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce')
        return d if len(d) > 200 else pd.DataFrame()
    except: return pd.DataFrame()

# --- Logic: Fetch Fundamental Data (NEW) ---
def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "pe": info.get('trailingPE', 0),
            "sector": info.get('sector', 'Unknown'),
            "market_cap": info.get('marketCap', 0),
            "beta": info.get('beta', 0),
            "website": info.get('website', '#')
        }
    except:
        return None

# --- Logic: Fetch News (CACHED) ---
@st.cache_data(ttl=14400) # Cache news for 4 hours
def get_news(query):
    try:
        clean_query = query.replace(" Index", "").replace(" Limited", "").replace(" Ltd", "")
        url = f"https://news.google.com/rss/search?q={clean_query}+stock+news+India&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text
            news_items.append({"title": title, "link": link, "date": pub_date})
        return news_items
    except: return []

# --- Logic: Advanced Indicators ---
def calculate(df):
    if df.empty: return df
    
    # Trend
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # RSI
    d = df['Close'].diff()
    g, l = d.where(d>0, 0), -d.where(d<0, 0)
    df['RSI'] = 100 - (100 / (1 + (g.rolling(14).mean() / l.rolling(14).mean())))
    
    # MACD
    e12 = df['Close'].ewm(span=12, adjust=False).mean()
    e26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = e12 - e26
    df['MACD_SIG'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_UP'], df['BB_LO'] = sma + (std*2), sma - (std*2)
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['STOCH_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean() 
    
    # VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    return df

# --- Logic: Pivot Points (NEW) ---
def calculate_pivots(df):
    last = df.iloc[-1]
    high = last['High']
    low = last['Low']
    close = last['Close']
    
    # Classic Pivot Formula
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {"P": pivot, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

# --- Logic: THE SNIPER DECISION ENGINE ---
def get_verdict(row):
    buy_score = 0
    sell_score = 0
    reasons = []

    if row['Close'] > row['SMA_200']:
        buy_score += 1
        reasons.append("✅ PRICE > 200 SMA (Long Term Uptrend)")
    else:
        sell_score += 1
        reasons.append("❌ PRICE < 200 SMA (Long Term Downtrend)")

    if row['Close'] > row['VWAP']:
        buy_score += 1
        reasons.append("✅ PRICE > VWAP (Buyers in control)")
    else:
        sell_score += 1
        reasons.append("❌ PRICE < VWAP (Sellers in control)")

    if row['RSI'] < 30 and row['STOCH_K'] < 20:
        buy_score += 3
        reasons.append("🔥 SNIPER BUY: RSI & Stochastic Oversold!")
    elif row['RSI'] < 40:
        buy_score += 1
        reasons.append("✅ RSI is Low (Cheap)")
    
    if row['RSI'] > 70 and row['STOCH_K'] > 80:
        sell_score += 3
        reasons.append("⚠️ SNIPER SELL: RSI & Stochastic Overbought!")
    elif row['RSI'] > 60:
        sell_score += 1
        reasons.append("❌ RSI is High (Expensive)")

    if row['MACD'] > row['MACD_SIG']:
        buy_score += 1
        reasons.append("✅ MACD is Positive")
    else:
        sell_score += 1
        reasons.append("❌ MACD is Negative")

    if row['Close'] < row['BB_LO']:
        buy_score += 2
        reasons.append("✅ PRICE CRASHED (Below Lower Band)")
    elif row['Close'] > row['BB_UP']:
        sell_score += 2
        reasons.append("❌ PRICE SPIKED (Above Upper Band)")

    total_score = buy_score - sell_score
    
    if total_score >= 4: return "STRONG BUY", "green", "High Confidence Setup", reasons
    elif total_score >= 2: return "BUY", "lightgreen", "Positive Outlook", reasons
    elif total_score <= -4: return "STRONG SELL", "red", "High Risk Warning", reasons
    elif total_score <= -2: return "SELL", "orange", "Negative Outlook", reasons
    else: return "HOLD / WAIT", "gray", "Market is sideways", reasons

# --- NEW: MARKET SCANNER LOGIC ---
def scan_market():
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get Nifty tickers (skip indices)
    tickers = [v + ".NS" for k, v in STOCK_DICT.items() if not v.startswith("^")]
    tickers = list(set(tickers)) 
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {i+1}/{total}: {ticker}...")
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if not df.empty and len(df) > 200:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = calculate(df).dropna()
                if not df.empty:
                    last = df.iloc[-1]
                    verdict, color, subtitle, reasons = get_verdict(last)
                    
                    # Only show BUY opportunities
                    if "BUY" in verdict:
                        results.append({
                            "Stock": ticker.replace(".NS", ""),
                            "Signal": verdict,
                            "Price": round(last['Close'], 2),
                            "RSI": round(last['RSI'], 1),
                            "Reason": reasons[0] if reasons else "Trend"
                        })
        except: pass
        progress_bar.progress((i + 1) / total)
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results).sort_values(by="RSI", ascending=True)

# --- MAIN EXECUTION ---

# 1. MARKET SCANNER EXECUTION
if run_scan:
    st.header("🎯 Nifty 100 Market Scanner")
    st.info("Scanning for High-Probability BUY signals... To STOP scan, click the 'X' (Stop) in your browser tab.")
    
    results_df = scan_market()
    
    if not results_df.empty:
        st.success(f"Scan Complete! Found {len(results_df)} opportunities.")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("No strong BUY signals found right now.")

# 2. SINGLE STOCK ANALYSIS EXECUTION
elif run_analysis:
    with st.spinner('Accessing Market Data...'):
        df = get_data(final_ticker, period)
        if not df.empty:
            df = calculate(df).dropna()
            if not df.empty:
                cur = df.iloc[-1]
                verdict, v_color, subtitle, reasons = get_verdict(cur)
                
                # --- FUNDAMENTALS SIDEBAR (NEW) ---
                if not final_ticker.startswith("^"): # Don't fetch for Indices
                    fund = get_fundamentals(final_ticker)
                    if fund:
                        st.sidebar.markdown("---")
                        st.sidebar.subheader("🏢 Company Profile")
                        st.sidebar.write(f"**Sector:** {fund['sector']}")
                        
                        # Color code PE
                        pe = fund['pe']
                        pe_color = "red" if pe > 80 else "orange" if pe > 30 else "green"
                        st.sidebar.markdown(f"**P/E Ratio:** :{pe_color}[{pe:.2f}]")
                        
                        beta = fund['beta']
                        st.sidebar.write(f"**Beta (Volatility):** {beta:.2f}")
                        
                        if fund['market_cap'] > 0:
                            mcap_cr = fund['market_cap'] / 10000000
                            st.sidebar.write(f"**Mkt Cap:** ₹{mcap_cr:,.0f} Cr")
                
                # --- THE BIG ANSWER BOX ---
                st.markdown(f"""
                <div style="text-align: center; border: 4px solid {v_color}; padding: 20px; border-radius: 10px; background-color: #0e1117;">
                    <h1 style="color: {v_color}; font-size: 60px; margin:0;">{verdict}</h1>
                    <h3 style="color: white; margin:0;">{subtitle}</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # --- DETAILS ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"₹{cur['Close']:.2f}")
                col2.metric("Stochastic %K", f"{cur['STOCH_K']:.1f}", help="Below 20 = Buy, Above 80 = Sell")
                col3.metric("VWAP", f"₹{cur['VWAP']:.2f}", help="Fair Price")
                
                # --- STRATEGY CHECKLIST ---
                with st.expander("See Strategy Logic (Why?)", expanded=True):
                    for r in reasons:
                        st.write(r)

                # --- PIVOT POINTS (TARGETS) - NEW ---
                st.markdown("---")
                st.subheader("🎯 Support & Resistance (Targets)")
                pivots = calculate_pivots(df)
                p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
                p_col1.metric("Support 2", f"₹{pivots['S2']:.2f}")
                p_col2.metric("Support 1", f"₹{pivots['S1']:.2f}")
                p_col3.metric("PIVOT POINT", f"₹{pivots['P']:.2f}")
                p_col4.metric("Resistance 1", f"₹{pivots['R1']:.2f}")
                p_col5.metric("Resistance 2", f"₹{pivots['R2']:.2f}")
                st.caption("Use Support as Stop Loss and Resistance as Target Booking levels.")

                # --- CHARTS ---
                st.subheader("Price Analysis (VWAP & Bollinger)")
                fig1 = go.Figure()
                fig1.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                fig1.add_trace(go.Scatter(x=df['Date'], y=df['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                fig1.add_trace(go.Scatter(x=df['Date'], y=df['BB_UP'], line=dict(color='gray', dash='dot'), name='Upper Band'))
                fig1.add_trace(go.Scatter(x=df['Date'], y=df['BB_LO'], line=dict(color='gray', dash='dot'), name='Lower Band'))
                fig1.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader("Cyclical Analysis (Stochastic)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df['Date'], y=df['STOCH_K'], line=dict(color='cyan', width=2), name='Stoch %K'))
                fig2.add_trace(go.Scatter(x=df['Date'], y=df['STOCH_D'], line=dict(color='blue', width=1), name='Stoch %D'))
                fig2.add_shape(type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=80, y1=80, line=dict(color="red", dash="dash"))
                fig2.add_shape(type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=20, y1=20, line=dict(color="green", dash="dash"))
                fig2.update_layout(height=250, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig2, use_container_width=True)
                
                # --- NEWS SECTION ---
                st.markdown("---")
                st.subheader(f"📰 Trending News: {news_query}")
                news = get_news(news_query)
                if news:
                    for n in news:
                        clean_date = n['date'].split("+")[0]
                        st.markdown(f"**[{n['title']}]({n['link']})**")
                        st.caption(f"📅 {clean_date}")
                else:
                    st.info("No recent news found (or connection blocked).")

                # --- STRATEGY EXPLANATION ---
                st.markdown("---")
                st.markdown("### 📘 Strategies Included")
                with st.expander("1. The Trend Master (SMA 200)", expanded=False):
                    st.write("**Buy:** Price > 200 SMA (Uptrend). **Sell:** Price < 200 SMA (Downtrend).")
                with st.expander("2. The Banker's Price (VWAP)", expanded=False):
                    st.write("**Buy:** Price > VWAP (Buyers in control). **Sell:** Price < VWAP (Sellers in control).")
                with st.expander("3. The Sniper Scope (Stochastic + RSI)", expanded=False):
                    st.write("**Buy:** Both Oversold. **Sell:** Both Overbought.")
                with st.expander("4. The Volatility Trap (Bollinger Bands)", expanded=False):
                    st.write("**Buy:** Price touches Lower Band. **Sell:** Price touches Upper Band.")

            else: st.error("Not enough data.")
        else: st.error("Stock not found.")
else:
    st.info("👈 Select a stock and click 'Run Sniper Analysis' in the sidebar.")