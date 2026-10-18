import html, json, os, time
from datetime import datetime
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    import requests
except Exception:
    requests = None
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="AI FX分析ツール", page_icon="📈", layout="wide")
JST=ZoneInfo("Asia/Tokyo")
SETTINGS_FILE="user_settings.json"; FEEDBACK_FILE="trade_feedback.json"
LOOKAHEAD_BARS=8; MODEL_VERSION=5
DEFAULT_SETTINGS={"account_balance":500000,"quantity_wan":0.10,"auto_refresh":False,
 "risk_percent":1.0,"rr_ratio":1.5,"trap_width":30,"ranges":{},"trade_memo":"","webhook_url":""}
PAIRS={"米ドル/円 (USD/JPY)":"USDJPY=X","英ポンド/円 (GBP/JPY)":"GBPJPY=X",
 "ユーロ/円 (EUR/JPY)":"EURJPY=X","豪ドル/円 (AUD/JPY)":"AUDJPY=X","ユーロ/米ドル (EUR/USD)":"EURUSD=X"}
TIMEFRAMES={"5分足":{"interval":"5m","period":"7d"},"15分足":{"interval":"15m","period":"30d"},
 "1時間足":{"interval":"1h","period":"60d"},"4時間足":{"interval":"4h","period":"90d"}}
SPREAD={"USDJPY=X":0.2,"EURUSD=X":0.4,"GBPJPY=X":1.0,"EURJPY=X":0.5,"AUDJPY=X":0.6}
FEATURES=["Return_1","Return_5","Dev_SMA20","Dev_EMA200","Dev_EMA20_200","Vol_Ratio","RSI","RSI_Diff",
 "MACD_Hist_Ratio","BB_PctB","ADX","ATR_Ratio","Upper_Wick_Ratio","Lower_Wick_Ratio","Stoch_K","CCI","ROC"]
LABELS={x:x.replace("_"," ") for x in FEATURES}
LABELS.update({"Return_1":"1本リターン","Return_5":"5本リターン","Dev_SMA20":"SMA20乖離",
 "Dev_EMA200":"EMA200乖離","Dev_EMA20_200":"EMA20/200乖離","Vol_Ratio":"ボラティリティ比",
 "RSI_Diff":"RSI変化","MACD_Hist_Ratio":"MACDヒストグラム比","BB_PctB":"ボリンジャー%B",
 "ATR_Ratio":"ATR比","Upper_Wick_Ratio":"上ヒゲ比率","Lower_Wick_Ratio":"下ヒゲ比率","Stoch_K":"ストキャス%K"})
st.markdown("""<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#0b1220!important;color:#e5e7eb!important}
[data-testid="stSidebar"]{background:#111827!important}
input,textarea,select,[data-baseweb="select"]>div{background:#1f2937!important;color:#fff!important}
.update-time,.param-box,.news-card{padding:12px;border:1px solid #334155;border-radius:10px;background:#111827;margin:8px 0;overflow-wrap:anywhere}
.warn-box,.alert-box{padding:12px;background:#422006;border:1px solid #f59e0b;border-radius:8px}
.badge-buy,.badge-sell,.badge-wait{padding:8px 12px;border-radius:8px;font-weight:bold;display:inline-block}
.badge-buy{background:#14532d;color:#86efac}.badge-sell{background:#7f1d1d;color:#fecaca}.badge-wait{background:#374151;color:#e5e7eb}
[data-testid="stTabs"] [role="tablist"]{display:flex;flex-wrap:wrap;gap:5px;background:#172033;padding:7px;border-radius:10px}
[data-testid="stTabs"] [role="tab"]{flex:1 1 145px;min-width:0;background:#111827;border-radius:6px;padding:8px;font-size:13px}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding-left:.7rem;padding-right:.7rem}[data-testid="stTabs"] [role="tab"]{flex:1 1 100%;font-size:14px}}
</style>""",unsafe_allow_html=True)

def clean(s): return pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan)
def load_settings():
    d=DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE,encoding="utf-8") as f:
            x=json.load(f)
        if isinstance(x,dict): d.update({k:v for k,v in x.items() if k in d})
    except (OSError,ValueError,TypeError): pass
    if not isinstance(d.get("ranges"),dict): d["ranges"]={}
    return d
def save_settings():
    try:
        with open(SETTINGS_FILE,"w",encoding="utf-8") as f:
            json.dump({k:st.session_state.get(k,v) for k,v in DEFAULT_SETTINGS.items()},f,ensure_ascii=False,indent=2,default=str)
    except OSError: st.warning("設定を保存できませんでした。")
def reset_settings():
    for k,v in DEFAULT_SETTINGS.items(): st.session_state[k]=v.copy() if isinstance(v,dict) else v
    try: os.remove(SETTINGS_FILE)
    except OSError: pass
def load_feedback():
    try:
        with open(FEEDBACK_FILE,encoding="utf-8") as f:
            x=json.load(f)
        return x if isinstance(x,list) else []
    except (OSError,ValueError,TypeError): return []
def save_feedback(x):
    try:
        with open(FEEDBACK_FILE,"w",encoding="utf-8") as f: json.dump(x,f,ensure_ascii=False,indent=2)
        return True
    except OSError: st.error("記録を保存できませんでした。"); return False
def flatten(df):
    if df is None or df.empty: return pd.DataFrame()
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        names=[]
        for c in x.columns:
            parts=[str(a) for a in c]
            names.append(next((a for a in parts if a.lower() in {"open","high","low","close","volume","adj close"}),parts[-1]))
        x.columns=names
    x.columns=[str(c).replace(" ","_").title() for c in x.columns]
    return x
def tokyo(df):
    if df is None or df.empty: return df
    x=df.copy(); idx=pd.to_datetime(x.index,errors="coerce")
    if idx.tz is None: idx=idx.tz_localize("UTC")
    x.index=idx.tz_convert(JST); return x
@st.cache_data(ttl=300,show_spinner=False)
def usdjpy():
    try:
        d=flatten(yf.download("USDJPY=X",period="5d",interval="1d",progress=False,threads=False))
        v=clean(d["Close"]).dropna()
        return float(v.iloc[-1]) if len(v) and v.iloc[-1]>0 else None
    except Exception: return None
def indicators(df):
    c,h,l,o=df.Close,df.High,df.Low,df.Open
    df["SMA_20"]=c.rolling(20,min_periods=10).mean(); df["EMA_20"]=c.ewm(span=20,adjust=False).mean()
    df["EMA_200"]=c.ewm(span=200,adjust=False).mean()
    prev=c.shift(); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    df["ATR"]=tr.rolling(14,min_periods=7).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14,min_periods=7).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=7).mean()
    df["RSI"]=(100-100/(1+gain/loss.replace(0,np.nan))).fillna(50); df["RSI_Diff"]=df.RSI.diff()
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    df["MACD"]=macd; df["MACD_Signal"]=macd.ewm(span=9,adjust=False).mean()
    df["MACD_Hist"]=macd-df.MACD_Signal; df["MACD_Hist_Ratio"]=df.MACD_Hist/c.replace(0,np.nan)
    sd=c.rolling(20,min_periods=10).std(); df["Upper_Band"]=df.SMA_20+2*sd; df["Lower_Band"]=df.SMA_20-2*sd
    df["BB_PctB"]=(c-df.Lower_Band)/(df.Upper_Band-df.Lower_Band).replace(0,np.nan)
    df["Return_1"]=c.pct_change(); df["Return_5"]=c.pct_change(5)
    df["Dev_SMA20"]=(c-df.SMA_20)/c.replace(0,np.nan); df["Dev_EMA200"]=(c-df.EMA_200)/c.replace(0,np.nan)
    df["Dev_EMA20_200"]=(df.EMA_20-df.EMA_200)/c.replace(0,np.nan); df["ATR_Ratio"]=df.ATR/c.replace(0,np.nan)
    df["Vol_Ratio"]=df.ATR/df.ATR.rolling(50,min_periods=10).mean().replace(0,np.nan)
    up,down=h.diff(),-l.diff(); plus=up.where((up>down)&(up>0),0); minus=down.where((down>up)&(down>0),0)
    pdi=100*plus.rolling(14,min_periods=7).mean()/df.ATR.replace(0,np.nan)
    mdi=100*minus.rolling(14,min_periods=7).mean()/df.ATR.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); df["ADX"]=dx.rolling(14,min_periods=7).mean().fillna(20)
    rng=(h-l).replace(0,np.nan); df["Upper_Wick_Ratio"]=(h-pd.concat([o,c],axis=1).max(axis=1))/rng
    df["Lower_Wick_Ratio"]=(pd.concat([o,c],axis=1).min(axis=1)-l)/rng
    lo,hi=l.rolling(14,min_periods=7).min(),h.rolling(14,min_periods=7).max()
    df["Stoch_K"]=100*(c-lo)/(hi-lo).replace(0,np.nan)
    tp=(h+l+c)/3; avg=tp.rolling(20,min_periods=10).mean()
    mad=tp.rolling(20,min_periods=10).apply(lambda z:np.mean(np.abs(z-np.mean(z))),raw=True)
    df["CCI"]=(tp-avg)/(.015*mad.replace(0,np.nan)); df["ROC"]=c.pct_change(10)*100
    return df
@st.cache_data(ttl=60,show_spinner=False)
def get_data(ticker,period,interval):
    try:
        source_interval="1h" if interval=="4h" else interval
        d=flatten(yf.download(ticker,period=period,interval=source_interval,auto_adjust=False,progress=False,threads=False,timeout=20))
        req=["Open","High","Low","Close"]
        if d.empty or not set(req).issubset(d.columns): return None
        d=d[req+(["Volume"] if "Volume" in d else [])].apply(pd.to_numeric,errors="coerce").dropna(subset=req).sort_index()
        if interval=="4h":
            agg={"Open":"first","High":"max","Low":"min","Close":"last"}
            if "Volume" in d: agg["Volume"]="sum"
            d=d.resample("4h",origin="start_day").agg(agg).dropna(subset=req)
        # Yahooの最終足は未確定の可能性があるため保守的に除外
        if len(d): d=d.iloc[:-1]
        if len(d)<80: return None
        d=indicators(d)
        pip=.01 if "JPY" in ticker else .0001; spread=SPREAD.get(ticker,.5)*pip
        atr=clean(d.ATR).to_numpy(float); c=clean(d.Close).to_numpy(float)
        hi=clean(d.High).to_numpy(float); lo=clean(d.Low).to_numpy(float)
        y=np.full(len(d),np.nan)
        # 同一足で上下両方に到達した曖昧例は除外。最後の8本は未ラベル。
        for i in range(max(0,len(d)-LOOKAHEAD_BARS)):
            barrier=c[i]+max(atr[i] if np.isfinite(atr[i]) else spread*2,spread*2)+spread
            upper=barrier; lower=c[i]-(barrier-c[i]); label=0; ambiguous=False
            for j in range(i+1,min(i+LOOKAHEAD_BARS+1,len(d))):
                u=hi[j]>=upper; dn=lo[j]<=lower
                if u and dn: ambiguous=True; break
                if u: label=1; break
                if dn: label=-1; break
            y[i]=np.nan if ambiguous else label
        d["Target"]=y; d=d.replace([np.inf,-np.inf],np.nan)
        return d.dropna(subset=FEATURES+["Close","ATR"])
    except Exception: return None
def model_file(ticker,interval): return f"model_{ticker.replace('=X','').replace('/','')}_{interval}.joblib"
def tsnorm(v):
    try:
        t=pd.Timestamp(v); t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.isoformat()
    except Exception: return str(v)
def analyze(df,ticker,interval,force=False):
    fallback=("WAIT (様子見)",0.,0,0.,"データ不足",{"buy":0,"sell":0,"wait":100},pd.Series(dtype=float),pd.Series(dtype=float))
    if df is None or len(df)<180: return fallback
    try:
        xx=df.reindex(columns=FEATURES).replace([np.inf,-np.inf],np.nan); lab=pd.to_numeric(df.Target,errors="coerce")
        mask=lab.notna()&xx.notna().all(axis=1); X=xx.loc[mask]; y=lab.loc[mask].astype(int)
        if len(X)<160 or y.nunique()<2: return fallback
        n=min(120,max(40,int(len(X)*.2))); split=len(X)-n; end=max(1,split-LOOKAHEAD_BARS)
        xt,yt=X.iloc[:end],y.iloc[:end]; xv,yv=X.iloc[split:],y.iloc[split:]
        if len(xt)<100 or len(xv)<20 or yt.nunique()<2: return fallback
        ev=RandomForestClassifier(n_estimators=180,max_depth=7,min_samples_leaf=8,class_weight="balanced_subsample",random_state=42,n_jobs=-1)
        ev.fit(xt,yt,sample_weight=compute_sample_weight("balanced",yt)); pred=ev.predict(xv)
        acc=accuracy_score(yv,pred)*100
        hist=pd.Series(np.nan,index=df.index,dtype=float); hist.loc[xv.index]=pred.astype(float)
        path=model_file(ticker,interval); model=None
        if not force and os.path.exists(path):
            try:
                saved=joblib.load(path)
                if saved.get("version")==MODEL_VERSION and saved.get("features")==FEATURES and time.time()-os.path.getmtime(path)<14400:
                    model=saved.get("model")
            except Exception: pass
        if model is None:
            model=RandomForestClassifier(n_estimators=220,max_depth=7,min_samples_leaf=8,class_weight="balanced_subsample",random_state=43,n_jobs=-1)
            model.fit(X,y,sample_weight=compute_sample_weight("balanced",y))
            try: joblib.dump({"version":MODEL_VERSION,"features":FEATURES,"model":model,"trained_at":datetime.now(JST).isoformat()},path)
            except Exception: pass
        latest=xx.dropna().iloc[[-1]]; raw=model.predict_proba(latest)[0]
        pm={int(c):float(p) for c,p in zip(model.classes_,raw)}; bp,sp=pm.get(1,0),pm.get(-1,0)
        probs={"buy":round(bp*100),"sell":round(sp*100),"wait":round(pm.get(0,0)*100)}
        rsi=float(df.RSI.iloc[-1]); adx=float(df.ADX.iloc[-1])
        status="BUY (買い)" if bp>=.53 and bp>sp and rsi<78 else "SELL (売り)" if sp>=.53 and sp>bp and rsi>22 else "WAIT (様子見)"
        market=("トレンド相場" if adx>=25 else "レンジ相場")+("・高ボラ" if float(df.ATR_Ratio.iloc[-1])>.01 else "")
        imp=pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False)
        return status,max(bp,sp)*100,len(yv),acc,market,probs,imp,hist
    except Exception as e:
        st.warning(f"AI分析に失敗したためWAITにしています ({type(e).__name__})"); return fallback

if "settings_loaded" not in st.session_state:
    for k,v in load_settings().items():
        if k not in st.session_state: st.session_state[k]=v
    st.session_state.settings_loaded=True

st.title("Pro FX Analyzer & Signal Pro")
st.caption("価格分析・注文補助専用。証券口座への接続・自動発注は行いません。")
c1,c2=st.columns(2)
selected_label=c1.selectbox("通貨ペア",list(PAIRS),key="selected_pair_label")
tf_label=c2.selectbox("時間足",list(TIMEFRAMES),key="selected_tf_label")
ticker=PAIRS[selected_label]; tf=TIMEFRAMES[tf_label]; is_jpy="JPY" in ticker
pip=.01 if is_jpy else .0001; fmt="%.3f" if is_jpy else "%.5f"; spread=SPREAD.get(ticker,.5)
st.sidebar.header("資金・リスク")
st.sidebar.number_input("口座資金 (円)",min_value=10000,step=50000,key="account_balance",on_change=save_settings)
st.sidebar.number_input("基本数量 (万通貨)",min_value=.01,step=.01,key="quantity_wan",on_change=save_settings)
st.sidebar.number_input("許容リスク (%)",min_value=.1,max_value=5.,step=.1,key="risk_percent",on_change=save_settings)
st.sidebar.number_input("リスクリワード比",min_value=.5,max_value=5.,step=.1,key="rr_ratio",on_change=save_settings)
st.sidebar.header("AIモデル")
if st.sidebar.button("モデルを強制再学習"):
    try: os.remove(model_file(ticker,tf["interval"]))
    except OSError: pass
    st.session_state.force_retrain_once=True; st.rerun()
if os.path.exists(model_file(ticker,tf["interval"])):
    st.sidebar.caption("最終学習: "+datetime.fromtimestamp(os.path.getmtime(model_file(ticker,tf["interval"])),JST).strftime("%m/%d %H:%M"))
if HAS_AUTOREFRESH:
    st.sidebar.checkbox("60秒ごとに自動更新",key="auto_refresh",on_change=save_settings)
    if st.session_state.auto_refresh: st_autorefresh(interval=60000,key="autorefresh")
with st.sidebar.expander("通知設定"):
    st.text_input("Webhook URL",key="webhook_url",on_change=save_settings)
    st.caption("共有環境ではURLを保存しないでください。")
if st.sidebar.button("設定を保存"): save_settings()
with st.sidebar.expander("設定バックアップ・復元"):
    bk=load_settings(); bk["webhook_url"]=""
    st.download_button("バックアップJSON",json.dumps(bk,ensure_ascii=False,indent=2),"fx_settings.json","application/json")
    up=st.file_uploader("設定JSONを復元",type=["json"])
    if up:
        try:
            v=json.load(up)
            if isinstance(v,dict):
                v={k:v[k] for k in DEFAULT_SETTINGS if k in v}; v["webhook_url"]=""
                with open(SETTINGS_FILE,"w",encoding="utf-8") as f: json.dump(v,f,ensure_ascii=False,indent=2)
                st.success("設定を復元しました。再読み込みしてください。")
        except Exception: st.error("設定ファイルを読み込めません。")
st.sidebar.button("設定を初期値に戻す",on_click=reset_settings)
if st.sidebar.button("最新データに更新"): st.cache_data.clear(); st.rerun()

with st.spinner("相場データを取得中..."):
    data=get_data(ticker,tf["period"],tf["interval"]); d4=get_data(ticker,"90d","4h")
    news=[]
    try:
        for n in (yf.Ticker(ticker).news or [])[:12]:
            if not isinstance(n,dict): continue
            x=n.get("content") if isinstance(n.get("content"),dict) else n
            title=x.get("title") or n.get("title") or "タイトルなし"
            txt=(title+" "+str(x.get("summary",""))).lower()
            impact="高" if any(k in txt for k in ["fomc","cpi","inflation","payroll","boj","gdp","interest rate"]) else "中" if any(k in txt for k in ["yen","dollar","treasury","employment"]) else "低"
            provider=x.get("provider"); publisher=provider.get("displayName","") if isinstance(provider,dict) else n.get("publisher","")
            url=(x.get("canonicalUrl") or {}).get("url","#") if isinstance(x.get("canonicalUrl"),dict) else n.get("link","#")
            news.append({"title":str(title),"publisher":str(publisher),"impact":impact,"url":str(url)})
    except Exception: pass
if d4 is None: d4=data
if data is None or data.empty: st.error("データ不足または取得エラーです。時間をおいて更新してください。"); st.stop()
last=tokyo(data).index[-1].strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">分析対象最終足: <b>{last} JST</b>（最終足を保守的に除外）</div>',unsafe_allow_html=True)
price=float(data.Close.iloc[-1]); atr=float(data.ATR.iloc[-1])
force=bool(st.session_state.pop("force_retrain_once",False))
status,conf,n_test,accuracy,market,probs,importance,history=analyze(data,ticker,tf["interval"],force)
a,b,c,d=st.columns(4); a.metric("確定足レート",fmt%price)
b.markdown("**AI参考アクション**"); b.markdown('<span class="badge-buy">BUY</span>' if status.startswith("BUY") else '<span class="badge-sell">SELL</span>' if status.startswith("SELL") else '<span class="badge-wait">WAIT</span>',unsafe_allow_html=True)
c.metric("テスト分類一致率",f"{accuracy:.1f}%",f"検証 {n_test}本"); d.metric("相場環境",market,f"ATR {atr/pip:.1f} pips")
st.caption("分類一致率は実際の取引勝率や利益率ではありません。")

if d4 is not None and not d4.empty:
    swing_hi=float(d4.High.tail(300).max()); swing_lo=float(d4.Low.tail(300).min()); atr4=float(d4.ATR.dropna().iloc[-1]) if d4.ATR.notna().any() else atr
else: swing_hi=price*1.03; swing_lo=price*.97; atr4=atr
ranges=st.session_state.get("ranges",{})
if ticker not in ranges or not isinstance(ranges[ticker],dict) or ranges[ticker].get("upper",0)<=ranges[ticker].get("lower",0):
    ranges[ticker]={"lower":swing_lo,"upper":swing_hi}; st.session_state.ranges=ranges; save_settings()
kl,ku="lower_"+ticker,"upper_"+ticker
if kl not in st.session_state: st.session_state[kl]=float(ranges[ticker]["lower"])
if ku not in st.session_state: st.session_state[ku]=float(ranges[ticker]["upper"])
with st.expander("リピートレンジ調整"):
    r1,r2=st.columns(2)
    r1.number_input("レンジ下限",key=kl,format=fmt,on_change=lambda:None)
    r2.number_input("レンジ上限",key=ku,format=fmt,on_change=lambda:None)
    if st.button("4時間足の高値・安値からレンジを再計算"):
        st.session_state[kl]=swing_lo; st.session_state[ku]=swing_hi; st.rerun()
user_lower=min(float(st.session_state[kl]),float(st.session_state[ku])); user_upper=max(float(st.session_state[kl]),float(st.session_state[ku]))
user_half=(user_lower+user_upper)/2; buffer=max(10.,atr4/pip*1.5); buy_sl=user_lower-buffer*pip; sell_sl=user_upper+buffer*pip

tab_chart,tab_single,tab_news,tab_repeat,tab_ai=st.tabs(["📈 メインチャート","⚡ 単発トレード","📰 ニュース分析","📋 リピート設定","🤖 AI詳細・ノート"])
with tab_chart:
    n=st.slider("表示本数",30,300,120,10,key="chart_n")
    sub=st.selectbox("サブ指標",["RSI (14)","MACD","なし"])
    m1,m2,m3,m4=st.columns(4)
    show_ma=m1.checkbox("移動平均線",True); show_bb=m2.checkbox("ボリンジャーバンド",True)
    show_range=m3.checkbox("リピートレンジ",tf_label=="4時間足"); show_sig=m4.checkbox("検証シグナル",True)
    chart=tokyo(data.tail(n)); x=chart.index.strftime("%m/%d %H:%M")
    has=sub!="なし"; fig=make_subplots(rows=2 if has else 1,cols=1,shared_xaxes=True,row_heights=[.75,.25] if has else [1],vertical_spacing=.06)
    fig.add_trace(go.Candlestick(x=x,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name="価格"),row=1,col=1)
    if show_ma:
        fig.add_trace(go.Scatter(x=x,y=chart.SMA_20,name="SMA20"),row=1,col=1); fig.add_trace(go.Scatter(x=x,y=chart.EMA_200,name="EMA200"),row=1,col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=x,y=chart.Upper_Band,name="BB上限",line=dict(dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=x,y=chart.Lower_Band,name="BB下限",line=dict(dash="dot")),row=1,col=1)
    if show_sig and history.notna().any():
        h=history.dropna(); common=chart.index.intersection(h.index)
        for val,name,symbol in [(1,"検証BUY","triangle-up"),(-1,"検証SELL","triangle-down")]:
            ix=common[h.loc[common].to_numpy()==val]
            if len(ix):
                yy=chart.loc[ix,"Low"]-atr*.25 if val==1 else chart.loc[ix,"High"]+atr*.25
                fig.add_trace(go.Scatter(x=tokyo(chart.loc[ix]).index.strftime("%m/%d %H:%M"),y=yy,mode="markers",name=name,marker=dict(symbol=symbol,size=9)),row=1,col=1)
    if show_range:
        fig.add_hrect(y0=user_lower,y1=user_upper,fillcolor="rgba(56,189,248,.08)",line_width=0,row=1,col=1)
        for yv,label in [(user_lower,"下限"),(user_half,"中央"),(user_upper,"上限"),(buy_sl,"買SL目安"),(sell_sl,"売SL目安")]:
            fig.add_hline(y=yv,annotation_text=label,line_dash="dot",row=1,col=1)
    if sub=="RSI (14)":
        fig.add_trace(go.Scatter(x=x,y=chart.RSI,name="RSI"),row=2,col=1); fig.add_hline(y=70,row=2,col=1); fig.add_hline(y=30,row=2,col=1)
        fig.update_yaxes(range=[0,100],row=2,col=1)
    elif sub=="MACD":
        fig.add_trace(go.Scatter(x=x,y=chart.MACD,name="MACD"),row=2,col=1); fig.add_trace(go.Scatter(x=x,y=chart.MACD_Signal,name="Signal"),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=500,margin=dict(l=5,r=40,t=25,b=5),xaxis_rangeslider_visible=False,hovermode="x unified")
    fig.update_xaxes(type="category"); fig.update_yaxes(side="right",tickformat=fmt.replace("%",""),row=1,col=1)
    st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displaylogo":False})

with tab_single:
    st.markdown("##### 単発トレード条件の計算（発注しません）")
    rate=price if ticker=="USDJPY=X" else usdjpy()
    pip_value=100. if is_jpy else (.0001*rate*10000 if rate else 0.)
    bal=float(st.session_state.account_balance); risk=bal*float(st.session_state.risk_percent)/100
    slp=max(1.,round(atr/pip*1.5,1)); rr=float(st.session_state.rr_ratio); tpp=round(slp*rr,1)
    loss_per=(slp+spread)*pip_value; qty=float(np.floor((risk/loss_per if loss_per else 0)*100)/100)
    qty=qty if qty>=.01 else 0.
    direction="BUY" if status.startswith("BUY") else "SELL" if status.startswith("SELL") else "WAIT"
    buy=direction=="BUY"; slprice=price-slp*pip if buy else price+slp*pip; tpprice=price+tpp*pip if buy else price-tpp*pip
    if direction=="WAIT" or qty==0 or pip_value<=0: st.warning("WAIT判定、または数量計算ができないため見送り候補です。")
    else:
        st.markdown(f"""<div class="param-box">方向: <b>{direction}</b><br>エントリー目安: <code>{fmt%price}</code><br>TP: <code>{fmt%tpprice}</code> (+{tpp:.1f} pips)<br>SL: <code>{fmt%slprice}</code> (-{slp:.1f} pips)<br>数量: <code>{qty:.2f} 万通貨</code><br>損失概算: <code>{(slp+spread)*pip_value*qty:,.0f}円</code><br>利益概算: <code>{(tpp-spread)*pip_value*qty:,.0f}円</code></div>""",unsafe_allow_html=True)
    st.caption("スプレッドの簡易計算です。スリッページ、スワップ、手数料、約定拒否などは含みません。")
    st.markdown("##### 取引結果の記録（AI学習には自動使用しません）")
    def feedback(win):
        items=load_feedback(); stamp=tsnorm(data.index[-1])
        found=False
        for item in items:
            if item.get("ticker")==ticker and item.get("interval")==tf["interval"] and tsnorm(item.get("timestamp"))==stamp:
                item.update({"is_win":win,"updated_at":datetime.now(JST).isoformat()}); found=True
        if not found: items.append({"ticker":ticker,"interval":tf["interval"],"timestamp":stamp,"is_win":win,"signal":status,"reference_close":price,"created_at":datetime.now(JST).isoformat()})
        return save_feedback(items)
    f1,f2=st.columns(2)
    if f1.button("👍 利益を記録",use_container_width=True):
        if feedback(True): st.success("記録しました。AI学習には反映していません。")
    if f2.button("👎 損失を記録",use_container_width=True):
        if feedback(False): st.warning("記録しました。AI学習には反映していません。")
    st.markdown("##### 手動条件の計算")
    side=st.radio("方向",["BUY","SELL"],horizontal=True)
    entry=st.number_input("エントリー想定レート",value=float(price),format=fmt,step=pip)
    slmode=st.radio("損切幅",["ATR倍率","固定pips"],horizontal=True)
    if slmode=="ATR倍率": mult=st.slider("ATR倍率",.5,3.,1.5,.1); manual_sl=max(1.,round(atr/pip*mult,1))
    else: manual_sl=st.number_input("損切pips",min_value=1.,value=float(slp),step=1.)
    manual_rr=st.slider("RR",.5,4.,float(min(4,max(.5,rr))),.1); manual_tp=manual_sl*manual_rr
    mqty=float(np.floor((risk/((manual_sl+spread)*pip_value) if pip_value else 0)*100)/100); mqty=mqty if mqty>=.01 else 0
    isbuy=side=="BUY"; stop=entry-manual_sl*pip if isbuy else entry+manual_sl*pip; take=entry+manual_tp*pip if isbuy else entry-manual_tp*pip
    q1,q2,q3,q4=st.columns(4); q1.metric("数量",f"{mqty:.2f} 万通貨"); q2.metric("SL",fmt%stop); q3.metric("TP",fmt%take)
    q4.metric("利益概算",f"{(manual_tp-spread)*pip_value*mqty:+,.0f}円",f"損失 -{(manual_sl+spread)*pip_value*mqty:,.0f}円")
    if (manual_tp-spread)<=0: st.warning("利確幅がスプレッド以下です。")

with tab_news:
    st.markdown("##### 📰 為替ニュースとキーワード参考判定")
    st.caption("見出し語の参考表示です。本文の理解や相場反応を保証しません。")
    with st.expander("主な経済イベント",expanded=True):
        st.markdown("| イベント | 注目点 | 一般的な影響例 |\n|---|---|---|\n| FOMC | 金利・議長発言 | 金利見通しを通じドル相場に影響する場合 |\n| 米雇用統計 | 雇用・賃金 | 予想との差が重要 |\n| 米CPI | インフレ | 金融政策見通しに影響する場合 |\n| 日銀会合 | 政策・会見 | 円相場に影響する場合 |")
    if not news: st.info("ニュースを取得できませんでした。")
    for item in news:
        url=item["url"] if item["url"].startswith(("https://","http://")) else "#"
        st.markdown(f"""<div class="news-card"><b>{html.escape(item['publisher'])} · 影響キーワード: {item['impact']}</b><p><a href="{html.escape(url,quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(item['title'])} 🔗</a></p><small>方向は本文と実際の市場反応を確認してください。</small></div>""",unsafe_allow_html=True)

with tab_repeat:
    st.markdown("##### 📋 リピート注文・ハーフ＆ハーフの参考設定")
    if price<user_lower or price>user_upper: st.warning("現在レートが設定レンジの外側です。")
    ai_width=int(np.clip(round((atr4/pip)*.8),10,500))
    width=st.number_input("注文幅 / 利確幅 (pips)",10,500,int(st.session_state.get("trap_width",ai_width)),5,key="repeat_width")
    if st.button(f"AI参考値 {ai_width} pips を適用"): st.session_state.trap_width=ai_width; st.rerun()
    spacing=max(1.,float(width)); half_count=max(1,int(np.floor(abs(user_upper-user_half)/(spacing*pip)))+1); total=half_count*2
    quantity=max(0.,float(st.session_state.quantity_wan)); units=int(quantity*10000)
    rate=price if ticker=="USDJPY=X" else usdjpy(); calc_ok=is_jpy or (rate is not None and rate>0)
    pipyen=quantity*100 if is_jpy else (units*.0001*rate if calc_ok else 0.)
    buy_levels=[user_lower+i*spacing*pip for i in range(half_count)]
    sell_levels=[user_upper-i*spacing*pip for i in range(half_count)]
    loss_pips=sum(max(0,(x-buy_sl)/pip) for x in buy_levels)+sum(max(0,(sell_sl-x)/pip) for x in sell_levels)
    scenario_loss=loss_pips*pipyen
    margin_order=(price*units/25 if is_jpy else price*rate*units/25) if calc_ok and units else 0.
    margin_total=margin_order*total; ratio=scenario_loss/max(1.,bal)*100 if calc_ok else float("nan")
    st.markdown(f"""<div class="param-box">買いレンジ: <code>{fmt%user_lower}</code>～<code>{fmt%user_half}</code>（SL目安 {fmt%buy_sl}）<br>売りレンジ: <code>{fmt%user_half}</code>～<code>{fmt%user_upper}</code>（SL目安 {fmt%sell_sl}）<br>注文幅: <code>{spacing:.0f} pips</code> · 片側本数目安: <code>{half_count}</code> · 合計: <code>{total}</code></div>""",unsafe_allow_html=True)
    r1,r2,r3=st.columns(3); r1.metric("含み損シナリオ",f"{scenario_loss:,.0f}円" if calc_ok else "算出不可")
    r2.metric("必要証拠金概算",f"{margin_total:,.0f}円" if calc_ok else "算出不可"); r3.metric("資金比率",f"{ratio:.1f}%" if np.isfinite(ratio) else "算出不可")
    st.warning("損失・証拠金は単純シナリオで、最大損失や証券会社の実際の必要証拠金ではありません。価格ギャップ、スワップ、スリッページ、保有ポジション、強制決済条件を含みません。")
    turns=max(0,int((atr/(spacing*pip))*6)); net=max(0.,spacing-spread); monthly=turns*net*pipyen
    st.markdown("##### 仮定に基づく回転数シミュレーション（利益予測ではありません）")
    z1,z2,z3,z4=st.columns(4); z1.metric("仮定回数/月",f"{turns}"); z2.metric("仮定損益/月",f"{monthly:,.0f}円" if calc_ok else "算出不可")
    z3.metric("仮定比率/月",f"{monthly/max(1,bal)*100:.1f}%" if calc_ok else "算出不可"); z4.metric("単純年換算",f"{monthly*12/max(1,bal)*100:.1f}%" if calc_ok else "算出不可")

with tab_ai:
    st.markdown("##### 🤖 AIモデルの詳細")
    a,b=st.columns(2)
    with a:
        for label,key in [("BUY","buy"),("SELL","sell"),("WAIT","wait")]:
            st.write(f"{label}: {probs[key]}%"); st.progress(max(0,min(1,probs[key]/100)))
    with b:
        st.metric("時系列テスト分類一致率",f"{accuracy:.1f}%",f"検証数 {n_test}本")
        st.caption("分類一致率は取引損益の勝率ではありません。")
        st.info(f"フィードバック記録: {len(load_feedback())}件。AIの教師データとは分離しています。")
        if st.button("フィードバック履歴をリセット"):
            try: os.remove(FEEDBACK_FILE)
            except OSError: pass
            st.rerun()
        exp=history.rename("予測ラベル").rename_axis("日時").reset_index()
        st.download_button("検証シグナルCSV",exp.to_csv(index=False).encode("utf-8-sig"),f"signals_{ticker.replace('=X','')}.csv","text/csv")
    st.markdown("##### テクニカル指標")
    def val(col,default):
        s=clean(data[col]).dropna() if col in data else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else default
    st.write(f"RSI: {val('RSI',50):.1f} · ADX: {val('ADX',20):.1f} · BB %B: {val('BB_PctB',.5):.2f} · MACD hist: {val('MACD_Hist',0):.5f} · Stoch %K: {val('Stoch_K',50):.1f}")
    if len(importance):
        top=importance.head(10).sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=top.values,y=[LABELS.get(k,k) for k in top.index],orientation="h")).update_layout(template="plotly_dark",height=350),use_container_width=True)
        csv=importance.rename(index=lambda x:LABELS.get(x,x)).rename("重要度").rename_axis("指標").reset_index()
        st.download_button("特徴量重要度CSV",csv.to_csv(index=False).encode("utf-8-sig"),"feature_importance.csv","text/csv")
    st.markdown("##### 📝 トレードノート")
    st.text_area("考察・反省メモ",key="trade_memo",height=150,on_change=save_settings,placeholder="例: 経済指標発表前のため見送り。")
    memo=st.session_state.get("trade_memo","")
    if memo.strip(): st.download_button("ノートをテキスト保存",memo.encode("utf-8"),"trade_memo.txt","text/plain")

st.caption("免責: 参考情報を表示するツールであり、利益や精度を保証しません。発注前にレート、数量、証拠金、損切条件を確認してください。")
