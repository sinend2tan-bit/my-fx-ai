# FXバックグラウンド分析・学習検証（参考情報のみ・自動発注なし）
# Discord通知は行わず、実行レポートを scan_state.json に保存します。
import json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

JST = ZoneInfo('Asia/Tokyo')
STATE_FILE = 'scan_state.json'
PAIRS = {
    '米ドル/円 (USD/JPY)': 'USDJPY=X',
    '英ポンド/円 (GBP/JPY)': 'GBPJPY=X',
    'ユーロ/円 (EUR/JPY)': 'EURJPY=X',
    '豪ドル/円 (AUD/JPY)': 'AUDJPY=X',
    'ユーロ/米ドル (EUR/USD)': 'EURUSD=X',
}
SPREAD = {'USDJPY=X': .2, 'EURUSD=X': .4, 'GBPJPY=X': 1.0, 'EURJPY=X': .5, 'AUDJPY=X': .6}
FEATURES = ['Return_1','Return_5','Dev_SMA20','Dev_EMA200','Dev_EMA20_200','Vol_Ratio','RSI','RSI_Diff','MACD_Hist_Ratio','BB_PctB','ADX','ATR_Ratio','Upper_Wick_Ratio','Lower_Wick_Ratio','Stoch_K','CCI','ROC']

def bars(ticker, interval, period):
    source = '1h' if interval == '4h' else interval
    d = yf.download(ticker, period=period, interval=source, auto_adjust=False, progress=False, threads=False, timeout=25)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        try:
            d = d.xs(ticker, axis=1, level=-1)
        except Exception:
            try:
                d = d.xs(ticker, axis=1, level=0)
            except Exception:
                d.columns = [x[0] if isinstance(x, tuple) else x for x in d.columns]
    need = ['Open','High','Low','Close']
    if not set(need).issubset(d.columns):
        return None
    d = d[need].apply(pd.to_numeric, errors='coerce').dropna().sort_index()
    if interval == '4h':
        d = d.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    # 形成中の足は使用しない
    if len(d) > 2:
        d = d.iloc[:-1]
    if len(d) < 220:
        return None
    c, h, l, o = d.Close, d.High, d.Low, d.Open
    d['SMA_20'] = c.rolling(20, min_periods=10).mean()
    d['EMA_20'] = c.ewm(span=20, adjust=False).mean()
    d['EMA_200'] = c.ewm(span=200, adjust=False).mean()
    prev = c.shift()
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14, min_periods=7).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = (100 - 100/(1+rs)).fillna(50)
    d['RSI_Diff'] = d.RSI.diff()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d['MACD_Hist_Ratio'] = hist / c.replace(0, np.nan)
    std = c.rolling(20, min_periods=10).std()
    upper, lower = d.SMA_20 + 2*std, d.SMA_20 - 2*std
    d['BB_PctB'] = (c-lower) / (upper-lower).replace(0, np.nan)
    d['Return_1'] = c.pct_change()
    d['Return_5'] = c.pct_change(5)
    d['Dev_SMA20'] = (c-d.SMA_20) / c.replace(0, np.nan)
    d['Dev_EMA200'] = (c-d.EMA_200) / c.replace(0, np.nan)
    d['Dev_EMA20_200'] = (d.EMA_20-d.EMA_200) / c.replace(0, np.nan)
    d['ATR_Ratio'] = d.ATR / c.replace(0, np.nan)
    d['Vol_Ratio'] = d.ATR / d.ATR.rolling(50, min_periods=10).mean().replace(0, np.nan)
    up, down = h.diff(), -l.diff()
    plus = up.where((up>down)&(up>0), 0.0)
    minus = down.where((down>up)&(down>0), 0.0)
    safe = d.ATR.replace(0, np.nan)
    pdi = 100*plus.rolling(14, min_periods=7).mean()/safe
    mdi = 100*minus.rolling(14, min_periods=7).mean()/safe
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    d['ADX'] = dx.rolling(14, min_periods=7).mean().fillna(20)
    rng = (h-l).replace(0, np.nan)
    d['Upper_Wick_Ratio'] = (h-pd.concat([o,c],axis=1).max(axis=1))/rng
    d['Lower_Wick_Ratio'] = (pd.concat([o,c],axis=1).min(axis=1)-l)/rng
    lo14, hi14 = l.rolling(14,min_periods=7).min(), h.rolling(14,min_periods=7).max()
    d['Stoch_K'] = 100*(c-lo14)/(hi14-lo14).replace(0,np.nan)
    tp = (h+l+c)/3
    tps = tp.rolling(20,min_periods=10).mean()
    mad = tp.rolling(20,min_periods=10).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
    d['CCI'] = (tp-tps)/(.015*mad.replace(0,np.nan))
    d['ROC'] = c.pct_change(10)*100
    pip = .01 if 'JPY' in ticker else .0001
    spread = SPREAD.get(ticker,.5)*pip
    barrier = d.ATR.clip(lower=spread*2).to_numpy()
    cv, hv, lv = c.to_numpy(), h.to_numpy(), l.to_numpy()
    y = np.full(len(d), np.nan)
    # 先読みラベル。最後の8本は将来データ不足のため未ラベル。
    for i in range(max(0, len(d)-8)):
        upb, lob, lab = cv[i]+barrier[i]+spread, cv[i]-barrier[i]-spread, 0
        for j in range(i+1, min(i+9, len(d))):
            hu, hlo = hv[j]>=upb, lv[j]<=lob
            if hu and hlo:
                lab = 0
                break
            if hu:
                lab = 1
                break
            if hlo:
                lab = -1
                break
        y[i] = lab
    d['Target'] = y
    return d.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['ATR'])

def trend(d):
    if d is None or d.empty:
        return '不明'
    c, e20, e200 = float(d.Close.iloc[-1]), float(d.EMA_20.iloc[-1]), float(d.EMA_200.iloc[-1])
    return '上昇' if c>e20>e200 else '下降' if c<e20<e200 else '中立'

def analyze(d):
    fallback = {'dir':'様子見','buy':0,'sell':0,'quality':False,'bal':0,'baseline_bal':0,'samples':0}
    if d is None or len(d)<180:
        return fallback
    xa = d.reindex(columns=FEATURES).replace([np.inf,-np.inf],np.nan)
    labels = pd.to_numeric(d.Target, errors='coerce')
    mask = labels.notna() & xa.notna().all(axis=1)
    X, y = xa.loc[mask], labels.loc[mask].astype(int)
    if len(X)<160 or y.nunique()<2:
        return fallback
    tn = min(120,max(40,int(len(X)*.2)))
    split = len(X)-tn
    end = max(1,split-8)  # ラベルの先読み区間を検証境界から除外
    xt, yt, xv, yv = X.iloc[:end], y.iloc[:end], X.iloc[split:], y.iloc[split:]
    if len(xt)<100 or len(xv)<20 or yt.nunique()<2:
        return fallback
    ev = RandomForestClassifier(n_estimators=220,max_depth=7,min_samples_leaf=8,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
    ev.fit(xt,yt,sample_weight=compute_sample_weight(class_weight='balanced',y=yt))
    bal = balanced_accuracy_score(yv,ev.predict(xv))*100
    base = DummyClassifier(strategy='most_frequent').fit(xt,yt)
    basebal = balanced_accuracy_score(yv,base.predict(xv))*100
    quality = bool(bal>=max(38,basebal+3))
    model = RandomForestClassifier(n_estimators=260,max_depth=7,min_samples_leaf=8,class_weight='balanced_subsample',random_state=43,n_jobs=-1)
    model.fit(X,y,sample_weight=compute_sample_weight(class_weight='balanced',y=y))
    latest = xa.dropna()
    if latest.empty:
        return fallback
    probs = {int(k):float(v) for k,v in zip(model.classes_,model.predict_proba(latest.iloc[[-1]])[0])}
    buy, sell = probs.get(1,0), probs.get(-1,0)
    rsi = float(d.RSI.iloc[-1]); direction = '様子見'
    if quality and buy>=.53 and buy>sell and rsi<78:
        direction = '買い'
    elif quality and sell>=.53 and sell>buy and rsi>22:
        direction = '売り'
    return {'dir':direction,'buy':round(buy*100),'sell':round(sell*100),'quality':quality,'bal':round(bal,1),'baseline_bal':round(basebal,1),'samples':int(len(X))}

def main():
    old = {}
    try:
        with open(STATE_FILE,encoding='utf-8') as f:
            old = json.load(f)
    except (OSError,ValueError,TypeError):
        pass
    interval = os.getenv('FX_SCAN_INTERVAL','1h')
    interval = interval if interval in ('1h','4h') else '1h'
    period = '60d' if interval=='1h' else '90d'
    report, metrics = [], {}
    for label,ticker in PAIRS.items():
        try:
            cur = bars(ticker,interval,period)
            four = bars(ticker,'4h','90d')
            daily = bars(ticker,'1d','2y')
            one = cur if interval=='1h' else bars(ticker,'1h','60d')
            if any(x is None for x in (cur,four,daily,one)):
                report.append(f'{label}: データ不足・判定保留')
                metrics[ticker] = {'status':'insufficient_data'}
                continue
            a = analyze(cur); direction = a['dir']; ts = [trend(one),trend(four),trend(daily)]
            matchdir = {'買い':'上昇','売り':'下降'}.get(direction)
            matching = sum(x==matchdir for x in ts) if matchdir else 0
            opposing = (direction=='買い' and '下降' in ts) or (direction=='売り' and '上昇' in ts)
            stamp = pd.Timestamp(cur.index[-1])
            stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
            age = (pd.Timestamp.now(tz='UTC')-stamp).total_seconds()/60
            pip = .01 if 'JPY' in ticker else .0001
            enough = float(cur.ATR.iloc[-1])/pip >= SPREAD.get(ticker,.5)*3
            if age>(210 if interval=='1h' else 600): decision,reason='判定保留','データが古い可能性'
            elif not a['quality']: decision,reason='見送り','時系列検証の品質条件を満たさない'
            elif not enough: decision,reason='見送り','推定スプレッド負担が大きい可能性'
            elif direction=='様子見': decision,reason='様子見','明確なAI方向シグナルなし'
            elif opposing: decision,reason='方向不一致・見送り','時間足の一部が逆方向'
            elif matching<2: decision,reason='様子見','時間足一致が不十分'
            else: decision,reason='候補あり',f'AI方向と複数時間足が一致（{matching}/3）'
            close = float(cur.Close.iloc[-1])
            barstamp = stamp.tz_convert(JST).strftime('%Y-%m-%d %H:%M')
            report.append(f'{label}: {decision} / {direction} / 買い{a["buy"]}%・売り{a["sell"]}% / 価格参考 {close:.5f} / {reason}')
            metrics[ticker] = {'pair':label,'status':'ok','decision':decision,'direction':direction,'close':close,'bar_time_jst':barstamp,'data_age_minutes':round(age,1),'validation_balanced_accuracy_pct':a['bal'],'baseline_balanced_accuracy_pct':a['baseline_bal'],'quality_gate_passed':a['quality'],'training_samples':a['samples'],'timeframe_trends':ts}
        except Exception as e:
            report.append(f'{label}: 取得・分析失敗 ({type(e).__name__})')
            metrics[ticker] = {'status':'error','error_type':type(e).__name__}
    state = {
        'schema_version':2,
        'last_run_jst':datetime.now(JST).isoformat(),
        'mode':'analysis_and_validation_no_discord_no_auto_orders',
        'interval':interval,
        'summary':report,
        'metrics':metrics,
        'note':'このレポートは検証記録です。学習済みモデルを永続保存・app.pyへ配布する機能はまだありません。予測確率・検証指標は将来の利益を保証しません。',
        'history':(old.get('history',[]) if isinstance(old.get('history',[]),list) else [])[-99:]
    }
    state['history'].append({'time_jst':state['last_run_jst'],'metrics':metrics})
    with open(STATE_FILE,'w',encoding='utf-8') as f:
        json.dump(state,f,ensure_ascii=False,indent=2)
    print('\n'.join(report))
    print('Completed; Discord notifications disabled; report saved to scan_state.json')

if __name__ == '__main__':
    main()
