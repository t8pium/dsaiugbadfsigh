import pandas as pd, numpy as np, json, math, os
from pathlib import Path
from numba import njit

BASE='/mnt/data/active_mnq.pkl'
OUT=Path('/mnt/data/fvg_ce_study'); OUT.mkdir(exist_ok=True)
TFS=[int(x) for x in os.environ.get('CE_TFS','1,2,3,5,10,15,30,60,120,240,360,480,720,1440').split(',') if x]
TFNAME={1:'1m',2:'2m',3:'3m',5:'5m',10:'10m',15:'15m',30:'30m',60:'1H',120:'2H',240:'4H',360:'6H',480:'8H',720:'12H',1440:'1D'}

print('loading base', flush=True)
a=pd.read_pickle(BASE).copy()
a=a.sort_values('ts_event').reset_index(drop=True)
a['base_idx']=np.arange(len(a),dtype=np.int64)
et=a.ts_event.dt.tz_convert('America/New_York')
a['_minute_et']=et.dt.hour.astype(np.int16)*60+et.dt.minute.astype(np.int16)
a['_session_min']=((a['_minute_et']-1080)%1440).astype(np.int16)
base_o=a.open.to_numpy(float); base_h=a.high.to_numpy(float); base_l=a.low.to_numpy(float); base_c=a.close.to_numpy(float)
base_sym=a.symbol.astype('category')
base_sym_code=base_sym.cat.codes.to_numpy(np.int32)
@njit(cache=True)
def segment_end_codes(code):
    n=len(code); out=np.empty(n,np.int64); end=n-1
    for i in range(n-1,-1,-1):
        if i==n-1 or code[i]!=code[i+1]: end=i
        out[i]=end
    return out
seg_end=segment_end_codes(base_sym_code)

@njit(cache=True)
def find_signals(o,h,l,c,sym, fidx, direction, lower, upper, mode):
    n=len(fidx)
    trig=np.full(n,-1,np.int64); depth=np.full(n,np.nan,np.float64); age=np.full(n,-1,np.int64)
    for k in range(n):
        i=fidx[k]; d=direction[k]; lo=lower[k]; up=upper[k]; w=up-lo; sc=sym[i]
        if w<=0: continue
        j=i+1
        while j<len(c) and sym[j]==sc:
            touched = (l[j] <= up) if d==1 else (h[j] >= lo)
            invalid = (c[j] < lo) if d==1 else (c[j] > up)
            opp = (c[j] < o[j]) if d==1 else (c[j] > o[j])
            dep = (up-c[j])/w if d==1 else (c[j]-lo)/w
            inside = dep>=0.0 and dep<1.0
            if mode==1 and touched:
                if opp and inside:
                    trig[k]=j; depth[k]=dep; age[k]=j-i
                break
            if touched and opp and inside:
                trig[k]=j; depth[k]=dep; age[k]=j-i
                break
            if invalid:
                break
            j+=1
    return trig,depth,age

@njit(cache=True)
def resolve_trades(base_h,base_l,base_sym_code,seg_end,entry_base,direction,lower,upper,entry):
    n=len(entry_base)
    outcome=np.zeros(n,np.int8)
    bars=np.full(n,-1,np.int64)
    mfe=np.full(n,np.nan,np.float64); mae=np.full(n,np.nan,np.float64)
    for k in range(n):
        bi=entry_base[k]
        if bi<0 or bi+1>=len(base_h): continue
        d=direction[k]; lo=lower[k]; up=upper[k]; en=entry[k]
        end=seg_end[bi]
        maxfav=0.0; maxadv=0.0
        for j in range(bi+1,end+1):
            if d==1:
                fav=max(0.0, base_h[j]-en); adv=max(0.0,en-base_l[j])
                ht=base_h[j]>=up; hs=base_l[j]<=lo
            else:
                fav=max(0.0,en-base_l[j]); adv=max(0.0,base_h[j]-en)
                ht=base_l[j]<=lo; hs=base_h[j]>=up
            if fav>maxfav: maxfav=fav
            if adv>maxadv: maxadv=adv
            if ht or hs:
                bars[k]=j-bi; mfe[k]=maxfav; mae[k]=maxadv
                if ht and hs: outcome[k]=2
                elif ht: outcome[k]=1
                else: outcome[k]=-1
                break
        if outcome[k]==0:
            mfe[k]=maxfav; mae[k]=maxadv
    return outcome,bars,mfe,mae

def build_bars(tf):
    if tf==1:
        b=a[['ts_event','open','high','low','close','volume','symbol','trade_date','base_idx']].copy()
        b['start_ts']=b.ts_event; b['end_ts']=b.ts_event; b['start_base']=b.base_idx; b['end_base']=b.base_idx
        return b.reset_index(drop=True)
    if tf==1440:
        g=a.groupby('trade_date',sort=False,observed=True)
    else:
        binv=(a['_session_min'].to_numpy()//tf).astype(np.int16)
        tmp=a.assign(_bin=binv)
        g=tmp.groupby(['trade_date','_bin'],sort=False,observed=True)
    b=g.agg(ts_event=('ts_event','first'),start_ts=('ts_event','first'),end_ts=('ts_event','last'),
            open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),
            symbol=('symbol','first'),symbol_last=('symbol','last'),start_base=('base_idx','first'),end_base=('base_idx','last'),n1m=('base_idx','size')).reset_index()
    b=b[b.symbol==b.symbol_last].copy().reset_index(drop=True)
    return b

def detect_fvgs(b):
    o=b.open.to_numpy(float);h=b.high.to_numpy(float);l=b.low.to_numpy(float);c=b.close.to_numpy(float);sym=b.symbol.astype('category').cat.codes.to_numpy(np.int32)
    bull=np.zeros(len(b),bool); bear=np.zeros(len(b),bool)
    if len(b)>=3:
        same=(sym[2:]==sym[1:-1])&(sym[1:-1]==sym[:-2])
        bull[2:]=(l[2:]>h[:-2])&same
        bear[2:]=(h[2:]<l[:-2])&same
    ix=np.flatnonzero(bull|bear)
    d=np.where(bull[ix],1,-1).astype(np.int8)
    lo=np.where(d==1,h[ix-2],h[ix])
    up=np.where(d==1,l[ix],l[ix-2])
    w=up-lo
    ok=w>=0.25-1e-12
    ix=ix[ok]; d=d[ok]; lo=lo[ok]; up=up[ok]; w=w[ok]
    return ix,d,lo,up,w,sym

all_events=[]; all_trades=[]
for tf in TFS:
    print('TF',TFNAME[tf],'build',flush=True)
    b=build_bars(tf)
    ix,d,lo,up,w,sym=detect_fvgs(b)
    print(' bars',len(b),'fvgs',len(ix),flush=True)
    o=b.open.to_numpy(float);h=b.high.to_numpy(float);l=b.low.to_numpy(float);c=b.close.to_numpy(float)
    tq,dep,age=find_signals(o,h,l,c,sym,ix,d,lo,up,0)
    tt,depf,agef=find_signals(o,h,l,c,sym,ix,d,lo,up,1)
    for mode,trig,depth,ag in [('qualifying',tq,dep,age),('first_touch',tt,depf,agef)]:
        m=trig>=0
        if not m.any(): continue
        fi=ix[m]; tr=trig[m]; dd=d[m]; llo=lo[m]; uup=up[m]; ww=w[m]; dp=depth[m]; aa=ag[m]
        en=c[tr]
        eb=b.end_base.to_numpy(np.int64)[tr]
        outcome,exitbars,mfe,mae=resolve_trades(base_h,base_l,base_sym_code,seg_end,eb,dd,llo,uup,en)
        reward=np.where(dd==1,uup-en,en-llo); risk=np.where(dd==1,en-llo,uup-en)
        rr=np.divide(reward,risk,out=np.full(len(reward),np.nan),where=risk>1e-12)
        realized=np.where(outcome==1,rr,np.where((outcome==-1)|(outcome==2),-1.0,np.nan))
        ts=b.end_ts.to_numpy()[tr]
        formts=b.end_ts.to_numpy()[fi]
        year=pd.to_datetime(ts,utc=True).year
        df=pd.DataFrame({'timeframe':TFNAME[tf],'tf_min':tf,'mode':mode,'fvg_bar':fi,'trigger_bar':tr,
                         'formation_ts':formts,'trigger_ts':ts,'direction':np.where(dd==1,'bullish','bearish'),
                         'lower':llo,'upper':uup,'width_pts':ww,'width_ticks':ww/.25,'entry':en,'depth':dp,'depth_pct':100*dp,
                         'age_tf_bars':aa,'reward_pts':reward,'risk_pts':risk,'rr':rr,'outcome_code':outcome,'exit_1m_bars':exitbars,
                         'mfe_pts':mfe,'mae_pts':mae,'realized_R_conservative':realized,'year':year})
        all_trades.append(df)
    ev=pd.DataFrame({'timeframe':TFNAME[tf],'tf_min':tf,'formation_ts':b.end_ts.to_numpy()[ix],'direction':np.where(d==1,'bullish','bearish'),
                     'lower':lo,'upper':up,'width_pts':w,'width_ticks':w/.25,
                     'qual_trigger':tq,'qual_depth':dep,'firsttouch_trigger':tt,'firsttouch_depth':depf})
    all_events.append(ev)

trades=pd.concat(all_trades,ignore_index=True)
events=pd.concat(all_events,ignore_index=True)
suffix='_'+'_'.join(map(str,TFS)) if len(TFS)<14 else ''
trades.to_pickle(OUT/f'trades{suffix}.pkl')
events.to_pickle(OUT/f'fvg_events{suffix}.pkl')

if os.environ.get('CE_CHUNK')=='1':
    print('chunk saved', suffix, len(trades), len(events), flush=True)
    raise SystemExit

primary=trades[trades['mode']=='qualifying'].copy()
bins=np.arange(0,1.000001,.1); labels=[f'{int(x*100)}-{int((x+.1)*100)}%' for x in bins[:-1]]
primary['depth_bucket']=pd.cut(primary.depth,bins=bins,labels=labels,include_lowest=True,right=False)

def agg(df,groupcols):
    z=df.copy(); resolved=z.outcome_code.isin([1,-1,2]); zr=z[resolved].copy()
    zr['win_cons']=(zr.outcome_code==1).astype(int)
    zr['loss_cons']=zr.outcome_code.isin([-1,2]).astype(int)
    out=zr.groupby(groupcols,observed=True).agg(N=('outcome_code','size'),wins=('win_cons','sum'),
            win_rate=('win_cons','mean'),mean_R=('realized_R_conservative','mean'),median_RR=('rr','median'),mean_RR=('rr','mean'),
            median_width_ticks=('width_ticks','median'),median_exit_min=('exit_1m_bars','median'),ambiguous=('outcome_code',lambda x:(x==2).sum())).reset_index()
    be=zr.groupby(groupcols,observed=True).depth.apply(lambda s: float(np.mean(1-s))).reset_index(name='avg_breakeven_winrate')
    out=out.merge(be,on=groupcols,how='left')
    out['edge_vs_breakeven_pp']=100*(out.win_rate-out.avg_breakeven_winrate)
    return out

by_tf=agg(primary,['timeframe','tf_min']).sort_values('tf_min')
by_depth=agg(primary,['depth_bucket'])
by_tf_depth=agg(primary,['timeframe','tf_min','depth_bucket']).sort_values(['tf_min','depth_bucket'])

spec=[]
for tf in ['ALL']+list(by_tf.sort_values('tf_min').timeframe):
    d0=primary if tf=='ALL' else primary[primary.timeframe==tf]
    slices=[('Exact 50%',np.isclose(d0.depth,.5,atol=1e-12)),('49-51%',(d0.depth>=.49)&(d0.depth<=.51)),('47.5-52.5%',(d0.depth>=.475)&(d0.depth<=.525)),('45-55%',(d0.depth>=.45)&(d0.depth<=.55)),('40-60%',(d0.depth>=.40)&(d0.depth<=.60))]
    for nm,ms in slices:
        zz=d0[ms & d0.outcome_code.isin([1,-1,2])].copy()
        if len(zz)==0: continue
        wins=(zz.outcome_code==1)
        spec.append([tf,nm,len(zz),wins.mean(),np.nanmean(zz.realized_R_conservative),np.nanmedian(zz.rr),np.mean(1-zz.depth),
                     100*(wins.mean()-np.mean(1-zz.depth)),np.median(zz.width_ticks),np.median(zz.exit_1m_bars),(zz.outcome_code==2).sum()])
special=pd.DataFrame(spec,columns=['timeframe','slice','N','win_rate','mean_R','median_RR','avg_breakeven_winrate','edge_vs_breakeven_pp','median_width_ticks','median_exit_min','ambiguous'])

ce=primary[(primary.depth>=.45)&(primary.depth<=.55)&primary.outcome_code.isin([1,-1,2])].copy()
ce['win_cons']=(ce.outcome_code==1).astype(int)
by_year=agg(ce,['year']) if len(ce) else pd.DataFrame()
spl=[]
for tf in ['ALL']+list(primary.timeframe.unique()):
    z=primary if tf=='ALL' else primary[primary.timeframe==tf]
    z=z[(z.depth>=.45)&(z.depth<=.55)&z.outcome_code.isin([1,-1,2])].sort_values('trigger_ts')
    if len(z)<20: continue
    cut=int(len(z)*.7)
    for name,zz in [('train70',z.iloc[:cut]),('test30',z.iloc[cut:])]:
        wr=(zz.outcome_code==1).mean(); be=np.mean(1-zz.depth); mr=np.nanmean(zz.realized_R_conservative)
        spl.append([tf,name,len(zz),str(zz.trigger_ts.min()),str(zz.trigger_ts.max()),wr,mr,be,100*(wr-be)])
split=pd.DataFrame(spl,columns=['timeframe','split','N','start','end','win_rate','mean_R','avg_breakeven_winrate','edge_vs_breakeven_pp'])

first=trades[(trades.mode=='first_touch') & (trades.depth>=.45)&(trades.depth<=.55)&trades.outcome_code.isin([1,-1,2])].copy()
first_summary=agg(first,['timeframe','tf_min']).sort_values('tf_min') if len(first) else pd.DataFrame()
ce_dir=agg(ce,['direction']) if len(ce) else pd.DataFrame()
ce_tf_dir=agg(ce,['timeframe','tf_min','direction']).sort_values(['tf_min','direction']) if len(ce) else pd.DataFrame()
setup_counts=events.groupby(['timeframe','tf_min'],observed=True).agg(FVGs=('formation_ts','size'),
    qualifying_signals=('qual_trigger',lambda x:(x>=0).sum()),firsttouch_signals=('firsttouch_trigger',lambda x:(x>=0).sum())).reset_index().sort_values('tf_min')
setup_counts['qualifying_rate']=setup_counts.qualifying_signals/setup_counts.FVGs

for name,df in [('summary_by_timeframe',by_tf),('summary_by_depth',by_depth),('summary_tf_depth',by_tf_depth),('midpoint_slices',special),('ce_by_year',by_year),('ce_train_test',split),('ce_first_touch_sensitivity',first_summary),('ce_direction',ce_dir),('ce_tf_direction',ce_tf_dir),('setup_counts',setup_counts)]:
    df.to_csv(OUT/f'{name}.csv',index=False)

rng=np.random.default_rng(20260918)
def boot_metric(z,B=1000):
    z=z[z.outcome_code.isin([1,-1,2])]
    if len(z)<10:return (np.nan,np.nan,np.nan,np.nan)
    arr=(z.outcome_code.to_numpy()==1).astype(float); r=z.realized_R_conservative.to_numpy(float)
    n=len(z); wr=np.empty(B); er=np.empty(B)
    for b in range(B):
        ii=rng.integers(0,n,n); wr[b]=arr[ii].mean(); er[b]=np.nanmean(r[ii])
    return (*np.quantile(wr,[.025,.975]),*np.quantile(er,[.025,.975]))
ci=[]
for bucket in labels:
    z=primary[primary.depth_bucket.astype(str)==bucket]
    wlo,whi,rlo,rhi=boot_metric(z,500)
    ci.append([bucket,len(z),wlo,whi,rlo,rhi])
z=ce;wlo,whi,rlo,rhi=boot_metric(z,1000);ci.append(['45-55%',len(z),wlo,whi,rlo,rhi])
ci=pd.DataFrame(ci,columns=['slice','N','win95_low','win95_high','meanR95_low','meanR95_high']);ci.to_csv(OUT/'bootstrap_ci.csv',index=False)

import matplotlib.pyplot as plt
bd=by_depth.copy();
plt.figure(figsize=(10,5));plt.bar(bd.depth_bucket.astype(str),100*bd.win_rate);plt.plot(bd.depth_bucket.astype(str),100*bd.avg_breakeven_winrate,marker='o',label='Breakeven WR');plt.ylabel('Win rate (%)');plt.xlabel('Trigger candle close depth into FVG');plt.xticks(rotation=45);plt.title('FVG rejection setup: win rate by body-close depth');plt.legend();plt.tight_layout();plt.savefig(OUT/'winrate_by_depth.png',dpi=160);plt.close()
plt.figure(figsize=(10,5));plt.bar(bd.depth_bucket.astype(str),bd.mean_R);plt.axhline(0,linewidth=1);plt.ylabel('Mean gross R / trade');plt.xlabel('Trigger close depth');plt.xticks(rotation=45);plt.title('Gross expectancy by FVG body-close depth');plt.tight_layout();plt.savefig(OUT/'expectancy_by_depth.png',dpi=160);plt.close()
ceplot=special[(special.slice=='45-55%') & (special.timeframe!='ALL')].copy(); ceplot['order']=ceplot.timeframe.map({TFNAME[x]:i for i,x in enumerate(TFS)});ceplot=ceplot.sort_values('order')
plt.figure(figsize=(11,5));plt.bar(ceplot.timeframe,100*ceplot.win_rate);plt.plot(ceplot.timeframe,100*ceplot.avg_breakeven_winrate,marker='o',label='Breakeven WR');plt.ylabel('Win rate (%)');plt.xlabel('FVG timeframe');plt.title('45-55% CE rejection: win rate by timeframe');plt.legend();plt.tight_layout();plt.savefig(OUT/'ce_winrate_by_timeframe.png',dpi=160);plt.close()
plt.figure(figsize=(11,5));plt.bar(ceplot.timeframe,ceplot.mean_R);plt.axhline(0,linewidth=1);plt.ylabel('Mean gross R / trade');plt.xlabel('FVG timeframe');plt.title('45-55% CE rejection: expectancy by timeframe');plt.tight_layout();plt.savefig(OUT/'ce_expectancy_by_timeframe.png',dpi=160);plt.close()

summary={
 'data_start':str(a.ts_event.min()),'data_end':str(a.ts_event.max()),'base_rows':len(a),
 'timeframes':[TFNAME[x] for x in TFS], 'total_fvgs':int(len(events)), 'primary_signals':int((events.qual_trigger>=0).sum()),
 'primary_resolved_trades':int(primary.outcome_code.isin([1,-1,2]).sum()), 'primary_censored':int((primary.outcome_code==0).sum()),
 'ce45_55_resolved':int(len(ce)), 'execution':'entry at signal timeframe candle close; TP/SL resolved on future 1-minute bars; ambiguous same-1m TP+SL treated as loss in conservative stats',
 'signal_definition':'bullish FVG requires later bearish candle; bearish FVG requires later bullish candle; wick must touch gap; close inside 0-100% gap; wick length irrelevant; search stops if a prior candle body closes through far edge',
}
json.dump(summary,open(OUT/'summary.json','w'),indent=2)

print('\nSETUP COUNTS\n',setup_counts.to_string(index=False),flush=True)
print('\nDEPTH\n',by_depth.to_string(index=False),flush=True)
print('\nMIDPOINT\n',special[special.timeframe=='ALL'].to_string(index=False),flush=True)
print('\nCE TF\n',ceplot[['timeframe','N','win_rate','mean_R','median_RR','avg_breakeven_winrate','edge_vs_breakeven_pp','median_width_ticks','median_exit_min','ambiguous']].to_string(index=False),flush=True)
print('\nCE YEAR\n',by_year.to_string(index=False),flush=True)
print('\nCE SPLIT\n',split[split.timeframe=='ALL'].to_string(index=False),flush=True)
print('\nCE DIR\n',ce_dir.to_string(index=False),flush=True)
print('\nFIRST TOUCH CE\n',first_summary.to_string(index=False),flush=True)
print('\nSUMMARY',json.dumps(summary),flush=True)
