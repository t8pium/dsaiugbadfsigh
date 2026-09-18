import pandas as pd, numpy as np, math, json, os, time
from pathlib import Path
from collections import defaultdict
from numba import njit

OUT=Path('/mnt/data/fvg_strength_project_single'); OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(260918)
base=pd.read_pickle('/mnt/data/active_mnq.pkl')
base=base.sort_values('ts_event').reset_index(drop=True)
roll_ts=base.loc[base.symbol.ne(base.symbol.shift()),'ts_event'].astype('int64').to_numpy()[1:]

@njit(cache=True)
def trail_min(x,w):
    n=len(x); out=np.empty(n,np.float64); dq=np.empty(n,np.int64); head=0; tail=0
    for i in range(n):
        while head<tail and dq[head] < i-w+1: head+=1
        while head<tail and x[dq[tail-1]] >= x[i]: tail-=1
        dq[tail]=i; tail+=1; out[i]=x[dq[head]]
    return out
@njit(cache=True)
def trail_max(x,w):
    n=len(x); out=np.empty(n,np.float64); dq=np.empty(n,np.int64); head=0; tail=0
    for i in range(n):
        while head<tail and dq[head] < i-w+1: head+=1
        while head<tail and x[dq[tail-1]] <= x[i]: tail-=1
        dq[tail]=i; tail+=1; out[i]=x[dq[head]]
    return out
@njit(cache=True)
def first_touch_reaction(rows, dirs, lowsz, upsz, low, high, close, atr, max_touch, react_h):
    n=len(rows); ft=np.full(n,-1,np.int32); race=np.zeros(n,np.int8); r3=np.full(n,np.nan,np.float64)
    N=len(close)
    for q in range(n):
        i=rows[q]; d=dirs[q]; lo=lowsz[q]; up=upsz[q]; width=up-lo
        jtouch=-1
        end=min(N-1,i+max_touch)
        for j in range(i+1,end+1):
            if (d==1 and low[j] <= up) or (d==-1 and high[j] >= lo):
                jtouch=j; break
        if jtouch<0: continue
        ft[q]=jtouch-i
        if jtouch+3 < N:
            near=up if d==1 else lo
            r3[q]=d*(close[jtouch+3]-near)/atr[i]
        near=up if d==1 else lo
        far=lo if d==1 else up
        targ=near + d*width
        end2=min(N-1,jtouch+react_h)
        for j in range(jtouch+1,end2+1):
            hitT = high[j] >= targ if d==1 else low[j] <= targ
            hitF = low[j] <= far if d==1 else high[j] >= far
            if hitT and hitF:
                race[q]=2; break
            elif hitT:
                race[q]=1; break
            elif hitF:
                race[q]=-1; break
    return ft,race,r3

trail_min(np.array([1.,2.]),1); trail_max(np.array([1.,2.]),1)
first_touch_reaction(np.array([0],dtype=np.int64),np.array([1],dtype=np.int8),np.array([0.]),np.array([1.]),np.array([0.,0.]),np.array([1.,1.]),np.array([1.,1.]),np.array([1.,1.]),1,1)

def session_labels(ts):
    et=pd.Series(ts).dt.tz_convert('America/New_York')
    minute=et.dt.hour.to_numpy()*60+et.dt.minute.to_numpy()
    sess=np.select([(minute>=1080)|(minute<120),minute<480,minute<570,minute<690,minute<810,minute<960],['Asia','London','NY premarket','NY AM','NY lunch','NY PM'],default='Post-market')
    return minute,sess,et

def resample_tf(minutes):
    if minutes==1:
        z=base[['ts_event','open','high','low','close','volume']].copy()
        return z
    x=base.set_index(base.ts_event.dt.tz_convert('America/New_York'))[['open','high','low','close','volume']]
    rule=f'{minutes}min'
    z=x.resample(rule,origin='start_day',offset='18h',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open'])
    z=z.reset_index().rename(columns={'ts_event':'ts_event_et'})
    if 'ts_event' in z.columns:
        z['ts_event']=z['ts_event'].dt.tz_convert('UTC')
    else:
        z=z.rename(columns={z.columns[0]:'ts_event'}); z['ts_event']=z['ts_event'].dt.tz_convert('UTC')
    return z[['ts_event','open','high','low','close','volume']]

def roll_block(ts):
    arr=pd.Series(ts).astype('int64').to_numpy()
    if len(roll_ts)==0:return np.zeros(len(arr),bool)
    pos=np.searchsorted(roll_ts,arr)
    out=np.zeros(len(arr),bool); day=86400*10**9
    m=pos<len(roll_ts); out[m]|=(np.abs(roll_ts[pos[m]]-arr[m])<=day)
    m=pos>0; out[m]|=(np.abs(roll_ts[pos[m]-1]-arr[m])<=day)
    return out

def bootstrap_diff(fvgv, ctrlv, dates, B=300):
    dif=fvgv-ctrlv
    ddf=pd.DataFrame({'date':dates,'x':dif}).dropna()
    groups=[g.x.to_numpy() for _,g in ddf.groupby('date')]
    if len(groups)<2:return (float(np.nanmean(dif)),np.nan,np.nan,np.nan)
    vals=np.empty(B)
    for b in range(B):
        sel=rng.integers(0,len(groups),len(groups)); vals[b]=np.concatenate([groups[j] for j in sel]).mean()
    return float(np.nanmean(dif)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),float(2*min((vals<=0).mean(),(vals>=0).mean()))

TFs=[int(os.environ.get('TF','1'))]
mag_rows=[]; form_rows=[]; react_rows=[]; decay_rows=[]; oos_rows=[]; counts=[]
all_key5=[]
for tf in TFs:
    t0=time.time(); z=resample_tf(tf); print('TF',tf,'bars',len(z),flush=True)
    T=z.ts_event; o=z.open.to_numpy(float); h=z.high.to_numpy(float); l=z.low.to_numpy(float); c=z.close.to_numpy(float); vol=z.volume.to_numpy(float); N=len(z)
    prev=np.r_[np.nan,c[:-1]]; tr=np.maximum(h-l,np.maximum(np.abs(h-prev),np.abs(l-prev)))
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy(); atr[atr<=0]=np.nan
    atrmed=pd.Series(atr).rolling(max(100,int(240/tf)),min_periods=50).median().to_numpy(); vr=atr/atrmed
    trend=np.full(N,np.nan); k=20; trend[k:]=(c[k:]-c[:-k])/atr[k:]
    minute,sess,et=session_labels(T)
    block=roll_block(T)
    bull=np.zeros(N,bool); bear=np.zeros(N,bool); bull[2:]=l[2:]>h[:-2]; bear[2:]=h[2:]<l[:-2]; isf=bull|bear
    valid=np.isfinite(atr)&np.isfinite(vr)&np.isfinite(trend)&~block
    ix=np.flatnonzero(isf&valid&(np.arange(N)<N-25))
    d=np.where(bull[ix],1,-1).astype(np.int8)
    lo=np.where(d==1,h[ix-2],h[ix]); up=np.where(d==1,l[ix],l[ix-2]); wd=up-lo
    dist=np.where(d==1,c[ix]-up,lo-c[ix]); keep=(wd>0)&(dist>=-1e-9)
    ix=ix[keep];d=d[keep];lo=lo[keep];up=up[keep];wd=wd[keep];dist=dist[keep]
    wa=wd/atr[ix]; da=dist/atr[ix]
    vreg=np.select([vr<.8,vr<=1.2],[0,1],default=2)
    treg=np.select([trend<=-1,trend<-.25,trend<=.25,trend<1],[-2,-1,0,1],default=2)
    horizons=[1,3,5,10,20]
    ext={}
    lowrev=np.r_[l[1:],np.inf][::-1]; highrev=np.r_[h[1:],-np.inf][::-1]
    for q in horizons:
        ext[q]=(trail_min(lowrev,q)[::-1],trail_max(highrev,q)[::-1])
    M=min(12000,len(ix)); choose=rng.choice(len(ix),M,replace=False); sx=ix[choose]; sd=d[choose]; slo=lo[choose]; sup=up[choose]; sw=wd[choose]; swa=wa[choose]; sda=da[choose]
    cand=np.flatnonzero((~isf)&valid&(np.arange(N)<N-25))
    timebin=(minute//60 if tf<=60 else minute//240)
    groups=defaultdict(list)
    for rr in cand:
        groups[(sess[rr],int(vreg[rr]),int(treg[rr]),int(timebin[rr]))].append(rr)
    groups={k:np.asarray(v,dtype=np.int64) for k,v in groups.items()}
    cr=[]; cp=[]; cd=[]; clo=[]; cup=[]
    K=3
    for q,rr0 in enumerate(sx):
        key=(sess[rr0],int(vreg[rr0]),int(treg[rr0]),int(timebin[rr0]))
        arr=groups.get(key)
        if arr is None or len(arr)==0: continue
        picks=rng.choice(arr,K,replace=len(arr)<K)
        for rr in picks:
            wid=swa[q]*atr[rr]; dst=sda[q]*atr[rr]
            if sd[q]==1:
                U=c[rr]-dst; L=U-wid
            else:
                L=c[rr]+dst; U=L+wid
            cr.append(rr); cp.append(q); cd.append(sd[q]); clo.append(L); cup.append(U)
    cr=np.asarray(cr,np.int64); cp=np.asarray(cp,np.int64); cd=np.asarray(cd,np.int8); clo=np.asarray(clo,float); cup=np.asarray(cup,float)
    parents=np.unique(cp)
    counts.append([tf,len(z),len(ix),len(parents),len(cr),float(np.median(wa)) if len(wa) else np.nan,float(np.median(da)) if len(da) else np.nan])
    fvg_out={}; ctrl_out={}
    for hor in horizons:
        fmin,fmax=ext[hor]
        fv=np.where(sd==1,fmin[sx]<=sup,fmax[sx]>=slo).astype(float)
        cv=np.where(cd==1,fmin[cr]<=cup,fmax[cr]>=clo).astype(float)
        fvg_out[hor]=fv;ctrl_out[hor]=cv
        tmp=pd.DataFrame({'p':cp,'y':cv}).groupby('p').y.mean()
        pp=tmp.index.to_numpy(); fvals=fv[pp]; cvals=tmp.to_numpy(); dates=et.iloc[sx[pp]].dt.date.to_numpy()
        diff,ci1,ci2,pv=bootstrap_diff(fvals,cvals,dates)
        mag_rows.append([tf,hor,len(pp),float(fvals.mean()),float(cvals.mean()),diff,ci1,ci2,pv])
        if hor==5:
            times=T.iloc[sx[pp]].to_numpy(); ordx=np.argsort(times); cutn=int(.7*len(ordx))
            for split,sel in [('train',ordx[:cutn]),('test',ordx[cutn:])]:
                if len(sel): oos_rows.append([tf,split,len(sel),float(fvals[sel].mean()),float(cvals[sel].mean()),float((fvals[sel]-cvals[sel]).mean())])
    tmp_by={}
    for hor in horizons:
        tmp_by[hor]=pd.DataFrame({'p':cp,'y':ctrl_out[hor]}).groupby('p').y.mean()
    common=parents
    for a_h,b_h in zip(horizons[:-1],horizons[1:]):
        fA=fvg_out[a_h][common]; fB=fvg_out[b_h][common]
        mask=fA==0; pf=float(((fB[mask]-fA[mask])>0).mean()) if mask.any() else np.nan
        A=ctrl_out[a_h];B=ctrl_out[b_h]; cm=A==0; pcv=float(((B[cm]-A[cm])>0).mean()) if cm.any() else np.nan
        decay_rows.append([tf,a_h,b_h,int(mask.sum()),int(cm.sum()),pf,pcv,pf-pcv if np.isfinite(pf) and np.isfinite(pcv) else np.nan])
    body=np.abs(c-o)
    move3=np.full(N,np.nan); move3[2:]=np.abs(c[2:]-c[:-2])/atr[2:]
    bodyB=np.full(N,np.nan); bodyB[2:]=body[1:-1]/atr[2:]
    dir3=np.zeros(N,np.int8); dir3[2:]=np.where(c[2:]>=c[:-2],1,-1)
    mbins=np.digitize(move3,[.5,1,1.5,2,3]); bbins=np.digitize(bodyB,[.25,.5,.75,1,1.5])
    fg=defaultdict(list)
    for rr in cand:
        if not np.isfinite(move3[rr]) or not np.isfinite(bodyB[rr]): continue
        key=(sess[rr],int(vreg[rr]),int(treg[rr]),int(timebin[rr]),int(dir3[rr]),int(mbins[rr]),int(bbins[rr]))
        fg[key].append(rr)
    fg={k:np.asarray(v,dtype=np.int64) for k,v in fg.items()}
    eidx=[]; midx=[]
    evsel=rng.choice(len(ix),min(8000,len(ix)),replace=False)
    for qi in evsel:
        rr0=ix[qi]
        key=(sess[rr0],int(vreg[rr0]),int(treg[rr0]),int(timebin[rr0]),int(d[qi]),int(mbins[rr0]),int(bbins[rr0]))
        arr=fg.get(key)
        if arr is None or len(arr)==0: continue
        eidx.append(rr0); midx.append(int(rng.choice(arr)))
    eidx=np.asarray(eidx,np.int64); midx=np.asarray(midx,np.int64)
    ed=np.where(bull[eidx],1,-1)
    for hor in [1,3,5,10]:
        ok=(eidx+hor<N)&(midx+hor<N)
        er=ed[ok]*(c[eidx[ok]+hor]-c[eidx[ok]])/atr[eidx[ok]]
        mr=ed[ok]*(c[midx[ok]+hor]-c[midx[ok]])/atr[midx[ok]]
        form_rows.append([tf,hor,int(ok.sum()),float(np.mean(er)),float(np.mean(mr)),float(np.mean(er-mr)),float(np.mean(er>0)),float(np.mean(mr>0))])
    ftF,raceF,r3F=first_touch_reaction(sx.astype(np.int64),sd,slo,sup,l,h,c,atr,20,10)
    ftC,raceC,r3C=first_touch_reaction(cr,cd,clo,cup,l,h,c,atr,20,10)
    ctl=pd.DataFrame({'p':cp,'ft':ftC,'race':raceC,'r3':r3C})
    for metric in ['r3']:
        pg=ctl.groupby('p')[metric].mean(); pp=pg.index.to_numpy(); m=(ftF[pp]>=0)&np.isfinite(r3F[pp])&np.isfinite(pg.to_numpy())
        if m.any(): react_rows.append([tf,'mean_3bar_away_ATR',int(m.sum()),float(np.mean(r3F[pp][m])),float(np.mean(pg.to_numpy()[m])),float(np.mean(r3F[pp][m]-pg.to_numpy()[m]))])
    f_res=(raceF==1)|(raceF==-1); fwin=np.where(f_res,raceF==1,np.nan)
    def ctrl_parent_race(g):
        r=g.race.to_numpy(); m=(r==1)|(r==-1)
        return np.mean(r[m]==1) if m.any() else np.nan
    pr=ctl.groupby('p').apply(ctrl_parent_race,include_groups=False)
    pp=pr.index.to_numpy(); m=f_res[pp]&np.isfinite(pr.to_numpy())
    if m.any(): react_rows.append([tf,'1gap_rejection_before_fullfill',int(m.sum()),float(np.nanmean(fwin[pp][m])),float(np.nanmean(pr.to_numpy()[m])),float(np.nanmean(fwin[pp][m]-pr.to_numpy()[m]))])
    print('done TF',tf,'fvgs',len(ix),'matched',len(parents),'sec',round(time.time()-t0,1),flush=True)

cols=['timeframe_min','horizon_bars','N_matched','FVG_touch','Control_touch','Difference','CI_low','CI_high','p_day_boot']
pd.DataFrame(mag_rows,columns=cols).to_csv(OUT/f"magnet_tf{TFs[0]}.csv",index=False)
pd.DataFrame(form_rows,columns=['timeframe_min','horizon_bars','N','FVG_dir_return_ATR','Matched_nonFVG_return_ATR','Difference_ATR','FVG_positive_rate','Matched_positive_rate']).to_csv(OUT/f"formation_tf{TFs[0]}.csv",index=False)
pd.DataFrame(react_rows,columns=['timeframe_min','metric','N','FVG','Control','Difference']).to_csv(OUT/f"reaction_tf{TFs[0]}.csv",index=False)
pd.DataFrame(decay_rows,columns=['timeframe_min','survived_through_bars','next_horizon_bars','N_FVG_survivors','N_control_survivors','FVG_cond_touch','Control_cond_touch','Difference']).to_csv(OUT/f"decay_tf{TFs[0]}.csv",index=False)
pd.DataFrame(oos_rows,columns=['timeframe_min','split','N','FVG_touch5','Control_touch5','Difference']).to_csv(OUT/f"oos_tf{TFs[0]}.csv",index=False)
pd.DataFrame(counts,columns=['timeframe_min','bars','FVG_count','matched_FVG','control_zones','median_width_ATR','median_distance_ATR']).to_csv(OUT/f"counts_tf{TFs[0]}.csv",index=False)
print('OUTPUT',OUT)
