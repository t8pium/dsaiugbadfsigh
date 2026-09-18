import pandas as pd, numpy as np, json
from pathlib import Path
from numba import njit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
OUT=Path('/mnt/data/fvg_study_outputs'); OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(42)
a=pd.read_pickle('/mnt/data/active_mnq.pkl')
T=a.ts_event.astype('int64').to_numpy(); o=a.open.to_numpy(float);h=a.high.to_numpy(float);l=a.low.to_numpy(float);c=a.close.to_numpy(float)
roll=a.symbol.ne(a.symbol.shift()).to_numpy();block=np.zeros(len(a),bool)
for x in T[roll]: block|=np.abs(T-x)<=86400*10**9
prev=np.r_[np.nan,c[:-1]];tr=np.maximum(h-l,np.maximum(np.abs(h-prev),np.abs(l-prev)));atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy();atr[atr<=0]=np.nan
ret60=np.full(len(a),np.nan);ret60[60:]=(c[60:]-c[:-60])/atr[60:];atrmed=pd.Series(atr).rolling(1380,min_periods=300).median().to_numpy();vr=atr/atrmed
et=a.ts_event.dt.tz_convert('America/New_York');minute=et.dt.hour.to_numpy()*60+et.dt.minute.to_numpy();year=et.dt.year.to_numpy()
sess=np.select([(minute>=1080)|(minute<120),minute<480,minute<570,minute<690,minute<810,minute<960],['Asia','London','NY premarket','NY AM','NY lunch','NY PM'],default='Post-market')
bull=np.zeros(len(a),bool);bear=np.zeros(len(a),bool);bull[2:]=l[2:]>h[:-2];bear[2:]=h[2:]<l[:-2];isf=bull|bear
valid=isf&np.isfinite(atr)&np.isfinite(ret60)&np.isfinite(vr)&~block
ix=np.flatnonzero(valid);d=np.where(bull[ix],1,-1).astype(np.int8);lo=np.where(d==1,h[ix-2],h[ix]);up=np.where(d==1,l[ix],l[ix-2]);mid=(lo+up)/2;wd=up-lo;dist=np.where(d==1,c[ix]-up,lo-c[ix]);keep=(wd>0)&(dist>=-1e-9);ix=ix[keep];d=d[keep];lo=lo[keep];up=up[keep];mid=mid[keep];wd=wd[keep];dist=dist[keep]
wa=wd/atr[ix];da=dist/atr[ix];vreg=np.select([vr[ix]<.8,vr[ix]<=1.2],[0,1],default=2);treg=np.select([ret60[ix]<=-1,ret60[ix]<-.25,ret60[ix]<=.25,ret60[ix]<1],[-2,-1,0,1],default=2);body=np.abs(c-o);disp=body[ix-1]/atr[ix]
f=pd.DataFrame({'row':ix,'timestamp':a.ts_event.iloc[ix].to_numpy(),'direction':d,'lower':lo,'upper':up,'mid':mid,'width_pts':wd,'width_ticks':wd/.25,'width_atr':wa,'distance_atr':da,'session':sess[ix],'tod':minute[ix],'vol_regime':vreg,'trend_regime':treg,'vol_ratio':vr[ix],'trend_score':ret60[ix],'disp_body_atr':disp,'year':year[ix]})
@njit(cache=True)
def trail_min(x,w):
 n=len(x);out=np.empty(n,np.float64);dq=np.empty(n,np.int64);head=0;tail=0
 for i in range(n):
  while head<tail and dq[head]<i-w+1: head+=1
  while head<tail and x[dq[tail-1]]>=x[i]: tail-=1
  dq[tail]=i;tail+=1;out[i]=x[dq[head]]
 return out
@njit(cache=True)
def trail_max(x,w):
 n=len(x);out=np.empty(n,np.float64);dq=np.empty(n,np.int64);head=0;tail=0
 for i in range(n):
  while head<tail and dq[head]<i-w+1: head+=1
  while head<tail and x[dq[tail-1]]<=x[i]: tail-=1
  dq[tail]=i;tail+=1;out[i]=x[dq[head]]
 return out
trail_min(np.array([1.,2.,3.]),2);trail_max(np.array([1.,2.,3.]),2)
horiz=[5,15,30,60,120,240,1380,4140]; ext={}
lowrev=np.r_[l[1:],np.inf][::-1];highrev=np.r_[h[1:],-np.inf][::-1]
for z in horiz:
 fmin=trail_min(lowrev,z)[::-1];fmax=trail_max(highrev,z)[::-1];ext[z]=(fmin,fmax)
 for typ,thr in [('touch',np.where(d==1,up,lo)),('half',mid),('full',np.where(d==1,lo,up))]: f[f'{typ}_{z}']=np.where(d==1,fmin[ix]<=thr,fmax[ix]>=thr).astype(np.int8)
smin=np.minimum.accumulate(l[::-1])[::-1];smax=np.maximum.accumulate(h[::-1])[::-1];nmin=np.r_[smin[1:],np.nan];nmax=np.r_[smax[1:],np.nan];f['touch_eventual']=np.where(d==1,nmin[ix]<=up,nmax[ix]>=lo).astype(np.int8);f['full_eventual']=np.where(d==1,nmin[ix]<=lo,nmax[ix]>=up).astype(np.int8)
for b in [1,3,5,10,15,30,60]:
 dest=ix+b;ok=dest<len(c);v=np.full(len(ix),np.nan);v[ok]=d[ok]*(c[ix[ok]]-c[dest[ok]])/atr[ix[ok]];f[f'toward_{b}']=v
fs=f.loc[rng.choice(f.index.to_numpy(),size=min(40000,len(f)),replace=False)]
base=(~isf)&np.isfinite(atr)&np.isfinite(ret60)&np.isfinite(vr)&~block&(np.arange(len(a))<len(a)-4140);cand=np.flatnonzero(base);cv=np.select([vr[cand]<.8,vr[cand]<=1.2],[0,1],default=2);ct=np.select([ret60[cand]<=-1,ret60[cand]<-.25,ret60[cand]<=.25,ret60[cand]<1],[-2,-1,0,1],default=2);ch=minute[cand]//60
from collections import defaultdict
g=defaultdict(list)
for rr,s0,h0,v0,t0 in zip(cand,sess[cand],ch,cv,ct): g[(s0,int(h0),int(v0),int(t0))].append(int(rr))
g={k:np.array(v,np.int64) for k,v in g.items()}
K=5;CR=[];CP=[];CD=[];CL=[];CU=[];CM=[]
for pi,r in fs.iterrows():
 arr=g.get((r.session,int(r.tod//60),int(r.vol_regime),int(r.trend_regime)))
 if arr is None or len(arr)==0:continue
 for rr in rng.choice(arr,size=K,replace=len(arr)<K):
  distp=float(r.distance_atr*atr[rr]);wid=float(r.width_atr*atr[rr]);
  if r.direction==1:u=c[rr]-distp;ll=u-wid
  else:ll=c[rr]+distp;u=ll+wid
  CR.append(rr);CP.append(pi);CD.append(r.direction);CL.append(ll);CU.append(u);CM.append((ll+u)/2)
cr=np.array(CR,np.int64);cp=np.array(CP,np.int64);cd=np.array(CD,np.int8);cl=np.array(CL);cu=np.array(CU);cm=np.array(CM)
ctrl=pd.DataFrame({'row':cr,'parent':cp,'direction':cd})
for z in horiz:
 fmin,fmax=ext[z]
 for typ,thr in [('touch',np.where(cd==1,cu,cl)),('half',cm),('full',np.where(cd==1,cl,cu))]:ctrl[f'{typ}_{z}']=np.where(cd==1,fmin[cr]<=thr,fmax[cr]>=thr).astype(np.int8)
for b in [1,3,5,10,15,30,60]:
 dest=cr+b;ok=dest<len(c);v=np.full(len(cr),np.nan);v[ok]=cd[ok]*(c[cr[ok]]-c[dest[ok]])/atr[cr[ok]];ctrl[f'toward_{b}']=v
rng2=np.random.default_rng(7)
def boot(col,B=200):
 pc=ctrl.groupby('parent')[col].mean();ids=pc.index.to_numpy();fv=f.loc[ids,col].to_numpy(float);cvv=pc.to_numpy(float);dif=fv-cvv;dates=pd.to_datetime(f.loc[ids,'timestamp'], utc=True).dt.tz_convert('America/New_York').dt.date.to_numpy();tmp=pd.DataFrame({'d':dates,'x':dif});cls=[q.x.to_numpy() for _,q in tmp.groupby('d')];bs=np.empty(B)
 for b in range(B):
  sel=rng2.integers(0,len(cls),len(cls));bs[b]=np.concatenate([cls[j] for j in sel]).mean()
 return fv.mean(),cvv.mean(),dif.mean(),*np.quantile(bs,[.025,.975]),2*min((bs<=0).mean(),(bs>=0).mean()),len(ids)
rows=[]
for z in horiz:
 for typ in (['touch','half','full'] if z in [60,1380,4140] else ['touch']):
  pf,pc,di,lci,hci,p,n=boot(f'{typ}_{z}');orr=(pf/(1-pf))/(pc/(1-pc)) if 0<pf<1 and 0<pc<1 else np.nan;rows.append([f'{typ}_{z}',pf,pc,di,di/pc if pc else np.nan,orr,lci,hci,p,n])
main=pd.DataFrame(rows,columns=['test','FVG','Control','Difference','RelativeDiff','OddsRatio','CI_low','CI_high','p_cluster_boot','N_matched']);main.to_csv(OUT/'main_results.csv',index=False)
def strat(col,bins=None,labels=None):
 vals=f[col];grp=pd.cut(vals,bins=bins,labels=labels,include_lowest=True,right=False) if bins is not None else vals.astype(str);out=[]
 for z in pd.Series(grp.loc[fs.index]).dropna().unique():
  ids=fs.index[grp.loc[fs.index]==z];cc=ctrl[ctrl.parent.isin(ids)]
  if len(ids)>=30: out.append([str(z),len(ids),f.loc[ids,'touch_60'].mean(),cc.touch_60.mean(),f.loc[ids,'touch_60'].mean()-cc.touch_60.mean()])
 return pd.DataFrame(out,columns=['group','N_FVG','FVG_touch60','Control_touch60','Difference'])
tabs={'distance':strat('distance_atr',[-np.inf,.25,.5,.75,1,1.5,2,3,np.inf],['<.25','.25-.5','.5-.75','.75-1','1-1.5','1.5-2','2-3','3+']),'size':strat('width_atr',[-np.inf,.1,.2,.3,.5,.75,1,np.inf],['<.10','.10-.20','.20-.30','.30-.50','.50-.75','.75-1','1+']),'session':strat('session'),'direction':strat('direction'),'volatility':strat('vol_regime'),'trend':strat('trend_regime'),'displacement':strat('disp_body_atr',[-np.inf,.25,.5,.75,1,1.5,np.inf],['<.25','.25-.5','.5-.75','.75-1','1-1.5','1.5+']),'year':strat('year')}
for n,t in tabs.items():t.to_csv(OUT/f'strat_{n}.csv',index=False)
cut=f.timestamp.quantile(.70);split=[]
for nm,ids in [('train',fs.index[fs.timestamp<=cut]),('test',fs.index[fs.timestamp>cut])]:
 cc=ctrl[ctrl.parent.isin(ids)];split.append([nm,str(cut),len(ids),f.loc[ids,'touch_60'].mean(),cc.touch_60.mean(),f.loc[ids,'touch_60'].mean()-cc.touch_60.mean()])
split=pd.DataFrame(split,columns=['split','cutoff','N_FVG','FVG_touch60','Control_touch60','Difference']);split.to_csv(OUT/'train_test.csv',index=False)
pcov=f.loc[ctrl.parent,['distance_atr','width_atr','direction']].reset_index(drop=True);uids=ctrl.parent.unique();fm=f.loc[uids,['touch_60','distance_atr','width_atr','vol_ratio','trend_score','tod','direction']].copy();fm['is_fvg']=1;cmx=pd.DataFrame({'touch_60':ctrl.touch_60.to_numpy(),'distance_atr':pcov.distance_atr.to_numpy(),'width_atr':pcov.width_atr.to_numpy(),'vol_ratio':vr[cr],'trend_score':ret60[cr],'tod':minute[cr],'direction':pcov.direction.to_numpy(),'is_fvg':0});mod=pd.concat([fm,cmx],ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna();X=mod[['is_fvg','distance_atr','width_atr','vol_ratio','trend_score','tod','direction']].copy();X['tod_sin']=np.sin(2*np.pi*X.tod/1440);X['tod_cos']=np.cos(2*np.pi*X.tod/1440);X=X.drop(columns='tod');y=mod.touch_60;lr=LogisticRegression(max_iter=200,C=1e6).fit(X,y);coef=dict(zip(X.columns,lr.coef_[0]));coef['intercept']=float(lr.intercept_[0]);coef['fvg_odds_ratio']=float(np.exp(coef['is_fvg']));coef['n']=len(mod);json.dump(coef,open(OUT/'logistic_60m.json','w'),indent=2)
sens=[]
for nm,mask in [('any',pd.Series(True,index=f.index)),('>=1tick',f.width_ticks>=1),('>=2ticks',f.width_ticks>=2),('>=.05ATR',f.width_atr>=.05),('>=.10ATR',f.width_atr>=.10),('>=.20ATR',f.width_atr>=.20)]:
 ids=fs.index[mask.loc[fs.index]];cc=ctrl[ctrl.parent.isin(ids)]
 for tar in ['touch_60','half_60','full_60']:sens.append([nm,tar,len(ids),f.loc[ids,tar].mean(),cc[tar].mean(),f.loc[ids,tar].mean()-cc[tar].mean()])
pd.DataFrame(sens,columns=['filter','target','N_FVG','FVG','Control','Difference']).to_csv(OUT/'sensitivity.csv',index=False)
D=[]
for b in [1,3,5,10,15,30,60]:D.append([b,f.loc[fs.index,f'toward_{b}'].mean(),ctrl[f'toward_{b}'].mean()])
pd.DataFrame(D,columns=['bars','FVG_mean_toward_ATR','Control_mean_toward_ATR']).to_csv(OUT/'directional.csv',index=False)
DI=pd.Series(a.ts_event).diff().dt.total_seconds().div(60);summary={'active_rows':len(a),'start':str(a.ts_event.min()),'end':str(a.ts_event.max()),'active_contracts':int(a.symbol.nunique()),'roll_boundaries':int(roll.sum()),'duplicates':int(a.ts_event.duplicated().sum()),'missing_ohlc':int(a[['open','high','low','close']].isna().sum().sum()),'gaps_gt_5min':int((DI>5).sum()),'fvg_count':len(f),'matched_fvg_sample':len(fs),'controls':len(ctrl),'raw_touch_5':float(f.touch_5.mean()),'raw_touch_60':float(f.touch_60.mean()),'raw_half_60':float(f.half_60.mean()),'raw_full_60':float(f.full_60.mean()),'raw_touch_1tradingday':float(f.touch_1380.mean()),'raw_touch_3tradingdays':float(f.touch_4140.mean()),'eventual_touch_within_dataset':float(f.touch_eventual.mean()),'eventual_full_within_dataset':float(f.full_eventual.mean()),'train_test_cutoff':str(cut)};json.dump(summary,open(OUT/'summary.json','w'),indent=2)
f.to_csv(OUT/'fvg_events.csv',index=False)
plt.figure(figsize=(8,5));plt.plot(horiz,[f[f'touch_{z}'].mean() for z in horiz],marker='o',label='FVG');plt.plot(horiz,[ctrl[f'touch_{z}'].mean() for z in horiz],marker='o',label='Matched control');plt.xscale('log');plt.xlabel('Trading minutes');plt.ylabel('Touch probability');plt.title('FVG vs matched controls');plt.legend();plt.tight_layout();plt.savefig(OUT/'01_touch_horizon.png',dpi=150);plt.close()
for nm in ['distance','size','session','volatility','trend','year']:
 t=tabs[nm];plt.figure(figsize=(9,5));x=np.arange(len(t));w=.38;plt.bar(x-w/2,t.FVG_touch60,w,label='FVG');plt.bar(x+w/2,t.Control_touch60,w,label='Control');plt.xticks(x,t.group,rotation=35,ha='right');plt.ylabel('60m touch probability');plt.title(nm.title());plt.legend();plt.tight_layout();plt.savefig(OUT/f'plot_{nm}.png',dpi=150);plt.close()
print('SUMMARY',json.dumps(summary));print('\nMAIN\n',main.to_string(index=False));print('\nSPLIT\n',split.to_string(index=False));print('\nDIST\n',tabs['distance'].to_string(index=False));print('\nSESSION\n',tabs['session'].to_string(index=False));print('\nYEAR\n',tabs['year'].to_string(index=False));print('\nDIR\n',pd.DataFrame(D,columns=['bars','FVG','Control']).to_string(index=False));print('\nLOGIT',coef)
