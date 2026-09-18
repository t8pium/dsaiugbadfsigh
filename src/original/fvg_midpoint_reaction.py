import pandas as pd, numpy as np, os, sys
from collections import defaultdict
from numba import njit
rng=np.random.default_rng(9917)
base=pd.read_pickle('/mnt/data/active_mnq.pkl').sort_values('ts_event').reset_index(drop=True)
roll_ts=base.loc[base.symbol.ne(base.symbol.shift()),'ts_event'].astype('int64').to_numpy()[1:]
@njit(cache=True)
def midrace(rows,dirs,los,ups,low,high,max_touch,race_h):
 n=len(rows); out=np.zeros(n,np.int8); ft=np.full(n,-1,np.int32); N=len(low)
 for q in range(n):
  i=rows[q]; d=dirs[q]; lo=los[q]; up=ups[q]; mid=(lo+up)/2; j0=-1
  for j in range(i+1,min(N-1,i+max_touch)+1):
   if (d==1 and low[j]<=mid) or (d==-1 and high[j]>=mid): j0=j;break
  if j0<0: continue
  ft[q]=j0-i
  for j in range(j0+1,min(N-1,j0+race_h)+1):
   hit_near = high[j]>=up if d==1 else low[j]<=lo
   hit_far = low[j]<=lo if d==1 else high[j]>=up
   if hit_near and hit_far: out[q]=2;break
   if hit_near: out[q]=1;break
   if hit_far: out[q]=-1;break
 return ft,out
midrace(np.array([0],np.int64),np.array([1],np.int8),np.array([0.]),np.array([1.]),np.array([0.,0.]),np.array([1.,1.]),1,1)
def resample(tf):
 if tf==1:return base[['ts_event','open','high','low','close','volume']].copy()
 x=base.set_index(base.ts_event.dt.tz_convert('America/New_York'))[['open','high','low','close','volume']]
 z=x.resample(f'{tf}min',origin='start_day',offset='18h',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
 z.ts_event=z.ts_event.dt.tz_convert('UTC');return z
def rollblock(ts):
 a=pd.Series(ts).astype('int64').to_numpy();p=np.searchsorted(roll_ts,a);out=np.zeros(len(a),bool);day=86400*10**9
 m=p<len(roll_ts);out[m]|=np.abs(roll_ts[p[m]]-a[m])<=day;m=p>0;out[m]|=np.abs(roll_ts[p[m]-1]-a[m])<=day;return out
def boot(vals,dates,B=500):
 d=pd.DataFrame({'date':dates,'x':vals}).dropna(); G=[g.x.to_numpy() for _,g in d.groupby('date')]; b=np.empty(B)
 for k in range(B): b[k]=np.concatenate([G[j] for j in rng.integers(0,len(G),len(G))]).mean()
 return float(np.mean(vals)),float(np.quantile(b,.025)),float(np.quantile(b,.975)),float(2*min((b<=0).mean(),(b>=0).mean()))
tf=int(sys.argv[1]);z=resample(tf);o=z.open.to_numpy(float);h=z.high.to_numpy(float);l=z.low.to_numpy(float);c=z.close.to_numpy(float);N=len(z)
prev=np.r_[np.nan,c[:-1]];tr=np.maximum(h-l,np.maximum(np.abs(h-prev),np.abs(l-prev)));atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy();med=pd.Series(atr).rolling(max(100,int(240/tf)),min_periods=50).median().to_numpy();vr=atr/med
trend=np.full(N,np.nan);trend[20:]=(c[20:]-c[:-20])/atr[20:]
et=z.ts_event.dt.tz_convert('America/New_York');minute=et.dt.hour.to_numpy()*60+et.dt.minute.to_numpy();sess=np.select([(minute>=1080)|(minute<120),minute<480,minute<570,minute<690,minute<810,minute<960],['Asia','London','NY premarket','NY AM','NY lunch','NY PM'],default='Post-market');vreg=np.select([vr<.8,vr<=1.2],[0,1],default=2);treg=np.select([trend<=-1,trend<-.25,trend<=.25,trend<1],[-2,-1,0,1],default=2)
bull=np.zeros(N,bool);bear=np.zeros(N,bool);bull[2:]=l[2:]>h[:-2];bear[2:]=h[2:]<l[:-2];isf=bull|bear;valid=np.isfinite(atr)&np.isfinite(vr)&np.isfinite(trend)&~rollblock(z.ts_event)&(np.arange(N)<N-30)
ix=np.flatnonzero(isf&valid);d=np.where(bull[ix],1,-1).astype(np.int8);lo=np.where(d==1,h[ix-2],h[ix]);up=np.where(d==1,l[ix],l[ix-2]);wd=up-lo;dist=np.where(d==1,c[ix]-up,lo-c[ix]);keep=(wd>0)&(dist>=0);ix=ix[keep];d=d[keep];lo=lo[keep];up=up[keep];wd=wd[keep];dist=dist[keep];wa=wd/atr[ix];da=dist/atr[ix]
sel=rng.choice(len(ix),min(12000,len(ix)),replace=False);sx=ix[sel];sd=d[sel];slo=lo[sel];sup=up[sel];swa=wa[sel];sda=da[sel]
cand=np.flatnonzero((~isf)&valid);tb=minute//60 if tf<=60 else minute//240;G=defaultdict(list)
for rr in cand:G[(sess[rr],int(vreg[rr]),int(treg[rr]),int(tb[rr]))].append(rr)
G={k:np.asarray(v,np.int64) for k,v in G.items()};cr=[];cp=[];cd=[];clo=[];cup=[]
for q,rr0 in enumerate(sx):
 arr=G.get((sess[rr0],int(vreg[rr0]),int(treg[rr0]),int(tb[rr0])))
 if arr is None:continue
 for rr in rng.choice(arr,3,replace=len(arr)<3):
  wid=swa[q]*atr[rr];dst=sda[q]*atr[rr]
  if sd[q]==1:U=c[rr]-dst;L=U-wid
  else:L=c[rr]+dst;U=L+wid
  cr.append(rr);cp.append(q);cd.append(sd[q]);clo.append(L);cup.append(U)
cr=np.asarray(cr,np.int64);cp=np.asarray(cp,np.int64);cd=np.asarray(cd,np.int8);clo=np.asarray(clo);cup=np.asarray(cup)
ftf,rf=midrace(sx.astype(np.int64),sd,slo,sup,l,h,20,10);ftc,rc=midrace(cr,cd,clo,cup,l,h,20,10)
ctl=pd.DataFrame({'p':cp,'r':rc})
def rate(g):
 r=g.r.to_numpy();m=(r==1)|(r==-1);return np.mean(r[m]==1) if m.any() else np.nan
pr=ctl.groupby('p').apply(rate,include_groups=False);pp=pr.index.to_numpy();fr=np.where(rf==1,1.,np.where(rf==-1,0.,np.nan));m=np.isfinite(fr[pp])&np.isfinite(pr.to_numpy());diff=fr[pp][m]-pr.to_numpy()[m];dates=et.iloc[sx[pp][m]].dt.date.to_numpy();mean,loCI,hiCI,p=boot(diff,dates)
row={'timeframe_min':tf,'N_FVG_total':len(ix),'N_matched':len(sx),'N_resolved_paired':int(m.sum()),'FVG_midpoint_reject_rate':float(np.mean(fr[pp][m])),'Control_midpoint_reject_rate':float(np.mean(pr.to_numpy()[m])),'Difference':mean,'CI_low':loCI,'CI_high':hiCI,'p_day_boot':p,'FVG_ambiguous_rate':float(np.mean(rf==2)),'Control_ambiguous_rate':float(np.mean(rc==2)),'Control_midpoint_touched_20bars':float(np.mean(ftc>=0)),'FVG_midpoint_touched_20bars':float(np.mean(ftf>=0))}
out=f'/mnt/data/midpoint_tf{tf}.csv';pd.DataFrame([row]).to_csv(out,index=False);print(row)
years=et.iloc[sx[pp][m]].dt.year.to_numpy(); yy=[]
for y in np.unique(years):
 mm=years==y
 yy.append({'timeframe_min':tf,'year':int(y),'N':int(mm.sum()),'FVG_rate':float(np.mean(fr[pp][m][mm])),'Control_rate':float(np.mean(pr.to_numpy()[m][mm])),'Difference':float(np.mean(diff[mm]))})
pd.DataFrame(yy).to_csv(f'/mnt/data/midpoint_year_tf{tf}.csv',index=False)
