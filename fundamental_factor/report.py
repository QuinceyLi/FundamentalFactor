"""The single offline research-report entry point, using frozen real artifacts."""
from __future__ import annotations
import base64
import gzip
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ASSETS = Path(__file__).with_name('report_assets')
MARKET_DATA = Path(__file__).with_name('report_data')
VARIANT_NAMES = {'raw':'原始信号','industry':'行业中性','size':'市值中性',
                 'industry_size':'行业及市值中性','full':'行业及风格中性'}

def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}

def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15))

def _resolve_root(run_dir) -> Path:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Run directory does not exist: {root}')
    if root.parent.name == 'factors' and (root.parent.parent/'config.frozen.json').is_file():
        root = root.parent.parent
    return root

def _factor_folder(root: Path, factor: dict):
    name = str(factor.get('name',''))
    if not name or name in {'.','..'} or Path(name).name != name or '\\' in name:
        return None
    expected = root/'factors'/name
    given = root/str(factor.get('output_dir',f'factors/{name}'))
    if given.resolve() != expected.absolute() or not given.resolve().is_relative_to(root):
        return None
    return given

def _execution(folder: Path, settings: dict) -> dict:
    account={'status':settings.get('status','ready' if settings.get('enabled') else 'disabled'),
             'issues':list(settings.get('blocking',[])), 'weekly':[], 'performance':[]}
    if not settings.get('enabled'):
        return account
    weekly=_csv(folder/'execution_weekly.csv');performance=_csv(folder/'execution_performance.csv')
    required={'entry','exit','variant','portfolio','nav_start'}
    if weekly.empty or performance.empty or not required.issubset(weekly):
        account['issues'].append('执行账本或绩效文件缺失，未展示残留执行值。')
    elif weekly.duplicated(['entry','exit','variant','portfolio']).any():
        account['issues'].append('执行账本存在重复账户期间。')
    else:
        first=weekly.sort_values('entry').groupby(['variant','portfolio'],sort=False).head(1)
        initial=pd.to_numeric(first.nav_start,errors='coerce')
        if not (np.isfinite(initial)&initial.gt(0)).all():
            account['issues'].append('执行账户初始资产无效。')
    if account['issues']:
        account['status']='unavailable'
    else:
        account['weekly']=_records(weekly);account['performance']=_records(performance)
    return account

def build_regimes(entries, source: Path | None = None) -> tuple[dict, dict]:
    """Labels use a close strictly before the entry day, without peak backfill."""
    source = source or MARKET_DATA/'csi_all_share.csv'
    meta = load_json(source.with_suffix('.json')) | {'short_window':20,'long_window':120,
        'threshold':.03,'timing':'入场前最近交易日收盘','benchmark_role':'行情分类，不替代收益基准',
        'max_source_lag_days':7,'staleness_policy':'仅在指数历史末端检查7日陈旧；历史覆盖内保留节假日前最近收盘'}
    out = {str(d)[:10]:{'state':'unknown','date':None,'gap':None} for d in entries}
    if not source.is_file():
        meta['status']='missing_source'
        return out,meta
    frame = pd.read_csv(source,usecols=['date','close'])
    frame['date']=pd.to_datetime(frame.date)
    frame=frame.sort_values('date')
    if frame.date.duplicated().any() or not np.isfinite(frame.close).all() or frame.close.le(0).any():
        raise ValueError('行情分类指数存在重复日期或无效价格')
    frame['gap']=frame.close.rolling(20,min_periods=20).mean()/frame.close.rolling(120,min_periods=120).mean()-1
    times=frame.date.to_numpy()
    for day in out:
        entry=pd.Timestamp(day)
        i=times.searchsorted(entry.to_datetime64(),side='left')-1
        if i<0:
            continue
        row=frame.iloc[i]
        if pd.isna(row.gap) or (entry>frame.date.max() and (entry-row.date).days>7):
            continue
        out[day]={'state':'bull' if row.gap>.03 else 'bear' if row.gap<-.03 else 'sideways',
                  'date':row.date.strftime('%Y-%m-%d'),'gap':float(row.gap)}
    meta.update(status='ready',first_date=str(frame.date.min().date()),last_date=str(frame.date.max().date()),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        unknown_periods=sum(x['state']=='unknown' for x in out.values()))
    return out,meta

def build_report_data(run_dir) -> dict:
    root=_resolve_root(run_dir)
    config=load_json(root/'config.frozen.json');audit=load_json(root/'audit.json')
    manifest=load_json(root/'panel_manifest.json');results=load_json(root/'results.json')
    failure=load_json(root/'failure.json');issues=list(audit.get('blocking',[]))
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() if config else None
    if config.get('schema_version')!=2: issues.append('冻结配置不是 schema_version=2。')
    if not audit.get('gate_passed'): issues.append('尚未通过数据审计。')
    if manifest.get('status')!='ready': issues.append('面板尚未就绪。')
    if results.get('status')!='completed' or results.get('schema_version')!=2: issues.append('尚无已完成的当前回测结果。')
    if results and not (digest==audit.get('config_digest')==manifest.get('config_digest')==results.get('config_digest')):
        issues.append('冻结配置、审计、面板和结果的版本指纹不一致。')
    if results and results.get('dataset_kind')!=config.get('dataset_kind'): issues.append('结果数据类型与冻结配置不一致。')
    if failure: issues.append(str(failure.get('message','存在失败记录。')))
    frequency=config.get('backtest',{}).get('rebalance_frequency','weekly')
    names=config.get('neutralization',{}).get('variants',[])
    if not names: names=list(dict.fromkeys(v for f in results.get('factors',[]) for v in f.get('variants',[])))
    labels={v:VARIANT_NAMES.get(v,v) for v in names}
    if config.get('smoothing',{}).get('method')=='ema' and 'raw' in labels: labels['raw']='未中性化平滑信号'
    payload={'config':config,'meta':{'frequency':frequency,'period_label':'月' if frequency=='monthly' else '周',
        'annualization':config.get('metrics',{}).get('annualization',12 if frequency=='monthly' else 52),
        'variants':labels,'sample_policy':config.get('neutralization',{}).get('sample_policy','common'),
        'industry_classification':config.get('industry',{}).get('classification','native'),
        'dataset_kind':config.get('dataset_kind'),'config':config,
        'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'config_digest':digest,'report_version':1,'issues':issues},
        'factors':[],'benchmark':[],'regimes':{},'profiles':{},
        'industry_mapping':_records(_csv(root/'industry_mapping.csv')),
        'audit':audit,'quality':manifest,'data_usage':_records(_csv(root/'data_usage.csv')),'links':[],'issues':issues}
    configured={str(f.get('name')):f for f in config.get('factors',[])}
    for f in results.get('factors',[]):
        if f.get('name') not in configured or _factor_folder(root,f) is None: issues.append('无效因子输出路径或因子不属于当前冻结配置。')
    for filename in ['config.frozen.json','audit.json','industry_mapping.csv','coverage.csv','data_usage.csv','data_quality_summary.csv','unresolved_labels.csv']:
        if (root/filename).is_file(): payload['links'].append({'name':filename,'href':filename})
    if issues: return payload
    coverage=_csv(root/'coverage.csv')
    for item in results.get('factors',[]):
        folder=_factor_folder(root,item);name=item['name']
        f={'name':name,'label':item.get('label',configured[name].get('label',name)),
           'summary':load_json(folder/'summary.json') or item,
           'coverage':_records(coverage.loc[coverage.factor.eq(name)]) if 'factor' in coverage else [],
           'sample':item.get('sample',{}),
           'evaluation_window':item.get('evaluation_window',{})}
        if not f['evaluation_window']:
            f['evaluation_window']=f['summary'].get('evaluation_window',{})
        for key,filename in [('portfolio','portfolio_weekly.csv'),('ic','ic_weekly.csv'),('performance','performance.csv'),('annual','annual_performance.csv'),('ic_summary','ic_summary.csv'),('excluded_periods','excluded_periods.csv')]:
            f[key]=_records(_csv(folder/filename))
            if (folder/filename).is_file(): payload['links'].append({'name':f'{name}/{filename}','href':f'factors/{quote(name)}/{filename}'})
        f['execution']=_execution(folder,item.get('execution',{}))
        payload['factors'].append(f)
    if not labels:
        labels.update({v:VARIANT_NAMES.get(v,v) for f in payload['factors'] for v in dict.fromkeys(r.get('variant') for r in f['portfolio']) if v})
    payload['benchmark']=_records(_csv(root/'benchmark.csv'))
    entries=sorted({str(r['entry'])[:10] for f in payload['factors'] for r in f['portfolio']})
    source=config.get('report',{}).get('market_index')
    payload['regimes'],payload['meta']['regime']=build_regimes(entries,Path(source) if source else None)
    from .report_profiles import build_profiles
    payload['profiles']=build_profiles(root,config,[f['name'] for f in payload['factors']])
    return payload

def report(run_dir) -> Path:
    root=_resolve_root(run_dir);payload=build_report_data(root)
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=str)
    (root/'report_data.json').write_text(encoded,encoding='utf-8')
    # Large historical runs stay self-contained while avoiding a 100 MB HTML file.
    if len(encoded)>1_000_000:
        embedded=json.dumps({'encoding':'gzip-base64','data':base64.b64encode(
            gzip.compress(encoded.encode('utf-8'),compresslevel=6,mtime=0)).decode('ascii')},separators=(',',':'))
    else:
        embedded=encoded
    css=(ASSETS/'style.css').read_text(encoding='utf-8')
    font=ASSETS/'report.woff'
    if font.is_file(): css="@font-face{font-family:ReportSans;src:url(data:font/woff;base64,"+base64.b64encode(font.read_bytes()).decode()+") format('woff');font-display:swap}"+css
    document=(ASSETS/'template.html').read_text(encoding='utf-8')
    document=document.replace('__CSS__',css).replace('__JS__',(ASSETS/'app.js').read_text(encoding='utf-8'))
    document=document.replace('__DATA__',embedded.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029'))
    output=root/'report.html';output.write_text(document,encoding='utf-8')
    # Old bookmarks open the same UI; no legacy rendering code is retained.
    folders={p.parent for p in (root/'factors').glob('*/report.html') if not p.is_symlink()}
    folders.update(root/'factors'/f['name'] for f in payload['factors'])
    for folder in folders:
        if not folder.resolve().is_relative_to(root) or folder.is_symlink(): continue
        folder.mkdir(parents=True,exist_ok=True)
        href=html.escape('../../report.html?factor='+quote(folder.name,safe=''),quote=True)
        (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+href+'"><title>因子研究报告</title><a href="'+href+'">打开统一报告</a>',encoding='utf-8')
    return output
