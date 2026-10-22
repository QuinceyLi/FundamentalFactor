"""Compute daily 10-session IC for the local f1_new ... f7_new factors.

The reference catalog specifies the date window, horizon, minimum paired
cross-section and summary fields, but does not specify its return-price or
universe rules. This script records the local choices with the results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/mongo')
OUTPUT = ROOT / 'outputs' / 'ff_ic_2019_2024'
FACTOR_NAMES = tuple(f'f{i}_new' for i in range(1, 8))
SIGNAL_START = 20190102
TARGET_CUTOFF = 20241231
HORIZON = 10
MIN_ASSETS = 5


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    return feather.read_table(path, columns=columns).to_pandas()


def _calendar() -> pd.DataFrame:
    days = np.sort(_read(DATA / 'public_trade_calendar_sina.feather', ['date']).date.unique())
    if len(days) != len(set(days)):
        raise ValueError('Duplicate calendar dates')
    valid = (days[:-HORIZON] >= SIGNAL_START) & (days[HORIZON:] <= TARGET_CUTOFF)
    return pd.DataFrame({'date': days[:-HORIZON][valid], 'target_date': days[HORIZON:][valid]})


def _nonduplicate(frame: pd.DataFrame, name: str) -> None:
    duplicate = frame.duplicated(['date', 'code'])
    if duplicate.any():
        example = frame.loc[duplicate, ['date', 'code']].head(3).to_dict('records')
        raise ValueError(f'{name} has {int(duplicate.sum())} duplicate date/code keys: {example}')


def _paired_data(calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    factors = _read(DATA / 'fs_quality_fnew.feather', ['date', 'code', *FACTOR_NAMES])
    factors = factors[factors.date.isin(calendar.date)].copy()
    _nonduplicate(factors, 'factors')

    needed_dates = np.union1d(calendar.date.to_numpy(), calendar.target_date.to_numpy())
    quotes = _read(DATA / 'quotefac_repaired.feather', ['date', 'code', 'acc_close'])
    quotes = quotes[quotes.date.isin(needed_dates)].copy()
    _nonduplicate(quotes, 'quotes')

    paired = factors.merge(calendar, on='date', how='inner', validate='many_to_one')
    paired = paired.merge(quotes.rename(columns={'acc_close': 'start_close'}),
                          on=['date', 'code'], how='left', validate='many_to_one')
    future = quotes.rename(columns={'date': 'target_date', 'acc_close': 'target_close'})
    paired = paired.merge(future, on=['target_date', 'code'], how='left', validate='many_to_one')
    start = paired.start_close.to_numpy(dtype=float)
    end = paired.target_close.to_numpy(dtype=float)
    valid_price = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0)
    paired['forward_return'] = np.where(valid_price, end / start - 1, np.nan)
    metadata = {
        'factor_rows_in_signal_window': int(len(factors)),
        'quote_rows_in_needed_dates': int(len(quotes)),
        'paired_rows': int(len(paired)),
        'valid_forward_returns': int(np.isfinite(paired.forward_return).sum()),
        'signal_dates_in_calendar': int(len(calendar)),
        'first_signal_date_in_calendar': str(int(calendar.date.min())),
        'last_signal_date_in_calendar': str(int(calendar.date.max())),
    }
    return paired[['date', 'target_date', 'code', *FACTOR_NAMES, 'forward_return']], metadata


def _daily_ic(paired: pd.DataFrame) -> pd.DataFrame:
    results = []
    for (date, target_date), block in paired.groupby(['date', 'target_date'], sort=True):
        y_all = block.forward_return.to_numpy(dtype=float)
        for factor in FACTOR_NAMES:
            x_all = block[factor].to_numpy(dtype=float)
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            x, y = x_all[valid], y_all[valid]
            ic = rank_ic = np.nan
            if len(x) >= MIN_ASSETS and np.ptp(x) > 0 and np.ptp(y) > 0:
                ic = float(np.corrcoef(x, y)[0, 1])
                rank_ic = float(np.corrcoef(rankdata(x, method='average'),
                                            rankdata(y, method='average'))[0, 1])
            results.append({'factor': factor, 'date': int(date), 'target_date': int(target_date),
                            'n_assets': int(len(x)), 'ic': ic, 'rank_ic': rank_ic})
    return pd.DataFrame(results)


def _summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, block in daily.groupby('factor', sort=False):
        valid = block[np.isfinite(block.ic) & np.isfinite(block.rank_ic)]
        if valid.empty:
            raise ValueError(f'{factor} has no valid IC dates')
        rows.append({
            'factor': factor,
            'mean_ic': float(valid.ic.mean()),
            'mean_rank_ic': float(valid.rank_ic.mean()),
            'rank_ic_std': float(valid.rank_ic.std(ddof=1)),
            'ic_dates': int(len(valid)),
            'mean_assets': float(valid.n_assets.mean()),
            'first_ic_date': pd.to_datetime(str(int(valid.date.min()))).strftime('%Y-%m-%d'),
            'last_ic_date': pd.to_datetime(str(int(valid.date.max()))).strftime('%Y-%m-%d'),
            'signal_start': '2019-01-02',
            'target_cutoff': '2024-12-31',
            'horizon_trading_days': HORIZON,
            'min_cross_section_assets': MIN_ASSETS,
            'source': '/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather',
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar = _calendar()
    paired, metadata = _paired_data(calendar)
    daily = _daily_ic(paired)
    summary = _summary(daily)
    daily.to_csv(OUTPUT / 'daily_ic.csv', index=False, float_format='%.15g')
    summary.to_csv(OUTPUT / 'summary.csv', index=False, float_format='%.15g')
    metadata.update({
        'factor_source': str(DATA / 'fs_quality_fnew.feather'),
        'price_source': str(DATA / 'quotefac_repaired.feather'),
        'calendar_source': str(DATA / 'public_trade_calendar_sina.feather'),
        'factor_name_mapping': {f'f{i}': f'f{i}_new' for i in range(1, 8)},
        'return_definition': 'acc_close[target_date] / acc_close[signal_date] - 1',
        'calendar_definition': 'target_date is 10 market-calendar sessions after signal_date',
        'universe_definition': 'date/code intersection of factor and adjusted close files; no additional stock filters',
        'ic_definition': 'daily cross-sectional Pearson; Spearman as Pearson of average-tie ranks',
        'cross_section_rule': 'at least 5 finite factor/return pairs and nonconstant inputs',
        'summary_definition': 'unweighted mean across valid dates; rank_ic_std uses sample ddof=1; mean_assets averages paired counts on valid dates',
        'reference_limitations': 'Reference does not specify price field, universe, PIT timing, missing returns, or standard-deviation convention; these results are local-method estimates.',
    })
    (OUTPUT / 'method.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps({k: metadata[k] for k in ['factor_rows_in_signal_window', 'paired_rows',
                                                  'valid_forward_returns', 'signal_dates_in_calendar']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
