from __future__ import annotations

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm


def mean_interval(values, lag: int) -> dict:
    series = np.asarray(values, dtype=float)
    good = np.isfinite(series)
    if good.sum() < lag + 2:
        return {"n": int(good.sum()), "mean": float(np.nanmean(series)) if good.any() else None,
                "se": None, "lower": None, "upper": None, "status": "insufficient_length"}
    # No compression across missing calendar weeks: formal inference fails visibly.
    if not good.all():
        return {"n": int(good.sum()), "mean": float(np.nanmean(series)), "se": None,
                "lower": None, "upper": None, "status": "missing_calendar_weeks"}
    fit = sm.OLS(series, np.ones((len(series), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True})
    mean, se = float(fit.params[0]), float(fit.bse[0])
    width = norm.ppf(.975) * se
    return {"n": len(series), "mean": mean, "se": se, "lower": mean - width,
            "upper": mean + width, "status": "ok"}

