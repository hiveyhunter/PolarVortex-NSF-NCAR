import numpy as np
import pandas as pd
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from dataclasses import dataclass, field, asdict
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage as ward_linkage


# vortex-state classifier; decision tree
# One winter season of per-day TDA diagnostics (from compute_flags, which
# reads the vtxplt NetCDF) is turned into one state label per day. There are
# nine states. Temperature triggers a warming; wind reformation ends it;
# geometry (geopotential lobes, easterly ring) only refines the *type*.
#
# per-day inputs (compute_flags):
# warm7_max        max 7-day in-vortex ΔT over 3-D temperature blocks [K]
# warming temperature footprint: T_in_geo_* (geopotential-contour)
# peak_U, mean_U   westerly ring peak / volume-mean zonal wind [m/s]
# region_speed     watershed-refined coherent-jet-core mean speed [m/s]
# (primary strength signal; tracks the jet core, not the
# broad mask; preferred over mean_U)
# n_ring_levels    # levels carrying a circumpolar (H1-confirmed) westerly
# ring. 0 when no component encloses the pole, so weak
# non-circumpolar flow does not read as a ring.
# east_intact_level  a closed easterly ring exists on some level
# comp1_present, area_fracs, geo_tilt, base_aspect, bottom-centroid lat
# second geopotential lobe / displacement / split geometry
#
# the hierarchy (top wins). Each day resolves to the first match:
#
# warming?  warm7_max >= 25 K (the max rise over any 1-7 day window at a
# │            fixed temperature block; a single day qualifies) and the
# │            geopotential lobe is still intact, i.e. enclosed by the jet.
# │            -> this is a warming day. Temperature is the trigger and sits
# │               at the top. Once the lobe is displaced/split/tilted/
# │               stretched its temperature samples lower-latitude air and no
# │               longer measures polar warming, so such days do not count.
# │   The warming continues until the wind reforms (strength risen back
# │   toward strong and rings returned across levels) or end-of-season. Wind
# │   reformation, not temperature dropping, closes the event, so the close
# │   is the same regardless of which temperature source drove the trigger.
# │
# │   type of warming (per event window):
# │     major vs minor : 7-day temperature rise reaches the major threshold
# │                      (>=30 K at/below 10 hPa, or >=40 K above) -> major
# │     morphology     : two large geo lobes        -> split
# │                      one displaced lobe         -> displaced
# │                      Y-shaped partial separation-> partial split
# │                      both displaced & split runs-> mixed
# │
# ├─ no warming today, but inside/adjacent an event window -> stays in that
# │     warming event (windows are contiguous; recovery may split a long one
# │     only where warm7 has actually dropped below 25 K).
# │
# ├─ end of season?  late-season, winds slowing, temperature relaxing,
# │     no further coherent vortex -> end of season (absorbs everything
# │     after; resets the post-warming guard).
# │
# ├─ post-warming gap (between an event and the next event / EOS):
# │     the gap is recovery by default; a warming is never handed straight
# │     to another warming or to a geo disturbance.
# │       weak recovery : winds rising from the warming low (short positive
# │                       trend in region_speed); need not regain full
# │                       strength. No high absolute threshold, no fixed
# │                       multi-day persistence.
# │       strong vortex : strength (region_speed, or mean_U) >= 0.7 * season-
# │                       strong and at least half the vortex altitudes carry
# │                       a circumpolar ring (a collapsed ring is never strong,
# │                       however fast the wind) and temperature stable. Strong ends the
# │                       recovery span; the rest of the gap is strong.
# │
# ├─ geo disturbance?  wind disruption (slowed and/or fewer ring altitudes)
# │     plus a geopotential change with no valid warming. The change is one
# │     of: a small component coming off the vortex, a large vertical centroid
# │     tilt (either sign), or a large horizontal aspect-ratio stretch; the
# │     type is reported in the numerical summary. Allowed only before the
# │     first warming of the season; never after one (the post-warming guard
# │     demotes any later geo-disturbed day to recovery or strong).
# │
# ├─ strong vortex?  strong steady westerlies + stable vortex geopotential.
# │
# └─ early season   (default, pre-vortex): weak winds, cold cap, low-level
# geopotential anomalies, before the season's first strong vortex.
#
# implementation; classify_season() runs these passes in order. Each later
# pass may override earlier paint only as noted; the order encodes the
# hierarchy above:
# 1. detect warming events        (_detect_warming_events_ev): open on
# sustained 25 K, close on wind reformation / recovery / EOS; classify
# type (_classify_event_ev).
# 2. end of season                (_detect_end_of_season_ev), before
# recovery so the last gap knows where to stop.
# 3. recovery painting            : fill every post-warming gap with weak
# recovery, flipping to strong once region_speed + ring + temp say so.
# 4. canonical 25 K enforcement   : reclaim any sustained-25 K day that an
# earlier pass painted non-warming (warming is top of hierarchy);
# brief residual thermal echoes after recovery are not reclaimed.
# 5. geo disturbance (no warming) : paint remaining disrupted days; warming
# days are never overwritten.
# 6. stable-strong check          : a strong day with a real geo-morphology
# anomaly becomes geo-disturbed; otherwise it stays a (quieter) strong
# vortex. (Wind-disturbance is not a separate output state.)
# 7. early-season prefix + despeckle.
# 8. post-warming hierarchy guard : after any warming, no geo-disturbance
# until EOS; demote to recovery (inside a recovery span) or strong.
# finalize: split a long warming only where warm7 actually fell below 25 K;
# merge stray warm-no-geo into adjacent events; enforce a recovery buffer
# between distinct warmings; assign major/minor; early-season prefix.
#
# so the same season can be classified from each and compared.

# Integer state codes. All 11 states are always defined so that hand-mapping
# a cluster to "displaced minor" (7) or "split minor" (9) is always possible;
# the automatic rule-based classifier still only assigns the merged displaced
# (6) / split (8) codes until we add a major/minor rule.
STATE_EARLY              = 1
STATE_STRONG             = 2
STATE_GEO_DISTURBED      = 3   # geo oddity (+ maybe wind), no warming
STATE_WARM_NO_GEO        = 4   # warming, no geo oddity
STATE_SSW_DISPLACED      = 5   # warming + displaced, major
STATE_SSW_DISPLACED_MIN  = 6   # warming + displaced, minor
STATE_SSW_SPLIT          = 7   # warming + split, major
STATE_SSW_SPLIT_MIN      = 8   # warming + split, minor
STATE_RECOVERING         = 9
STATE_END                = 10
STATE_SSW_MIXED_MAJ      = 11  # warming + mixed morphology (displaced → split
                                # or split → rejoin/persistent displaced), major
STATE_SSW_MIXED_MIN      = 12  # warming + mixed morphology, minor
STATE_SSW_PARTIAL_SPLIT      = 13  # warming + partial split (slice b0≥2, no 2nd lobe)
STATE_SSW_PARTIAL_SPLIT_MIN  = 14  # warming + partial split, minor

# Minimum comp-1 / comp-0 area ratio for genuine split morphology.
SPLIT_AREAFRAC = 0.15

# Temperature-based major/minor SSW thresholds (replaces the ring-based test).
# A warming (already >= 25 K, the event criterion) is MAJOR if its peak 7-day
# in-vortex temperature rise reaches MAJOR_DK_AT_OR_BELOW_10HPA anywhere at
# 10 hPa or below (pressure >= 10 hPa, i.e. altitude at/under the 10 hPa
# surface), OR MAJOR_DK_ABOVE_10HPA anywhere above 10 hPa (pressure < 10 hPa,
# higher altitude); otherwise it is MINOR. Minor stays at the 25 K event
# threshold. If the vertical sense is reversed, swap the two numbers.
MAJOR_DK_AT_OR_BELOW_10HPA = 30.0
MAJOR_DK_ABOVE_10HPA       = 40.0
MAJOR_MINOR_SPLIT_HPA      = 10.0

# Partial-split clean-signal threshold: a real partial split keeps its pinch
# in ONE band (bottom or top) with this fraction sustained for >=3 contiguous
# days. Weak, band-flip-flopping b0_partial at a messy event tail is rejected.
PARTIAL_BAND_FRAC          = 0.40

STATE_NAMES = {
    STATE_EARLY:             'early season',
    STATE_STRONG:            'strong',
    STATE_GEO_DISTURBED:     'geo disturbance (no warming)',
    STATE_WARM_NO_GEO:       'warming, no geo disturbance',
    STATE_SSW_DISPLACED:     'warming (displaced, major)',
    STATE_SSW_DISPLACED_MIN: 'warming (displaced, minor)',
    STATE_SSW_SPLIT:         'warming (split, major)',
    STATE_SSW_SPLIT_MIN:     'warming (split, minor)',
    STATE_RECOVERING:        'weak recovering',
    STATE_END:               'end of season',
    STATE_SSW_MIXED_MAJ:     'warming (mixed, major)',
    STATE_SSW_MIXED_MIN:     'warming (mixed, minor)',
    STATE_SSW_PARTIAL_SPLIT:     'warming (partial split, major)',
    STATE_SSW_PARTIAL_SPLIT_MIN: 'warming (partial split, minor)',
}

STATE_COLORS = {
    STATE_EARLY:             '#9fb6cc',
    STATE_STRONG:            '#1f5fa8',
    STATE_GEO_DISTURBED:     '#8e8c3a',  # khaki-olive (distinct from all)
    STATE_WARM_NO_GEO:       '#f5cb5c',
    STATE_SSW_DISPLACED:     '#c0392b',
    STATE_SSW_DISPLACED_MIN: '#e67e22',
    STATE_SSW_SPLIT:         '#7b241c',  # split major: deep red
    STATE_SSW_SPLIT_MIN:     '#d98880',  # split minor: light red
    STATE_RECOVERING:        '#6fc3df',
    STATE_END:               '#bdbdbd',
    STATE_SSW_MIXED_MAJ:     '#6c3483',  # purple, dark
    STATE_SSW_MIXED_MIN:     '#a569bd',  # purple, light
    STATE_SSW_PARTIAL_SPLIT:     '#c2185b',  # partial split major: magenta
    STATE_SSW_PARTIAL_SPLIT_MIN: '#e88aad',  # partial split minor: pink
}

# Legend display order. With column-major fill and ncol=7 this lays out as 7
# stacked pairs on two lines, grouping like with like:
#   early/end | strong/weak-recovery | geo-only/warm-only |
#   displaced(maj,min) | split(maj,min) | partial-split(maj,min) | mixed(maj,min)
LEGEND_ORDER = [
    STATE_EARLY,             STATE_END,
    STATE_STRONG,            STATE_RECOVERING,
    STATE_GEO_DISTURBED,     STATE_WARM_NO_GEO,
    STATE_SSW_DISPLACED,     STATE_SSW_DISPLACED_MIN,
    STATE_SSW_SPLIT,         STATE_SSW_SPLIT_MIN,
    STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN,
    STATE_SSW_MIXED_MAJ,     STATE_SSW_MIXED_MIN,
]


# Auto-calibrated thresholds. Fields default to None where the sensible
# default is a data-driven percentile or statistic of the input season;
# those get filled in by _calibrate_rules. Override explicitly if you
# want a fixed cutoff.
@dataclass
class StateRules:
    # auto-calibrated from the season itself (None => compute)
    strong_peak_U:         float = None     # median of top quartile of peak_U
    wind_disturb_peak_U:   float = None     # low pct: only *very* weak jets
    end_peak_U:            float = None     # 20th percentile
    end_dT_abs:            float = None     # 20th percentile of |dT_col|
    end_T_warmer_than:     float = None     # early-season T_polar median
    end_anom_floor_K:      float = None     # minimum T anom max to call end
    end_post_warming_days: int   = 2        # wait this many days after the
                                            # last warming-related run ends
    warming_T_anom_K:      float = None     # 1.5 * seasonal IQR above median
    warming_dT_K_per_day:  float = None     # 85th percentile of dT_upper

    # fixed structural parameters
    warming_alt_lo_km:     float = 20.0
    warming_alt_hi_km:     float = 50.0
    geo_disp_lat:          float = 72.0
    geo_aspect_bot:        float = 1.8
    geo_second_area_frac:  float = 0.25
    n_early_days:          int   = 25
    # Calendar convention: these leading days are *always* STATE_EARLY after
    # all passes (so the strip starts with early season, not stray warming).
    # Keep ≤ ~2 weeks so mid‑Nov geo / wind events are not erased.
    n_early_season_force_days: int = 16
    n_end_window_days:     int   = 60    # search window for end of season
    min_end_run_days:      int   = 10    # end segment must be at least this
    recovery_window_days:  int   = 14
    recovery_peak_U_frac:  float = 0.85  # below this fraction of strong_peak_U
    min_run_length:        int   = 3
    smooth:                int   = 3

    # Legacy major/minor demotion parameters, no longer used: severity is now
    # decided solely by the 7-day temperature rise (see refine_major_minor).
    ssw_major_min_days:     int   = 7
    ssw_major_disp_lat:     float = 70.0
    ssw_major_disp_drop:    float = 1.0
    ssw_major_split_hi_frac:float = 0.5  # fraction of event days with
                                         # >=2 components above 30 km
    # Minimum |geo centroid tilt| (deg/km) for a *stretched* geopotential
    # (high base aspect) to count toward geo disturbance without a split.
    geo_disturb_min_tilt:  float = None
    # When bridging split-SSW gaps: equatorward extent of the vortex (°N)
    split_bridge_lat_lo:   float = 38.0
    split_bridge_lat_hi:   float = 50.0
    split_bridge_max_gap:  int   = 14
    # If polar cap warms but geo flags lag, upgrade WARM_NO_GEO → SSW when
    # geo split/elongation appears within this many days.
    warming_geo_lag_days:  int   = 8
    # Centroid / lowest-lat must trend equatorward (°N / day) for displaced.
    disp_move_deg_per_day: float = 0.12
    # Days to look back for “wind disturbed” when classifying displaced SSW.
    wind_precursor_lookback: int = 5
    # Geo *displacement* without same-day d(lat)/dt: vortex bottom latitude
    # drops at least this far equatorward within disp_bottom_low_window_days.
    disp_bottom_low_lat_max: float = 64.0
    disp_bottom_low_window_days: int = 15
    equatorward_recent_days: int = 8
    # Strong jet: peak_U above this fraction of strong_peak_U resists
    # “broken jet” from topology alone (fewer false wind disturbances).
    jet_resilience_frac:     float = 0.62
    # Pull wind/geo labels into following SSW runs (sequential event).
    event_precursor_days:  int   = 14
    # SSW thermal gate: max polar-cap-mean ΔT over lag 1..7 days at the
    # most-warming altitude must reach this to open or label an event
    # (canonical 25 K trigger).
    ssw_warm25_K:          float = 25.0
    # Minimum number of warmed (>= ssw_warm25_K) 10deg x 30deg x 4km
    # blocks required for a day to count as a warming trigger.  A real
    # SSW warms a tall, broad region (tens of blocks); a shallow noisy
    warm_min_blocks:       int   = 15
    # Event criterion is warm7_max >= ssw_warm25_K (25 K); nothing below 25 K
    # opens or labels an event. Major vs minor is decided by the 7-day
    # temperature rise (>=30 K at/below 10 hPa, or >=40 K above), not the wind.
    warm_rate_min_K:       float = 0.0
    # Geo split support from component-1 geometry relative to component-0.
    split_comp1_near_lat_deg: float = 12.0
    split_comp1_near_lon_deg: float = 35.0
    split_comp1_min_lat:      float = 45.0
    # If peak_U is above this fraction of strong_peak_U and the jet is intact,
    # ignore marginal thermal bumps (no WARM_NO_GEO / SSW) unless polar
    # warming_for_ssw is true.
    ignore_marginal_warming_strong_frac: float = 0.82
    # Latest month a new warming ONSET is allowed (Nov, Dec, Jan..this month).
    # Month 1 = January … 12 = December; default 3 = onset only in Nov-Mar.
    warming_last_onset_month: int = 3
    # Event-extension cap only, NOT an onset limit: an event whose window runs
    # past the winter months may extend at most this many days into the
    # off-season before EOS takes over.
    warming_max_extension_april_days: int = 7


# Fill in None fields by computing sensible data-driven defaults from
# the per-day diagnostic arrays. Any explicit values are left untouched.
def calibrate_rules(flags, r):
    peak = flags['peak_U']
    finite_peak = peak[np.isfinite(peak)]
    dTabs       = np.abs(flags['dT_col'])
    Tp          = flags['Tp_col']
    anom        = flags['warm_anom_max']
    dT_up       = flags['warm_dT_max']

    def q(arr, p, default):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return default
        return float(np.percentile(arr, p))

    if r.strong_peak_U is None:
        # anchor "strong" at the upper-quartile of peak U seen this season
        r.strong_peak_U = q(finite_peak, 75, 40.0)
    if r.wind_disturb_peak_U is None:
        # Low percentile so only unusually weak peaks count; cuts false
        # “wind disturbance” during otherwise stable jets.
        r.wind_disturb_peak_U = q(finite_peak, 22, 22.0)
    if r.end_peak_U is None:
        r.end_peak_U = q(finite_peak, 20, 15.0)
    if r.end_dT_abs is None:
        r.end_dT_abs = q(dTabs, 25, 0.4)
    if r.end_T_warmer_than is None:
        # early-season polar cap T (first 25 days median)
        early = Tp[: min(r.n_early_days, len(Tp))]
        early = early[np.isfinite(early)]
        r.end_T_warmer_than = (float(np.median(early)) + 3.0
                               if early.size else 200.0)
    if r.warming_T_anom_K is None:
        # 1.5 * IQR above seasonal median
        med = q(anom, 50, 0.0)
        iqr = q(anom, 75, 0.0) - q(anom, 25, 0.0)
        r.warming_T_anom_K = max(5.0, med + 1.5 * iqr)
    if r.warming_dT_K_per_day is None:
        r.warming_dT_K_per_day = max(1.0, q(dT_up, 85, 1.0))
    if r.end_anom_floor_K is None:
        # "still warm" after a warming: T anom max at least ~half of the
        # warming threshold.
        r.end_anom_floor_K = max(2.0, 0.5 * r.warming_T_anom_K)
    if r.geo_disturb_min_tilt is None:
        gt = np.abs(np.asarray(flags['geo_tilt'], dtype=float))
        gt = gt[np.isfinite(gt)]
        r.geo_disturb_min_tilt = max(0.04, q(gt, 70, 0.06))
    return r


def rolling_mean_at(arr, k, win=3):
    # Trailing mean of *arr* ending at index *k* (ignores NaNs).
    lo = max(0, int(k) - int(win) + 1)
    seg = np.asarray(arr, dtype=float)[lo:int(k) + 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if seg.size else np.nan


def train_event_physics(fl, r):
    # Derive recovery / geo-motion parameters from one season's flags.
    n = len(fl['peak_U'])
    warm7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    dT    = np.asarray(fl.get('dT_col', np.full(n, np.nan)), dtype=float)
    peak  = np.asarray(fl.get('peak_U', np.full(n, np.nan)), dtype=float)
    nr    = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)), dtype=float)
    dbc   = np.asarray(fl.get('d_geo_c0_bottom_cent',
                               np.full(n, np.nan)), dtype=float)
    dbl   = np.asarray(fl.get('d_geo_c0_bottom_low',
                               np.full(n, np.nan)), dtype=float)
    rw    = fl.get('ring_present_west')

    def qseg(arr, mask, p, default):
        seg = np.asarray(arr, dtype=float)[mask]
        seg = seg[np.isfinite(seg)]
        return float(np.percentile(seg, p)) if seg.size >= 3 else default

    finite_w7 = warm7[np.isfinite(warm7)]
    q25_w7 = float(np.percentile(finite_w7, 25)) if finite_w7.size else 0.0
    q75_w7 = float(np.percentile(finite_w7, 75)) if finite_w7.size else 25.0
    thr25  = float(getattr(r, 'ssw_warm25_K', 25.0))

    # Quiet / strong-vortex days: low localized warming, no easterly.
    quiet = (np.isfinite(warm7) & (warm7 <= q25_w7))
    if int(quiet.sum()) < 5:
        quiet = np.isfinite(peak) & (peak >= np.nanpercentile(
            peak[np.isfinite(peak)], 75) if np.isfinite(peak).any() else 0)

    # 3-day trailing dT/dt on quiet days → cooling threshold.
    dT3 = np.array([rolling_mean_at(dT, k, 3) for k in range(n)])
    dT_cool_thr = qseg(dT3, quiet, 25, -0.05)

    peak_recover_thr = qseg(peak, quiet, 50,
                            float(np.nanpercentile(peak, 50))
                            if np.isfinite(peak).any() else 80.0)
    nr_recover_thr   = qseg(nr, quiet, 50,
                            float(np.nanpercentile(nr, 50))
                            if np.isfinite(nr).any() else 10.0)

    # Joint bottom-geo motion rate: both centroid and lowest lat changing
    # together (min of |dcent|, |dlow| captures "moving in tandem").
    joint = np.minimum(np.abs(dbc), np.abs(dbl))
    joint = np.where(np.isfinite(joint), joint, np.nan)
    warm_mask = (np.isfinite(warm7) &
                 (np.round(warm7, 2) >= thr25))
    warm_joint = joint[warm_mask & np.isfinite(joint) & (joint > 0)]
    all_joint  = joint[np.isfinite(joint) & (joint > 0)]
    if warm_joint.size >= 5:
        geo_warm_thr = float(np.percentile(warm_joint, 25))
    elif all_joint.size >= 5:
        geo_warm_thr = float(np.percentile(all_joint, 25))
    else:
        geo_warm_thr = 1.0
    quiet_joint = joint[quiet & np.isfinite(joint) & (joint > 0)]
    if quiet_joint.size >= 5:
        geo_quiet_thr = float(np.percentile(quiet_joint, 75))
    else:
        geo_quiet_thr = geo_warm_thr

    # Per-altitude ring count on strong-wind days (not only quiet-warm7
    # days, which can have no ring early in the season).
    strong_wind = (np.isfinite(peak) &
                     (peak >= np.nanpercentile(peak[np.isfinite(peak)], 75)
                      if np.isfinite(peak).any() else 0))
    if rw is not None:
        ring_counts = np.array([int(rw[k].sum()) for k in range(n)
                                if strong_wind[k]], dtype=float)
        if ring_counts.size == 0:
            ring_counts = np.array([int(rw[k].sum()) for k in range(n)
                                    if quiet[k]], dtype=float)
        ring_med_strong = (float(np.median(ring_counts))
                           if ring_counts.size else 10.0)
        ring_fracs = [int(rw[k].sum()) / max(ring_med_strong, 1.0)
                      for k in range(n) if strong_wind[k]]
        if len(ring_fracs) < 5:
            ring_fracs = [int(rw[k].sum()) / max(ring_med_strong, 1.0)
                          for k in range(n) if quiet[k]]
        ring_recover_frac = (max(0.50, float(np.percentile(ring_fracs, 25)))
                             if len(ring_fracs) >= 5 else 0.75)
    else:
        ring_med_strong = 10.0
        ring_recover_frac = 0.75

    # Per-altitude wind-match fraction threshold: on quiet days each day
    # should match itself at >= this fraction of reference levels.
    ring_match_frac = 0.80
    if rw is not None and int(quiet.sum()) >= 5:
        fracs = []
        ref_days = np.where(quiet)[0]
        for k in ref_days[::max(1, len(ref_days) // 20)]:
            ref_lv = np.where(rw[k])[0]
            if ref_lv.size == 0:
                continue
            cur = rw[k]
            cur_d = cur.copy()
            cur_d[:-1] |= cur[1:]
            cur_d[1:]  |= cur[:-1]
            fracs.append(float(cur_d[ref_lv].sum()) / ref_lv.size)
        if fracs:
            ring_match_frac = float(np.percentile(fracs, 25))

    # Speed-match fraction: quiet-day self-comparison of west_mean_U.
    speed_match_frac = 0.50
    wu = fl.get('west_mean_U')
    ref_days = np.where(quiet)[0]
    if wu is not None and rw is not None and ref_days.size >= 5:
        ratios = []
        for k in ref_days[::max(1, len(ref_days) // 20)]:
            ref_U = wu[k]; cur_U = wu[k]
            ref_lv = np.where(rw[k])[0]
            ok = tot = 0
            for lev in ref_lv:
                if np.isfinite(ref_U[lev]) and ref_U[lev] > 0:
                    tot += 1
                    if np.isfinite(cur_U[lev]) and cur_U[lev] >= ref_U[lev]:
                        ok += 1
            if tot > 0:
                ratios.append(ok / tot)
        if ratios:
            speed_match_frac = float(np.percentile(ratios, 25))

    # Update displacement rate rule from warming-day joint geo motion.
    warm_rates = joint[warm_mask & np.isfinite(joint)]
    if warm_rates.size >= 5:
        r.disp_move_deg_per_day = float(np.percentile(warm_rates, 25))

    min_rec_run = max(int(r.min_run_length), 3)
    partial_phys = dict(dT_cool_thr=dT_cool_thr,
                        peak_recover_thr=peak_recover_thr,
                        nr_recover_thr=nr_recover_thr,
                        ring_med_strong=ring_med_strong,
                        ring_recover_frac=ring_recover_frac,
                        ring_match_frac=ring_match_frac,
                        speed_match_frac=speed_match_frac)
    if rw is not None:
        sig = [recovery_signature(k, fl, partial_phys, ref_idx=None)
               for k in range(n)]
        run_lens = []
        i = 0
        while i < n:
            if sig[i]:
                j = i
                while j < n and sig[j]:
                    j += 1
                run_lens.append(j - i)
                i = j
            else:
                i += 1
        if run_lens:
            min_rec_run = max(2, int(np.percentile(run_lens, 25)))
    # Quiet-season recovery runs can inflate the 25th pct; cap so jet
    # collapse (2–3 days) can still terminate events like 97/98.
    min_rec_run = min(int(min_rec_run), 5)

    # Weak-recovery rises (3-day Δpeak U / ring-level count) on full-
    # recovery days → thresholds for cycle splitting inside long spells.
    peak_rises = []
    nr_rises = []
    for k in range(3, n):
        dT3 = rolling_mean_at(dT, k, 3)
        lo5 = max(0, k - 4)
        seg5 = dT[lo5:k + 1]
        seg5 = seg5[np.isfinite(seg5)] if seg5.size else seg5
        cooling = ((np.isfinite(dT3) and dT3 <= dT_cool_thr) or
                   (seg5.size >= 3 and float(seg5.sum()) < 0.0))
        if not cooling:
            continue
        j0 = k - 3
        pu_rise = (float(peak[k]) - float(peak[j0])
                   if np.isfinite(peak[k]) and np.isfinite(peak[j0])
                   else np.nan)
        nr_rise = (float(nr[k]) - float(nr[j0])
                   if np.isfinite(nr[k]) and np.isfinite(nr[j0])
                   else np.nan)
        if np.isfinite(pu_rise) and pu_rise > 0:
            peak_rises.append(pu_rise)
        if np.isfinite(nr_rise) and nr_rise > 0:
            nr_rises.append(nr_rise)
    peak_rise_thr = (float(np.percentile(peak_rises, 25))
                     if len(peak_rises) >= 5 else 3.0)
    nr_rise_thr = (float(np.percentile(nr_rises, 25))
                   if len(nr_rises) >= 5 else 1.0)
    partial_phys['peak_rise_thr'] = peak_rise_thr
    partial_phys['nr_rise_thr'] = nr_rise_thr

    weak_sig = [weak_recovery_signature(k, fl, partial_phys, ref_idx=None)
                for k in range(n)]
    weak_runs = []
    i = 0
    while i < n:
        if weak_sig[i]:
            j = i
            while j < n and weak_sig[j]:
                j += 1
            weak_runs.append(j - i)
            i = j
        else:
            i += 1
    min_weak_rec_run = 2
    if weak_runs:
        short = [r for r in weak_runs if r <= 4]
        pool = short if len(short) >= 5 else weak_runs
        min_weak_rec_run = max(2, min(3, int(np.percentile(pool, 25))))

    return dict(
        dT_cool_thr=dT_cool_thr,
        peak_recover_thr=peak_recover_thr,
        nr_recover_thr=nr_recover_thr,
        geo_warm_thr=geo_warm_thr,
        geo_quiet_thr=geo_quiet_thr,
        ring_med_strong=ring_med_strong,
        ring_recover_frac=ring_recover_frac,
        ring_match_frac=ring_match_frac,
        speed_match_frac=speed_match_frac,
        min_rec_run=min_rec_run,
        peak_rise_thr=peak_rise_thr,
        nr_rise_thr=nr_rise_thr,
        min_weak_rec_run=min_weak_rec_run,
        min_jet_rec_run=2,
    )


def jet_reformed(k, fl, phys, evt_start=None, lookback=5):
    # Wind has REFORMED after the warming: the coherent-jet strength has
    if k < 0:
        return False
    phys = phys or {}
    if evt_start is not None and (k - int(evt_start)) < 5:
        return False
    # strength signal: watershed region_speed, fall back to peak_U
    strg = np.asarray(fl.get('region_speed', []), dtype=float)
    if not (strg.size and np.isfinite(strg).any()):
        strg = np.asarray(fl.get('peak_U', []), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if k >= strg.size or not np.isfinite(strg[k]):
        return False
    sf = strg[np.isfinite(strg)]
    if sf.size == 0:
        return False
    strong_ref = float(np.percentile(sf, 75))
    # strength risen back toward strong
    strength_ok = strg[k] >= 0.75 * strong_ref
    # and rising over the recent window (reforming, not just high)
    lo = max(0, k - lookback)
    seg = strg[lo:k + 1]
    seg = seg[np.isfinite(seg)]
    rising = seg.size >= 2 and (seg[-1] - seg[0]) > 0
    # and ring present across enough levels
    ring_med = float(phys.get('ring_med_strong', 10.0))
    nrv = float(nr[k]) if k < nr.size and np.isfinite(nr[k]) else np.nan
    ring_ok = (not np.isfinite(nrv)) or nrv >= max(8.0, 0.6 * ring_med)
    return bool(strength_ok and (rising or strength_ok) and ring_ok)


def jet_collapse_recovery(k, fl, phys, evt_peak_u=None, evt_nr_peak=None,
                           evt_start=None):
    # Jet-ring collapse during an event: peak U drops sharply and the
    if k < 0:
        return False
    phys = phys or {}
    peak = np.asarray(fl.get('peak_U', []), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if k >= peak.size:
        return False
    pk = float(peak[k]) if np.isfinite(peak[k]) else np.nan
    nrv = float(nr[k]) if k < nr.size and np.isfinite(nr[k]) else np.nan
    if not np.isfinite(pk):
        return False

    if evt_start is not None and (k - int(evt_start)) < 12:
        return False

    lo7 = max(0, k - 7)
    recent_pk_max = float(np.nanmax(peak[lo7:k + 1]))
    if not np.isfinite(recent_pk_max) or recent_pk_max <= 0:
        return False
    pk_drop_frac = (recent_pk_max - pk) / recent_pk_max
    peak_collapsed = pk_drop_frac >= 0.35 and pk <= 0.75 * recent_pk_max

    recent_nr_max = (float(np.nanmax(nr[lo7:k + 1]))
                     if k < nr.size and np.isfinite(nr[lo7:k + 1]).any()
                     else np.nan)
    ring_med = float(phys.get('ring_med_strong', 10.0))
    nr_stripped = False
    if np.isfinite(nrv):
        if np.isfinite(recent_nr_max) and recent_nr_max > 0:
            nr_stripped = nrv <= 0.40 * recent_nr_max
        if not nr_stripped:
            nr_stripped = nrv <= 0.55 * ring_med

    had_strong_jet = recent_pk_max >= 0.80 * float(
        phys.get('peak_recover_thr', 80.0))

    lo3 = max(0, k - 2)
    hi3 = k + 1
    pk_now = float(np.nanmean(peak[lo3:hi3]))
    pk_before = float(np.nanmean(peak[max(0, k - 5):max(lo3, k)]))
    rise_thr = float(phys.get('peak_rise_thr', 3.0))
    jet_declining = (
        np.isfinite(pk_now) and np.isfinite(pk_before) and
        pk_now <= pk_before - 0.5 * rise_thr
    )

    severe_collapse = pk <= 0.55 * recent_pk_max

    return bool(
        nr_stripped and had_strong_jet and (
            (peak_collapsed and jet_declining) or
            (severe_collapse and peak_collapsed) or
            (severe_collapse and nr_stripped)
        ))


def wind_matches_reference(k, ref_idx, ring_west, westU, phys):
    # Per-altitude westerly-ring comparison against a reference day.
    if ring_west is None or ref_idx is None:
        return False
    if not (0 <= k < ring_west.shape[0] and 0 <= ref_idx < ring_west.shape[0]):
        return False
    ref_rp = ring_west[ref_idx]
    cur_rp = ring_west[k]
    ref_levels = np.where(ref_rp)[0]
    if ref_levels.size == 0:
        return False
    cur_d = cur_rp.copy()
    cur_d[:-1] |= cur_rp[1:]
    cur_d[1:]  |= cur_rp[:-1]
    frac_match = float(cur_d[ref_levels].sum()) / ref_levels.size
    if frac_match < phys['ring_match_frac']:
        return False
    if westU is not None:
        ref_U = westU[ref_idx]
        cur_U = westU[k]
        speeds_ok = speeds_tot = 0
        for lev in ref_levels:
            if np.isfinite(ref_U[lev]) and ref_U[lev] > 0 and cur_d[lev]:
                speeds_tot += 1
                cur_val = cur_U[lev]
                if (np.isfinite(cur_val) and
                        cur_val >= phys['speed_match_frac'] * ref_U[lev]):
                    speeds_ok += 1
        if speeds_tot > 0 and speeds_ok / speeds_tot < phys['speed_match_frac']:
            return False
    return True


def geo_motion_quiet(k, fl, phys):
    # Bottom geopotential motion rate below warming-active levels.
    dbc = np.asarray(fl.get('d_geo_c0_bottom_cent', []), dtype=float)
    dbl = np.asarray(fl.get('d_geo_c0_bottom_low', []), dtype=float)
    if not dbc.size or not dbl.size or k >= dbc.size:
        return True
    joint_k = np.minimum(np.abs(dbc), np.abs(dbl))
    gmov = rolling_mean_at(joint_k, k, 3)
    gthr = phys.get('geo_quiet_thr', phys.get('geo_warm_thr', np.inf))
    warm_thr = phys.get('geo_warm_thr', 0.0)
    if warm_thr > 0:
        gthr = min(gthr, warm_thr)
    return (not np.isfinite(gmov)) or gmov <= gthr


def geo_returning_poleward(k, fl, lookback=7):
    # Centroid moving poleward or sitting at a recovered latitude.
    bot = np.asarray(fl.get('geo_c0_bottom_cent', []), dtype=float)
    if k < lookback or k >= bot.size:
        return False
    if not (np.isfinite(bot[k]) and np.isfinite(bot[k - lookback])):
        return False
    jump = float(bot[k]) - float(bot[k - lookback])
    lo = max(0, k - lookback)
    seg = bot[lo:k + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size < 3:
        return False
    recent_rise = jump >= 3.0
    high_bc = float(bot[k]) >= float(np.nanpercentile(seg, 65))
    return bool(recent_rise or high_bc)


def ring_level_count(k, fl):
    # Per-altitude westerly ring count (preferred) or n_ring_levels.
    rw = fl.get('ring_present_west')
    if rw is not None and 0 <= k < rw.shape[0]:
        return int(rw[k].sum())
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if k < nr.size and np.isfinite(nr[k]):
        return int(nr[k])
    return 0


def ring_coherent(k, fl, phys):
    # Multi-level westerly ring intact; not a breakup day.
    if k < 0:
        return False
    jet = np.asarray(fl.get('jet_intact', np.zeros(1)), dtype=float)
    intact = (k < jet.size and float(jet[k]) >= 0.5)
    n_ring = ring_level_count(k, fl)
    med = float(phys.get('ring_med_strong', 10.0))
    need = max(8, int(round(0.45 * med)))
    return bool(intact and n_ring >= need)


def ring_disruption_active(k, fl, phys, lookback=7):
    # Ring breakup / jet weakening; warming+geo context, not recovery.
    if k < 0:
        return False
    if ring_coherent(k, fl, phys):
        return False
    lo = max(0, k - lookback)
    had_rings = any(ring_coherent(j, fl, phys) for j in range(lo, k))
    peak = np.asarray(fl.get('peak_U', []), dtype=float)
    peak_drop = False
    if k < peak.size and np.isfinite(peak[k]):
        seg = peak[lo:k + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            recent_max = float(np.nanmax(seg[:-1])) if seg.size > 1 else float(seg[0])
            if recent_max > 0 and float(peak[k]) <= 0.78 * recent_max:
                peak_drop = True
    jet = np.asarray(fl.get('jet_intact', []), dtype=float)
    jet_broken = (k < jet.size and float(jet[k]) < 0.5)
    n_now = ring_level_count(k, fl)
    n_hist = [ring_level_count(j, fl) for j in range(lo, k)]
    n_hist = [n for n in n_hist if n > 0]
    ring_collapse = False
    if n_hist and n_now <= 0.35 * float(max(n_hist)):
        ring_collapse = True
    return bool(had_rings and (jet_broken or ring_collapse or peak_drop or n_now < 8))


def vortex_objectively_strong(k, fl, phys):
    # Season-trained strong jet with coherent multi-level rings.
    if not ring_coherent(k, fl, phys):
        return False
    peak = np.asarray(fl.get('peak_U', []), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if k >= peak.size:
        return False
    pk = float(peak[k]) if np.isfinite(peak[k]) else np.nan
    nr_k = float(nr[k]) if k < nr.size and np.isfinite(nr[k]) else np.nan
    return bool(
        np.isfinite(pk) and pk >= phys.get('peak_recover_thr', 0.0) and
        (not np.isfinite(nr_k) or nr_k >= phys.get('nr_recover_thr', 0.0))
    )


def is_residual_thermal_onset(i, fl, phys):
    # Block warm7 spikes only on a coherent, recovered strong vortex.
    if fl is None or i < 1:
        return False
    # Duration gate: an echo is short. Count how many of the surrounding
    # days are at/above threshold; a sustained run is a warming, not residual.
    w7 = np.asarray(fl.get('warm7_max', []), dtype=float)
    thr = float(phys.get('ssw_thr', 25.0)) if phys else 25.0
    if i < w7.size:
        lo = max(0, i - 3); hi = min(w7.size, i + 4)
        seg = w7[lo:hi]
        n_at = int(np.sum(np.isfinite(seg) & (seg >= thr)))
        if n_at >= 4:          # >=4 of ~7 surrounding days warm -> sustained
            return False
    if geo_poleward_pause(i, fl, phys):
        return False
    if ring_disruption_active(i, fl, phys):
        return False
    if not (geo_recovered_position(i, fl) and
            vortex_objectively_strong(i, fl, phys)):
        return False
    pre_lo = max(0, i - 7)
    n_ok = sum(
        1 for k in range(pre_lo, i)
        if (ring_coherent(k, fl, phys) and
            geo_recovered_position(k, fl) and
            vortex_objectively_strong(k, fl, phys)))
    return n_ok >= 5


def wind_reforming(k, fl, phys, ref_idx=None):
    # Westerly jet strengthening and/or multi-level ring reformation.
    peak = np.asarray(fl.get('peak_U', []), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if k >= peak.size:
        return False
    pk = float(peak[k]) if np.isfinite(peak[k]) else np.nan
    nr_k = float(nr[k]) if k < nr.size and np.isfinite(nr[k]) else np.nan

    wind_strong = (
        np.isfinite(pk) and pk >= phys.get('peak_recover_thr', 0.0) and
        (not np.isfinite(nr_k) or nr_k >= phys.get('nr_recover_thr', 0.0))
    )

    j0 = max(0, k - 3)
    pu_rise = (float(peak[k]) - float(peak[j0])
               if (k < peak.size and j0 < peak.size and
                   np.isfinite(peak[k]) and np.isfinite(peak[j0]))
               else np.nan)
    nr_rise = (float(nr[k]) - float(nr[j0])
               if (k < nr.size and j0 < nr.size and
                   np.isfinite(nr[k]) and np.isfinite(nr[j0]))
               else np.nan)
    wind_rising = (
        (np.isfinite(pu_rise) and pu_rise >= phys.get('peak_rise_thr', 3.0)) or
        (np.isfinite(nr_rise) and nr_rise >= phys.get('nr_rise_thr', 1.0))
    )

    rw = fl.get('ring_present_west')
    wu = fl.get('west_mean_U')
    if rw is not None and ref_idx is not None:
        ring_match = wind_matches_reference(k, ref_idx, rw, wu, phys)
    elif rw is not None:
        need = (max(0.50, phys.get('ring_recover_frac', 0.75)) *
                phys.get('ring_med_strong', 10.0))
        ring_match = int(rw[k].sum()) >= need
    else:
        ring_match = False

    return bool(wind_strong or wind_rising or ring_match)


def geo_recovered_position(k, fl):
    # Centroid at a recovered (poleward) latitude; aligns with strong jet.
    bot = np.asarray(fl.get('geo_c0_bottom_cent', []), dtype=float)
    if k >= bot.size or not np.isfinite(bot[k]):
        return False
    bc = float(bot[k])
    if bc >= 72.0:
        return True
    lo = max(0, k - 10)
    seg = bot[lo:k + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size >= 5 and bc >= float(np.nanpercentile(seg, 75)):
        return True
    return geo_returning_poleward(k, fl)


def wind_geo_recovery(k, fl, phys, ref_idx=None, full=False):
    # Vortex recovering via wind + geopotential; NOT polar-cap cooling.
    if k < 0:
        return False
    if full and not ring_coherent(k, fl, phys):
        return False
    if ring_disruption_active(k, fl, phys):
        return False
    wind_ok = wind_reforming(k, fl, phys, ref_idx)
    if not wind_ok:
        return False
    geo_pos = geo_recovered_position(k, fl)
    if full:
        return bool(geo_pos and ring_coherent(k, fl, phys))
    return bool(geo_pos or geo_motion_quiet(k, fl, phys) or
                geo_returning_poleward(k, fl))


def warming_has_stopped(k, fl, thr, trig=None):
    # Thermal trigger inactive; residual elevated warm7 is not warming.
    w7 = np.asarray(fl.get('warm7_max', np.full(k + 1, np.nan)), dtype=float)
    if k >= w7.size:
        return True
    if trig is not None and k < len(trig) and bool(trig[k]):
        return False
    wv = float(w7[k]) if np.isfinite(w7[k]) else np.nan
    if np.isfinite(wv) and wv >= 0.85 * float(thr):
        j0 = max(0, k - 4)
        if np.isfinite(w7[j0]) and wv >= float(w7[j0]) - 1.5:
            return False
    return True


def recovery_signature(k, fl, phys, ref_idx=None):
    # Wind-ring reformation + geopotential at recovered latitude.
    return wind_geo_recovery(k, fl, phys, ref_idx=None, full=True)


def recovery_stable(k, fl, phys, ref_idx, thr, trig=None):
    # Fully recovered vortex: wind+geo recovery and warming has stopped.
    if not wind_geo_recovery(k, fl, phys, ref_idx, full=True):
        return False
    return warming_has_stopped(k, fl, thr, trig=trig)


def geo_poleward_pause(k, fl, phys):
    # Interior pause: bottom centroid jumps poleward after a low-latitude
    if k < 4:
        return False
    bot = np.asarray(fl.get('geo_c0_bottom_cent', []), dtype=float)
    dbc = np.asarray(fl.get('d_geo_c0_bottom_cent', []), dtype=float)
    if k >= bot.size:
        return False
    if not (np.isfinite(bot[k]) and np.isfinite(bot[k - 3])):
        return False
    jump3 = float(bot[k]) - float(bot[k - 3])
    lo10 = max(0, k - 10)
    seg = bot[lo10:k + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size < 4:
        return False
    recent_min = float(np.nanmin(seg[:-1])) if seg.size > 1 else float(seg[0])
    dbc3 = rolling_mean_at(dbc, k, 3)
    poleward_surge = jump3 >= 5.0 and np.isfinite(dbc3) and dbc3 >= 0.35
    was_low = recent_min <= float(np.nanpercentile(seg, 30))
    return bool(poleward_surge and was_low)


def weak_recovery_signature(k, fl, phys, ref_idx=None):
    # Interior pause: wind strengthening and/or geo poleward surge.
    if k < 1:
        return False
    if geo_poleward_pause(k, fl, phys):
        return True
    if wind_geo_recovery(k, fl, phys, ref_idx, full=False):
        return True
    peak = np.asarray(fl.get('peak_U', []), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    j0 = max(0, k - 3)
    pu_rise = (float(peak[k]) - float(peak[j0])
               if (k < peak.size and j0 < peak.size and
                   np.isfinite(peak[k]) and np.isfinite(peak[j0]))
               else np.nan)
    nr_rise = (float(nr[k]) - float(nr[j0])
               if (k < nr.size and j0 < nr.size and
                   np.isfinite(nr[k]) and np.isfinite(nr[j0]))
               else np.nan)
    return bool(
        (np.isfinite(pu_rise) and pu_rise >= phys.get('peak_rise_thr', 3.0)) or
        (np.isfinite(nr_rise) and nr_rise >= phys.get('nr_rise_thr', 1.0))
    )


def cycle_boundary(k, fl, phys, ref_idx):
    # Pause inside a prolonged spell (weak recovery or geo poleward surge).
    return (weak_recovery_signature(k, fl, phys, ref_idx) or
            geo_poleward_pause(k, fl, phys))


def warming_resumes(k, fl, trig, warm7, look=4):
    # True when warming / triggers pick up again after a recovery pause.
    n = len(trig)
    if k >= n:
        return False
    if trig[k]:
        return True
    w7 = np.asarray(warm7, dtype=float)
    if k < w7.size and np.isfinite(w7[k]):
        j0 = max(0, k - 3)
        if np.isfinite(w7[j0]) and w7[k] >= w7[j0] + 1.0:
            return True
    for j in range(k, min(n, k + look)):
        if trig[j]:
            return True
    return False


def split_envelope_at_recovery_cycles(env_lo, env_hi, fl, phys, ref_idx,
                                       trig, min_weak_rec, warm7,
                                       min_span=14):
    # Split a long warming envelope at pause-and-resume cycles only.
    env_lo = int(env_lo)
    env_hi = int(env_hi)
    if env_hi - env_lo + 1 < int(min_span):
        return [(env_lo, env_hi)]

    segments = []
    seg_lo = env_lo
    k = env_lo
    rec_run = 0
    rec_lo = None
    while k <= env_hi:
        if cycle_boundary(k, fl, phys, ref_idx):
            if rec_lo is None:
                rec_lo = k
            rec_run += 1
        else:
            if rec_run >= min_weak_rec and rec_lo is not None:
                pause_end = k - 1
                if (warming_resumes(k, fl, trig, warm7) and
                        pause_end >= seg_lo):
                    seg_hi = rec_lo - 1
                    if (seg_hi >= seg_lo and
                            int(trig[seg_lo:seg_hi + 1].sum()) >= 2):
                        segments.append((seg_lo, seg_hi))
                    nxt = k
                    while nxt <= env_hi and not trig[nxt]:
                        nxt += 1
                    seg_lo = nxt if nxt <= env_hi else env_hi + 1
            rec_run = 0
            rec_lo = None
        k += 1
    if seg_lo <= env_hi and int(trig[seg_lo:env_hi + 1].sum()) >= 2:
        segments.append((seg_lo, env_hi))
    return segments if segments else [(env_lo, env_hi)]


def dedupe_event_spans(events):
    # Drop overlapping / duplicate spans; keep chronological non-overlap.
    if not events:
        return []
    events = sorted(events, key=lambda x: (x[0], x[1]))
    out = []
    for lo, hi in events:
        lo, hi = int(lo), int(hi)
        if out and lo <= out[-1][1]:
            if hi <= out[-1][1]:
                continue
            lo = out[-1][1] + 1
            if lo > hi:
                continue
        out.append((lo, hi))
    return out


def max_consecutive_true(mask, lo, hi):
    # Longest run of True in mask[lo:hi] inclusive.
    lo, hi = int(lo), int(hi)
    mask = np.asarray(mask, dtype=bool)
    best = run = 0
    for k in range(lo, min(len(mask), hi + 1)):
        if mask[k]:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def max_consecutive_false(mask, lo, hi):
    # Longest run of False in mask[lo:hi] inclusive.
    lo, hi = int(lo), int(hi)
    mask = np.asarray(mask, dtype=bool)
    best = run = 0
    for k in range(lo, min(len(mask), hi + 1)):
        if not mask[k]:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def sustained_warming_trigger_mask(w7, thr, min_trig=1, bridge=2):
    # Days that count as a 25 K warming. A single day qualifies (minor
    w7 = np.asarray(w7, dtype=float)
    n = w7.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    w_rounded = np.round(w7, 2)
    trig = np.isfinite(w_rounded) & (w_rounded >= float(thr))
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not trig[i]:
            i += 1
            continue
        j = i
        while j < n and trig[j]:
            j += 1
        if (j - i) >= min_trig:
            out[i:j] = True
        else:
            for k in range(i, j):
                lo = max(0, k - bridge)
                hi = min(n, k + bridge + 1)
                if int(trig[lo:hi].sum()) >= min_trig:
                    out[k] = True
        i = j
    return out


def max_consecutive_wind_match(lo, hi, ref_idx, ring_west, westU, phys):
    # Longest run of per-altitude wind-ring match to a reference day.
    if ref_idx is None or ring_west is None:
        return 0
    lo, hi = int(lo), int(hi)
    best = run = 0
    for k in range(lo, hi + 1):
        if wind_matches_reference(k, ref_idx, ring_west, westU, phys):
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def allow_new_warming_onset(i, last_event_end, last_event_ref, ref_idx,
                             ring_west, westU, phys, trig=None, fl=None,
                             after_jet_collapse=False, after_geo_pause=False):
    # Re-trigger gate: prior reference wind match, gap ring recovery, or a
    if last_event_end < 0:
        return True
    # Final-warming / wind-reversal exception: an intact easterly ring at the
    # trigger day is definitive evidence of a genuine warming (the jet has
    # reversed). Allow the onset regardless of wind "recovery"; a final
    if fl is not None:
        eint = np.asarray(fl.get('east_intact_level', np.zeros(len(trig)
                           if trig is not None else 0)), dtype=float)
        if eint.size > i and eint[i] >= 1.0:
            return True
    if fl is not None and geo_poleward_pause(i, fl, phys):
        return True
    if fl is not None and ring_disruption_active(i, fl, phys):
        return True
    gap_lo = last_event_end + 1
    gap_hi = i - 1
    if after_geo_pause and i > last_event_end and trig is not None and trig[i]:
        return True
    if after_jet_collapse and i > last_event_end:
        pause_need = int(phys.get('retrigger_pause_days', 3))
        if gap_lo <= gap_hi:
            if max_consecutive_false(trig, gap_lo, gap_hi) >= pause_need:
                return True
            if fl is not None:
                jet_best = jet_run = 0
                for k in range(gap_lo, gap_hi + 1):
                    if jet_collapse_recovery(k, fl, phys, evt_start=gap_lo):
                        jet_run += 1
                        jet_best = max(jet_best, jet_run)
                    else:
                        jet_run = 0
                if jet_best >= 2:
                    return True
        return False
    if last_event_ref is None:
        return True
    if ring_west is not None and wind_matches_reference(
            i, last_event_ref, ring_west, westU, phys):
        return True
    gap_lo = last_event_end + 1
    gap_hi = i - 1
    if gap_lo <= gap_hi:
        min_wind_rec = int(phys.get(
            'min_wind_rec_run',
            max(3, int(phys.get('min_rec_run', 3)) - 1)))
        if (max_consecutive_wind_match(gap_lo, gap_hi, last_event_ref,
                                        ring_west, westU, phys)
                >= min_wind_rec):
            return True
    # Fresh cycle: local strong-day reference + thermal pause in the gap.
    if (ref_idx is not None and ring_west is not None and trig is not None
            and i > last_event_end):
        if wind_matches_reference(i, ref_idx, ring_west, westU, phys):
            pause_need = int(phys.get('retrigger_pause_days', 3))
            if gap_lo > gap_hi:
                return False
            if max_consecutive_false(trig, gap_lo, gap_hi) >= pause_need:
                return True
            if fl is not None:
                rec_run = 0
                rec_best = 0
                for k in range(gap_lo, gap_hi + 1):
                    if recovery_signature(k, fl, phys, ref_idx):
                        rec_run += 1
                        rec_best = max(rec_best, rec_run)
                    else:
                        rec_run = 0
                if rec_best >= 2:
                    return True
                jet_best = 0
                jet_run = 0
                for k in range(gap_lo, gap_hi + 1):
                    if jet_collapse_recovery(k, fl, phys, evt_start=gap_lo):
                        jet_run += 1
                        jet_best = max(jet_best, jet_run)
                    else:
                        jet_run = 0
                if jet_best >= 2:
                    return True
    # Long-gap fresh warming: after >= 15 days, a sustained new 25 K
    # cycle can open even if the per-altitude ring has not yet matched
    # the pre-season reference (typical post-SSW residual, e.g. 97/98).
    if (trig is not None and fl is not None and i > last_event_end and
            trig[i]):
        min_gap_days = int(phys.get('retrigger_min_gap_days', 15))
        if (i - last_event_end) >= min_gap_days:
            pause_need = int(phys.get('retrigger_pause_days', 3))
            if gap_lo <= gap_hi and (
                    max_consecutive_false(trig, gap_lo, gap_hi)
                    >= pause_need):
                return True
    return False


# Fixed 3-D regional-mean warming scheme: 4 km alt bins × 10° × 30°.
TBIN_SCHEME_SPECS = [
    (dict(lat_group=2, lon_group=1, alt_km=4.0),
     '4 km x 10 x 30'),
]

TBIN_ALT_KM_OPTIONS = [4.0]


def alt_km_bin_options():
    # Altitude bin widths (km) used by the fixed 3-D bin scheme set.
    return list(TBIN_ALT_KM_OPTIONS)


def tbin_scheme_combinations(for_plot=True):
    # Return fixed 3-D bin grouping schemes for warming tests.
    return list(TBIN_SCHEME_SPECS)


def ring_U_by_alt(ds):
    nt    = ds.sizes['time']
    alt   = np.asarray((ds['altitude_km'].values if 'altitude_km' in ds
                        else ds['lev'].values), dtype=np.float32)
    nz    = alt.size

    # Bin edges around each level centre (needed by pcolormesh).
    if nz >= 2:
        mids  = 0.5 * (alt[:-1] + alt[1:])
        edges = np.concatenate([
            [alt[0] - (alt[1] - alt[0]) / 2.0],
            mids,
            [alt[-1] + (alt[-1] - alt[-2]) / 2.0],
        ]).astype(np.float32)
    else:
        edges = np.array([alt[0] - 0.5, alt[0] + 0.5], dtype=np.float32) \
                if nz else np.array([0.0, 1.0], dtype=np.float32)

    mean_field = np.full((nt, nz), np.nan, np.float32)
    peak_field = np.full((nt, nz), np.nan, np.float32)

    # Strict ring-only: a (time, level) cell is only painted when the
    # level falls inside a valid wind_ring_alt_bands entry for some
    # component. Inside a band we prefer the per-level signed
    bands = ds['wind_ring_alt_bands'].values      # (t, comp, band, 2)
    mU    = ds['wind_mean_U'].values
    pU    = (ds['wind_greatest_mag_U'].values
             if 'wind_greatest_mag_U' in ds else mU)
    sign  = (ds['wind_sign'].values
             if 'wind_sign' in ds else np.full_like(mU, np.nan))
    lev_U = (np.asarray(ds['wind_lev_mean_U'].values, dtype=np.float32)
             if 'wind_lev_mean_U' in ds else None)

    nc    = bands.shape[1]
    nb    = bands.shape[2]
    EPS   = 1e-4

    def sign_of(ti, ci):
        s_raw = sign[ti, ci] if ci < sign.shape[1] else np.nan
        if np.isfinite(s_raw):
            return 1.0 if int(round(float(s_raw))) == 0 else -1.0
        if ci < mU.shape[1] and np.isfinite(mU[ti, ci]):
            return 1.0 if mU[ti, ci] >= 0 else -1.0
        return np.nan

    for ti in range(nt):
        for ci in range(nc):
            s_sign = sign_of(ti, ci)
            u_mean_s = (s_sign * abs(float(mU[ti, ci]))
                        if (ci < mU.shape[1] and np.isfinite(mU[ti, ci])
                            and np.isfinite(s_sign))
                        else np.nan)
            u_peak_s = (s_sign * abs(float(pU[ti, ci]))
                        if (ci < pU.shape[1] and np.isfinite(pU[ti, ci])
                            and np.isfinite(s_sign))
                        else u_mean_s)

            for bi in range(nb):
                lo = bands[ti, ci, bi, 0]
                hi = bands[ti, ci, bi, 1]
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                a, b = float(min(lo, hi)), float(max(lo, hi))
                sel = (edges[1:] > a - EPS) & (edges[:-1] < b + EPS)
                if not sel.any():
                    k_near = int(np.argmin(np.abs(alt - 0.5 * (a + b))))
                    sel = np.zeros(nz, dtype=bool)
                    sel[k_near] = True

                # pull per-level signed U where we have it, else the
                # scalar component mean with inferred sign
                if lev_U is not None:
                    u_lev = lev_U[ti, ci]
                    u_per = np.where(np.isfinite(u_lev), u_lev, u_mean_s)
                else:
                    u_per = np.full(nz, u_mean_s, dtype=np.float32)

                finite_in_band = np.isfinite(u_per[sel])
                if not finite_in_band.any():
                    continue

                cur = mean_field[ti, sel]
                new = np.where(
                    finite_in_band & (np.isnan(cur) |
                                      (np.abs(u_per[sel]) > np.abs(cur))),
                    u_per[sel], cur)
                mean_field[ti, sel] = new

                # same rule for peak field; if per-level peak isn't
                # available we reuse the per-level mean (above), or
                # the scalar peak of the component, whichever is larger.
                cand = np.where(np.isfinite(u_per[sel]),
                                u_per[sel], u_peak_s)
                cur = peak_field[ti, sel]
                new = np.where(
                    np.isfinite(cand) & (np.isnan(cur) |
                                         (np.abs(cand) > np.abs(cur))),
                    cand, cur)
                peak_field[ti, sel] = new

    return dict(alt=alt, edges=edges, U_mean=mean_field, U_peak=peak_field)


def apply_t_bin_alt_nlevels(Tc_mean, Tc_max, alt, n_bins, Tc_n=None):
    # Merge **all** native altitude levels into *n_bins* contiguous groups.
    alt = np.asarray(alt, dtype=float)
    nlev = alt.size
    if nlev == 0:
        return Tc_mean, Tc_max, alt
    n_bins = min(max(1, int(n_bins)), nlev)
    chunks = np.array_split(np.arange(nlev), n_bins)
    parts_m = []
    parts_x = []
    alt_out = []
    for ch in chunks:
        if ch.size == 0:
            continue
        Tm_ch = Tc_mean[:, ch, :, :]
        if Tc_n is not None:
            w = np.asarray(Tc_n[:, ch, :, :], dtype=np.float64)
            w = np.where(np.isfinite(Tm_ch) & (w > 0), w, 0.0)
            num = np.nansum(Tm_ch * w, axis=1, keepdims=True)
            den = np.nansum(w, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                parts_m.append(np.where(den > 0, num / den, np.nan)
                               .astype(np.float32))
        else:
            parts_m.append(np.nanmean(Tm_ch, axis=1, keepdims=True)
                          .astype(np.float32))
        parts_x.append(np.nanmax(Tc_max[:, ch, :, :], axis=1, keepdims=True)
                       .astype(np.float32))
        alt_out.append(float(np.mean(alt[ch])))
    return (np.concatenate(parts_m, axis=1).astype(np.float32),
            np.concatenate(parts_x, axis=1).astype(np.float32),
            np.asarray(alt_out, dtype=float))


def apply_t_bin_alt_km(Tc_mean, Tc_max, alt, km_width, Tc_n=None):
    # Merge native levels into fixed-width altitude bins (km).
    alt = np.asarray(alt, dtype=float)
    if alt.size == 0:
        return Tc_mean, Tc_max, alt
    km_width = float(km_width)
    if km_width <= 0:
        raise ValueError(f"t_bin_alt_km must be positive, got {km_width}")
    lo = float(np.nanmin(alt))
    hi = float(np.nanmax(alt))
    edges = np.arange(lo, hi + 1e-9, km_width)
    if edges.size < 2:
        edges = np.array([lo, lo + km_width], dtype=float)
    elif edges[-1] < hi - 1e-6:
        edges = np.append(edges, hi + 1e-9)
    n_bins = len(edges) - 1
    parts_m = []
    parts_x = []
    alt_out = []
    for b in range(n_bins):
        if b < n_bins - 1:
            sel = (alt >= edges[b]) & (alt < edges[b + 1])
        else:
            sel = (alt >= edges[b]) & (alt <= edges[-1])
        ch = np.where(sel)[0]
        if ch.size == 0:
            continue
        Tm_ch = Tc_mean[:, ch, :, :]
        if Tc_n is not None:
            w = np.asarray(Tc_n[:, ch, :, :], dtype=np.float64)
            w = np.where(np.isfinite(Tm_ch) & (w > 0), w, 0.0)
            num = np.nansum(Tm_ch * w, axis=1, keepdims=True)
            den = np.nansum(w, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                parts_m.append(np.where(den > 0, num / den, np.nan)
                               .astype(np.float32))
        else:
            parts_m.append(np.nanmean(Tm_ch, axis=1, keepdims=True)
                          .astype(np.float32))
        parts_x.append(np.nanmax(Tc_max[:, ch, :, :], axis=1, keepdims=True)
                       .astype(np.float32))
        alt_out.append(float(np.mean(alt[ch])))
    if not parts_m:
        return Tc_mean, Tc_max, alt
    return (np.concatenate(parts_m, axis=1).astype(np.float32),
            np.concatenate(parts_x, axis=1).astype(np.float32),
            np.asarray(alt_out, dtype=float))


def compute_warm7_regional_means(T_reg, alt_reg, lat_reg, thr=25.0,
                                  max_d=7, reset_mask=None,
                                  split_alt_km=None):
    # SSW warming trigger from 3-D regional means.
    nt = T_reg.shape[0]
    nlev = T_reg.shape[1]
    nlat = T_reg.shape[2]
    nlon = T_reg.shape[3]
    alt_a = np.asarray(alt_reg, dtype=float)
    lat_a = np.asarray(lat_reg, dtype=float)
    if reset_mask is not None:
        reset_mask = np.asarray(reset_mask, dtype=bool)
        last_reset = np.zeros(nt, dtype=int)
        lr = 0
        for i in range(nt):
            if reset_mask[i]:
                lr = i
            last_reset[i] = lr
    else:
        last_reset = None

    warm7_max = np.full(nt, np.nan, np.float32)
    warm7_alt_km = np.full(nt, np.nan, np.float32)
    warm7_lat = np.full(nt, np.nan, np.float32)
    warm7_lookback_d = np.full(nt, -1, np.int16)
    warm7_end_off = np.full(nt, -1, np.int16)   # days-before-ti of warm day b
    # Peak 7-day rise split at the 10 hPa boundary (split_alt_km): _below10 is
    # the max over levels at/below 10 hPa (pressure >= 10 hPa, altitude <=
    # split_alt_km); _above10 is the max over levels above 10 hPa.
    warm7_below10 = np.full(nt, np.nan, np.float32)
    warm7_above10 = np.full(nt, np.nan, np.float32)
    warm1_max = np.full(nt, np.nan, np.float32)
    warm1_alt_km = np.full(nt, np.nan, np.float32)
    warm1_lat = np.full(nt, np.nan, np.float32)
    warm_area_frac = np.full(nt, np.nan, np.float32)
    warm_bin_count = np.full(nt, np.nan, np.float32)

    if nlev == 0 or nlat == 0 or nlon == 0:
        return dict(warm7_max=warm7_max, warm7_alt_km=warm7_alt_km,
                    warm7_lat=warm7_lat, warm7_lookback_d=warm7_lookback_d,
                    warm7_end_off=warm7_end_off,
                    warm7_below10=warm7_below10, warm7_above10=warm7_above10,
                    warm1_max=warm1_max, warm1_alt_km=warm1_alt_km,
                    warm1_lat=warm1_lat, warm_area_frac=warm_area_frac,
                    warm_bin_count=warm_bin_count)

    nblock = nlat * nlon

    for ti in range(nt):
        lo = max(0, ti - max_d)
        if last_reset is not None:
            lo = max(lo, int(last_reset[ti]))
        win = T_reg[lo:ti + 1]                      # (w, lev, lat, lon)
        w = win.shape[0]
        if w < 2:
            continue
        flat = win.reshape(w, nlev, nblock)         # (w, lev, block)

        # Best rise per block over all pairs a<b in the window:
        # for each end b, rise = T(b) - running_min over a<=b; take max over b.
        # Track which span (a..b) gave the per-block best for the global best.
        best_block = np.full((nlev, nblock), -np.inf, dtype=np.float64)
        best_b = np.full((nlev, nblock), -1, dtype=np.int32)
        best_a = np.full((nlev, nblock), -1, dtype=np.int32)
        run_min = flat[0].astype(np.float64).copy()           # min T(a), a<=b
        run_min_idx = np.zeros((nlev, nblock), dtype=np.int32)
        run_min[~np.isfinite(run_min)] = np.inf
        for bi in range(1, w):
            cur = flat[bi].astype(np.float64)
            rise = cur - run_min                              # T(b)-min_a T(a)
            valid = np.isfinite(cur) & np.isfinite(run_min)
            upd = valid & (rise > best_block)
            best_block[upd] = rise[upd]
            best_b[upd] = bi
            best_a[upd] = run_min_idx[upd]
            # advance running min (including current day b as a candidate a)
            lower = np.isfinite(cur) & (cur < run_min)
            run_min[lower] = cur[lower]
            run_min_idx[lower] = bi

        if not np.isfinite(best_block).any():
            continue
        # Exclude the single topmost altitude level (the ~50 km model top):
        # warmings that only the very top level produces are not counted.
        # The winning block is taken from levels strictly below the highest
        if alt_a.size:
            top_lev = int(np.nanargmax(alt_a))
            if 0 <= top_lev < nlev:
                best_block[top_lev, :] = -np.inf
        if not np.isfinite(best_block).any():
            continue
        gi = int(np.nanargmax(np.where(np.isfinite(best_block),
                                       best_block, -np.inf)))
        lev_i = gi // nblock
        blk_i = gi % nblock
        lat_i = blk_i // nlon
        best_v = float(best_block[lev_i, blk_i])

        if best_v > -np.inf:
            warm7_max[ti] = best_v
            warm7_alt_km[ti] = (float(alt_a[lev_i])
                                if 0 <= lev_i < alt_a.size else np.nan)
            warm7_lat[ti] = (float(lat_a[lat_i])
                             if 0 <= lat_i < lat_a.size else np.nan)
            bb = int(best_b[lev_i, blk_i]); aa = int(best_a[lev_i, blk_i])
            warm7_lookback_d[ti] = int(bb - aa) if (bb >= 0 and aa >= 0) else -1
            warm7_end_off[ti] = int(ti - (lo + bb)) if bb >= 0 else -1

        # Peak 7-day rise on each side of the 10 hPa boundary. best_block holds
        # the per-(level, block) best rise (top level already excluded above);
        # reduce it over the two level groups.
        if split_alt_km is not None:
            below = np.where(alt_a <= float(split_alt_km))[0]
            above = np.where(alt_a >  float(split_alt_km))[0]
            if below.size:
                vb = best_block[below]; vb = vb[np.isfinite(vb)]
                if vb.size:
                    warm7_below10[ti] = float(vb.max())
            if above.size:
                va = best_block[above]; va = va[np.isfinite(va)]
                if va.size:
                    warm7_above10[ti] = float(va.max())

        # warm1: largest single-day rise ending at ti
        if ti >= 1:
            T_now = T_reg[ti]
            T_yest = T_reg[ti - 1]
            with np.errstate(invalid='ignore'):
                d1 = np.where(np.isfinite(T_now) & np.isfinite(T_yest),
                              T_now - T_yest, -np.inf)
            if np.isfinite(d1).any():
                fi1 = int(np.nanargmax(d1))
                v1 = float(d1.flat[fi1])
                if v1 > -np.inf:
                    li = fi1 // nblock
                    bi1 = fi1 % nblock
                    la1 = bi1 // nlon
                    warm1_max[ti] = v1
                    warm1_alt_km[ti] = (float(alt_a[li])
                                        if 0 <= li < alt_a.size else np.nan)
                    warm1_lat[ti] = (float(lat_a[la1])
                                     if 0 <= la1 < lat_a.size else np.nan)

        # area fraction: blocks whose best in-window rise reaches threshold
        fin = np.isfinite(best_block)
        n_bins = int(fin.sum())
        if n_bins > 0:
            n_warm = int((fin & (best_block >= thr)).sum())
            warm_area_frac[ti] = float(n_warm) / float(n_bins)
            warm_bin_count[ti] = float(n_warm)

    return dict(warm7_max=warm7_max, warm7_alt_km=warm7_alt_km,
                warm7_lat=warm7_lat, warm7_lookback_d=warm7_lookback_d,
                warm7_end_off=warm7_end_off,
                warm7_below10=warm7_below10, warm7_above10=warm7_above10,
                warm1_max=warm1_max, warm1_alt_km=warm1_alt_km,
                warm1_lat=warm1_lat, warm_area_frac=warm_area_frac,
                warm_bin_count=warm_bin_count)


def vertical_run_filter(mask_2d, min_run):
    # Keep only levels belonging to runs of >= min_run consecutive True.
    out = np.zeros_like(mask_2d, dtype=bool)
    nt, nlev = mask_2d.shape
    min_run = max(1, int(min_run))
    for ti in range(nt):
        run = 0
        start = 0
        for lev in range(nlev):
            if mask_2d[ti, lev]:
                if run == 0:
                    start = lev
                run += 1
            else:
                if run >= min_run:
                    out[ti, start:start + run] = True
                run = 0
        if run >= min_run:
            out[ti, start:start + run] = True
    return out


def compute_b0_morphology_flags(big_b0, sec_b0, alt,
                                   top_alt_min_km=45.0,
                                   bottom_alt_max_km=35.0,
                                   min_vert=3):
    # Daily split / partial-split signatures from per-level Betti-0 profiles.
    big_b0 = np.asarray(big_b0, dtype=float)
    sec_b0 = np.asarray(sec_b0, dtype=float)
    alt = np.asarray(alt, dtype=float)
    nt, nlev = big_b0.shape
    fin = np.isfinite(big_b0)
    nlev_pop = fin.sum(axis=1).astype(float)
    nlev_pop[nlev_pop == 0] = np.nan
    no_second = ~np.isfinite(sec_b0) | (sec_b0 <= 0)
    sec_present = np.isfinite(sec_b0) & (sec_b0 > 0)

    raw_partial = fin & (big_b0 >= 2) & no_second
    coherent_all = vertical_run_filter(raw_partial, min_vert)

    top_m = alt >= float(top_alt_min_km)
    bot_m = alt <= float(bottom_alt_max_km)
    top_raw = raw_partial & top_m[None, :]
    bot_raw = raw_partial & bot_m[None, :]
    top_coherent = vertical_run_filter(top_raw, min_vert)
    bot_coherent = vertical_run_filter(bot_raw, min_vert)

    frac_second = np.where(
        nlev_pop > 0, sec_present.sum(axis=1) / nlev_pop, np.nan)
    frac_coherent = np.where(
        nlev_pop > 0, coherent_all.sum(axis=1) / nlev_pop, np.nan)

    top_pop = fin[:, top_m].sum(axis=1).astype(float)
    top_pop[top_pop == 0] = np.nan
    bot_pop = fin[:, bot_m].sum(axis=1).astype(float)
    bot_pop[bot_pop == 0] = np.nan
    frac_top_partial = np.where(
        top_pop > 0, top_coherent.sum(axis=1) / top_pop, np.nan)
    frac_bot_partial = np.where(
        bot_pop > 0, bot_coherent.sum(axis=1) / bot_pop, np.nan)

    sec_fracs = frac_second[np.isfinite(frac_second) & (frac_second > 0)]
    full_thr = (float(np.percentile(sec_fracs, 25))
                if sec_fracs.size >= 5 else 0.40)
    full_thr = float(np.clip(full_thr, 0.25, 0.55))

    pop_fracs = frac_coherent[np.isfinite(frac_coherent) & (frac_coherent > 0)]
    partial_floor = (float(np.percentile(pop_fracs, 20))
                     if pop_fracs.size >= 5 else 0.12)
    partial_floor = float(np.clip(partial_floor, 0.08, 0.30))

    raw_b02 = fin & (big_b0 >= 2)
    coherent_b02 = vertical_run_filter(raw_b02, min_vert)
    frac_b02 = np.where(
        nlev_pop > 0, coherent_b02.sum(axis=1) / nlev_pop, np.nan)

    def consecutive(mask, min_run):
        # True only on days that belong to a run of >= min_run consecutive
        m = np.asarray(mask, dtype=bool)
        out = np.zeros_like(m)
        n = len(m)
        i = 0
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                if (j - i) >= min_run:
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out

    def consecutive_bridged(mask, min_run, max_gap=1):
        # Like ``consecutive`` but bridges gaps of up to ``max_gap``
        m = np.asarray(mask, dtype=bool)
        n = len(m)
        out = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if not m[i]:
                i += 1
                continue
            last_true = i
            while True:
                nxt = -1
                for k in range(last_true + 1, min(n, last_true + max_gap + 2)):
                    if m[k]:
                        nxt = k
                        break
                if nxt == -1:
                    break
                last_true = nxt
            if int(m[i:last_true + 1].sum()) >= min_run:
                out[i:last_true + 1] = True
            i = last_true + 1
        return out

    MIN_SPLIT_DAYS = 3   # a split phase must persist >= 3 consecutive days

    # per-day "substantial pinch" (a real chunk of the column shows b0>=2,
    # or a separate second lobe is present on a real fraction of levels)
    day_full_level = (
        (np.isfinite(frac_second) & (frac_second >= full_thr)) |
        (np.isfinite(frac_b02) & (frac_b02 >= full_thr))
    )
    day_partial_level = (
        ~day_full_level &
        (
            (np.isfinite(frac_second) & (frac_second >= partial_floor)) |
            (np.isfinite(frac_b02) & (frac_b02 >= partial_floor)) |
            (np.isfinite(frac_bot_partial) & (frac_bot_partial >= PARTIAL_BAND_FRAC)) |
            (np.isfinite(frac_top_partial) & (frac_top_partial >= PARTIAL_BAND_FRAC))
        )
    )

    # full split: a near-complete second column sustained >= 3 consecutive
    # days. A single-day spike (even many levels) is not a full split.
    full_split = consecutive(day_full_level, MIN_SPLIT_DAYS)

    # partial split: a substantial pinch sustained over >= 3 partial days that
    # never reaches the full-column second component. Single-day gaps inside
    # the phase are bridged (the pinch can momentarily reconnect), but >= 3
    partial_level_or_full = day_partial_level | day_full_level
    partial_runs = consecutive_bridged(partial_level_or_full, MIN_SPLIT_DAYS,
                                       max_gap=1)
    partial = partial_runs & ~full_split

    # Top pinch only counts if it, too, persists >= 3 consecutive days.
    top_pinch_day = (
        np.isfinite(frac_second) & (frac_second < full_thr) &
        (top_raw.sum(axis=1) >= min_vert) &
        (top_coherent.sum(axis=1) >= min_vert)
    )
    top_pinch = consecutive(top_pinch_day, MIN_SPLIT_DAYS) & ~full_split

    # No separate progressive flag: full_split already spans the sustained
    # run, and the surrounding partial days are captured by `partial`.
    b0_full_split_prog = full_split.copy()

    return dict(
        b0_full_split=full_split.astype(np.float32),
        b0_full_split_prog=b0_full_split_prog.astype(np.float32),
        b0_top_pinch_split=top_pinch.astype(np.float32),
        b0_partial_split=partial.astype(np.float32),
        b0_frac_second=frac_second.astype(np.float32),
        b0_frac_top_partial=frac_top_partial.astype(np.float32),
        b0_frac_bot_partial=frac_bot_partial.astype(np.float32),
        b0_split_full_thr=np.float32(full_thr),
        b0_split_partial_floor=np.float32(partial_floor),
    )


def compute_flags(ds, smooth=3, warming_alt_lo=20.0, warming_alt_hi=49.0,
                  t_bin_lat_group=1, t_bin_lon_group=1, t_bin_alt_group=1,
                  t_bin_alt_nlevels=None, t_bin_alt_km=None):
    # Per-timestep flags from a vortex_full_timeseries dataset.
    nt  = ds.sizes['time']
    lat = ds['lat'].values
    alt_native = (ds['altitude_km'].values if 'altitude_km' in ds
                  else ds['lev'].values)
    alt = np.asarray(alt_native, dtype=float)
    mid_upper = (alt >= warming_alt_lo) & (alt <= warming_alt_hi)
    lat_polar = lat >= 60.0

    # Temperature source.
    # New datasets store per-coarse-bin (lat_bin × lon_bin) T inside
    # Warming flags are driven by the geopotential-contour temperature
    tpref = 'T_in_geo'
    has_new_T = (f'{tpref}_mean' in ds.data_vars and
                 f'{tpref}_max'  in ds.data_vars and
                 'lat_bin_center' in ds.variables and
                 'lon_bin_center' in ds.variables)

    if has_new_T:
        Tc_mean = np.asarray(ds[f'{tpref}_mean'].values, dtype=np.float32)
        Tc_max  = np.asarray(ds[f'{tpref}_max' ].values, dtype=np.float32)
        Tc_n    = (np.asarray(ds[f'{tpref}_n'].values, dtype=np.float64)
                   if f'{tpref}_n' in ds.data_vars
                   else np.where(np.isfinite(Tc_mean), 1.0, 0.0))
        lat_bin_center = np.asarray(ds['lat_bin_center'].values, dtype=float)
        lon_bin_center = np.asarray(ds['lon_bin_center'].values, dtype=float)
        # Optional re-aggregation onto a coarser grid (averaging
        # contiguous lat_group × lon_group blocks).
        lg, mg = int(t_bin_lat_group), int(t_bin_lon_group)
        if lg > 1 or mg > 1:
            nlat_b, nlon_b = Tc_mean.shape[2], Tc_mean.shape[3]
            if nlat_b % lg or nlon_b % mg:
                raise ValueError(
                    f"t_bin_lat_group/t_bin_lon_group = ({lg}, {mg}) do "
                    f"not divide the native ({nlat_b}, {nlon_b}) grid.")
            nlat_n = nlat_b // lg
            nlon_n = nlon_b // mg
            shp = (Tc_mean.shape[0], Tc_mean.shape[1],
                   nlat_n, lg, nlon_n, mg)
            Tm5 = Tc_mean.reshape(shp)
            Tx5 = Tc_max.reshape(shp)
            Tn5 = Tc_n.reshape(shp)
            # weighted mean across the merged sub-blocks
            num = np.nansum(Tm5 * Tn5, axis=(3, 5))
            den = np.nansum(Tn5,         axis=(3, 5))
            with np.errstate(invalid='ignore'):
                Tc_mean = np.where(den > 0,
                                   num / den,
                                   np.nan).astype(np.float32)
            Tc_max  = np.nanmax(Tx5,  axis=(3, 5)).astype(np.float32)
            Tc_n    = np.nansum(Tn5, axis=(3, 5))
            lat_bin_center = lat_bin_center.reshape(nlat_n, lg).mean(axis=1)
            lon_bin_center = lon_bin_center.reshape(nlon_n, mg).mean(axis=1)
        latb    = lat_bin_center
        # Optional altitude re-aggregation (full native column).
        if t_bin_alt_km is not None:
            Tc_mean, Tc_max, alt = apply_t_bin_alt_km(
                Tc_mean, Tc_max, alt, float(t_bin_alt_km), Tc_n=Tc_n)
            mid_upper = (alt >= warming_alt_lo) & (alt <= warming_alt_hi)
        elif t_bin_alt_nlevels is not None:
            Tc_mean, Tc_max, alt = apply_t_bin_alt_nlevels(
                Tc_mean, Tc_max, alt, int(t_bin_alt_nlevels), Tc_n=Tc_n)
            mid_upper = (alt >= warming_alt_lo) & (alt <= warming_alt_hi)
        else:
            # Legacy: merge all native levels in contiguous blocks.
            ag = int(t_bin_alt_group)
            if ag > 1:
                nlev0 = Tc_mean.shape[1]
                if nlev0 % ag:
                    raise ValueError(
                        f"t_bin_alt_group={ag} does not divide native "
                        f"nlev={nlev0}.")
                nlev_n = nlev0 // ag
                shp_a = (Tc_mean.shape[0], nlev_n, ag,
                         Tc_mean.shape[2], Tc_mean.shape[3])
                Tc_mean = np.nanmean(
                    Tc_mean.reshape(shp_a), axis=2).astype(np.float32)
                Tc_max  = np.nanmax(
                    Tc_max.reshape(shp_a), axis=2).astype(np.float32)
                alt_arr = np.asarray(alt, dtype=float)
                alt = alt_arr.reshape(nlev_n, ag).mean(axis=1).astype(float)
                mid_upper = (alt >= warming_alt_lo) & (alt <= warming_alt_hi)
        # Polar-cap weighted mean per (t, lev) from T_in_geo_mean.
        # cos(lat)-weight the lat_bin dimension, simple nanmean over lon_bin.
        polar_b = latb >= 60.0
        if polar_b.any():
            Tcm_lat = np.nanmean(Tc_mean[:, :, polar_b, :], axis=3)  # (t, lev, polar_lat)
            wlat = np.cos(np.deg2rad(latb[polar_b])).astype(np.float32)
            wlat = np.where(np.isfinite(wlat) & (wlat > 0.0), wlat, 0.0)
            wmat = wlat[None, None, :]
            fin  = np.isfinite(Tcm_lat).astype(np.float32)
            T_sum = (np.where(fin > 0, Tcm_lat, 0.0) * wmat).sum(axis=2)
            w_sum = (fin * wmat).sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                Tp = np.where(w_sum > 0,
                              T_sum / np.maximum(w_sum, 1e-12),
                              np.nan).astype(np.float32)
        else:
            Tp = np.nanmean(np.nanmean(Tc_mean, axis=3), axis=2).astype(np.float32)
    else:
        # polar-cap zonal-mean T (time, lev)
        T  = ds['T_zonal'].values.astype(np.float32)
        Tp = np.nanmean(T[:, :, lat_polar], axis=2)
        Tc_mean = None
        Tc_max  = None
        latb    = None

    if smooth and smooth > 1:
        Tp = uniform_filter1d(Tp, size=int(smooth),
                              axis=0, mode='nearest')
    Tp_col  = np.nanmean(Tp, axis=1).astype(np.float32)
    dT_col  = np.gradient(Tp_col).astype(np.float32)
    # day-to-day change of the tendency itself: "dT/dt barely moves"
    d2T_col = np.gradient(dT_col).astype(np.float32)

    # mid/upper-strat warming diagnostics
    T_anom = Tp - np.nanmean(Tp, axis=0, keepdims=True)
    if mid_upper.any():
        warm_anom_max = np.nanmax(T_anom[:, mid_upper], axis=1).astype(np.float32)
        dT_up         = np.gradient(Tp[:, mid_upper], axis=0)
        warm_dT_max   = np.nanmax(dT_up, axis=1).astype(np.float32)
    else:
        warm_anom_max = np.zeros(nt, np.float32)
        warm_dT_max   = np.zeros(nt, np.float32)

    # local (no-lat-average) polar-stratosphere warming metrics.
    # This is the primary SSW thermal signal: if warming is sharp anywhere
    # in the polar stratosphere, it should be visible even when averages are not.
    if has_new_T:
        # Use T_in_geo_mean over (lat_bin >= 50°N) ∩ mid-upper-strat.
        # Each bin already lives inside the polar wind-ring perimeter.
        mu_lev = ((np.asarray(alt, dtype=float) >= warming_alt_lo) &
                  (np.asarray(alt, dtype=float) <= warming_alt_hi))
        lat_b50 = latb >= 50.0
        loc_mask3 = np.broadcast_to(
            (mu_lev[:, None, None] & lat_b50[None, :, None]),
            Tc_mean.shape[1:]).copy()
        warm_local_anom_max = np.full(nt, np.nan, np.float32)
        warm_local_dT_max   = np.full(nt, np.nan, np.float32)
        if loc_mask3.any():
            base = np.nanmean(Tc_mean, axis=0, keepdims=True)
            Tan  = Tc_mean - base
            dT_b = np.gradient(Tc_mean, axis=0)
            for ti in range(nt):
                a = Tan[ti][loc_mask3]
                b = dT_b[ti][loc_mask3]
                if np.isfinite(a).any():
                    warm_local_anom_max[ti] = float(np.nanmax(a))
                if np.isfinite(b).any():
                    warm_local_dT_max[ti] = float(np.nanmax(b))
    else:
        Tloc = np.asarray(ds['T_zonal'].values, dtype=np.float32)  # (t, lev, lat)
        lat_loc = np.asarray(lat, dtype=float)
        loc_mask = ((np.asarray(alt, dtype=float) >= warming_alt_lo) &
                    (np.asarray(alt, dtype=float) <= warming_alt_hi))[:, None] & \
                   (lat_loc[None, :] >= 50.0)
        warm_local_anom_max = np.full(nt, np.nan, np.float32)
        warm_local_dT_max   = np.full(nt, np.nan, np.float32)
        if np.any(loc_mask):
            base = np.nanmean(Tloc, axis=0, keepdims=True)  # seasonal local baseline
            Tan = Tloc - base
            dT  = np.gradient(Tloc, axis=0)
            for ti in range(nt):
                a = Tan[ti][loc_mask]
                b = dT [ti][loc_mask]
                if np.isfinite(a).any():
                    warm_local_anom_max[ti] = float(np.nanmax(a))
                if np.isfinite(b).any():
                    warm_local_dT_max[ti] = float(np.nanmax(b))

    # westerly ring + easterly occupancy
    is_h1   = ds['wind_is_h1'].values
    sign    = ds['wind_sign'].values
    mU      = ds['wind_mean_U'].values
    pU      = ds['wind_greatest_mag_U'].values
    rsp     = (ds['wind_grad_refined_region_speed'].values
               if 'wind_grad_refined_region_speed' in ds
               else np.full_like(mU, np.nan))
    pct10   = ds['wind_pct_10hPa_60lat'].values
    inner_l = ds['wind_mean_inner_lat'].values
    outer_l = ds['wind_mean_outer_lat'].values
    alt_m   = ds['wind_mean_alt'].values

    # per-longitude core voxel (written by vtxplt per westerly ring)
    core_lat_lon = ds['wind_core_lat'].values  # (t, wind_comp, lon)
    core_alt_lon = ds['wind_core_alt'].values
    tilt_slope_v = ds['wind_tilt_slope'].values if 'wind_tilt_slope' in ds \
                   else np.full_like(mU, np.nan)

    area_by_c = ds['wind_total_area_km2'].values if 'wind_total_area_km2' in ds \
                else np.full_like(mU, np.nan)

    peak_U      = np.full(nt, np.nan, np.float32)
    mean_U      = np.full(nt, np.nan, np.float32)
    region_speed = np.full(nt, np.nan, np.float32)
    inner_lat   = np.full(nt, np.nan, np.float32)
    outer_lat   = np.full(nt, np.nan, np.float32)
    ring_alt    = np.full(nt, np.nan, np.float32)
    east_inner  = np.full(nt, np.nan, np.float32)
    east_outer  = np.full(nt, np.nan, np.float32)
    east_alt    = np.full(nt, np.nan, np.float32)
    east_U      = np.full(nt, np.nan, np.float32)
    core_lat_m  = np.full(nt, np.nan, np.float32)
    core_lat_s  = np.full(nt, np.nan, np.float32)
    core_alt_m  = np.full(nt, np.nan, np.float32)
    core_alt_s  = np.full(nt, np.nan, np.float32)
    ecore_lat_m = np.full(nt, np.nan, np.float32)
    ecore_lat_s = np.full(nt, np.nan, np.float32)
    ecore_alt_m = np.full(nt, np.nan, np.float32)
    ecore_alt_s = np.full(nt, np.nan, np.float32)
    wind_tilt   = np.full(nt, np.nan, np.float32)
    ring_area   = np.full(nt, np.nan, np.float32)
    east_area   = np.full(nt, np.nan, np.float32)
    # %-at-10hPa/60°N: NaN when no ring exists rather than 0, so the
    # line plot has a gap instead of a fake zero floor.
    west_pct    = np.full(nt, np.nan, np.float32)
    east_pct    = np.full(nt, np.nan, np.float32)
    west_active = np.zeros(nt,       np.float32)
    east_active = np.zeros(nt,       np.float32)

    for ti in range(nt):
        # Include every component that has some evidence of a ring:
        # finite is_h1, finite wind_mean_U, or any finite per-longitude
        # core data. Sign is inferred from wind_sign when finite,
        n_comp = mU.shape[1]
        has_core = np.isfinite(core_lat_lon[ti]).any(axis=-1)
        active = np.zeros(n_comp, dtype=bool)
        for c in range(n_comp):
            h = is_h1[ti, c] if c < is_h1.shape[1] else np.nan
            if np.isfinite(h) and h >= 0.5:
                active[c] = True
            elif np.isfinite(mU[ti, c]) or has_core[c]:
                active[c] = True
        if not active.any():
            continue

        def sign_of(c):
            s_raw = sign[ti, c]
            if np.isfinite(s_raw):
                return int(round(float(s_raw)))           # 0=west, 1=east
            if np.isfinite(mU[ti, c]):
                return 0 if mU[ti, c] >= 0 else 1
            return -1

        w_idx = [int(c) for c in np.where(active)[0] if sign_of(c) == 0]
        e_idx = [int(c) for c in np.where(active)[0] if sign_of(c) == 1]
        if w_idx:
            ci = max(w_idx, key=lambda c: abs(mU[ti, c])
                     if np.isfinite(mU[ti, c]) else -1.0)
            peak_U   [ti] = pU[ti, ci]
            mean_U   [ti] = mU[ti, ci]
            region_speed[ti] = rsp[ti, ci] if np.isfinite(rsp[ti, ci]) else np.nan
            inner_lat[ti] = inner_l[ti, ci]
            outer_lat[ti] = outer_l[ti, ci]
            ring_alt [ti] = alt_m  [ti, ci]
            ring_area[ti] = area_by_c[ti, ci]
            if np.isfinite(pct10[ti, ci]):
                west_pct[ti] = float(pct10[ti, ci])
            west_active[ti] = 1.0
            # per-longitude core: use whatever longitudes have finite
            # values; if none do, fall back to the ring's own mid-lat
            # and mean alt so we still get a useful line point.
            cl = core_lat_lon[ti, ci]; ca = core_alt_lon[ti, ci]
            if np.isfinite(cl).any():
                core_lat_m[ti] = float(np.nanmean(cl))
                core_lat_s[ti] = float(np.nanstd(cl))
            elif (np.isfinite(inner_l[ti, ci])
                  and np.isfinite(outer_l[ti, ci])):
                core_lat_m[ti] = 0.5 * (float(inner_l[ti, ci])
                                        + float(outer_l[ti, ci]))
            if np.isfinite(ca).any():
                core_alt_m[ti] = float(np.nanmean(ca))
                core_alt_s[ti] = float(np.nanstd(ca))
            elif np.isfinite(alt_m[ti, ci]):
                core_alt_m[ti] = float(alt_m[ti, ci])
            wind_tilt[ti]  = tilt_slope_v[ti, ci]
        if e_idx:
            ei = max(e_idx, key=lambda c: abs(mU[ti, c])
                     if np.isfinite(mU[ti, c]) else -1.0)
            east_inner[ti]  = inner_l[ti, ei]
            east_outer[ti]  = outer_l[ti, ei]
            east_alt  [ti]  = alt_m  [ti, ei]
            east_U    [ti]  = mU     [ti, ei]
            east_area [ti]  = area_by_c[ti, ei]
            east_active[ti] = 1.0
            # easterly core (same per-longitude fields vtxplt writes for
            # any ring component; we extract for the dominant easterly)
            ecl = core_lat_lon[ti, ei]; eca = core_alt_lon[ti, ei]
            if np.isfinite(ecl).any():
                ecore_lat_m[ti] = float(np.nanmean(ecl))
                ecore_lat_s[ti] = float(np.nanstd(ecl))
            elif (np.isfinite(inner_l[ti, ei])
                  and np.isfinite(outer_l[ti, ei])):
                ecore_lat_m[ti] = 0.5 * (float(inner_l[ti, ei])
                                         + float(outer_l[ti, ei]))
            if np.isfinite(eca).any():
                ecore_alt_m[ti] = float(np.nanmean(eca))
                ecore_alt_s[ti] = float(np.nanstd(eca))
            elif np.isfinite(alt_m[ti, ei]):
                ecore_alt_m[ti] = float(alt_m[ti, ei])
        if e_idx:
            east_pct[ti] = float(np.nansum(pct10[ti, e_idx]))

    jet_intact   = ds['jet_intact'].values.astype(np.float32)
    grad_rev     = ds['gradient_reversed'].values.astype(np.float32)
    geo_b0       = ds['geo_b0'].values.astype(np.float32)

    geo_bot_lat  = ds['geo_bottom_lat'].values
    geo_aspect   = ds['geo_aspect_ratio'].values
    geo_area     = ds['geo_total_area_km2'].values
    # Per-timestep ranking of geopotential components by area so "comp 0/1"
    # diagnostics always refer to largest/second-largest lobes.
    n_geo_comp = geo_area.shape[1] if geo_area.ndim >= 2 else 1
    rank_idx = np.zeros((nt, n_geo_comp), dtype=int)
    for ti in range(nt):
        aa = np.asarray(geo_area[ti], dtype=float)
        aa2 = np.where(np.isfinite(aa), aa, -np.inf)
        rank_idx[ti] = np.argsort(-aa2)

    def ranked(arr2):
        out = np.asarray(arr2, dtype=float).copy()
        if out.ndim != 2:
            return out
        rr = np.empty_like(out)
        for ti in range(nt):
            rr[ti] = out[ti, rank_idx[ti]]
        return rr

    geo_bot_lat_r = ranked(geo_bot_lat)
    geo_aspect_r  = ranked(geo_aspect)
    geo_area_r    = ranked(geo_area)
    big_bot_lat  = (geo_bot_lat_r[:, 0] if geo_bot_lat_r.shape[1]
                    else np.full(nt, np.nan))
    big_aspect   = (geo_aspect_r[:, 0]  if geo_aspect_r.shape[1]
                    else np.full(nt, np.nan))
    area_fracs   = np.zeros(nt, np.float32)
    for ti in range(nt):
        aa = geo_area_r[ti]
        aa = aa[np.isfinite(aa) & (aa > 0)]
        if aa.size >= 2:
            area_fracs[ti] = float(aa[1] / np.nansum(aa))

    # number of geopotential components whose upper extent reaches above
    # 30 km; a real split should have >= 2 lobes alive up there.
    geo_alt_hi   = ds['geo_alt_hi'].values
    geo_alt_lo   = ds['geo_alt_lo'].values
    geo_lowest   = (ds['geo_lowest_lat'].values
                    if 'geo_lowest_lat' in ds
                    else np.full(np.shape(geo_alt_hi), np.nan,
                                 dtype=np.float32))
    geo_alt_hi_r = ranked(geo_alt_hi)
    geo_alt_lo_r = ranked(geo_alt_lo)
    geo_lowest_r = ranked(geo_lowest)
    with np.errstate(invalid='ignore'):
        geo_lowest_min = np.nanmin(geo_lowest_r, axis=1).astype(np.float32)
    geo_alt_max = np.full(nt, np.nan, np.float32)
    for ti in range(nt):
        ah = geo_alt_hi_r[ti]
        if ah.size:
            geo_alt_max[ti] = float(np.nanmax(ah))
    hi_comps_30  = np.zeros(nt, np.float32)
    for ti in range(nt):
        ah = geo_alt_hi_r[ti]
        hi_comps_30[ti] = float(np.sum(np.isfinite(ah) & (ah >= 30.0)))

    # per-level aspect ratio of each component -> take the value at the
    # lowest altitude level where the lobe exists. "Base aspect ratio"
    # is what we use for the geo_disturbance test instead of the mean,
    lev_ar = (ds['geo_lev_aspect_ratio'].values
              if 'geo_lev_aspect_ratio' in ds
              else np.full((nt, 1, len(alt)), np.nan))
    if lev_ar.ndim == 3:
        lev_ar_r = np.empty_like(lev_ar, dtype=float)
        for ti in range(nt):
            lev_ar_r[ti] = lev_ar[ti, rank_idx[ti], :]
        lev_ar = lev_ar_r
    ncomp  = lev_ar.shape[1] if lev_ar.ndim >= 2 else 0
    alt_np = np.asarray(alt, dtype=float)
    order  = np.argsort(alt_np)        # ascending altitude
    base_aspect = np.full((nt, ncomp), np.nan, np.float32)
    for ti in range(nt):
        for c in range(ncomp):
            col = lev_ar[ti, c]
            for k in order:
                if np.isfinite(col[k]):
                    base_aspect[ti, c] = col[k]
                    break

    # geopotential tilt of the biggest lobe: slope of centroid latitude
    # with altitude (deg / km). Positive => leans equatorward going down.
    geo_lev_lat = (ds['geo_lev_centroid_lat'].values
                   if 'geo_lev_centroid_lat' in ds
                   else np.full((nt, 1, len(alt)), np.nan))
    if geo_lev_lat.ndim == 3:
        geo_lev_lat_r = np.empty_like(geo_lev_lat, dtype=float)
        for ti in range(nt):
            geo_lev_lat_r[ti] = geo_lev_lat[ti, rank_idx[ti], :]
        geo_lev_lat = geo_lev_lat_r

    geo_lev_eq = None
    if 'geo_lev_lat_equatorward' in ds:
        geo_lev_eq = np.asarray(ds['geo_lev_lat_equatorward'].values, dtype=float)
        if geo_lev_eq.ndim == 3:
            geo_lev_eq_r = np.empty_like(geo_lev_eq, dtype=float)
            for ti in range(nt):
                geo_lev_eq_r[ti] = geo_lev_eq[ti, rank_idx[ti], :]
            geo_lev_eq = geo_lev_eq_r

    geo_lev_po = None
    if 'geo_lev_lat_poleward' in ds:
        geo_lev_po = np.asarray(ds['geo_lev_lat_poleward'].values, dtype=float)
        if geo_lev_po.ndim == 3:
            geo_lev_po_r = np.empty_like(geo_lev_po, dtype=float)
            for ti in range(nt):
                geo_lev_po_r[ti] = geo_lev_po[ti, rank_idx[ti], :]
            geo_lev_po = geo_lev_po_r

    # Per-level aspect ratio of comp 0 (= width / height of the 2-D
    # geopotential lobe at that level).  Used to distinguish localized
    # tilt-driven disturbances from filamentation/fragmentation events.
    geo_lev_aspect = None
    if 'geo_lev_aspect_ratio' in ds:
        geo_lev_aspect = np.asarray(ds['geo_lev_aspect_ratio'].values, dtype=float)
        if geo_lev_aspect.ndim == 3:
            geo_lev_aspect_r = np.empty_like(geo_lev_aspect, dtype=float)
            for ti in range(nt):
                geo_lev_aspect_r[ti] = geo_lev_aspect[ti, rank_idx[ti], :]
            geo_lev_aspect = geo_lev_aspect_r

    # Whole-component aspect ratio (one number per (t, comp)); used as
    # a complement to per-level aspect ratio for the time-series plot.
    geo_aspect_ranked = None
    if 'geo_aspect_ratio' in ds:
        ga = np.asarray(ds['geo_aspect_ratio'].values, dtype=float)
        if ga.ndim == 2:
            ga_r = np.empty_like(ga, dtype=float)
            for ti in range(nt):
                ga_r[ti] = ga[ti, rank_idx[ti]]
            geo_aspect_ranked = ga_r

    # per-component mean centroid latitude (nanmean over levels where
    # the lobe exists); comp 0 alone is the classic "displacement"
    # proxy, any-comp > 50°N is used to test for a real split above
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='Mean of empty slice')
        with np.errstate(invalid='ignore'):
            comp_cent_lat = (np.nanmean(geo_lev_lat, axis=2)
                             if geo_lev_lat.shape[1]
                             else np.full((nt, 1), np.nan))
    comp_cent_lat = np.asarray(comp_cent_lat, dtype=np.float32)
    big_cent_lat  = (comp_cent_lat[:, 0] if comp_cent_lat.shape[1]
                     else np.full(nt, np.nan, np.float32))
    dcent_lat     = np.gradient(big_cent_lat).astype(np.float32)
    d_lowest_min  = np.gradient(geo_lowest_min.astype(np.float64)
                                ).astype(np.float32)

    # Bottom of the largest lobe (comp0 after area ranking): **largest lev index**
    # with a finite per-level centroid; same convention as vtxplt
    # (bottom_lev_idx = max(level_indices)), not the column mean and not
    geo_c0_bottom_cent = np.full(nt, np.nan, np.float32)
    geo_c0_bottom_low = np.full(nt, np.nan, np.float32)
    order_asc = np.argsort(alt_np)
    if geo_lev_lat.shape[1] >= 1:
        for ti2 in range(nt):
            cl = geo_lev_lat[ti2, 0]
            el = (geo_lev_eq[ti2, 0] if geo_lev_eq is not None
                  and geo_lev_eq.shape[1] >= 1 else None)
            idxs = np.where(np.isfinite(cl))[0]
            if idxs.size:
                kk = int(idxs.max())
                geo_c0_bottom_cent[ti2] = float(cl[kk])
                if (el is not None and kk < el.shape[0]
                        and np.isfinite(el[kk])):
                    geo_c0_bottom_low[ti2] = float(el[kk])
    elif geo_lowest_r.shape[1]:
        # Degenerate: no per-level centroids in file; legacy scalar only.
        geo_c0_bottom_low = geo_lowest_r[:, 0].astype(np.float32)
    d_geo_c0_bottom_cent = np.gradient(
        geo_c0_bottom_cent.astype(np.float64)).astype(np.float32)
    d_geo_c0_bottom_low = np.gradient(
        geo_c0_bottom_low.astype(np.float64)).astype(np.float32)

    # "second-H0-above-50°N": second-largest component exists and its
    # mean centroid is pole-ward of 50°N. Used only to tighten the
    # geo_disturbance-without-warming test.
    second_above_50 = np.zeros(nt, dtype=bool)
    if comp_cent_lat.shape[1] >= 2:
        lat2 = comp_cent_lat[:, 1]
        second_above_50 = np.isfinite(lat2) & (lat2 >= 50.0)
    geo_tilt   = np.full(nt, np.nan, np.float32)
    alt_arr    = np.asarray(alt_native, dtype=np.float64)
    for ti in range(nt):
        clat = geo_lev_lat[ti, 0] if geo_lev_lat.shape[1] else None
        if clat is None:
            continue
        m = np.isfinite(clat) & np.isfinite(alt_arr)
        if m.sum() < 3:
            continue
        # simple linear slope (least-squares): d(lat) / d(alt)
        x = alt_arr[m]; y = clat[m].astype(np.float64)
        cov = np.polyfit(x, y, 1)
        geo_tilt[ti] = float(cov[0])

    # component-1 split hints: must be a real second lobe that is seeded
    # near comp-0 at some level (split origin), not a detached low-lat
    # blob that never overlaps the intact vortex column.
    geo_lev_lon = (ds['geo_lev_centroid_lon'].values
                   if 'geo_lev_centroid_lon' in ds
                   else np.full_like(geo_lev_lat, np.nan))
    if geo_lev_lon.ndim == 3:
        geo_lev_lon_r = np.empty_like(geo_lev_lon, dtype=float)
        for ti in range(nt):
            geo_lev_lon_r[ti] = geo_lev_lon[ti, rank_idx[ti], :]
        geo_lev_lon = geo_lev_lon_r
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='Mean of empty slice')
        with np.errstate(invalid='ignore'):
            comp_cent_lon = (np.nanmean(geo_lev_lon, axis=2)
                             if geo_lev_lon.shape[1]
                             else np.full((nt, 1), np.nan))
    comp_cent_lon = np.asarray(comp_cent_lon, dtype=np.float32)
    split_c1_hint = np.zeros(nt, np.float32)
    split_seeded = np.zeros(nt, np.float32)
    split_comp1_debris = np.zeros(nt, np.float32)
    comp1_present = np.zeros(nt, np.float32)
    if geo_area_r.shape[1] >= 2:
        comp1_present = (np.isfinite(geo_area_r[:, 1]) &
                         (geo_area_r[:, 1] > 0.0)).astype(np.float32)
    if comp_cent_lat.shape[1] >= 2:
        lat0 = comp_cent_lat[:, 0]; lat1 = comp_cent_lat[:, 1]
        lon0 = comp_cent_lon[:, 0] if comp_cent_lon.shape[1] >= 1 \
               else np.full(nt, np.nan, np.float32)
        lon1 = comp_cent_lon[:, 1] if comp_cent_lon.shape[1] >= 2 \
               else np.full(nt, np.nan, np.float32)
        dlon = np.abs(((lon1 - lon0 + 180.0) % 360.0) - 180.0)
        near_mean = (np.isfinite(lat0) & np.isfinite(lat1) &
                     np.isfinite(dlon) &
                     (np.abs(lat1 - lat0) <= 6.0) & (dlon <= 20.0))
        upper = np.isfinite(lat1) & (lat1 >= 50.0)
        low_only = np.isfinite(lat1) & (lat1 < 45.0)
        if geo_alt_hi_r.shape[1] >= 2 and geo_alt_lo_r.shape[1] >= 2:
            tall = (np.isfinite(geo_alt_hi_r[:, 1]) & np.isfinite(geo_alt_lo_r[:, 1]) &
                    ((geo_alt_hi_r[:, 1] - geo_alt_lo_r[:, 1]) >= 10.0) &
                    (geo_alt_hi_r[:, 1] >= 30.0))
        else:
            tall = np.zeros(nt, dtype=bool)
        # level-by-level split seed: at least one level where comp1 is near
        # Split-seeding: comp 1 is a real split product if it has
        # substantial area (>=18% of comp 0) and its centroid sits
        seeded_now = np.zeros(nt, dtype=bool)
        if (geo_lev_lat.shape[1] >= 2) and geo_area_r.shape[1] >= 2:
            a0 = np.maximum(np.asarray(geo_area_r[:, 0],
                                        dtype=np.float64), 1e-6)
            a1 = np.asarray(geo_area_r[:, 1], dtype=np.float64)
            af = a1 / a0
            substantial = np.isfinite(af) & (af >= 0.18)
            # comp 1's mean centroid lat across levels; at least
            # 47°N to distinguish a real polar-vortex split lobe
            # (mean ~47-60°N) from an edge-of-cropped-grid fragment
            lat1_lev = geo_lev_lat[:, 1, :]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                lat1_mean = np.where(np.isfinite(lat1_lev).any(axis=1),
                                      np.nanmean(lat1_lev, axis=1),
                                      np.nan)
            off_edge = np.isfinite(lat1_mean) & (lat1_mean >= 47.0)
            seeded_now = substantial & off_edge
        seeded_recent = np.zeros(nt, dtype=bool)
        for ti in range(nt):
            j0 = max(0, ti - 12)
            seeded_recent[ti] = bool(np.any(seeded_now[j0:ti + 1]))
        split_seeded = seeded_recent.astype(np.float32)

        split_c1_hint = ((comp1_present >= 0.5) &
                         (split_seeded >= 0.5) &
                         (upper | near_mean | tall) &
                         (~low_only))
        split_c1_hint = split_c1_hint.astype(np.float32)

        # Second lobe that is tiny, shallow, lives at low latitude, and was
        # never seeded next to comp0 is subtropical / mask artefact; not split.
        a0 = np.maximum(np.asarray(geo_area_r[:, 0], dtype=np.float64), 1e-6)
        a1 = np.asarray(geo_area_r[:, 1], dtype=np.float64)
        frac1 = (a1 / a0).astype(np.float64)
        ah1 = geo_alt_hi_r[:, 1]
        al1 = geo_alt_lo_r[:, 1]
        depth1 = np.where(np.isfinite(ah1) & np.isfinite(al1), ah1 - al1, np.nan)
        lat0c = comp_cent_lat[:, 0]
        debris = (
            (comp1_present >= 0.5) &
            np.isfinite(lat1) & (lat1 < 47.0) &
            np.isfinite(frac1) & (frac1 < 0.22) &
            (~seeded_recent) &
            (
                (~np.isfinite(ah1)) | (ah1 < 30.0) |
                (~np.isfinite(depth1)) | (depth1 < 10.0)
            ) &
            (np.isfinite(lat0c) & (lat0c >= 52.0))
        )
        split_comp1_debris = debris.astype(np.float32)

    # level-aware wind reversal: negative wind with enough longitude
    # span at any level, regardless of primary component ordering.
    lev_U_all = (np.asarray(ds['wind_lev_mean_U'].values, dtype=np.float32)
                 if 'wind_lev_mean_U' in ds
                 else np.full((nt, 1, len(alt)), np.nan, np.float32))
    lev_S_all = (np.asarray(ds['wind_lev_lon_span'].values, dtype=np.float32)
                 if 'wind_lev_lon_span' in ds
                 else np.full_like(lev_U_all, np.nan))
    rev_mask = np.isfinite(lev_U_all) & np.isfinite(lev_S_all) & \
               (lev_U_all < 0.0) & (lev_S_all >= 0.5)
    full_mask = np.isfinite(lev_S_all) & (lev_S_all >= 0.95)
    rev_lev_any = rev_mask.any(axis=1)  # (time, lev)
    full_lev_any = full_mask.any(axis=1)
    rev_lev_count = np.sum(rev_lev_any, axis=1).astype(np.float32)
    full_lev_count = np.sum(full_lev_any, axis=1).astype(np.float32)
    upper_lev = np.asarray(alt_native, dtype=float) >= 30.0
    if upper_lev.any():
        upper_rev = rev_lev_any[:, upper_lev].any(axis=1).astype(np.float32)
    else:
        upper_rev = np.zeros(nt, np.float32)

    # intact easterly ring at any level: at least one (comp,lev) cell with
    # negative U and near-closed longitude coverage.
    east_intact_level = (
        np.isfinite(lev_U_all) & np.isfinite(lev_S_all) &
        (lev_U_all < 0.0) & (lev_S_all >= 0.95)
    ).any(axis=(1, 2)).astype(np.float32)

    # in-vortex warming signals
    # the polar cap mean is too broad; temperature
    # tracking should follow where the vortex actually is.  We compute
    warm7_max     = np.full(nt, np.nan, np.float32)
    warm7_alt_km  = np.full(nt, np.nan, np.float32)
    warm7_lat     = np.full(nt, np.nan, np.float32)
    warm7_lookback_d = np.full(nt, -1,  np.int16)
    warm7_end_off = np.full(nt, -1, np.int16)
    # Max single-day ΔT inside the vortex (separate from the 7-day
    # max-rise above).  Used by print_onsets so multi-spike events
    # report the fastest single-day jump in addition to the
    warm1_max     = np.full(nt, np.nan, np.float32)
    warm1_alt_km  = np.full(nt, np.nan, np.float32)
    warm1_lat     = np.full(nt, np.nan, np.float32)
    warm_area_frac = np.full(nt, np.nan, np.float32)
    warm_bin_count = np.full(nt, np.nan, np.float32)
    alt_a    = np.asarray(alt, dtype=np.float64)
    nlev = len(alt_a)
    T_vortex       = np.full((nt, nlev), np.nan, np.float32)
    lat_full = np.asarray(lat, dtype=float)

    # Polar-cap area-weighted mean (kept for plot/diagnostic backwards
    # compatibility); exposed as Tp_w.
    if has_new_T:
        # Already computed above for the T_in_geo_mean path (cos-lat
        # weighted polar-cap mean per (t, lev)).
        Tp_w = Tp.copy()
    else:
        lat_pol60 = lat_full >= 60.0
        if np.any(lat_pol60):
            w = np.cos(np.deg2rad(lat_full[lat_pol60]))
            w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0).astype(np.float32)
            T_polar = T[:, :, lat_pol60]
            finite_mask = np.isfinite(T_polar)
            T_w_sum = (np.where(finite_mask, T_polar, 0.0) *
                       w[None, None, :]).sum(axis=2)
            w_sum  = (finite_mask * w[None, None, :]).sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                Tp_w = np.where(w_sum > 0,
                                T_w_sum / np.maximum(w_sum, 1e-12),
                                np.nan).astype(np.float32)
        else:
            Tp_w = np.nanmean(T, axis=2).astype(np.float32)

    # in-vortex gridpoint 7-day and 1-day rise
    # T_in_geo path: T_in_geo_max[t, lev, lat_bin, lon_bin] is
    # already restricted to the polar wind-ring perimeter, so each
    warm7_max_below10 = np.full(nt, np.nan, np.float32)
    warm7_max_above10 = np.full(nt, np.nan, np.float32)
    if has_new_T:
        nlat_b = Tc_max.shape[2]
        nlon_b = Tc_max.shape[3]
        # T_vortex: mean T_in_geo_mean over all populated bins per (t, lev)
        with np.errstate(invalid='ignore'):
            T_vortex = np.nanmean(np.nanmean(Tc_mean, axis=3),
                                  axis=2).astype(np.float32)

        # SSW trigger: 3-D regional mean ΔT (all altitudes)
        T_reg = Tc_mean
        alt_reg = np.asarray(alt, dtype=float)
        thr25 = 25.0
        # Restart the 7-day comparison after a warming ends. The warm7 window
        # must never difference today's temperature against a day from before
        # the vortex reconvened; that would manufacture a false rise from
        ring_lev = (np.asarray(ds['wind_ring_n_levels'].values, dtype=float)
                     if 'wind_ring_n_levels' in ds
                     else np.full((nt, is_h1.shape[1]), np.nan))
        nrl = np.full(nt, np.nan, dtype=float)
        for tix in range(nt):
            best = np.nan
            for cc in range(ring_lev.shape[1]):
                h = is_h1[tix, cc] if cc < is_h1.shape[1] else np.nan
                if np.isfinite(h) and h >= 0.5 and np.isfinite(ring_lev[tix, cc]):
                    v = float(ring_lev[tix, cc])
                    if not np.isfinite(best) or v > best:
                        best = v
            nrl[tix] = 0.0 if not np.isfinite(best) else best
        nrf = nrl[np.isfinite(nrl) & (nrl > 0)]
        reset_mask = None
        if nrf.size:
            half = 0.5 * float(np.percentile(nrf, 98))
            bc = np.asarray(geo_c0_bottom_cent, dtype=float)
            ring_hi = np.isfinite(nrl) & (nrl >= half)
            geo_ctr = np.isfinite(bc) & (bc >= 70.0)
            reconv_day = ring_hi & geo_ctr
            # genuine reconvene: rings up and geo centered, >= 2 consecutive
            # days. A brief ring spike without geo centering is not a
            # reconvene and must not reset the temperature baseline.
            reconvened = np.zeros(nt, dtype=bool)
            run = 0
            for kkx in range(nt):
                if reconv_day[kkx]:
                    run += 1
                    if run >= 2:
                        reconvened[kkx - 1:kkx + 1] = True
                else:
                    run = 0
            # reset only on the rising edge: the first reconvened day after a
            # non-reconvened day. The baseline restarts there once; warm7 then
            # computes normally forward through the stable period.
            reset_mask = np.zeros(nt, dtype=bool)
            reset_mask[1:] = reconvened[1:] & ~reconvened[:-1]
            if reconvened[0]:
                reset_mask[0] = True
        # Altitude of the 10 hPa surface, from the raw pressure/altitude
        # coordinate; used to split the warming rise into the at/below-10 hPa
        # and above-10 hPa bands for the temperature-based major/minor test.
        lev_hpa_full = np.asarray(ds['lev'].values, dtype=float)
        alt_km_full  = np.asarray(ds['altitude_km'].values, dtype=float)
        z10_km = float(np.interp(MAJOR_MINOR_SPLIT_HPA, lev_hpa_full,
                                 alt_km_full))
        w7 = compute_warm7_regional_means(
            T_reg, alt_reg, latb, thr=thr25, max_d=7, reset_mask=reset_mask,
            split_alt_km=z10_km)
        warm7_max = w7['warm7_max']
        warm7_max_below10 = w7['warm7_below10']
        warm7_max_above10 = w7['warm7_above10']
        warm7_alt_km = w7['warm7_alt_km']
        warm7_lat = w7['warm7_lat']
        warm7_lookback_d = w7['warm7_lookback_d']
        warm7_end_off = w7.get('warm7_end_off',
                                np.full(nt, -1, dtype=np.int16))
        warm1_max = w7['warm1_max']
        warm1_alt_km = w7['warm1_alt_km']
        warm1_lat = w7['warm1_lat']
        warm_area_frac = w7['warm_area_frac']
        warm_bin_count = w7['warm_bin_count']
    else:
        # Build the per-level lat-mask for comp 0: in_vortex[t, lev, lat]
        # is True iff the lat is inside the comp-0 geopotential lobe at
        # (t, lev).  Also store T_vortex (mean T over those gridpoints).
        use_vortex = ((geo_lev_eq is not None) and (geo_lev_po is not None) and
                      lat_full.size > 0)
        if use_vortex:
            nlat = lat_full.size
            in_vortex = np.zeros((nt, nlev, nlat), dtype=bool)
            for ti in range(nt):
                for lv in range(nlev):
                    eq = (geo_lev_eq[ti, 0, lv]
                          if geo_lev_eq.shape[1] else np.nan)
                    po = (geo_lev_po[ti, 0, lv]
                          if geo_lev_po.shape[1] else np.nan)
                    if not (np.isfinite(eq) and np.isfinite(po)):
                        continue
                    lo = min(float(eq), float(po))
                    hi = max(float(eq), float(po))
                    in_vortex[ti, lv, :] = (lat_full >= lo) & (lat_full <= hi)
                    seg = T[ti, lv, in_vortex[ti, lv, :]]
                    seg = seg[np.isfinite(seg)]
                    if seg.size:
                        T_vortex[ti, lv] = float(seg.mean())

            # Per-gridpoint 7-day rise on lightly lat-smoothed T_zonal
            T_s = uniform_filter1d(
                np.where(np.isfinite(T), T, np.nan),
                size=3, axis=2, mode='nearest').astype(np.float32)
            T_s = np.where(np.isfinite(T), T_s, np.nan)

            for ti in range(nt):
                best_v = -np.inf; best_lev = -1; best_lat_idx = -1; best_d = -1
                mask_t = in_vortex[ti]              # (nlev, nlat)
                T_now  = T_s[ti]                    # (nlev, nlat)
                for d in range(1, min(8, ti + 1)):
                    T_then = T_s[ti - d]            # (nlev, nlat)
                    with np.errstate(invalid='ignore'):
                        delta = np.where(mask_t & np.isfinite(T_now) &
                                         np.isfinite(T_then),
                                         T_now - T_then, -np.inf)
                    if np.isfinite(delta).any():
                        flat_idx = int(np.argmax(delta))
                        v = float(delta.flat[flat_idx])
                        if v > best_v:
                            best_v = v
                            best_lev = flat_idx // delta.shape[1]
                            best_lat_idx = flat_idx % delta.shape[1]
                            best_d = d
                if np.isfinite(best_v) and best_v > -np.inf:
                    warm7_max[ti]     = best_v
                    warm7_alt_km[ti]  = (float(alt_a[best_lev])
                                         if 0 <= best_lev < alt_a.size else np.nan)
                    warm7_lat[ti]     = (float(lat_full[best_lat_idx])
                                         if 0 <= best_lat_idx < nlat else np.nan)
                    warm7_lookback_d[ti] = int(best_d)
                if ti >= 1:
                    T_yest = T_s[ti - 1]
                    with np.errstate(invalid='ignore'):
                        d1 = np.where(mask_t & np.isfinite(T_now) &
                                      np.isfinite(T_yest),
                                      T_now - T_yest, -np.inf)
                    if np.isfinite(d1).any():
                        fi1 = int(np.argmax(d1))
                        v1 = float(d1.flat[fi1])
                        if v1 > -np.inf:
                            warm1_max[ti]    = v1
                            warm1_alt_km[ti] = (
                                float(alt_a[fi1 // d1.shape[1]])
                                if 0 <= fi1 // d1.shape[1] < alt_a.size
                                else np.nan)
                            warm1_lat[ti] = (
                                float(lat_full[fi1 % d1.shape[1]])
                                if 0 <= fi1 % d1.shape[1] < nlat
                                else np.nan)
        else:
            # Fallback: polar-cap-mean lookback (Tp_w) per level.
            for ti in range(nt):
                best_v = np.nan; best_lev = -1; best_d = -1
                for d in range(1, 8):
                    j0 = ti - d
                    if j0 < 0:
                        break
                    d7 = Tp_w[ti] - Tp_w[j0]
                    if np.isfinite(d7).any():
                        k = int(np.nanargmax(d7))
                        v = float(d7[k])
                        if not np.isfinite(best_v) or v > best_v:
                            best_v = v; best_lev = k; best_d = d
                if np.isfinite(best_v):
                    warm7_max[ti]     = best_v
                    warm7_alt_km[ti]  = (float(alt_a[best_lev])
                                         if 0 <= best_lev < alt_a.size else np.nan)
                    warm7_lookback_d[ti] = int(best_d)

    # warm_v_max; kept as alias for plot backwards compat (same as
    # warm7 now since we already track local-gridpoint max changes).
    warm_v_max    = warm7_max.copy()
    warm_v_alt_km = warm7_alt_km.copy()

    ring_field = ring_U_by_alt(ds)

    # widest (lo, hi) altitude band of the dominant westerly ring, useful
    # both as a clustering feature and for per-cluster physics summaries.
    bands    = ds['wind_ring_alt_bands'].values   # (t, comp, band, 2)
    band_lo  = np.full(nt, np.nan, np.float32)
    band_hi  = np.full(nt, np.nan, np.float32)
    band_span= np.full(nt, np.nan, np.float32)
    for ti in range(nt):
        active = np.isfinite(is_h1[ti]) & (is_h1[ti] >= 0.5) \
                 & np.isfinite(sign[ti]) \
                 & (np.round(sign[ti].astype(float)) == 0)
        if not active.any():
            continue
        ci = int(np.where(active)[0][np.nanargmax(
            np.abs(mU[ti, active]))])
        widths = []
        for bi in range(bands.shape[2]):
            lo = bands[ti, ci, bi, 0]; hi = bands[ti, ci, bi, 1]
            if np.isfinite(lo) and np.isfinite(hi):
                widths.append((float(min(lo, hi)), float(max(lo, hi))))
        if not widths:
            continue
        lo_b, hi_b = max(widths, key=lambda ab: ab[1] - ab[0])
        band_lo[ti]   = lo_b
        band_hi[ti]   = hi_b
        band_span[ti] = hi_b - lo_b

    # per-level polar-cap T anomaly and dT/dt (used for SVD)
    Tp_anom_lev = (Tp - np.nanmean(Tp, axis=0, keepdims=True)).astype(np.float32)
    dTp_lev     = np.gradient(Tp, axis=0).astype(np.float32)

    # Ring multiplicity: number of altitude levels where the dominant
    # westerly ring (comp 0 by area) has an intact ring (>= 50% of
    # longitudes contain westerly wind).  A "stable strong" vortex
    wind_ring_n_lev_per_comp = (
        np.asarray(ds['wind_ring_n_levels'].values, dtype=float)
        if 'wind_ring_n_levels' in ds
        else np.full((nt, 4), np.nan))

    # Per-altitude westerly ring presence and mean U for the dominant
    # westerly component at each level. These are the per-altitude inputs the
    # event detector uses to compare against the most recent strong reference.
    alt_km_arr = np.asarray(alt_native, dtype=float)
    n_alt = alt_km_arr.size
    ring_present_west = np.zeros((nt, n_alt), dtype=bool)
    west_mean_U       = np.full((nt, n_alt), np.nan, dtype=np.float32)
    lev_U_for_state = (np.asarray(ds['wind_lev_mean_U'].values, dtype=float)
                       if 'wind_lev_mean_U' in ds else None)
    for ti in range(nt):
        for ci in range(bands.shape[1]):
            # westerly only (sign == 0; 1 == easterly)
            if np.isfinite(sign[ti, ci]) and np.round(sign[ti, ci]) == 0:
                for bi in range(bands.shape[2]):
                    lo = bands[ti, ci, bi, 0]
                    hi = bands[ti, ci, bi, 1]
                    if np.isfinite(lo) and np.isfinite(hi):
                        lo_k, hi_k = min(lo, hi), max(lo, hi)
                        in_band = (alt_km_arr >= lo_k) & (alt_km_arr <= hi_k)
                        ring_present_west[ti] |= in_band
                if lev_U_for_state is not None:
                    u_row = lev_U_for_state[ti, ci]
                    pos = np.isfinite(u_row) & (u_row > 0)
                    if pos.any():
                        cur = west_mean_U[ti]
                        new = np.where(pos, u_row, -np.inf)
                        cur_filled = np.where(np.isfinite(cur), cur, -np.inf)
                        west_mean_U[ti] = np.where(
                            new > cur_filled, new,
                            np.where(np.isfinite(cur), cur, np.nan))

    # n_ring_levels = number of altitudes carrying a westerly ring, from the
    # per-altitude ring_present_west mask (the field the plot's ring panel
    # draws). This is the authoritative ring-coverage count.
    n_ring_levels = ring_present_west.sum(axis=1).astype(np.float32)

    # Genuine-two-lobe indicator: True iff two distinct geopotential
    # components are present whose per-level centroids are far apart
    # in physical km on enough shared levels to be a real split, not
    two_lobes_genuine = np.zeros(nt, dtype=np.float32)
    if (geo_lev_lat.ndim == 3 and geo_lev_lat.shape[1] >= 2 and
            geo_lev_lon.ndim == 3 and geo_lev_lon.shape[1] >= 2):
        # We need the per-level centroids of components 0 and 1
        # together; reuse the rank-sorted (geo_lev_lat, geo_lev_lon)
        # that were built above for compatibility with comp1_present.
        if 'geo_lev_centroid_lat' in ds and 'geo_lev_centroid_lon' in ds:
            glat = np.asarray(ds['geo_lev_centroid_lat'].values, dtype=float)
            glon = np.asarray(ds['geo_lev_centroid_lon'].values, dtype=float)
            if glat.ndim == 3 and glat.shape[1] >= 2:
                # Apply rank-sort by area to align comp-0 / comp-1 with
                # the rest of the diagnostics.
                g0_lat = np.empty(glat.shape[2], dtype=float)
                g0_lon = np.empty(glat.shape[2], dtype=float)
                g1_lat = np.empty(glat.shape[2], dtype=float)
                g1_lon = np.empty(glat.shape[2], dtype=float)
                GREAT_CIRCLE_THR_KM = 2500.0  # ~22.5 deg great-circle
                MIN_FAR_LEVELS      = 5
                R_KM = 6371.0
                for ti in range(nt):
                    ri = rank_idx[ti]
                    if ri.size < 2:
                        continue
                    g0_lat[:] = glat[ti, ri[0], :]
                    g0_lon[:] = glon[ti, ri[0], :]
                    g1_lat[:] = glat[ti, ri[1], :]
                    g1_lon[:] = glon[ti, ri[1], :]
                    finite = (np.isfinite(g0_lat) & np.isfinite(g0_lon) &
                              np.isfinite(g1_lat) & np.isfinite(g1_lon))
                    if not finite.any():
                        continue
                    # Great-circle distance, vectorised over shared levels
                    phi1 = np.deg2rad(g0_lat[finite])
                    phi2 = np.deg2rad(g1_lat[finite])
                    dphi = phi2 - phi1
                    dlam = np.deg2rad(g1_lon[finite] - g0_lon[finite])
                    a = (np.sin(dphi / 2.0)**2 +
                         np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2)
                    a = np.clip(a, 0.0, 1.0)
                    d_km = 2.0 * R_KM * np.arcsin(np.sqrt(a))
                    n_far = int(np.sum(d_km >= GREAT_CIRCLE_THR_KM))
                    if n_far >= MIN_FAR_LEVELS:
                        two_lobes_genuine[ti] = 1.0

    # Per-level Betti-0 profiles (_3 datasets): slice-view split / partial split.
    b0_morph = {}
    if ('geo_largest_b0_profile' in ds.data_vars and
            'geo_second_b0_profile' in ds.data_vars):
        big_b0 = np.asarray(ds['geo_largest_b0_profile'].values, dtype=float)
        sec_b0 = np.asarray(ds['geo_second_b0_profile'].values, dtype=float)
        b0_morph = compute_b0_morphology_flags(
            big_b0, sec_b0, np.asarray(alt_native, dtype=float),
            top_alt_min_km=45.0, bottom_alt_max_km=35.0, min_vert=3)
    else:
        z = np.zeros(nt, np.float32)
        b0_morph = dict(b0_full_split=z, b0_top_pinch_split=z,
                        b0_partial_split=z, b0_frac_second=z,
                        b0_frac_top_partial=z, b0_frac_bot_partial=z,
                        b0_full_split_prog=z,
                        b0_split_full_thr=np.float32(0.40),
                        b0_split_partial_floor=np.float32(0.12))

    return dict(
        peak_U=peak_U, mean_U=mean_U, region_speed=region_speed,
        inner_lat=inner_lat, outer_lat=outer_lat, ring_alt=ring_alt,
        ring_area=ring_area, east_area=east_area,
        east_inner=east_inner, east_outer=east_outer,
        east_alt=east_alt, east_U=east_U, east_active=east_active,
        band_lo=band_lo, band_hi=band_hi, band_span=band_span,
        n_ring_levels=n_ring_levels,
        core_lat_mean=core_lat_m,  core_lat_std=core_lat_s,
        core_alt_mean=core_alt_m,  core_alt_std=core_alt_s,
        east_core_lat_mean=ecore_lat_m, east_core_lat_std=ecore_lat_s,
        east_core_alt_mean=ecore_alt_m, east_core_alt_std=ecore_alt_s,
        wind_tilt=wind_tilt, geo_tilt=geo_tilt,
        east_pct=east_pct, west_pct=west_pct,
        Tp_col=Tp_col, dT_col=dT_col, d2T_col=d2T_col,
        Tp_anom_lev=Tp_anom_lev, dTp_lev=dTp_lev,
        warm_anom_max=warm_anom_max, warm_dT_max=warm_dT_max,
        warm_local_anom_max=warm_local_anom_max,
        warm_local_dT_max=warm_local_dT_max,
        jet_intact=jet_intact,
        gradient_reversed=grad_rev,
        geo_b0=geo_b0,
        big_bot_lat=big_bot_lat.astype(np.float32),
        big_aspect=big_aspect.astype(np.float32),
        area_fracs=area_fracs,
        hi_comps_30=hi_comps_30,
        big_cent_lat=big_cent_lat,
        dcent_lat=dcent_lat,
        d_lowest_min=d_lowest_min,
        geo_c0_bottom_cent=geo_c0_bottom_cent,
        geo_c0_bottom_low=geo_c0_bottom_low,
        d_geo_c0_bottom_cent=d_geo_c0_bottom_cent,
        d_geo_c0_bottom_low=d_geo_c0_bottom_low,
        comp_cent_lat=comp_cent_lat,
        comp_cent_lon=comp_cent_lon,
        split_c1_hint=split_c1_hint,
        split_seeded=split_seeded,
        split_comp1_debris=split_comp1_debris,
        comp1_present=comp1_present,
        second_above_50=second_above_50.astype(np.float32),
        rev_lev_count=rev_lev_count,
        full_lev_count=full_lev_count,
        upper_rev=upper_rev,
        east_intact_level=east_intact_level,
        ring_present_west=ring_present_west,
        west_mean_U=west_mean_U,
        two_lobes_genuine=two_lobes_genuine,
        warm7_max=warm7_max,
        warm7_alt_km=warm7_alt_km,
        warm7_lat=warm7_lat,           # latitude where the 7-day max-rise gridpoint lives
        warm7_lookback_d=warm7_lookback_d,
        warm7_end_off=warm7_end_off,
        warm7_max_below10=warm7_max_below10,   # peak 7-d rise, p>=10 hPa band
        warm7_max_above10=warm7_max_above10,   # peak 7-d rise, p<10 hPa band
        warm1_max=warm1_max,           # max single-day in-vortex ΔT
        warm1_alt_km=warm1_alt_km,
        warm1_lat=warm1_lat,
        warm_area_frac=warm_area_frac,
        warm_bin_count=warm_bin_count,
        Tp_w=Tp_w,            # (nt, nlev) polar-cap area-weighted mean T
        T_vortex=T_vortex,    # (nt, nlev) mean T inside comp-0 lat band per level
        warm_v_max=warm_v_max,         # peak-to-trough warming (in-vortex)
        warm_v_alt_km=warm_v_alt_km,   # altitude where peak-trough max'd
        base_aspect=base_aspect,
        geo_alt_lo=geo_alt_lo_r.astype(np.float32),
        geo_alt_hi=geo_alt_hi_r.astype(np.float32),
        geo_lowest_min=geo_lowest_min,
        geo_alt_max=geo_alt_max,
        geo_area=geo_area_r.astype(np.float32),
        geo_aspect=(geo_aspect_ranked.astype(np.float32)
                    if geo_aspect_ranked is not None
                    else np.full((nt, n_geo_comp), np.nan, np.float32)),
        geo_lev_aspect=(geo_lev_aspect.astype(np.float32)
                        if geo_lev_aspect is not None
                        else np.full((nt, n_geo_comp, nlev), np.nan, np.float32)),
        geo_lev_lat_eq=(geo_lev_eq.astype(np.float32)
                        if geo_lev_eq is not None
                        else np.full((nt, n_geo_comp, nlev), np.nan, np.float32)),
        geo_lev_lat_po=(geo_lev_po.astype(np.float32)
                        if geo_lev_po is not None
                        else np.full((nt, n_geo_comp, nlev), np.nan, np.float32)),
        west_active=west_active,
        # altitude Hovmöller: signed mean/peak U per (time, alt)
        ring_U_alt=ring_field['U_mean'],
        ring_U_alt_peak=ring_field['U_peak'],
        alt_km=ring_field['alt'],
        alt_edges=ring_field['edges'],
        **b0_morph,
    )


# core decision logic

def wind_bad_day(ti, fl, r):
    # True wind disturbance: easterly ring, genuinely weak peak, or
    peak = fl['peak_U'][ti]
    if np.isfinite(peak) and peak <= r.wind_disturb_peak_U:
        return True
    jet_broken = fl['jet_intact'][ti] < 0.5
    thr = r.strong_peak_U * r.jet_resilience_frac
    strong_west = np.isfinite(peak) and peak >= thr
    if jet_broken and not strong_west:
        return True
    return False


def wind_recent(ti, fl, r):
    lb = int(max(1, r.wind_precursor_lookback))
    for k in range(max(0, ti - lb + 1), ti + 1):
        if wind_bad_day(k, fl, r):
            return True
    return False


def equatorward_motion(ti, fl, r):
    # Biggest-lobe centroid and vortex lowest latitude both drifting
    dll_a = fl.get('d_lowest_min')
    if dll_a is None:
        return False
    thr = float(r.disp_move_deg_per_day)
    dcl = float(fl['dcent_lat'][ti])
    dll = float(dll_a[ti])
    if not (np.isfinite(dcl) and np.isfinite(dll)):
        return False
    return (dcl < -thr) and (dll < -thr)


def equatorward_comp0_bottom(ti, fl, r):
    # Comp0: lowest-level centroid and lowest contour latitude both move
    dc = fl.get('d_geo_c0_bottom_cent')
    dl = fl.get('d_geo_c0_bottom_low')
    if dc is None or dl is None:
        return False
    thr = float(r.disp_move_deg_per_day)
    d1 = float(dc[ti])
    d2 = float(dl[ti])
    if not (np.isfinite(d1) and np.isfinite(d2)):
        return False
    return (d1 < -thr) and (d2 < -thr)


def wind_context_ssw(ti, fl, r):
    # SSW sequence is normally wind → T → geo; require same-day or recent wind.
    return wind_recent(ti, fl, r) or wind_bad_day(ti, fl, r)


def equatorward_motion_recent(ti, fl, r, days=None):
    # True if column or comp0-bottom equatorward motion on any recent day.
    if days is None:
        days = int(getattr(r, 'equatorward_recent_days', 8))
    for k in range(max(0, ti - int(days)), ti + 1):
        if equatorward_motion(k, fl, r) or equatorward_comp0_bottom(k, fl, r):
            return True
    return False


def bottom_low_displaced(ti, fl, r):
    # Comp0 lowest contour latitude reaches unusually low °N recently; the
    low = fl.get('geo_c0_bottom_low')
    if low is None:
        return False
    low = np.asarray(low, dtype=float)
    n = low.size
    if ti < 0 or ti >= n:
        return False
    win = int(getattr(r, 'disp_bottom_low_window_days', 15))
    thr = float(getattr(r, 'disp_bottom_low_lat_max', 64.0))
    j0 = max(0, ti - win)
    seg = low[j0:ti + 1]
    return bool(np.isfinite(seg).any() and float(np.nanmin(seg)) <= thr)


def geo_displacement_signal(ti, fl, r, geo_bad_here: bool):
    # Any evidence the *main* lobe is displaced (not just column-mean elongation).
    return bool(
        geo_bad_here
        or bottom_low_displaced(ti, fl, r)
        or equatorward_motion_recent(ti, fl, r)
    )


def geo_disturbance_type(fl, r):
    # Per-day geopotential disturbance classification.
    n = len(fl['geo_b0'])
    split = build_geo_split_array(fl, r)
    comp1 = np.asarray(fl.get('comp1_present', np.zeros(n)), dtype=float) >= 0.5
    tilt = np.abs(np.asarray(fl.get('geo_tilt', np.full(n, np.nan)),
                             dtype=float))
    asp = np.asarray(fl.get('big_aspect', np.full(n, np.nan)), dtype=float)
    tilt_thr = float(getattr(r, 'geo_disturb_min_tilt', 0.06) or 0.06)
    asp_thr = float(getattr(r, 'geo_aspect_bot', 1.8))
    code = np.zeros(n, dtype=np.int16)
    stretch = np.isfinite(asp) & (asp >= asp_thr)
    big_tilt = np.isfinite(tilt) & (tilt >= tilt_thr)
    # Priority (last assignment wins): filamentation > stretching > tilting.
    # Tilting is lowest because a stretched lobe is often also tilted, but a
    # tilt need not stretch; a real second component (filament) outranks both.
    code[big_tilt] = 2          # tilting
    code[stretch] = 3           # stretching (overrides tilting)
    code[split | comp1] = 1     # filamentation (overrides all)
    return code


GEO_DIST_TYPE_NAME = {0: 'intact', 1: 'filamentation',
                       2: 'tilting', 3: 'stretching'}


def build_geo_split_array(fl, r):
    # Two-cell *polar* split mask; veto small low-latitude debris lobes.
    n = len(fl['geo_b0'])
    geo_b0 = np.asarray(fl['geo_b0'], dtype=float)
    area_f = np.asarray(fl['area_fracs'], dtype=float)
    sec50 = np.asarray(fl['second_above_50'], dtype=float) >= 0.5
    c1_hint = np.asarray(fl.get('split_c1_hint', np.zeros(n)), dtype=float) >= 0.5
    c1_seed = np.asarray(fl.get('split_seeded', np.zeros(n)), dtype=float) >= 0.5
    c1_present = np.asarray(fl.get('comp1_present', np.zeros(n)), dtype=float) >= 0.5
    debris = np.asarray(fl.get('split_comp1_debris', np.zeros(n)), dtype=float) >= 0.5
    area_relax = max(0.17, 0.68 * r.geo_second_area_frac)
    split_strict = (geo_b0 >= 2) & (area_f >= r.geo_second_area_frac) & sec50
    split_hint = ((geo_b0 >= 2) & c1_present & c1_seed & c1_hint &
                  (area_f >= area_relax) & sec50)
    return (split_strict | split_hint) & (~debris)


def rapid_polar_warm_gate(ti, fl, r):
    # True if canonical SSW thermal criterion was met recently: ≥25 K rise
    w7 = np.asarray(fl.get('warm7_max', []), dtype=float)
    if w7.size == 0 or ti < 0 or ti >= w7.size:
        return False
    lb = int(max(0, getattr(r, 'ssw_thermal_lookback_days', 14)))
    j0 = max(0, ti - lb)
    seg = w7[j0:ti + 1]
    return bool(np.isfinite(seg).any() and (np.nanmax(seg) >= r.ssw_warm25_K))


def polar_warming_for_ssw(ti, fl, r):
    # Thermal context for SSW-class morphology (split/displaced): canonical
    if rapid_polar_warm_gate(ti, fl, r):
        return True
    w7 = np.asarray(fl.get('warm7_max', []), dtype=float)
    anm = np.asarray(fl.get('warm_anom_max', []), dtype=float)
    if w7.size == 0 or ti < 0 or ti >= w7.size:
        return False
    lb = int(max(0, getattr(r, 'ssw_thermal_lookback_days', 14)))
    j0 = max(0, ti - lb)
    seg = w7[j0:ti + 1]
    wmx = float(np.nanmax(seg)) if np.isfinite(seg).any() else float('nan')
    # there is no sub-25 K minor threshold.  The 25 K criterion is the single
    # trigger; major vs minor is then set by the 7-day temperature rise, not
    # the wind.
    if not (np.isfinite(wmx) and wmx >= float(r.ssw_warm25_K)):
        return False
    return True


def vortex_quiet_strong(ti, fl, r):
    # Stable, strong westerly vortex; marginal T noise should not flag warming.
    pk = fl['peak_U'][ti]
    frac = float(getattr(r, 'ignore_marginal_warming_strong_frac', 0.82))
    if not (np.isfinite(pk) and pk >= frac * r.strong_peak_U):
        return False
    if fl['jet_intact'][ti] < 0.5:
        return False
    if wind_bad_day(ti, fl, r):
        return False
    return True


def segment_has_ssw_thermal(w7_seg, anom_seg, r):
    # only 25 K counts.  No sub-25 K minor-warming
    # branch exists in this hierarchy.
    if w7_seg.size and np.isfinite(w7_seg).any():
        mx = float(np.nanmax(w7_seg))
        if mx >= r.ssw_warm25_K:
            return True
    return False


def force_early_season_prefix(states, r):
    # First n_early_days are early season (winter spin-up) UNLESS a
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    e = min(int(getattr(r, 'n_early_season_force_days', 16)), n)
    seen_warm = False
    for i in range(e):
        cur = int(s[i])
        if cur == STATE_END:
            continue
        if cur in WARMING_STATES:
            seen_warm = True
            continue
        if seen_warm:
            # after a warming: do not paint early season
            continue
        s[i] = STATE_EARLY
    return s


def despeckle(states, min_run):
    # Merge short runs into neighbors.  If a run shorter than
    if min_run <= 1:
        return states
    s = states.copy()
    n = len(s)
    protected = WARMING_STATES   # don't absorb these even if short

    def runlen(start, dirn):
        # length of contiguous same-label run starting at `start`,
        # going in direction dirn (+1 = forward, -1 = backward).
        if start < 0 or start >= n:
            return 0
        v = s[start]; k = start; L = 0
        while 0 <= k < n and s[k] == v:
            L += 1; k += dirn
        return L

    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        run_len = j - i
        # Brief strong runs (>=2 days) are genuine brief reconvenes and must
        # not be absorbed into surrounding disturbance labels: a 2-day strong
        # period separating two disturbed periods is the recovery that
        is_short_strong = (int(s[i]) == STATE_STRONG and 2 <= run_len < min_run)
        if (run_len < min_run and int(s[i]) not in protected
                and not is_short_strong):
            left_ok  = (i > 0)
            right_ok = (j < n)
            if left_ok and right_ok and s[i - 1] == s[j]:
                # symmetric neighbors: absorb into them
                s[i:j] = s[i - 1]
            elif left_ok and right_ok:
                # asymmetric neighbors: absorb into the longer
                left_len  = runlen(i - 1, -1)
                right_len = runlen(j, +1)
                s[i:j] = s[i - 1] if left_len >= right_len else s[j]
            elif left_ok:
                s[i:j] = s[i - 1]
            elif right_ok:
                s[i:j] = s[j]
            # else: only run in the array, leave it
        i = j
    return s


WARMING_STATES = {STATE_WARM_NO_GEO,
                   STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
                   STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
                   STATE_SSW_SPLIT, STATE_SSW_SPLIT_MIN,
                   STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN}


# Any run of STATE_GEO_DISTURBED that starts within `window` days of
# the end of a warming run (SSW or WARM_NO_GEO) is absorbed into that
# warming event. This is the "geo keeps wobbling after a warming even
def merge_post_warming_geo(states, window=10):
    n = len(states)
    s = np.asarray(states).copy()
    # anchor stays pinned to the last *true* warming day so a long
    # geo-disturbance tail can't keep the warming "alive" forever.
    last_warm_end  = -1
    last_warm_code = -1
    for i in range(n):
        sc = int(s[i])
        if sc in WARMING_STATES:
            last_warm_end  = i
            last_warm_code = sc
            continue
        if (sc == STATE_GEO_DISTURBED and last_warm_end >= 0 and
            (i - last_warm_end) <= window):
            s[i] = last_warm_code
    return s


# Second geopotential lobe drops out briefly during a split SSW; if the
# gap is ≤ split_bridge_max_gap days and the vortex's equatorward edge
# (min lowest-lat across lobes) sits in 40–45°N throughout the gap,
def bridge_split_ssw_component_gaps(states, fl, r):
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    low_min = np.asarray(fl.get('geo_lowest_min',
                              np.full(n, np.nan)), dtype=float)
    hi30 = np.asarray(fl['hi_comps_30'], dtype=float)
    geo_hi = np.asarray(fl['geo_alt_hi'], dtype=float)

    lo_lat, hi_lat = r.split_bridge_lat_lo, r.split_bridge_lat_hi
    max_gap = int(r.split_bridge_max_gap)

    def early_low_only(ti):
        if ti >= r.n_early_days:
            return False
        if hi30[ti] >= 2:
            return False
        gh = geo_hi[ti] if geo_hi.ndim > 1 else geo_hi
        mx = float(np.nanmax(gh)) if np.size(gh) else float('nan')
        return np.isfinite(mx) and (mx < 30.0)

    # Second lobe: relax area vs canonical split test so transient loss
    # of comp-2 in the mask still chains across short gaps.
    second_big = build_geo_split_array(fl, r)
    anchors = np.where(second_big)[0]
    if anchors.size < 2:
        return s

    def gap_lat_ok(g):
        if early_low_only(g):
            return False
        lm = low_min[g]
        if np.isfinite(lm):
            return lo_lat <= lm <= hi_lat
        # tolerate missing lowest-lat for a day inside the gap
        if g > 0 and g + 1 < n:
            a, b = low_min[g - 1], low_min[g + 1]
            if np.isfinite(a) and np.isfinite(b):
                mid = 0.5 * (float(a) + float(b))
                return lo_lat <= mid <= hi_lat
        return True

    blocks = []
    idx = 0
    while idx < len(anchors):
        a_start = anchors[idx]
        a_end = a_start
        j = idx
        while j + 1 < len(anchors):
            prev = anchors[j]
            nxt = anchors[j + 1]
            if nxt - prev > max_gap:
                break
            gap_ok = True
            for g in range(prev + 1, nxt):
                if not gap_lat_ok(g):
                    gap_ok = False
                    break
            if not gap_ok:
                break
            a_end = nxt
            j += 1
        blocks.append((a_start, a_end))
        idx = j + 1

    anom = np.asarray(fl['warm_anom_max'], dtype=float)
    dTu = np.asarray(fl['warm_dT_max'], dtype=float)
    dTc = np.asarray(fl['dT_col'], dtype=float)
    promote = {
        STATE_STRONG, STATE_GEO_DISTURBED,
        STATE_WARM_NO_GEO, STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
        STATE_SSW_SPLIT_MIN,
    }
    for a0, a1 in blocks:
        if a1 <= a0:
            continue
        warm_any = False
        w7loc = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
        for ti in range(a0, a1 + 1):
            if np.isfinite(w7loc[ti]) and w7loc[ti] >= (0.35 * r.ssw_warm25_K):
                warm_any = True
                break
            if np.isfinite(anom[ti]) and anom[ti] >= 0.30 * r.warming_T_anom_K:
                warm_any = True
                break
            if np.isfinite(dTu[ti]) and dTu[ti] >= 0.5 * r.warming_dT_K_per_day:
                warm_any = True
                break
            if np.isfinite(dTc[ti]) and dTc[ti] >= 0.5 * r.warming_dT_K_per_day:
                warm_any = True
                break
        if not warm_any:
            continue
        for ti in range(a0, a1 + 1):
            c = int(s[ti])
            if c in (STATE_END, STATE_EARLY, STATE_RECOVERING):
                continue
            if early_low_only(ti):
                continue
            if c in promote:
                s[ti] = STATE_SSW_SPLIT
    return s


def merge_trailing_warm_into_ssw(states, fl, r):
    # After an SSW run, trailing WARM_NO_GEO that still satisfies the thermal
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    lag = int(r.event_precursor_days)
    if lag <= 0 or n == 0:
        return s
    i = 0
    while i < n:
        if int(s[i]) not in SSW_STATES:
            i += 1
            continue
        j = i
        while j < n and int(s[j]) in SSW_STATES:
            j += 1
        last_code = int(s[j - 1])
        for t in range(j, min(n, j + lag)):
            if int(s[t]) != STATE_WARM_NO_GEO:
                break
            if not polar_warming_for_ssw(t, fl, r):
                break
            s[t] = last_code
        i = j
    return s


# Geopotential split/elongation often appears a few days after the wind
# / temperature signal.  Upgrade WARM_NO_GEO when geo_bad is nearby in
# time so the SSW label isn't withheld until the centroid catches up.
def bridge_warming_near_geo(states, fl, r):
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    lag = int(r.warming_geo_lag_days)
    geo_split = build_geo_split_array(fl, r)
    base_ar = (fl['base_aspect'][:, 0] if fl['base_aspect'].shape[1]
               else np.full(n, np.nan))
    tilt = np.asarray(fl['geo_tilt'], dtype=float)
    upper_rev = np.asarray(fl.get('upper_rev', np.zeros(n)), dtype=float) >= 0.5
    tilt_ok = np.isfinite(tilt) & (
        (tilt <= -r.geo_disturb_min_tilt) |
        (upper_rev & (np.abs(tilt) >= 0.6 * r.geo_disturb_min_tilt))
    )
    geo_elong = (np.isfinite(base_ar) & (base_ar >= r.geo_aspect_bot)
                 & tilt_ok)
    bot_mov = np.zeros(n, dtype=bool)
    bot_low = np.zeros(n, dtype=bool)
    for ti in range(n):
        bot_mov[ti] = equatorward_comp0_bottom(ti, fl, r)
        bot_low[ti] = bottom_low_displaced(ti, fl, r)
    geo_for_bridge = geo_elong | bot_mov | bot_low

    for ti in range(n):
        if int(s[ti]) != STATE_WARM_NO_GEO:
            continue
        if not polar_warming_for_ssw(ti, fl, r):
            continue
        j0 = max(0, ti - lag)
        j1 = min(n, ti + lag + 1)
        geo_active = bool(geo_for_bridge[ti] or bot_mov[ti] or bot_low[ti] or
                           geo_for_bridge[j0:j1].any())
        if not (wind_context_ssw(ti, fl, r) or geo_active):
            continue
        if geo_for_bridge[j0:j1].any() or geo_split[j0:j1].any():
            # Never bridge WARM_NO_GEO into split/partial; only displaced.
            s[ti] = STATE_SSW_DISPLACED
    return s


def align_warming_block_to_geo_start(states):
    # Before each SSW run, make a single precursor block and start it at
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    precursor = {STATE_WARM_NO_GEO, STATE_GEO_DISTURBED}
    i = 0
    while i < n:
        if int(s[i]) not in SSW_STATES:
            i += 1
            continue
        # contiguous SSW run [i, j)
        j = i
        while j < n and int(s[j]) in SSW_STATES:
            j += 1
        # immediate precursor run [k, i)
        k = i - 1
        while k >= 0 and int(s[k]) in precursor:
            k -= 1
        p0 = k + 1
        if p0 < i:
            # warming block starts where geopotential disturbance begins;
            # earlier wind-only days remain as-is.
            gidx = [t for t in range(p0, i) if int(s[t]) == STATE_GEO_DISTURBED]
            if gidx:
                g0 = gidx[0]
                s[g0:i] = STATE_WARM_NO_GEO
        i = j
    return s


def promote_geo_with_local_warming(states, fl, r, lag_days=3):
    # If a GEO_DISTURBED day sits near a clear local warming signal,
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    warm25 = np.isfinite(w7) & (w7 >= r.ssw_warm25_K)
    lag = int(max(0, lag_days))
    for i in range(n):
        if int(s[i]) != STATE_GEO_DISTURBED:
            continue
        j0 = max(0, i - lag); j1 = min(n, i + lag + 1)
        if np.any(int(s[k]) == STATE_RECOVERING for k in range(j0, j1)):
            continue
        if np.any(warm25[j0:j1]):
            # Geo + thermal → displaced morphology, not open-ended WARM_NO_GEO.
            s[i] = STATE_SSW_DISPLACED
    return s


def end_warming_on_wind_restabilize(states, fl, r, stable_days=2):
    # Terminate warming-related blocks once wind rings re-stabilize.
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    peak = np.asarray(fl.get('peak_U', np.full(n, np.nan)), dtype=float)
    jet  = np.asarray(fl.get('jet_intact', np.zeros(n)), dtype=float)
    warmish = {STATE_WARM_NO_GEO, STATE_SSW_DISPLACED, STATE_SSW_SPLIT,
               STATE_SSW_DISPLACED_MIN, STATE_SSW_SPLIT_MIN,
               STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN}
    i = 0
    while i < n:
        if int(s[i]) not in warmish:
            i += 1
            continue
        j = i
        while j < n and int(s[j]) in warmish:
            j += 1
        k0 = -1
        for k in range(i, j):
            stable = ((jet[k] >= 0.5) and
                      np.isfinite(peak[k]) and (peak[k] > 0))
            if not stable:
                continue
            if k + stable_days - 1 < j:
                ok = True
                for kk in range(k, k + stable_days):
                    st = ((jet[kk] >= 0.5) and
                          np.isfinite(peak[kk]) and (peak[kk] > 0))
                    if not st:
                        ok = False
                        break
                if ok:
                    k0 = k
                    break
        if k0 >= 0:
            # only end warming if thermal signal also subsides
            w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)),
                            dtype=float)
            still_warm = np.isfinite(w7[k0:j]) & (w7[k0:j] >= (0.55 * r.ssw_warm25_K))
            if np.any(still_warm):
                i = j
                continue
            s[k0:j] = STATE_GEO_DISTURBED
        i = j
    return s


# Wind / geo / early warming days that immediately precede an SSW run
# are part of the same sequential event (wind → T rise → geo).
def absorb_event_precursors(states, r):
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    lag = int(r.event_precursor_days)
    if lag <= 0 or n == 0:
        return s
    bridge = {STATE_GEO_DISTURBED, STATE_WARM_NO_GEO}
    i = 0
    while i < n:
        c = int(s[i])
        if c not in SSW_STATES:
            i += 1
            continue
        j = i + 1
        while j < n and int(s[j]) == c:
            j += 1
        k = i - 1
        while k >= 0 and (i - k) <= lag and int(s[k]) in bridge:
            k -= 1
        for t in range(k + 1, i):
            if int(s[t]) in bridge:
                s[t] = c
        i = j
    return s


# End-of-season pass. Physical picture: once the vortex gives up, the
# cold air over the pole is no longer confined, so the polar cap warms
# slowly (positive-but-modest dT/dt), the T anomaly stays well above
def apply_end_of_season(states, fl, r):
    n = len(states)
    if n == 0:
        return states
    s    = np.asarray(states).copy()
    peak  = np.asarray(fl['peak_U'],  dtype=float)
    eastu = np.asarray(fl.get('east_U', np.full(n, np.nan)), dtype=float)
    dT    = np.asarray(fl['dT_col'],        dtype=float)      # signed
    anom  = np.asarray(fl['warm_anom_max'], dtype=float)

    # After the last *major* SSW (codes 6 / 8), final warming + weak vortex
    # should transition to end more readily (mid‑Feb → spring in your seasons).
    last_major_end = -1
    ii = 0
    while ii < n:
        if int(s[ii]) not in (STATE_SSW_DISPLACED, STATE_SSW_SPLIT):
            ii += 1
            continue
        jj = ii
        while jj < n and int(s[jj]) in (STATE_SSW_DISPLACED, STATE_SSW_SPLIT):
            jj += 1
        last_major_end = jj - 1
        ii = jj
    post_major = last_major_end >= 0
    wind_slack = 1.40 if post_major else 1.0
    min_run_eos = max(5, r.min_end_run_days - 4) if post_major \
        else r.min_end_run_days

    # last index of any warming-related state; we only relabel after
    # that point so a real final warming can still be seen.
    warm_mask   = np.array([int(c) in WARMING_STATES for c in s],
                           dtype=bool)
    last_warm   = int(np.where(warm_mask)[0][-1]) if warm_mask.any() else -1
    had_warming = last_warm >= 0

    w_mag = np.where(np.isfinite(peak),  np.abs(peak),  0.0)
    e_mag = np.where(np.isfinite(eastu), np.abs(eastu), 0.0)
    wind_mag = np.maximum(w_mag, e_mag)

    dT_lo = -r.end_dT_abs             # not cooling hard
    # end-of-season can keep warming steadily; allow modest positive dT/dt.
    dT_hi = max(0.25, 4.0 * r.end_dT_abs)

    settle = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(anom[i]) and np.isfinite(dT[i])):
            continue
        if wind_mag[i] > r.end_peak_U * wind_slack:
            continue
        if anom[i] < r.end_anom_floor_K:
            continue
        if not (dT_lo <= dT[i] <= dT_hi):
            continue
        settle[i] = True

    # Gradual-breakup mode: wind steadily weakens, polar-cap anomaly
    # creeps upward, and dT/dt has no large spikes.
    dw = np.gradient(wind_mag)
    da = np.gradient(anom)
    gradual = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(wind_mag[i]) and np.isfinite(anom[i]) and np.isfinite(dT[i])):
            continue
        if wind_mag[i] > (1.25 * r.end_peak_U * wind_slack):
            continue
        if anom[i] < (0.85 * r.end_anom_floor_K):
            continue
        if dT[i] > max(0.30, 0.90 * r.warming_dT_K_per_day):
            continue
        if not (dT_lo <= dT[i] <= dT_hi):
            continue
        if np.isfinite(dw[i]) and dw[i] > 0.2:
            continue
        if np.isfinite(da[i]) and da[i] < -0.05:
            continue
        gradual[i] = True

    if had_warming:
        scan_start = last_warm + 1 + int(r.end_post_warming_days)
    else:
        scan_start = n // 3
    if post_major:
        scan_start = min(scan_start, last_major_end + 2)
    if scan_start >= n:
        scan_start = max(0, n - r.n_end_window_days)

    i = scan_start
    run_start = -1
    while i < n:
        j = i
        while j < n and settle[j]:
            j += 1
        if (j - i) >= min_run_eos:
            run_start = i
            break
        i = max(j + 1, i + 1)

    if run_start < 0:
        # fallback: sustained gradual-breakup run
        i = scan_start
        while i < n:
            j = i
            while j < n and gradual[j]:
                j += 1
            if (j - i) >= max(5, min_run_eos - 5):
                run_start = i
                break
            i = max(j + 1, i + 1)

    if run_start < 0:
        # very-late-season safety net: warm cap + weak winds, regardless of
        # chaotic geopotential, with no rapid warming spikes.
        late_start = max(scan_start, int(0.55 * n))
        weakwarm = np.zeros(n, dtype=bool)
        for i in range(late_start, n):
            if not (np.isfinite(wind_mag[i]) and np.isfinite(anom[i]) and np.isfinite(dT[i])):
                continue
            if wind_mag[i] > (1.45 * r.end_peak_U * wind_slack):
                continue
            if anom[i] < (0.70 * r.end_anom_floor_K):
                continue
            if dT[i] > max(0.40, 1.10 * r.warming_dT_K_per_day):
                continue
            weakwarm[i] = True
        i = late_start
        while i < n:
            j = i
            while j < n and weakwarm[j]:
                j += 1
            if (j - i) >= 4:
                run_start = i
                break
            i = max(j + 1, i + 1)

    if run_start >= 0:
        s[run_start:] = STATE_END
    return s


def collapse_non_ssw_states(states):
    # Reduce non-SSW non-END clutter:
    s = np.asarray(states, dtype=np.int16).copy()
    return s


SSW_STATES = {STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
               STATE_SSW_SPLIT,     STATE_SSW_SPLIT_MIN,
               STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
               STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN}


def merge_warm_no_geo_into_adjacent_ssw(states, r=None, max_precursor_days=5):
    # Relabel short WARM_NO_GEO precursors immediately before SSW.
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    max_pre = int(max_precursor_days)
    if r is not None:
        max_pre = min(max_pre, int(getattr(r, 'warming_geo_lag_days', 8)))
    i = 0
    while i < n:
        if int(s[i]) not in SSW_STATES:
            i += 1
            continue
        code = int(s[i])
        j = i
        while j < n and int(s[j]) == code:
            j += 1
        k = i - 1
        while k >= 0 and int(s[k]) == STATE_WARM_NO_GEO:
            k -= 1
        p0 = k + 1
        if p0 < i and (i - p0) <= max_pre:
            for t in range(p0, i):
                s[t] = code
        i = j
    return s


def trim_warming_tail_to_recovery(states, fl, r):
    # Trim the TAIL of a warming run to weak recovery once the jet has
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)), dtype=float)
    bc = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                    dtype=float)
    thr = float(getattr(r, 'ssw_warm25_K', 25.0))
    trig = sustained_warming_trigger_mask(w7, thr)
    phys = fl.get('event_physics') or train_event_physics(fl, r)
    nr_pos = nr[np.isfinite(nr) & (nr > 0)]
    # weak recovery: the ring need only rebuild PARTIALLY (the jet is coming
    # back but is not yet a full strong ring), so use a fraction well below
    # the half-column "reformed" bar.
    recover_ring = (0.35 * float(np.percentile(nr_pos, 95))
                    if nr_pos.size else 5.0)
    min_rec = max(3, int(r.min_run_length))
    reformed = (np.isfinite(nr) & (nr >= recover_ring) &
                np.isfinite(bc) & (bc >= 70.0))
    i = 0
    while i < n:
        if int(s[i]) not in WARMING_STATES:
            i += 1
            continue
        j = i
        while j < n and int(s[j]) in WARMING_STATES:
            j += 1
        seg = w7[i:j]
        pk = (i + int(np.nanargmax(np.where(np.isfinite(seg), seg, -np.inf)))
              if np.isfinite(seg).any() else i)
        # the run must actually recover by its end: a sustained PARTIAL ring
        # rebuild past the peak. Residual warmth (still-elevated warm7) is
        # expected during a jet recovery and is NOT excluded here.
        tail_reformed = int(np.sum(reformed[pk + 1:j]))
        if tail_reformed >= min_rec:
            onset = -1
            for k in range(pk + 1, j):
                if weak_recovery_signature(k, fl, phys, ref_idx=None):
                    onset = k
                    break
            if onset >= 0:
                # Extend recovery from the onset, but stop at a genuine
                # re-disturbance: once the warming has subsided below threshold
                # during the recovery, a fresh day that climbs back to >= thr
                end = j
                k = onset + 1
                while k < j:
                    subsided = bool(np.any(w7[onset:k] < thr))
                    if subsided and w7[k] >= thr and not reformed[k]:
                        end = k
                        break
                    k += 1
                s[onset:end] = STATE_RECOVERING
        i = j
    return s


def split_warming_runs_at_internal_recovery(states, fl, r):
    # Break long warming runs at interior recovery pauses -- but only where
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    phys = fl.get('event_physics') or train_event_physics(fl, r)
    min_rec = int(phys.get('min_rec_run', max(int(r.min_run_length), 3)))
    min_span = max(8, 2 * int(r.min_run_length))
    w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    thr = float(getattr(r, 'ssw_warm25_K', 25.0))
    warm_sustained = sustained_warming_trigger_mask(w7, thr)
    # half-of-full-coverage westerly ring threshold
    nr_arr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)),
                        dtype=float)
    bc_arr = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                        dtype=float)
    nr_pos = nr_arr[np.isfinite(nr_arr) & (nr_arr > 0)]
    half_ring = (0.5 * float(np.percentile(nr_pos, 95))
                 if nr_pos.size else 8.0)

    def ring_back(k):
        # genuine reconvene: ring across >= half column and geo centered
        return (k < nr_arr.size and np.isfinite(nr_arr[k]) and
                nr_arr[k] >= half_ring and
                k < bc_arr.size and np.isfinite(bc_arr[k]) and
                bc_arr[k] >= 70.0)

    i = 0
    while i < n:
        if int(s[i]) != STATE_WARM_NO_GEO:
            i += 1
            continue
        j = i
        while j < n and int(s[j]) == STATE_WARM_NO_GEO:
            j += 1
        if (j - i) >= min_span:
            k = i + min_rec
            while k < j - min_rec:
                # split only where warming has stopped and the vortex has
                # genuinely reconvened (sustained westerly ring), not a spike
                if (not warm_sustained[k] and ring_back(k) and
                        recovery_signature(k, fl, phys, ref_idx=None)):
                    rec_lo = k
                    rec_run = 0
                    while (k < j and not warm_sustained[k] and ring_back(k)
                           and recovery_signature(k, fl, phys,
                                                   ref_idx=None)):
                        rec_run += 1
                        k += 1
                    if rec_run >= min_rec:
                        for t in range(rec_lo, rec_lo + rec_run):
                            s[t] = STATE_RECOVERING
                    continue
                k += 1
        i = j
    return s


def enforce_warming_separation(states, r):
    # Never allow two separate warming event runs without recovery buffer.
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    min_gap = max(3, int(r.min_run_length) + 1)

    def is_warming(k):
        return int(s[k]) in WARMING_STATES

    runs = []
    i = 0
    while i < n:
        if not is_warming(i):
            i += 1
            continue
        j = i
        while j < n and is_warming(j):
            j += 1
        runs.append((i, j - 1))
        i = j

    for idx in range(len(runs) - 1):
        lo1, hi1 = runs[idx]
        lo2, hi2 = runs[idx + 1]
        gap = lo2 - hi1 - 1
        if gap >= min_gap:
            continue
        need = min_gap - gap
        for k in range(hi1 + 1, lo2):
            s[k] = STATE_RECOVERING
        for k in range(lo2, min(n, lo2 + need)):
            if is_warming(k):
                s[k] = STATE_RECOVERING
    return s


def finalize_classified_states(states, fl, r):
    # Post-process event-based states: bridge geo morphology, split long
    s = np.asarray(states, dtype=np.int16).copy()
    s = split_warming_runs_at_internal_recovery(s, fl, r)
    s = merge_warm_no_geo_into_adjacent_ssw(s, r)
    s = bridge_warming_near_geo(s, fl, r)
    s = promote_geo_with_local_warming(s, fl, r, lag_days=3)
    s = enforce_warming_separation(s, r)
    s = refine_major_minor(s, fl, r)
    s = force_early_season_prefix(s, r)
    s = despeckle(s, r.min_run_length)
    # Final guard (runs last): a weak recovery can only follow a warming.
    # Any recovering day with no unresolved warming since the last EOS /
    # strong vortex is invalid; later passes (e.g. recovery fill) can
    s = enforce_recovery_follows_warming(s, fl, r)
    # hierarchy safety net: a sustained >=25 K warming is a warming, full
    # stop. If any such day was left as a geo disturbance or folded into EOS
    # (e.g. a final warming the event machinery skipped because the vortex
    s = rescue_unlabeled_warmings(s, fl, r)
    # Trim warming-run tails to weak recovery once the jet has come back. Runs
    # AFTER the warming rescue so a genuine post-peak recovery is not undone by
    # the >=25 K residual-warmth hierarchy (the jet recovers while the cap is
    s = trim_warming_tail_to_recovery(s, fl, r)
    # Re-evaluate the morphology family on the trimmed warming runs: a partial
    # / split / displaced signal that fell in the (now-recovery) tail must not
    # define the event. A warming whose only partial pinch was post-peak
    s = refine_major_minor(s, fl, r)
    # terminal EOS: once the season ends and the vortex never again sustains a
    # strong centered ring, it stays ended; w7 spikes in the broken-down
    # vortex are not warmings (0910 mid-March) and must not fragment EOS.
    eos_days = np.where(s == STATE_END)[0]
    if eos_days.size:
        eos0 = int(eos_days[0])
        nt = len(s)
        nr_t = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                          dtype=float)
        bc_t = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                          dtype=float)
        nr_tp = nr_t[np.isfinite(nr_t) & (nr_t > 0)]
        half_t = (0.5 * float(np.percentile(nr_tp, 95))
                  if nr_tp.size else 8.0)
        strong_t = (np.isfinite(nr_t) & (nr_t >= half_t) &
                    np.isfinite(bc_t) & (bc_t >= 70.0))
        if max_consecutive_true(strong_t, eos0, nt - 1) < 5:
            s[eos0:] = STATE_END
    return s


def rescue_unlabeled_warmings(states, fl, r):
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    thr = float(getattr(r, 'ssw_warm25_K', 25.0))
    trig = sustained_warming_trigger_mask(w7, thr)
    # A recovering period where the vortex has genuinely reconvened (a
    # sustained westerly ring across >= half the column with the geopotential
    # recentered) is a reforming vortex, not a warming: the >=25 K net must
    nr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)), dtype=float)
    bc = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                    dtype=float)
    nr_pos = nr[np.isfinite(nr) & (nr > 0)]
    half_ring = (0.5 * float(np.percentile(nr_pos, 95))
                 if nr_pos.size else 8.0)
    min_rec = max(int(r.min_run_length), 3)
    reconv_day = (np.isfinite(nr) & (nr >= half_ring) &
                  np.isfinite(bc) & (bc >= 70.0))
    protected = np.zeros(n, dtype=bool)
    k = 0
    while k < n:
        if int(s[k]) == STATE_RECOVERING:
            a = k
            while k < n and int(s[k]) == STATE_RECOVERING:
                k += 1
            if max_consecutive_true(reconv_day, a, k - 1) >= min_rec:
                protected[a:k] = True
        else:
            k += 1
    # find maximal runs of sustained-warming days currently labeled geo,
    # recovery, or EOS (skipping protected reconvened recovery)
    i = 0
    while i < n:
        if (trig[i] and not protected[i] and
                int(s[i]) in (STATE_GEO_DISTURBED, STATE_RECOVERING,
                              STATE_END)):
            j = i
            while (j < n and trig[j] and not protected[j] and
                   int(s[j]) in (STATE_GEO_DISTURBED, STATE_RECOVERING,
                                 STATE_END)):
                j += 1
            # relabel [i, j) as a displaced warming (a final warming is a
            # displacement of the breaking-down vortex); major/minor and the
            # precise family are then set by _refine_major_minor on a re-pass.
            s[i:j] = STATE_SSW_DISPLACED
            i = j
        else:
            i += 1
    # re-run morphology so the rescued run gets the correct family + major.
    s = refine_major_minor(s, fl, r)
    return s


def enforce_recovery_follows_warming(states, fl, r):
    s = np.asarray(states, dtype=np.int16).copy()
    n = len(s)
    nr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)), dtype=float)
    ei = np.asarray(fl.get('east_intact_level', np.zeros(n)), dtype=float)
    nrsm = uniform_filter1d(np.where(np.isfinite(nr), nr, 0.0),
                            size=5, mode='nearest')
    nrp = nrsm[nrsm > 0]
    # "substantial ring" = 40% of the season's full coverage (more lenient
    # than half-col: a 13-14 level ring just under half is still a vortex).
    ring_thr = 0.4 * float(np.percentile(nrp, 95)) if nrp.size else 6.0
    seen_warm = False
    for k in range(n):
        sc = int(s[k])
        if sc == STATE_END or sc == STATE_STRONG:
            seen_warm = False
        elif sc in WARMING_STATES:
            seen_warm = True
        elif (not seen_warm) and sc == STATE_RECOVERING:
            ring_ok = (nrsm[k] >= ring_thr and ei[k] < 1.0 and
                       (nr[k] if np.isfinite(nr[k]) else 0.0) > 0)
            s[k] = STATE_STRONG if ring_ok else STATE_GEO_DISTURBED
    return despeckle(s, r.min_run_length)


# Each SSW run must have thermal evidence: warm7_max >= ssw_warm25_K
# anywhere in the event window plus event_precursor_days lookback.
def validate_ssw_warming(states, fl, r, min_warm_days=None):
    s    = np.asarray(states).copy()
    n    = len(s)
    w7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    lb = int(getattr(r, 'ssw_thermal_lookback_days', 14))
    pre = int(getattr(r, 'event_precursor_days', 0))
    geo_split = build_geo_split_array(fl, r)
    base_ar   = (fl['base_aspect'][:, 0] if fl['base_aspect'].shape[1]
                 else np.full(n, np.nan))
    tilt      = np.asarray(fl['geo_tilt'], dtype=float)
    upper_rev = np.asarray(fl.get('upper_rev', np.zeros(n)), dtype=float) >= 0.5
    tilt_ok   = np.isfinite(tilt) & (
        (tilt <= -r.geo_disturb_min_tilt) |
        (upper_rev & (np.abs(tilt) >= 0.6 * r.geo_disturb_min_tilt))
    )
    bot = np.zeros(n, dtype=bool)
    for ti in range(n):
        bot[ti] = equatorward_comp0_bottom(ti, fl, r)
    # stretch: a large lower-latitude aspect ratio is a geopotential
    # disturbance in its own right, independent of tilt. A stretch may carry
    # a tilt, but need not. We use the largest-lobe aspect (big_aspect),
    big_asp = np.asarray(fl.get('big_aspect', np.full(n, np.nan)),
                         dtype=float)
    stretch = ((np.isfinite(big_asp) & (big_asp >= r.geo_aspect_bot)) |
               (np.isfinite(base_ar) & (base_ar >= r.geo_aspect_bot)))
    # tilt: an independent disturbance signal (negative tilt, or an upper
    # reversal with substantial tilt magnitude).
    tilt_dist = tilt_ok
    geo_bad   = geo_split | stretch | tilt_dist | bot
    i = 0
    while i < n:
        if int(s[i]) not in SSW_STATES:
            i += 1; continue
        j = i
        while j < n and int(s[j]) == int(s[i]):
            j += 1
        i0 = max(0, i - lb - pre)
        win = w7[i0:j]
        an_win = np.asarray(fl['warm_anom_max'], dtype=float)[i0:j]
        if not segment_has_ssw_thermal(win, an_win, r):
            for k in range(i, j):
                s[k] = (STATE_GEO_DISTURBED if bool(geo_bad[k])
                        else STATE_STRONG)
        i = j
    return s


# Assign major vs minor SSW per run.  major iff there
# is an intact easterly ring on any level during the event window;
# minor otherwise.  The 25 K warm7 trigger is the single warming
def refine_major_minor(states, fl, r):
    n = len(states)
    s = np.asarray(states).copy()
    warm7 = np.asarray(fl.get('warm7_max', np.full(n, np.nan)), dtype=float)
    warm7_below = np.asarray(fl.get('warm7_max_below10', np.full(n, np.nan)),
                             dtype=float)
    warm7_above = np.asarray(fl.get('warm7_max_above10', np.full(n, np.nan)),
                             dtype=float)
    # Major/minor is decided by the 7-day temperature rise (below); east_intact
    # is kept only as a diagnostic and no longer gates severity.
    east_intact = np.asarray(fl.get('east_intact_level', np.zeros(n)),
                             dtype=float) >= 1.0

    def morph_family(code):
        # Partial-split is its own family: only when a genuine full split
        # never forms (the second 3-D B0 never fully appears).
        c = int(code)
        if c == STATE_WARM_NO_GEO:
            return 'nogeo'
        if c in (STATE_SSW_SPLIT, STATE_SSW_SPLIT_MIN):
            return 'split'
        if c in (STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN):
            return 'partial'
        if c in (STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN):
            return 'mixed'
        return 'displaced'

    major_map = {
        'split': STATE_SSW_SPLIT,
        'partial': STATE_SSW_PARTIAL_SPLIT,
        'mixed': STATE_SSW_MIXED_MAJ,
        'displaced': STATE_SSW_DISPLACED,
        'nogeo': STATE_WARM_NO_GEO,
    }
    minor_map = {
        'split': STATE_SSW_SPLIT_MIN,
        'partial': STATE_SSW_PARTIAL_SPLIT_MIN,
        'mixed': STATE_SSW_MIXED_MIN,
        'displaced': STATE_SSW_DISPLACED_MIN,
        'nogeo': STATE_WARM_NO_GEO,
    }

    ssw_codes = set([STATE_WARM_NO_GEO])
    for maj, mn in ((STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN),
                    (STATE_SSW_SPLIT, STATE_SSW_SPLIT_MIN),
                    (STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN),
                    (STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN)):
        ssw_codes.add(maj)
        ssw_codes.add(mn)

    comp1_arr = np.asarray(fl.get('comp1_present', np.zeros(n)), dtype=float)
    nr_arr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)),
                        dtype=float)
    bc_arr = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                        dtype=float)
    # genuine full split (a real second 3-D B0 component, or the lobe pinched
    # into >=2 pieces across most levels) vs a partial pinch only.
    full_split_arr = np.asarray(fl.get('b0_full_split_prog',
                                       fl.get('b0_full_split', np.zeros(n))),
                                dtype=float)
    top_pinch_arr = np.asarray(fl.get('b0_top_pinch_split', np.zeros(n)),
                               dtype=float)
    partial_arr = np.asarray(fl.get('b0_partial_split', np.zeros(n)),
                             dtype=float)
    frac_bot_arr = np.nan_to_num(np.asarray(
        fl.get('b0_frac_bot_partial', np.zeros(n)), dtype=float))
    frac_top_arr = np.nan_to_num(np.asarray(
        fl.get('b0_frac_top_partial', np.zeros(n)), dtype=float))
    two_genuine_arr = np.asarray(
        fl.get('two_lobes_genuine', np.zeros(n)), dtype=float)
    tilt_arr = np.abs(np.asarray(fl.get('geo_tilt', np.full(n, np.nan)),
                                 dtype=float))
    asp_arr = np.asarray(fl.get('big_aspect', np.full(n, np.nan)),
                         dtype=float)
    tilt_thr = float(getattr(r, 'geo_disturb_min_tilt', 0.06) or 0.06)
    asp_thr = float(getattr(r, 'geo_aspect_bot', 1.8))
    bl_arr = np.asarray(fl.get('geo_c0_bottom_low', np.full(n, np.nan)),
                        dtype=float)
    bc_all_bl = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                            dtype=float)
    bl_strong = bl_arr[np.isfinite(bl_arr) &
                        np.isfinite(nr_arr) & (nr_arr > 15) &
                        np.isfinite(bc_all_bl) & (bc_all_bl >= 75.0)]
    if bl_strong.size >= 5:
        # "normal low" for the undisturbed vortex; a dip clearly below its
        # lower quartile means the lobe reaches anomalously far equatorward
        # (a displacement) even with a centered centroid.
        bl_dip_thr = float(np.percentile(bl_strong, 25)) - 2.0
    else:
        bl_dip_thr = 58.0

    i = 0
    while i < n:
        code = int(s[i])
        if code not in ssw_codes:
            i += 1
            continue
        j = i
        while j < n and int(s[j]) in ssw_codes:
            j += 1

        # Major/minor by the 7-day temperature rise (replaces the easterly-
        # ring test): MAJOR if any day in the run reaches the at/below-10 hPa
        # threshold in that band, or the (higher) above-10 hPa threshold in
        below_seg = warm7_below[i:j]
        above_seg = warm7_above[i:j]
        major_below = (np.isfinite(below_seg) &
                       (np.round(below_seg, 2) >= MAJOR_DK_AT_OR_BELOW_10HPA))
        major_above = (np.isfinite(above_seg) &
                       (np.round(above_seg, 2) >= MAJOR_DK_ABOVE_10HPA))
        run_is_major = bool(major_below.any() or major_above.any())

        # morphology family for the whole run
        # A full-split day = the genuine second full 3-D B0 column
        # (b0_full_split). A partial-split day = a partial pinch or a top
        full_day = (full_split_arr[i:j] >= 0.5)
        partial_day = (((partial_arr[i:j] >= 0.5) |
                        (top_pinch_arr[i:j] >= 0.5)) & ~full_day)
        # displaced phase = vortex intact as one lobe and displaced off-pole:
        # no second component, ring present, and either the centroid is
        # clearly off-pole (bc < 70) or the lobe extends anomalously far
        no_second = (comp1_arr[i:j] < 0.5) & ~full_day & ~partial_day
        nr_seg = nr_arr[i:j]
        bc_seg = bc_arr[i:j]
        bl_seg = bl_arr[i:j]
        eq_dip_seg = np.isfinite(bl_seg) & (bl_seg <= bl_dip_thr)
        intact_displ_days = (no_second &
                             np.isfinite(nr_seg) & (nr_seg > 0) &
                             ((np.isfinite(bc_seg) & (bc_seg < 70.0)) |
                              eq_dip_seg))

        def max_run(mask):
            mx = 0; cur = 0
            for v in mask:
                cur = cur + 1 if v else 0
                mx = max(mx, cur)
            return mx
        full_run = max_run(full_day)
        partial_run = max_run(partial_day)
        disp_run = max_run(intact_displ_days)
        # Clean partial-split signature: the pinch stays in ONE band across a
        # contiguous run (bottom fraction sustained, as in 0506-Dec / 1112-Mar,
        # or top fraction sustained), OR genuine two-lobe geometry supports it
        fb_seg = frac_bot_arr[i:j]
        ft_seg = frac_top_arr[i:j]
        tg_seg = two_genuine_arr[i:j]
        bot_band_run = max_run(partial_day & (fb_seg >= PARTIAL_BAND_FRAC))
        top_band_run = max_run(partial_day & (ft_seg >= PARTIAL_BAND_FRAC))
        clean_partial_signal = (bot_band_run >= 3 or top_band_run >= 3 or
                                int(tg_seg.sum()) >= 1)
        has_full = full_run >= 3
        has_partial = (partial_run >= 3) and clean_partial_signal
        has_disp = disp_run >= 3

        # A day with no second component of any kind, a ring present, the
        # geopotential centroid still centered on the pole (bc >= 70), and no
        # tilt or stretch signal is an undisturbed-cap day: the vortex is
        tilt_seg = tilt_arr[i:j]
        asp_seg = asp_arr[i:j]
        no_tilt = ~(np.isfinite(tilt_seg) & (tilt_seg >= tilt_thr))
        no_stretch = ~(np.isfinite(asp_seg) & (asp_seg >= asp_thr))
        no_eq_dip = ~eq_dip_seg
        intact_centered_days = (no_second &
                                np.isfinite(nr_seg) & (nr_seg > 0) &
                                np.isfinite(bc_seg) & (bc_seg >= 70.0) &
                                no_tilt & no_stretch & no_eq_dip)
        # Sustained disturbance signals within the run (tilt, stretch, ring
        # dropout, or an anomalous equatorward dip of the lobe).
        tilt_day = np.isfinite(tilt_seg) & (tilt_seg >= tilt_thr)
        stretch_day = np.isfinite(asp_seg) & (asp_seg >= asp_thr)
        ring_drop_day = ~(np.isfinite(nr_seg) & (nr_seg > 0))
        disturb_day = tilt_day | stretch_day | ring_drop_day | eq_dip_seg
        run_len = max(1, (j - i))
        disturb_frac = float(disturb_day.sum()) / run_len
        max_tilt = (float(np.nanmax(tilt_seg))
                    if np.isfinite(tilt_seg).any() else 0.0)
        no_disturb = (disturb_frac <= 0.15 and max_tilt < tilt_thr)
        centered_majority = (intact_centered_days.sum() >= 0.6 * run_len)
        has_centered = (max_run(intact_centered_days) >= 3 and
                        centered_majority and no_disturb)

        # Mixed: a genuine split phase and a genuine intact-displaced phase.
        # But a displaced run bracketed by full-split days on BOTH sides is not
        # a distinct displacement period -- it is a messy-B0 lull within one
        def longest_true_span(mask):
            best_len = best_lo = best_hi = 0
            cur = lo = 0
            for idx, v in enumerate(mask):
                if v:
                    if cur == 0:
                        lo = idx
                    cur += 1
                    if cur > best_len:
                        best_len, best_lo, best_hi = cur, lo, idx
                else:
                    cur = 0
            return best_len, best_lo, best_hi
        disp_len, disp_lo, disp_hi = longest_true_span(intact_displ_days)
        disp_sandwiched = (disp_len > 0 and
                           bool(full_day[:disp_lo].any()) and
                           bool(full_day[disp_hi + 1:].any()))
        if has_full and has_disp and not disp_sandwiched:
            run_fam = 'mixed'
        elif has_full:
            run_fam = 'split'
        elif has_partial:
            # only ever a partial pinch, never a full split -> partial split.
            run_fam = 'partial'
        elif has_disp:
            run_fam = 'displaced'
        elif has_centered and not (has_full or has_partial or has_disp):
            # no split, no partial, no displacement; the cap is intact and
            # centered. This is a warming with no geopotential disturbance.
            run_fam = 'nogeo'
        else:
            # nothing sustained: pick by which signal has the most days. A
            # centered intact cap only counts toward no-geo if there is no
            # sustained disturbance in the run.
            counts = {'split': int(full_day.sum()),
                      'partial': (int(partial_day.sum())
                                  if clean_partial_signal else 0),
                      'displaced': int(intact_displ_days.sum())}
            if no_disturb:
                counts['nogeo'] = int(intact_centered_days.sum())
            run_fam = max(counts, key=counts.get)
            if counts[run_fam] == 0:
                run_fam = 'displaced'

        out_code = major_map[run_fam] if run_is_major else minor_map[run_fam]
        for k in range(i, j):
            s[k] = out_code
        i = j
    return s


# Collapse adjacent major/minor SSW runs into a single dominant label.
# The rule-based classifier only emits the major codes today, but in
# cluster mapping users often assign neighbouring clusters to (major,
def collapse_major_minor(states):
    # Do not merge adjacent major/minor days: _refine_major_minor assigns
    return np.asarray(states, dtype=np.int16).copy()


DISTURBANCE_SET = {
    STATE_GEO_DISTURBED, STATE_WARM_NO_GEO,
    STATE_SSW_DISPLACED, STATE_SSW_SPLIT,
    STATE_SSW_DISPLACED_MIN, STATE_SSW_SPLIT_MIN,
    STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
}


def vortex_reconvened(k, fl, half_ring, bc_pole=70.0):
    # Genuine recovery signal: the vortex has reconvened at the pole --
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    bc = np.asarray(fl.get('geo_c0_bottom_cent', []), dtype=float)
    if k < 0 or k >= nr.size:
        return False
    ring_ok = np.isfinite(nr[k]) and nr[k] >= half_ring
    cent_ok = (k < bc.size and np.isfinite(bc[k]) and bc[k] >= bc_pole)
    return bool(ring_ok and cent_ok)


def half_ring_threshold(fl):
    # Half the vortex's full westerly-ring coverage (smoothed peak).
    nr = np.asarray(fl.get('n_ring_levels', []), dtype=float)
    if nr.size == 0:
        return 8.0
    nrs = uniform_filter1d(np.where(np.isfinite(nr), nr, 0.0),
                           size=5, mode='nearest')
    pos = nrs[nrs > 0]
    return 0.5 * float(np.percentile(pos, 95)) if pos.size else 8.0


# Recovery pass: after a disturbance ends, weak-wind days within
# recovery_window_days that aren't already end become recovering
# but only while the vortex has genuinely reconvened (rings + centered geo).
def apply_recovery(states, fl, r):
    n = len(states); s = states.copy()
    peak = fl['peak_U']
    thr  = r.strong_peak_U * r.recovery_peak_U_frac
    half_ring = half_ring_threshold(fl)
    in_event = False
    last_event = -1
    for i in range(n):
        if s[i] in DISTURBANCE_SET:
            in_event = True; last_event = i
            continue
        if in_event and s[i] not in (STATE_END,):
            # Only call it recovery if the vortex has actually reconvened
            # (rings + centered geo). Weak winds with no ring / displaced geo
            # is the disturbance continuing, not recovery.
            if ((i - last_event) <= r.recovery_window_days and
                    np.isfinite(peak[i]) and peak[i] <= thr and
                    vortex_reconvened(i, fl, half_ring)):
                s[i] = STATE_RECOVERING
            elif (i - last_event) > r.recovery_window_days:
                in_event = False
    return s



# Event-based classifier (replaces the older multi-pass
# rule chain).  The decision flow is:
#


def day_month(times, ti):
    # Return calendar month (1-12) for a time index; None if unavailable.
    if times is None or ti < 0 or ti >= len(times):
        return None
    t = times[ti]
    try:
        return int(t.month)
    except Exception:
        try:
            return int(np.asarray(t).astype('datetime64[M]').astype(int) % 12 + 1)
        except Exception:
            return None


def onset_month_ok(ti, times, r):
    # True if a warming onset on day `ti` is allowed.  new warming onsets are only allowed in NH winter months Nov-Mar.
    if times is None:
        return True
    m = day_month(times, ti)
    if m is None:
        return True
    last_mo = int(getattr(r, 'warming_last_onset_month', 3))
    # Allowed winter months are Nov, Dec, then Jan..last_mo.
    return (m == 11) or (m == 12) or (1 <= m <= last_mo)


def april_cap_index(times, r, nt):
    # Return the last time index that an event window is allowed to occupy.
    if times is None:
        return nt - 1
    max_ext = int(getattr(r, 'warming_max_extension_april_days', 7))
    last_mo = int(getattr(r, 'warming_last_onset_month', 3))

    # Walk forward; the first index whose month is not a winter-season
    # month and that comes after we've already seen a late-winter month
    # (or whose month is strictly April..October encountered after the
    seen_late_winter = False
    eos_start = -1
    for k in range(nt):
        m = day_month(times, k)
        if m is None:
            continue
        if 1 <= m <= last_mo:
            seen_late_winter = True
            continue
        if m == 11 or m == 12:
            continue  # head of season
        # m is in {4..10} or > last_mo on a non-winter month
        if seen_late_winter or 4 <= m <= 10:
            eos_start = k
            break
    if eos_start < 0:
        return nt - 1
    return min(nt - 1, eos_start + max_ext - 1)


def is_eos_regime(idx, times, fl, r):
    # Return True if day `idx` is part of the late-season EOS
    if times is None or idx >= len(times):
        return False
    t = times[idx]
    m = getattr(t, 'month', None) or getattr(t, 'tm_mon', None)
    if m is None:
        return False
    # March 1 or later for the Northern Hemisphere winter calendar.
    # Pre-March we explicitly do not consider EOS regime; the
    # final warming can still occur in late February.
    if m < 3 or m > 10:
        return False

    nt = len(times)
    if idx < 7 or idx >= nt:
        return False
    peak  = np.asarray(fl.get('peak_U',  np.full(nt, np.nan)), dtype=float)
    dT    = np.asarray(fl.get('dT_col',  np.full(nt, np.nan)), dtype=float)
    strong_U = float(getattr(r, 'strong_peak_U', 100.0))

    # 14-day smoothed peak and dT around idx (use what's available
    # backward; recovery_window-style smoothing).
    win = max(7, int(getattr(r, 'recovery_window_days', 14)))
    lo  = max(0, idx - win + 1)
    seg_p  = peak[lo: idx + 1]
    seg_dT = dT[lo: idx + 1]
    seg_p  = seg_p[np.isfinite(seg_p)]
    seg_dT = seg_dT[np.isfinite(seg_dT)]
    if seg_p.size == 0 or seg_dT.size == 0:
        return False
    peak_smooth = float(seg_p.mean())
    dT_smooth   = float(seg_dT.mean())

    # Same conditions used by _detect_end_of_season_ev for the
    # late-season relaxed EOS rule.
    wind_slow_enough = (peak_smooth <= 0.70 * strong_U)
    dT_calm          = (-0.5 <= dT_smooth <= 0.6)
    return bool(wind_slow_enough and dT_calm)


def detect_warming_events_ev(fl, r, gap_tol=10, mode='cum25', times=None):
    # Return list of (start_idx, end_idx) for each warming event.
    if mode == 'rate':
        w = np.asarray(fl.get('warm1_max', []), dtype=float)
        thr = float(getattr(r, 'warm_rate_min_K', 0.0))
        if thr <= 0.0:
            # No threshold configured; diagnostic mode does nothing.
            return []
    else:
        w = np.asarray(fl.get('warm7_max', []), dtype=float)
        thr = float(getattr(r, 'ssw_warm25_K', 25.0))
    n = w.size
    if n == 0:
        return []
    # the 25 K threshold means "about 25", not "strictly
    # greater than or equal to 25.0".  Round to hundredths so a w7 of
    # 24.9991 K rounds up to 25.00 and crosses the threshold, while
    w_rounded = np.round(w, 2)
    trig = np.isfinite(w_rounded) & (w_rounded >= thr)
    # Vortex-reconvened gate: a warming cannot trigger or continue on a day
    # where the circumpolar ring has rebuilt across at least half the column.
    # A re-formed ring means the cold polar cap is back; the warming is over
    nrt = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)), dtype=float)
    bct = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                     dtype=float)
    nrf = nrt[np.isfinite(nrt) & (nrt > 0)]
    if nrf.size:
        half_col = 0.5 * float(np.percentile(nrf, 98))
        # reconvene = ring altitudes up to >= half column and geopotential
        # returned poleward (centered). The geo return can lag the ring
        # return by a few days, so accept rings high now or within the
        ring_hi = np.isfinite(nrt) & (nrt >= half_col)
        ring_recent = np.zeros(n, dtype=bool)
        for kk in range(n):
            lo = max(0, kk - 3)
            ring_recent[kk] = bool(np.any(ring_hi[lo:kk + 1]))
        geo_centered = np.isfinite(bct) & (bct >= 70.0)
        reconv_day = ring_recent & geo_centered
        reconvened = np.zeros(n, dtype=bool)
        run = 0
        for kk in range(n):
            if reconv_day[kk]:
                run += 1
                if run >= 2:
                    reconvened[kk - 1:kk + 1] = True
            else:
                run = 0
        trig = trig & ~reconvened
        # Polar-cap-exists gate: a warming can only trigger when there has
        # been a polar vortex (a cold cap) to warm within the recent past.
        # "A cap to warm" requires only that a substantial westerly ring
        nr_cap = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)),
                             dtype=float)
        ring_any = np.isfinite(nr_cap) & (nr_cap > 0)
        cap_ring_present = np.zeros(n, dtype=bool)
        for kk in range(n):
            lo = max(0, kk - 3)
            cap_ring_present[kk] = bool(np.any(ring_any[lo:kk + 1]))
        cap_lookback = 21
        cap_exists = np.zeros(n, dtype=bool)
        for kk in range(n):
            lo = max(0, kk - cap_lookback)
            cap_exists[kk] = bool(np.any(cap_ring_present[lo:kk + 1]))
        # An intact easterly circumpolar ring is definitive evidence of a
        # wind-reversal (major) warming; the jet has reversed. Such a day
        # is a warming regardless of the cap-exists lookback, so it bypasses
        eint_trig = np.asarray(fl.get('east_intact_level', np.zeros(n)),
                                dtype=float)
        cap_exists = cap_exists | (eint_trig >= 1.0)
        # only gate the trigger (onset): once a warming begins, the cap
        # disrupts and we don't keep requiring a cap during it. The trigger
        # detector treats `trig` as the onset signal; gate it directly.
        trig = trig & cap_exists
    # Require the warming to occupy a tall, broad region, not a single
    # shallow high-altitude block.  A day triggers only if enough
    # 10x30x4km blocks warmed past threshold (warm_bin_count).  This
    min_blocks = float(getattr(r, 'warm_min_blocks', 0) or 0)
    if min_blocks > 0:
        wbc = np.asarray(fl.get('warm_bin_count', np.full(n, np.nan)),
                         dtype=float)
        # apply the gate only where the block count exists; a missing
        # (NaN) count must not silently drop the day (older files lack it).
        if np.isfinite(wbc).any():
            block_ok = (~np.isfinite(wbc)) | (wbc >= min_blocks)
            trig = trig & block_ok
    if not trig.any():
        return []

    # Auxiliary fields used by event detection.
    peak_U   = np.asarray(fl.get('peak_U', np.full(n, np.nan)),
                          dtype=float)
    dT_col   = np.asarray(fl.get('dT_col', np.full(n, np.nan)),
                          dtype=float)
    warm7    = np.asarray(fl.get('warm7_max', np.full(n, np.nan)),
                          dtype=float)
    # Per-altitude westerly state. These are the canonical
    # wind diagnostics for recovery: per-event we capture the per-
    # altitude state at the most recent strong-vortex day before
    ring_west = fl.get('ring_present_west', None)
    westU     = fl.get('west_mean_U', None)
    phys = fl.get('event_physics')
    if not phys:
        phys = train_event_physics(fl, r)
    min_rec = int(phys.get('min_rec_run', max(int(r.min_run_length), 3)))

    # events cannot have their onset (first 25 K day)
    # in April or later.
    april_cap = april_cap_index(times, r, n)

    def is_strong_day_for_reference(k):
        # Quiet strong-vortex reference day (data-driven quantiles).
        if not np.isfinite(warm7[k]):
            return False
        finite_w7 = warm7[np.isfinite(warm7)]
        if finite_w7.size == 0:
            return False
        q25 = float(np.percentile(finite_w7, 25))
        if warm7[k] > q25:
            return False
        if ring_west is not None:
            rp = ring_west[k]
            n_ring_today = int(rp.sum())
            counts = [int(ring_west[kk].sum()) for kk in range(n)]
            if counts and n_ring_today < float(np.median(counts)):
                return False
        return True

    def find_reference_day(start_idx):
        for k in range(start_idx - 1, -1, -1):
            if is_strong_day_for_reference(k):
                return k
        return None

    events = []
    last_event_end = -1   # index of the last day belonging to a previous event
    last_event_ref = None # ref_idx used by the previous event (its pre-event strong-day snapshot)
    last_ended_jet_collapse = False
    last_ended_geo_pause = False
    i = 0
    while i < n:
        if not trig[i]:
            i += 1
            continue
        # Block onset in April or later
        if not onset_month_ok(i, times, r):
            i += 1
            continue
        # A sustained >=25K warming during the late-season breakdown is
        # the final warming; the canonical season-ending SSW; and must be
        # registered as a warming (hierarchy: a warming is a warming). We do

        # Locate the per-event reference day (most recent strong-
        # vortex day before the trigger).  This freezes the per-
        # altitude wind state we'll compare against for recovery.
        ref_idx = find_reference_day(i)

        # for a re-trigger after a previous event,
        # require that the per-altitude westerly state at the trigger
        # day matches the pre-previous-event strong-day reference
        if not allow_new_warming_onset(i, last_event_end, last_event_ref,
                                        ref_idx, ring_west, westU, phys,
                                        trig=trig, fl=fl,
                                        after_jet_collapse=(
                                            last_ended_jet_collapse and
                                            i <= last_event_end + 20),
                                        after_geo_pause=(
                                            last_ended_geo_pause and
                                            i <= last_event_end + 5)):
            i += 1
            continue

        # Residual thermal on an already-recovered vortex (strong jet +
        # poleward centroid) is not a new warming cycle.
        if is_residual_thermal_onset(i, fl, phys):
            i += 1
            continue

        # One relaxed re-trigger per jet-collapse / geo-pause termination.
        last_ended_jet_collapse = False
        last_ended_geo_pause = False

        start = i
        last  = i
        j = i + 1
        evt_peak_w7 = float(warm7[start]) if np.isfinite(warm7[start]) else 25.0
        peak_day    = start
        evt_peak_U = (float(peak_U[start])
                      if np.isfinite(peak_U[start]) else np.nan)
        nr_arr = np.asarray(fl.get('n_ring_levels', np.full(n, np.nan)),
                            dtype=float)
        evt_nr_peak = (float(nr_arr[start])
                       if np.isfinite(nr_arr[start]) else np.nan)
        min_jet_rec = int(phys.get('min_jet_rec_run', 2))
        severe_jet_rec = 1

        # Event ends after min_rec consecutive recovery-signature days.
        # Recovery = polar-cap cooling + wind-ring reformation +
        # bottom-geo motion quiet; trained from the season itself.
        rec_run = 0
        rec_start = None
        wind_rec_run = 0
        wind_rec_start = None
        jet_rec_run = 0
        jet_rec_start = None
        ended_with_recovery = False
        ended_jet_collapse = False
        ended_geo_pause = False
        # Wind reconvene = the authoritative end (block-independent).
        # Compute the half-column westerly-ring threshold once for this event.
        nrf_ev = nr_arr[np.isfinite(nr_arr) & (nr_arr > 0)]
        half_ev = (0.5 * float(np.percentile(nrf_ev, 98))
                    if nrf_ev.size else 8.0)
        bc_ev = np.asarray(fl.get('geo_c0_bottom_cent', np.full(n, np.nan)),
                            dtype=float)
        while j < n:
            if np.isfinite(peak_U[j]) and (
                    not np.isfinite(evt_peak_U) or peak_U[j] > evt_peak_U):
                evt_peak_U = float(peak_U[j])
            if j < nr_arr.size and np.isfinite(nr_arr[j]) and (
                    not np.isfinite(evt_nr_peak) or nr_arr[j] > evt_nr_peak):
                evt_nr_peak = float(nr_arr[j])
            if is_eos_regime(j, times, fl, r):
                break
            # authoritative end: the vortex has reconvened; geopotential
            # returned poleward (centered) and ring altitudes >= half column
            # now or within the preceding 3 days (geo may lag the rings)
            if ((j - start) >= 5 and j + 1 < nr_arr.size):
                ok2 = True
                for jj in (j, j + 1):
                    lo = max(start, jj - 3)
                    ring_recent = np.any(
                        np.isfinite(nr_arr[lo:jj + 1]) &
                        (nr_arr[lo:jj + 1] >= half_ev))
                    geo_ok = (jj < bc_ev.size and np.isfinite(bc_ev[jj]) and
                              bc_ev[jj] >= 70.0)
                    if not (ring_recent and geo_ok):
                        ok2 = False
                        break
                if ok2:
                    last = max(start, j - 1)
                    ended_with_recovery = True
                    break
            if (not trig[j]) and (j - last - 1) >= int(gap_tol):
                # warm7 gap alone does not end the event. End only if the
                # vortex is recovering; ring rebuilding toward half column
                # or geopotential centering poleward. A still-displaced,
                bc = np.asarray(fl.get('geo_c0_bottom_cent', []),
                                 dtype=float)
                ring_prog = (j < nr_arr.size and np.isfinite(nr_arr[j]) and
                             nr_arr[j] >= 0.5 * half_ev)
                geo_prog = (j < bc.size and np.isfinite(bc[j]) and
                            bc[j] >= 70.0)
                if ring_prog or geo_prog or (j - last - 1) >= 3 * int(gap_tol):
                    break
            if jet_collapse_recovery(j, fl, phys,
                                      evt_peak_u=evt_peak_U,
                                      evt_nr_peak=evt_nr_peak,
                                      evt_start=start):
                if jet_rec_start is None:
                    jet_rec_start = j
                jet_rec_run += 1
                peak = np.asarray(fl.get('peak_U', []), dtype=float)
                lo7 = max(int(start), j - 7)
                recent_pk_max = float(np.nanmax(peak[lo7:j + 1]))
                pk_j = float(peak[j]) if j < peak.size else np.nan
                need = (severe_jet_rec if (np.isfinite(pk_j) and
                                           np.isfinite(recent_pk_max) and
                                           recent_pk_max > 0 and
                                           pk_j <= 0.55 * recent_pk_max)
                        else min_jet_rec)
                if jet_rec_run >= need and jet_rec_start > start:
                    last = min(last, jet_rec_start - 1)
                    last = max(last, start)
                    ended_with_recovery = True
                    ended_jet_collapse = True
                    break
            else:
                jet_rec_run = 0
                jet_rec_start = None
            if (geo_poleward_pause(j, fl, phys) and (j - start) >= 21 and
                    rec_run < min_rec):
                mo = day_month(times, j)
                if mo is not None and 2 <= mo <= 3:
                    last = min(last, j - 1)
                    last = max(last, start)
                    ended_with_recovery = True
                    ended_geo_pause = True
                    break
            # Full recovery ends the trigger continuation; wind + geo
            # returning, not polar-cap cooling.  Residual warm7 above
            # 25 K must not sustain an event once the jet has recovered.
            if ((j - start) >= 7 and
                    recovery_signature(j, fl, phys, ref_idx)):
                if rec_start is None:
                    rec_start = j
                rec_run += 1
                if rec_run >= min_rec and rec_start > start:
                    last = min(last, rec_start - 1)
                    last = max(last, start)
                    ended_with_recovery = True
                    break
            else:
                rec_run = 0
                rec_start = None
            # Wind-reformation close: the vortex is reconvening (jet strength
            # risen and the circumpolar ring rebuilt across the column). This
            # ends the warming regardless of residual warm7; a warming ends
            if ((j - start) >= 5 and
                    jet_reformed(j, fl, phys, evt_start=start)):
                # strong reconvening: ring rebuilt across >= half the column
                nr_j = float(nr_arr[j]) if (j < nr_arr.size and
                                            np.isfinite(nr_arr[j])) else 0.0
                nrf = nr_arr[np.isfinite(nr_arr) & (nr_arr > 0)]
                half_col = (0.5 * float(np.percentile(nrf, 98))
                            if nrf.size else 8.0)
                if nr_j >= half_col:
                    # unambiguous recovery -> close now, no 2-day wait
                    last = max(start, j - 1)
                    ended_with_recovery = True
                    break
                if wind_rec_start is None:
                    wind_rec_start = j
                wind_rec_run += 1
                if wind_rec_run >= 2 and wind_rec_start > start:
                    last = min(last, wind_rec_start - 1)
                    last = max(last, start)
                    ended_with_recovery = True
                    break
            else:
                wind_rec_run = 0
                wind_rec_start = None
            if trig[j] and rec_run < min_rec:
                if not wind_geo_recovery(j, fl, phys, ref_idx, full=False):
                    last = j
                    if np.isfinite(warm7[j]) and warm7[j] > evt_peak_w7:
                        evt_peak_w7 = float(warm7[j])
                        peak_day    = j
            j += 1
        # no arbitrary 12.5 K threshold.  Event opens
        # only when 25 K is reached and the warming is sustained
        # enough to register >= 2 trigger days within the gap-tol /
        n_trig_days = int(trig[start:last + 1].sum())
        env_span = last - start + 1
        if n_trig_days >= 2 and env_span >= 5:
            # Event window
            # the event start is the first day on which
            # warm7_max actually reaches 25 K (the trigger day).  Do
            env_lo = start
            env_hi = last
            # Do not walk forward through post-warming recovery; the
            # continuation loop already backed ``last`` up to the last
            # trigger before a sustained recovery block.

            # Require >= 3 days total; drop 1- and 2-day blips.
            if (env_hi - env_lo + 1) >= 3:
                env_span = env_hi - env_lo + 1
                if env_span >= 30:
                    min_weak = int(phys.get('min_weak_rec_run', 2))
                    segments = split_envelope_at_recovery_cycles(
                        env_lo, env_hi, fl, phys, ref_idx, trig, min_weak,
                        warm7, min_span=14)
                    last_accepted_end = None
                    for seg_lo, seg_hi in segments:
                        if ((seg_hi - seg_lo + 1) < 5 or
                                int(trig[seg_lo:seg_hi + 1].sum()) < 2):
                            continue
                        events.append((seg_lo, seg_hi))
                        last_accepted_end = seg_hi
                else:
                    events.append((env_lo, env_hi))
                    last_accepted_end = env_hi
                if last_accepted_end is not None:
                    last_event_end = last_accepted_end
                    last_ended_jet_collapse = ended_jet_collapse
                    last_ended_geo_pause = ended_geo_pause
                else:
                    last_event_end = env_hi
                    last_ended_jet_collapse = ended_jet_collapse
                    last_ended_geo_pause = ended_geo_pause
                # Keep the pre-event reference until a later onset passes
                # the wind-recovery gate (per-altitude ring reformation).
                if ref_idx is not None:
                    last_event_ref = ref_idx
        i = max(j, last + 1)
    return dedupe_event_spans(events)


def event_has_second_component(fl, evt_start, evt_end, nt, r):
    # True only when a genuine second 3-D lobe exists during the warming
    evt_start = int(evt_start)
    evt_end = int(min(nt - 1, evt_end))
    if evt_end < evt_start:
        return False
    sl = slice(evt_start, evt_end + 1)

    comp1_present = (np.asarray(fl.get('comp1_present', np.zeros(nt)),
                                dtype=float) >= 0.5)
    seeded = (np.asarray(fl.get('split_seeded', np.zeros(nt)),
                         dtype=float) >= 0.5)
    area_fracs = np.asarray(fl.get('area_fracs', np.zeros(nt)), dtype=float)
    two_genuine = (np.asarray(fl.get('two_lobes_genuine', np.zeros(nt)),
                              dtype=float) >= 0.5)
    geo_alt_lo = np.asarray(fl.get('geo_alt_lo'), dtype=float)
    geo_alt_hi = np.asarray(fl.get('geo_alt_hi'), dtype=float)
    if geo_alt_lo.ndim == 2 and geo_alt_lo.shape[1] >= 2:
        al1 = geo_alt_lo[:, 1]
        ah1 = geo_alt_hi[:, 1] if geo_alt_hi.ndim == 2 else np.full(nt, np.nan)
        depth1 = np.where(np.isfinite(ah1) & np.isfinite(al1), ah1 - al1, np.nan)
    else:
        al1 = np.full(nt, np.nan)
        depth1 = np.full(nt, np.nan)

    comp1_reaches_mid = np.isfinite(al1) & (al1 <= 25.0)
    comp1_deep = np.isfinite(depth1) & (depth1 >= 15.0)
    comp1_touches_bottom = np.isfinite(al1) & (al1 <= 22.0)
    lobe_ok = (comp1_present & seeded & two_genuine & comp1_reaches_mid &
               (comp1_touches_bottom | comp1_deep) &
               np.isfinite(area_fracs) & (area_fracs >= SPLIT_AREAFRAC))
    if lobe_ok[sl].any():
        return True

    b0_full = (np.asarray(fl.get('b0_full_split', np.zeros(nt)),
                          dtype=float) >= 0.5)
    b0_top = (np.asarray(fl.get('b0_top_pinch_split', np.zeros(nt)),
                         dtype=float) >= 0.5)
    split_geo = b0_full | b0_top
    if int(split_geo[sl].sum()) >= 2:
        if int(b0_full[sl].sum()) >= 2:
            return True
        if (seeded[sl].any() and
                np.isfinite(area_fracs[sl]).any() and
                float(np.nanmax(area_fracs[sl])) >= SPLIT_AREAFRAC):
            return True
    return False


def mixed_is_genuine_sequence(disp_runs, split_runs, bot_cent, bc_base,
                               comp1_present, split_sig_mix, flavor):
    # Mixed requires a true monolithic-displacement ↔ split sequence.
    if not disp_runs or not split_runs:
        return False
    disp_first = disp_runs[0]
    split_first = split_runs[0]
    d0, d1 = disp_first
    s0, s1 = split_first
    n = bot_cent.size

    if flavor == 'disp_then_split':
        # Split must follow a period at a displaced *single* lobe; not
        # concurrent equatorward drift while the vortex is already splitting.
        if s0 < d0 + 4:
            return False
        pre_end = min(d1, s0 - 1)
        if pre_end >= d0:
            pre = slice(d0, pre_end + 1)
            n_pre = pre_end - d0 + 1
            single = (~comp1_present[pre] & ~split_sig_mix[pre])
            if int(single.sum()) < max(3, int(0.55 * n_pre)):
                return False
        # Before the split crystallizes, the centroid must have been
        # substantially lower than baseline (intact displaced monolith),
        # not merely drifting down while already multi-lobe.
        look_lo = max(0, s0 - 10)
        before = bot_cent[look_lo:s0]
        before = before[np.isfinite(before)]
        if before.size >= 4:
            if float(np.nanmean(before[-4:])) >= bc_base - 4.0:
                return False
        already_low = (np.isfinite(bot_cent[look_lo:s0]) &
                       (bot_cent[look_lo:s0] < bc_base - 5.0))
        if int(already_low.sum()) >= 5:
            return False
        return True

    # split_then_rejoin; require comp 1 absent for several days between
    # the split spell ending and the later displaced monolith.
    split_last = split_runs[0]
    disp_after = [r for r in disp_runs
                  if (r[0] - split_last[1]) > 2]
    if not disp_after:
        return False
    gap_lo = split_last[1] + 1
    gap_hi = disp_after[0][0] - 1
    if gap_lo > gap_hi or gap_hi >= n:
        return False
    gap = slice(gap_lo, min(n, gap_hi + 1))
    if int((~comp1_present[gap]).sum()) < max(3, gap_hi - gap_lo):
        return False
    return True


def classify_event_ev(evt, fl, r, nt, next_evt_start=None):
    # Classify one warming event (evt = (start, end)) by:
    evt_start, evt_end = evt
    east = np.asarray(fl.get('east_intact_level', np.zeros(nt)), dtype=float)
    # In-vortex per-gridpoint warming: max d-day rise of T (d=1..7)
    # at any (lev, lat) inside the comp-0 vortex region.
    w7 = np.asarray(fl.get('warm7_max', []), dtype=float)
    if w7.size == 0 or not np.isfinite(w7).any():
        w7 = np.full(nt, np.nan)

    pre  = int(getattr(r, 'event_precursor_days', 14))
    post = max(int(getattr(r, 'recovery_window_days', 14)),
               int(getattr(r, 'split_bridge_max_gap', 14)))
    nrl_m = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                       dtype=float)
    nrf_m = nrl_m[np.isfinite(nrl_m) & (nrl_m > 0)]
    # Lower bound of the morphology window: the vortex disturbance onset
    # (wind, block-independent); the last day at/before evt_start on which
    # the ring was still rebuilt across half the column, i.e. where the lobe
    disturb_onset = max(0, evt_start - pre)
    if nrf_m.size:
        half_m2 = 0.5 * float(np.percentile(nrf_m, 98))
        for kk in range(int(evt_start), max(0, evt_start - pre) - 1, -1):
            if np.isfinite(nrl_m[kk]) and nrl_m[kk] >= half_m2:
                disturb_onset = kk
                break
    win_lo = max(0, min(int(evt_start), disturb_onset))
    # Upper bound of the morphology window: the vortex reconvene point (wind,
    # block-independent), not the temperature-driven evt_end. The warming's
    # type and geopotential state must not depend on which temperature block
    reconv_end = evt_end
    if nrf_m.size:
        half_m = 0.5 * float(np.percentile(nrf_m, 98))
        for kk in range(int(evt_start) + 1, nt):
            if np.isfinite(nrl_m[kk]) and nrl_m[kk] >= half_m:
                reconv_end = kk
                break
    win_hi = min(nt, max(int(evt_end), int(reconv_end)) + post + 1)
    # Clip morphology search at next event start
    if next_evt_start is not None:
        win_hi = min(win_hi, int(next_evt_start))

    # provisional within-event flag from the easterly ring; the FINAL
    # major/minor is set by the 7-day temperature rise (refine_major_minor)
    has_east_ring = np.asarray(fl.get('east_intact_level', np.zeros(nt)),
                               dtype=float)
    has_east_in_event = bool((has_east_ring[evt_start:evt_end + 1] >= 0.5).any())
    major = has_east_in_event

    # Event labelling starts on the first 25 K day (evt_start).
    label_start = evt_start

    # shared component arrays
    comp1_present = (np.asarray(fl.get('comp1_present', np.zeros(nt)),
                                dtype=float) >= 0.5)
    geo_alt_hi    = np.asarray(fl.get('geo_alt_hi'), dtype=float)
    geo_alt_lo    = np.asarray(fl.get('geo_alt_lo'), dtype=float)
    if geo_alt_hi.ndim == 2 and geo_alt_hi.shape[1] >= 2:
        ah0 = geo_alt_hi[:, 0]
        ah1 = geo_alt_hi[:, 1]
        comp1_high = np.isfinite(ah1) & (ah1 >= 30.0)
        comp0_high = np.isfinite(ah0) & (ah0 >= 30.0)
    else:
        ah0 = np.full(nt, np.nan); ah1 = np.full(nt, np.nan)
        comp1_high = np.zeros(nt, dtype=bool)
        comp0_high = np.zeros(nt, dtype=bool)

    cent_lat = np.asarray(fl.get('comp_cent_lat'), dtype=float)
    has_lat1 = cent_lat.ndim == 2 and cent_lat.shape[1] >= 2
    if has_lat1:
        lat0 = cent_lat[:, 0]; lat1 = cent_lat[:, 1]
    else:
        lat0 = np.full(nt, np.nan); lat1 = np.full(nt, np.nan)

    area_fracs = np.asarray(fl.get('area_fracs', np.zeros(nt)), dtype=float)

    # displaced signal (computed early so we can test for the
    # mixed (displaced→split / split→displaced) temporal ordering
    # before committing to "split" or "displaced")
    bot_cent = np.asarray(fl.get('geo_c0_bottom_cent',
                                  np.full(nt, np.nan)), dtype=float)
    base_lo = max(0, evt_start - 30)
    base_hi = max(base_lo + 1, evt_start)
    segx = bot_cent[base_lo:base_hi]
    finite_seg = segx[np.isfinite(segx)]
    if finite_seg.size >= 5:
        bc_base = float(np.percentile(finite_seg, 75))
    elif finite_seg.size > 0:
        bc_base = float(np.nanmax(finite_seg))
    else:
        bc_base = 70.0
    DISP_DROP = 5.0
    sustained_low_full = (np.isfinite(bot_cent) &
                          (bot_cent < bc_base - DISP_DROP))

    # split detection
    # A real split has three properties that distinguish it from
    # transient upper-level fragments and detached low-lat blobs:

    # Per-level seed signal: was already computed in compute_flags as
    # split_seeded; at least one (lev) where comp 0 and comp 1 are
    # close in (lat, lon).  We require this to be true on the same
    seeded = (np.asarray(fl.get('split_seeded', np.zeros(nt)),
                         dtype=float) >= 0.5)

    # comp 1 must span most altitudes and touch the bottom.  Per user
    # spec; "it can't be a small blob on the other edge or at the
    # top, it's a big portion of geopotential splitting.  if the
    if geo_alt_lo.ndim == 2 and geo_alt_lo.shape[1] >= 2:
        al1 = geo_alt_lo[:, 1]
    else:
        al1 = np.full(nt, np.nan)
    # touch the bottom: comp 1 reaches down to <= 22 km (bottom of
    # the stratospheric grid in this dataset is ~16 km, so allow a
    # few km of buffer).  A split lobe touches the lower stratosphere.
    comp1_touches_bottom = (np.isfinite(al1) & (al1 <= 22.0))
    # reject upper-only / low-latitude debris that never
    # reaches below ~25 km altitude.
    comp1_reaches_mid = np.isfinite(al1) & (al1 <= 25.0)

    if geo_alt_hi.ndim == 2 and geo_alt_hi.shape[1] >= 2:
        depth1 = np.where(np.isfinite(ah1) & np.isfinite(al1),
                          ah1 - al1, np.nan)
    else:
        depth1 = np.full(nt, np.nan)
    # span most altitudes: depth >= 15 km of the ~35 km column.
    # Looser than the previous 20 km bar; real splits can have
    # somewhat shallower second lobes (e.g. ~17-19 km).
    comp1_deep = np.isfinite(depth1) & (depth1 >= 15.0)

    # Two-lobe (split) signature requires all of:
    # • comp 1 present, high (top >=30km) and deep (>=15km vertical span)
    # or touches the bottom (alt_lo <= 22 km); either is acceptable
    two_lobe = (
        comp1_present & comp1_high & comp0_high &
        comp1_reaches_mid &
        (comp1_touches_bottom | comp1_deep) & seeded &
        np.isfinite(lat0) & np.isfinite(lat1) &
        np.isfinite(area_fracs) & (area_fracs >= SPLIT_AREAFRAC)
    )

    # reject longitude-wrap artifacts that present as
    # two components but are actually one wrapped lobe.  The
    # `two_lobes_genuine` flag (computed in compute_flags from the
    two_genuine = (np.asarray(fl.get('two_lobes_genuine', np.zeros(nt)),
                              dtype=float) >= 0.5)
    two_lobe = two_lobe & two_genuine

    # Per-level Betti-0 profiles (_3 datasets): multi-day / partial evolution.
    b0_full = (np.asarray(fl.get('b0_full_split', np.zeros(nt)),
                          dtype=float) >= 0.5)
    b0_full_prog = (np.asarray(fl.get('b0_full_split_prog', np.zeros(nt)),
                               dtype=float) >= 0.5)
    b0_top = (np.asarray(fl.get('b0_top_pinch_split', np.zeros(nt)),
                         dtype=float) >= 0.5)
    b0_partial = (np.asarray(fl.get('b0_partial_split', np.zeros(nt)),
                             dtype=float) >= 0.5)
    frac_bot_partial = np.nan_to_num(
        np.asarray(fl.get('b0_frac_bot_partial', np.zeros(nt)), dtype=float))
    frac_top_partial = np.nan_to_num(
        np.asarray(fl.get('b0_frac_top_partial', np.zeros(nt)), dtype=float))
    # Combined split signature: component geometry or instantaneous full
    # B0 split or top pinch (≥ 45 km only).
    split_sig_full = two_lobe | b0_full
    split_sig = split_sig_full | b0_top
    split_sig_ext = split_sig | b0_full_prog

    # Restrict the split-evidence search to the warming event window
    # itself (plus a short ±5-day buffer for boundary cases).  Any
    # qualifying day must fall inside the window for the event to be
    post_split_window = max(int(getattr(r, 'recovery_window_days', 14)),
                            int(getattr(r, 'split_bridge_max_gap', 14)),
                            21)
    inner_lo = max(0, evt_start - 5)
    inner_hi = min(nt, evt_end + post_split_window + 1)
    if next_evt_start is not None:
        inner_hi = min(inner_hi, int(next_evt_start))
    sig_w = split_sig[inner_lo:inner_hi]
    n_split_days = int(sig_w.sum())
    # A real split has substantial two-lobe activity during the
    # warming event itself, not just at the tail.  # "the split shouldn't just be at the end of the warming, that's
    # usually indicative of a messy weak vortex region".  Count
    sig_inside_evt = split_sig[evt_start:min(nt, evt_end + 1)]
    n_split_inside = int(sig_inside_evt.sum())
    n_split_tail   = n_split_days - n_split_inside
    # mixed (displaced ↔ split) temporal ordering
    # mixed exists only when:
    # (a) the vortex fully moves to a displaced state, then splits
    MIXED_DISP_DROP = 8.0  # stricter than the displaced-detection -5°
    MIXED_DISP_RUN  = 5    # stricter than the looser run length below
    MIXED_SPLIT_RUN = 3    # stricter than the looser split run length
    MIXED_GAP_MIN   = 2    # require a real temporal separation
    sustained_low_mixed = (np.isfinite(bot_cent) &
                            (bot_cent < bc_base - MIXED_DISP_DROP))
    mix_lo = max(0, evt_start - 3)
    mix_hi = min(nt, evt_end + 14 + 1)  # short buffer, not recovery_window
    if next_evt_start is not None:
        mix_hi = min(mix_hi, int(next_evt_start))

    def find_runs(mask, lo, hi, min_len):
        runs = []
        i = lo
        while i < hi:
            if not mask[i]:
                i += 1; continue
            j = i
            while j < hi and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        return runs

    disp_runs_mix  = find_runs(sustained_low_mixed, mix_lo, mix_hi,
                                 min_len=MIXED_DISP_RUN)
    # Mixed split phase must use genuine two-lobe geometry (seeded,
    # comp1 reaches below 25 km) or Betti-0 full/top pinch; not
    # transient upper fragments from lower latitudes.
    split_sig_mix = b0_full | b0_top | two_lobe
    split_runs_mix = find_runs(split_sig_mix,        mix_lo, mix_hi,
                                 min_len=MIXED_SPLIT_RUN)

    mixed_window = None
    if disp_runs_mix and split_runs_mix:
        # a mixed event is a definite displacement
        # followed by a clear split that either (a) develops while
        # the vortex is still in the displaced state, or (b) develops
        disp_first  = disp_runs_mix[0]
        split_first = split_runs_mix[0]
        if split_first[0] >= disp_first[0]:
            first = max(disp_first[0], label_start)
            last  = max(split_runs_mix[-1][1], disp_runs_mix[-1][1])
            mixed_window = (first, last, 'disp_then_split')
        # "split → rejoin": split phase ends, with clear gap, before
        # a persistent fully-displaced phase begins.  Keep the gap
        # requirement on this flavor only; a split that immediately
        if mixed_window is None:
            split_last = split_runs_mix[0]
            disp_after = [r for r in disp_runs_mix
                          if (r[0] - split_last[1]) > MIXED_GAP_MIN]
            if disp_after:
                first = max(split_last[0], label_start)
                last  = disp_after[-1][1]
                mixed_window = (first, last, 'split_then_rejoin')

    # mixed cannot be assigned in recovery or
    # end-of-season / final-warming territory because the geopotential
    # signals are unreliable there.
    late_season = bool(evt_start > 0.70 * nt)
    if mixed_window is not None and not late_season:
        # Full B0 split inside the warming window → commit to split,
        # not mixed (partial-then-full evolution over multiple days).
        n_b0_full_inside = int(
            b0_full[evt_start:min(nt, evt_end + 1)].sum())
        flavor = mixed_window[2]
        genuine = mixed_is_genuine_sequence(
            disp_runs_mix, split_runs_mix, bot_cent, bc_base,
            comp1_present, split_sig_mix, flavor)
        if n_b0_full_inside >= 2 or not genuine:
            mixed_window = None
        else:
            first, last, flavorx = mixed_window
            first = max(first, label_start)
            return dict(kind='mixed', window=(first, last),
                        major=major, label_start=label_start)

    # Split detection; real splits sometimes don't crystallize until
    # after the warming peak (the temperature spike heats the cap and
    # the geopotential separates a few days later, as in the user-
    seeded_in_window = int(seeded[inner_lo:inner_hi].sum())
    max_af_in_window = (float(np.nanmax(area_fracs[inner_lo:inner_hi]))
                         if inner_hi > inner_lo else 0.0)
    b0_evt_hi = min(nt, evt_end + 1)
    b0_full_days = int(b0_full[evt_start:b0_evt_hi].sum())
    has_during_split = (
        (b0_full_days >= 2) or
        (int(two_lobe[evt_start:b0_evt_hi].sum()) >= 2) or
        (int((two_lobe & b0_full)[evt_start:b0_evt_hi].sum()) >= 1)
    )
    lag_hi = min(nt, evt_end + post_split_window, inner_hi)
    two_near = int(two_lobe[evt_start:lag_hi].sum())
    seeded_lobe_near = int(
        (two_lobe & seeded & comp1_present & (area_fracs >= SPLIT_AREAFRAC))[evt_start:lag_hi].sum())
    max_af_near = (float(np.nanmax(area_fracs[evt_start:lag_hi]))
                   if lag_hi > evt_start else 0.0)
    has_lagged_split = (
        two_near >= 2 or
        (two_near >= 1 and seeded_lobe_near >= 1 and
         max_af_near >= max(SPLIT_AREAFRAC, 0.18))
    )
    b0_inside_full = split_sig_full[evt_start:b0_evt_hi]
    n_b0_full_inside = int(b0_inside_full.sum())
    n_b0_full_consec = max_consecutive_true(
        split_sig_full, evt_start, min(nt - 1, evt_end))
    max_af_evt = (float(np.nanmax(area_fracs[evt_start:b0_evt_hi]))
                  if b0_evt_hi > evt_start else 0.0)
    seeded_inside = int(seeded[evt_start:b0_evt_hi].sum())
    # Full split: B0-full or genuine two-lobe only; not top pinch /
    # partial B0 alone (8485 Feb has seed/b0part but is not a split).
    tail_hi = min(nt, evt_end + post_split_window + 1)
    if next_evt_start is not None:
        tail_hi = min(tail_hi, int(next_evt_start))
    b0_full_tail = int(b0_full[evt_end + 1:tail_hi].sum()) if tail_hi > evt_end + 1 else 0
    two_lobe_tail = int(two_lobe[evt_end + 1:tail_hi].sum()) if tail_hi > evt_end + 1 else 0
    has_b0_split = (
        (b0_full_days >= 2) or
        (n_b0_full_inside >= 2 and n_b0_full_consec >= 2) or
        (b0_full_days >= 1 and n_b0_full_consec >= 2 and seeded_inside >= 1) or
        (b0_full_tail >= 2 and two_lobe_tail >= 1) or
        (b0_full_tail >= 2 and max_af_evt >= SPLIT_AREAFRAC)
    )
    if (has_during_split or has_lagged_split or has_b0_split):
        if not event_has_second_component(fl, evt_start, evt_end, nt, r):
            # Allow lagged split evidence in the post-event tail of this
            # event (e.g. 8485 Mar b0-full at day 153).
            tail_sl = slice(evt_end + 1, tail_hi)
            tail_ok = False
            if tail_sl.stop > tail_sl.start:
                tail_ok = (
                    int(b0_full[tail_sl].sum()) >= 2 or
                    int(two_lobe[tail_sl].sum()) >= 2
                )
            if not tail_ok:
                has_during_split = has_lagged_split = has_b0_split = False
            elif not has_during_split and not has_b0_split:
                has_lagged_split = True
    if has_during_split or has_lagged_split or has_b0_split:
        sig_idx = np.where(split_sig_full[inner_lo:inner_hi])[0]
        if sig_idx.size:
            first = inner_lo + int(sig_idx[0])
            last  = inner_lo + int(sig_idx[-1])
        else:
            search_idx = np.where(split_sig_full[evt_start:tail_hi])[0]
            if search_idx.size:
                first = label_start
                last  = evt_start + int(search_idx[-1])
            else:
                first = label_start
                last  = evt_end
        # Extend "until stabilized": comp 1 absent, or comp 1 present
        # but no longer extends below upper altitudes for >= 2
        # consecutive days.
        scan_hi = min(nt, last + 60)
        absent_run = 0
        for k in range(last + 1, scan_hi):
            still_split = (
                split_sig_ext[k] or
                (comp1_present[k] and comp1_touches_bottom[k] and
                 ((np.isfinite(lat1[k]) and lat1[k] >= 45.0) or
                  (np.isfinite(ah1[k])  and ah1[k]  >= 30.0))))
            if still_split:
                last = k
                absent_run = 0
            else:
                absent_run += 1
                if absent_run >= 2:
                    break
        first = max(first, label_start)

        # Ambiguous-event check: within the same warming run, was the
        # vortex also strongly displaced for a comparable number of
        # days?  Per user; these "starts displaced, becomes split"
        bc_chk = np.asarray(fl.get('geo_c0_bottom_cent',
                                    np.full(nt, np.nan)), dtype=float)
        # Compute the same baseline as the displaced branch
        baselo = max(0, evt_start - 30)
        basehi = max(baselo + 1, evt_start)
        segx = bc_chk[baselo:basehi]
        segx = segx[np.isfinite(segx)]
        if segx.size >= 5:
            bcbase = float(np.percentile(segx, 75))
        elif segx.size > 0:
            bcbase = float(np.nanmax(segx))
        else:
            bcbase = 70.0
        DROPX = 5.0
        evt_lo = max(0, evt_start - 5)
        evt_hi = min(nt, evt_end + post_split_window + 1)
        if next_evt_start is not None:
            evt_hi = min(evt_hi, int(next_evt_start))
        n_evt_days = max(1, evt_hi - evt_lo)
        disp_chk = (np.isfinite(bc_chk[evt_lo:evt_hi]) &
                    (bc_chk[evt_lo:evt_hi] < bcbase - DROPX))
        out_split_mask = np.ones(n_evt_days, dtype=bool)
        sf = max(first, evt_lo); sl = min(last,  evt_hi - 1)
        if sl >= sf:
            out_split_mask[(sf - evt_lo):(sl - evt_lo + 1)] = False
        n_disp_outside_split = int((disp_chk & out_split_mask).sum())
        # The mixed (displaced ↔ split) decision is driven exclusively
        # by the strict temporal-ordering test above (mixed_window).
        # In this split branch we never re-tag as mixed; if the
        cap_last = cap_geo_window_at_recovery(
            last, evt_end, fl, nt, post=post, evt_start=evt_start)
        return dict(kind='split', window=(first, max(first, cap_last)),
                    major=major, label_start=label_start)

    # partial split
    # One 3-D component: slice b0≥2 at top (≥ 45 km) or bottom band for
    # a sustained period without a full second 3-D lobe.
    sig_part = b0_partial[inner_lo:inner_hi]
    n_part_days = int(sig_part.sum())
    n_part_inside = int(b0_partial[evt_start:min(nt, evt_end + 1)].sum())
    n_part_consec = max_consecutive_true(
        b0_partial, evt_start, min(nt - 1, evt_end))
    n_full_in_window = int(b0_full[inner_lo:inner_hi].sum())
    bc_in_evt = bot_cent[evt_start:min(nt, evt_end + 1)]
    bc_in_evt = bc_in_evt[np.isfinite(bc_in_evt)] if bc_in_evt.size else bc_in_evt
    bc_median_evt = (float(np.nanmedian(bc_in_evt))
                     if bc_in_evt.size else bc_base)
    has_comp1_evt = (int(comp1_present[evt_start:min(nt, evt_end + 1)].sum())
                     >= 2)
    # Clean-signal requirement. A real partial split is a PERSISTENT pinch in
    # one band: the bottom (or top) fraction stays high across a contiguous
    # b0_partial run, as in the canonical 0506-Dec and 1112-Mar bottom pinches
    eend = min(nt - 1, evt_end)
    bp_on = b0_partial
    bot_clean = max_consecutive_true(
        bp_on & (frac_bot_partial >= PARTIAL_BAND_FRAC), evt_start, eend)
    top_clean = max_consecutive_true(
        bp_on & (frac_top_partial >= PARTIAL_BAND_FRAC), evt_start, eend)
    tg_support = int(two_lobe[evt_start:min(nt, evt_end + 1)].sum())
    clean_pinch = (bot_clean >= 3) or (top_clean >= 3) or (tg_support >= 1)
    partial_ok = (
        n_part_days >= 3 and n_part_inside >= 1 and n_part_consec >= 3
        and n_full_in_window < 2 and clean_pinch and
        (has_comp1_evt or bc_median_evt < bc_base - 5.0)
    )
    if partial_ok:
        pidx = np.where(sig_part)[0]
        first = inner_lo + int(pidx[0])
        last  = inner_lo + int(pidx[-1])
        scan_hi = min(nt, last + 30)
        absent_run = 0
        for k in range(last + 1, scan_hi):
            if b0_partial[k]:
                last = k
                absent_run = 0
            else:
                absent_run += 1
                if absent_run >= 3:
                    break
        first = max(first, label_start)
        cap_last = cap_geo_window_at_recovery(
            last, evt_end, fl, nt, post=post, evt_start=evt_start)
        return dict(kind='partial_split',
                    window=(first, max(first, cap_last)),
                    major=major, label_start=label_start)

    # displaced detection
    # Reuse sustained_low_full + bc_base computed earlier (used by
    # the mixed temporal-ordering test).
    sustained_low = sustained_low_full
    DROP = DISP_DROP

    # In-event swing: large equatorward excursion inside the event -
    # captures "vortex flapping around" on a major SSW even if the
    # centroid is high on average.
    SWING_DEG = 8.0
    SWING_LAT = 65.0
    bc_v = np.where(np.isfinite(bot_cent), bot_cent, np.nan)
    swing_sig = np.zeros(nt, dtype=bool)
    win_swing = 7
    half = win_swing // 2
    for k in range(max(0, evt_start - 2),
                   min(nt, evt_end + post + 1)):
        a = max(0, k - half); b = min(nt, k + half + 1)
        seg = bc_v[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 3:
            sw = float(np.nanmax(seg) - np.nanmin(seg))
            if sw >= SWING_DEG and float(np.nanmin(seg)) <= SWING_LAT:
                swing_sig[k] = True

    disp_sig = sustained_low | swing_sig

    sig_d = disp_sig[win_lo:win_hi]
    if sig_d.any():
        first = win_lo + int(np.where(sig_d)[0][0])
        last  = win_lo + int(np.where(sig_d)[0][-1])

        scan_hi = min(nt, last + 90)
        ok_run = 0
        stab_run = 0
        STAB_WIN = 5; STAB_RSTD = 2.5; STAB_NEED = 4
        for k in range(last + 1, scan_hi):
            keep = bool(disp_sig[k])
            if not keep and np.isfinite(bot_cent[k]):
                keep = (bot_cent[k] < bc_base - 0.5 * DROP)
            if keep:
                last = k
                ok_run = 0
            else:
                ok_run += 1
                if ok_run >= 2:
                    break

            a = max(0, k - STAB_WIN + 1)
            seg = bc_v[a:k + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 3 and float(np.std(seg)) < STAB_RSTD:
                stab_run += 1
            else:
                stab_run = 0
            if stab_run >= STAB_NEED:
                break

        first = max(first, label_start)
        last = cap_geo_window_at_recovery(
            last, evt_end, fl, nt, post=post, evt_start=evt_start)
        last = max(first, last)
        # In this displaced branch the split signature was already
        # judged too weak to be the dominant morphology (the split
        # branch above didn't fire).  Mixed is determined exclusively
        return dict(kind='displaced', window=(first, last),
                    major=major, label_start=label_start)

    # No clear morphology associated with this warming.
    return dict(kind='none', window=None,
                major=major, label_start=label_start)


def cap_geo_window_at_recovery(last, evt_end, fl, nt, post=14, evt_start=None):
    # Clip morphology tail at the first sustained post-event recovery.
    last = int(last)
    evt_end = int(evt_end)
    phys = fl.get('event_physics') or {}
    peak = np.asarray(fl.get('peak_U', np.full(nt, np.nan)), dtype=float)
    nr = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)), dtype=float)
    evt_peak_u = evt_nr_peak = None
    if evt_start is not None:
        sl = slice(max(0, int(evt_start)), min(nt, evt_end + 1))
        if sl.stop > sl.start:
            evt_peak_u = float(np.nanmax(peak[sl]))
            evt_nr_peak = float(np.nanmax(nr[sl]))
    rec_run = 0
    cap = min(nt - 1, evt_end + max(int(post), 10))
    for k in range(evt_end + 1, cap + 1):
        if recovery_signature(k, fl, phys, ref_idx=None):
            rec_run += 1
            if rec_run >= 2:
                cap_at = max(int(evt_start or 0), k - 2)
                return min(last, cap_at)
        else:
            rec_run = 0
    return min(last, cap)


def stable_strong_mask_ev(fl, r, smooth=3):
    # Stable strong wind ring days.  Requires high peak_U AND a
    nt   = len(fl['peak_U'])
    peak = np.asarray(fl['peak_U'], dtype=float)
    band_lo = np.asarray(fl.get('band_lo', np.full(nt, np.nan)), dtype=float)
    band_hi = np.asarray(fl.get('band_hi', np.full(nt, np.nan)), dtype=float)
    n_ring = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                        dtype=float)

    # 5-day rolling-mean peak so a single low-day doesn't disqualify a
    # strong-jet stretch.
    win_p = max(3, int(smooth) + 2)
    peak_smooth = uniform_filter1d(
        np.where(np.isfinite(peak), peak, np.nan), size=win_p, mode='nearest')
    n_ring_smooth = uniform_filter1d(
        np.where(np.isfinite(n_ring), n_ring, 0.0),
        size=win_p, mode='nearest')
    strong_U = float(r.strong_peak_U)
    peak_thr_factor_A = float(getattr(r, 'stable_peak_frac', 0.85))
    peak_thr_factor_B = 0.55   # wide-altitude path threshold
    high_peak_A = (np.isfinite(peak_smooth) &
                   (peak_smooth >= peak_thr_factor_A * strong_U))
    high_peak_B = (np.isfinite(peak_smooth) &
                   (peak_smooth >= peak_thr_factor_B * strong_U))

    # Multi-level ring requirement.  Threshold of 12 levels (out of
    # ~50) is ~25%; well above the 6-10 levels that the user
    # identified as a wind-disturbed signature.  Use the 5-day
    NRING_OK = 12
    has_ring_data = bool(np.isfinite(n_ring).any())
    if has_ring_data:
        multi_ring = n_ring_smooth >= NRING_OK
    else:
        multi_ring = np.ones(nt, dtype=bool)

    def abs_smooth(a):
        ag = np.gradient(np.where(np.isfinite(a), a, np.nan))
        absag = np.where(np.isfinite(ag), np.abs(ag), 0.0)
        win = max(1, int(smooth))
        return uniform_filter1d(absag, size=win, mode='nearest')

    dpeak_s = abs_smooth(peak)
    dlo_s   = abs_smooth(band_lo)
    dhi_s   = abs_smooth(band_hi)

    PEAK_VAR_OK = 9.0   # m/s per day (smoothed |gradient|)
    BAND_VAR_OK = 6.0   # km per day  (smoothed |gradient|)
    BAND_VAR_OK_B = 8.0 # slightly looser band variability for path B

    # Path A: original stability test + ring multiplicity
    stable_A = high_peak_A & multi_ring & (
        (dpeak_s <= PEAK_VAR_OK) &
        (dlo_s   <= BAND_VAR_OK) &
        (dhi_s   <= BAND_VAR_OK))

    # Path B: wide-altitude span; band must span >= 15 km of altitude.
    # A vortex that's only present at one or two levels is not stable
    # by this path, no matter how high peak_U is.
    span = np.where(np.isfinite(band_hi) & np.isfinite(band_lo),
                    band_hi - band_lo, 0.0)
    span_smooth = uniform_filter1d(span, size=win_p, mode='nearest')
    wide_span = span_smooth >= 15.0
    stable_B = high_peak_B & wide_span & multi_ring & (
        (dpeak_s <= PEAK_VAR_OK) &
        (dlo_s   <= BAND_VAR_OK_B) &
        (dhi_s   <= BAND_VAR_OK_B))

    return stable_A | stable_B


def geo_morphology_anomaly_mask(fl, r):
    # Permissive geopotential-anomaly mask used to gate the
    nt = len(fl.get('peak_U', []))
    if nt == 0:
        return np.zeros(0, dtype=bool)

    out = np.zeros(nt, dtype=bool)
    tilt = np.abs(np.asarray(fl.get('geo_tilt', np.full(nt, np.nan)),
                              dtype=float))
    out |= np.isfinite(tilt) & (tilt >= 0.30)

    bc = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                    dtype=float)
    out |= np.isfinite(bc) & (bc <= 55.0)

    # Comp-0 aspect-ratio anomaly at top or bottom level
    lev_ar = np.asarray(fl.get('geo_lev_aspect',
                                np.full((nt, 4, 1), np.nan)),
                        dtype=float)
    if lev_ar.ndim == 3 and lev_ar.shape[1] >= 1 and lev_ar.shape[2] > 1:
        for ti in range(nt):
            col = lev_ar[ti, 0, :]
            finite = np.where(np.isfinite(col))[0]
            if not finite.size:
                continue
            kb = int(finite.max())   # bottom
            kt = int(finite.min())   # top
            if (col[kb] >= 2.0) or (col[kt] >= 2.0):
                out[ti] = True

    # comp 1 fragment present (tighter area threshold)
    c1_present = np.asarray(fl.get('comp1_present', np.zeros(nt)),
                             dtype=float) >= 0.5
    af = np.asarray(fl.get('area_fracs', np.full(nt, np.nan)), dtype=float)
    geo_alt_hi = np.asarray(fl.get('geo_alt_hi'), dtype=float)
    if geo_alt_hi.ndim == 2 and geo_alt_hi.shape[1] >= 2:
        ah1 = geo_alt_hi[:, 1]
        c1_high = np.isfinite(ah1) & (ah1 >= 30.0)
    else:
        c1_high = np.zeros(nt, dtype=bool)
    out |= (c1_present & c1_high &
            np.isfinite(af) & (af >= SPLIT_AREAFRAC))

    return out


def geo_disturbance_no_warming_mask_ev(fl, r):
    # Days with structural geopotential disturbance.  Whether the day
    nt = len(fl['geo_b0'])
    tilt = np.asarray(fl.get('geo_tilt', np.full(nt, np.nan)), dtype=float)
    tilt_abs = np.where(np.isfinite(tilt), np.abs(tilt), np.nan)
    big_tilt = np.isfinite(tilt_abs) & (tilt_abs >= float(r.geo_disturb_min_tilt))

    comp1_present = (np.asarray(fl.get('comp1_present', np.zeros(nt)),
                                dtype=float) >= 0.5)
    seeded = (np.asarray(fl.get('split_seeded', np.zeros(nt)),
                          dtype=float) >= 0.5)
    area_fracs = np.asarray(fl.get('area_fracs', np.zeros(nt)), dtype=float)

    geo_alt_hi = np.asarray(fl.get('geo_alt_hi'), dtype=float)
    geo_alt_lo = np.asarray(fl.get('geo_alt_lo'), dtype=float)
    cent_lat   = np.asarray(fl.get('comp_cent_lat'), dtype=float)
    if geo_alt_hi.ndim == 2 and geo_alt_hi.shape[1] >= 2:
        ah1 = geo_alt_hi[:, 1]
        c1_high = np.isfinite(ah1) & (ah1 >= 30.0)
    else:
        ah1 = np.full(nt, np.nan); c1_high = np.zeros(nt, dtype=bool)
    if geo_alt_lo.ndim == 2 and geo_alt_lo.shape[1] >= 2:
        al1 = geo_alt_lo[:, 1]
    else:
        al1 = np.full(nt, np.nan)
    if cent_lat.ndim == 2 and cent_lat.shape[1] >= 2:
        lat1 = cent_lat[:, 1]
    else:
        lat1 = np.full(nt, np.nan)

    # Trough: low-lat, low-alt, small comp 1; exclude from disturbance.
    trough = (
        np.isfinite(lat1) & (lat1 < 45.0) &
        np.isfinite(ah1)  & (ah1  < 25.0) &
        np.isfinite(area_fracs) & (area_fracs < 0.15)
    )

    # Real seeded comp-1 (any non-tiny area, near comp 0 in lat/lon).
    seeded_comp1 = (comp1_present & seeded &
                    np.isfinite(area_fracs) &
                    (area_fracs >= 0.05) & (~trough))

    # comp 1 reaches stratospheric altitude (structural twin).
    high_comp1 = comp1_present & c1_high & (~trough)

    # Comp-0 bottom centroid wandering: 5-day rolling stddev >= 3°.
    # Was 7-day; the 7-day window kept the signal elevated for ~7 days
    # after the actual disturbance ended, making geo-disturbance runs
    bot_cent = np.asarray(fl.get('geo_c0_bottom_cent',
                                  np.full(nt, np.nan)), dtype=float)
    bc = np.where(np.isfinite(bot_cent), bot_cent, np.nan)
    win = 5
    bc_filled = np.where(np.isfinite(bc), bc, 0.0)
    valid     = np.isfinite(bc).astype(float)
    n_valid   = uniform_filter1d(valid, size=win, mode='nearest') * win
    sum_x     = uniform_filter1d(bc_filled, size=win, mode='nearest') * win
    sum_xx    = uniform_filter1d(bc_filled**2, size=win, mode='nearest') * win
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(n_valid > 0, sum_x / np.maximum(n_valid, 1), np.nan)
        var  = np.where(n_valid > 1,
                        (sum_xx - n_valid * mean**2) / np.maximum(n_valid - 1, 1),
                        np.nan)
    rstd = np.sqrt(np.maximum(var, 0.0))
    cent_wander = np.isfinite(rstd) & (rstd >= 3.0) & (n_valid >= 3)

    # Sustained equatorward excursion: bc at least 6° below the
    # 21-day rolling 75th percentile (the "undisturbed-vortex"
    # baseline).  Catches Canadian-warming-style displacements that
    win_b = 21
    bc_drop = np.zeros(nt, dtype=bool)
    DROP_OFF = 6.0
    for k in range(nt):
        a = max(0, k - win_b + 1); b = k + 1
        seg = bc[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 7 and np.isfinite(bc[k]):
            base75 = float(np.percentile(seg, 75))
            if bc[k] < base75 - DROP_OFF:
                bc_drop[k] = True

    # Fast-response signal for very short (1-2 day) geo disturbances:
    # a single-day jump in bc of ≥ 6° equatorward, or an absolute bc
    # excursion where bc(k) is ≥ 5° below the centered 5-day median
    fast_jump = np.zeros(nt, dtype=bool)
    JUMP_DEG = 6.0   # day-to-day equatorward jump
    SHORT_DEV = 5.0  # deviation from local 5-day median
    for k in range(1, nt):
        # day-to-day jump
        if (np.isfinite(bc[k]) and np.isfinite(bc[k - 1]) and
                (bc[k - 1] - bc[k]) >= JUMP_DEG):
            fast_jump[k] = True
            if k - 1 >= 0:
                fast_jump[k - 1] = True
    for k in range(nt):
        # local-median deviation (5-day centered window, excluding k)
        a = max(0, k - 2); b = min(nt, k + 3)
        seg = np.concatenate([bc[a:k], bc[k + 1:b]])
        seg = seg[np.isfinite(seg)]
        if seg.size >= 3 and np.isfinite(bc[k]):
            loc_med = float(np.median(seg))
            if (loc_med - bc[k]) >= SHORT_DEV:
                fast_jump[k] = True

    # aspect-ratio stretch: large horizontal elongation of the lobe.
    big_asp_v = np.asarray(fl.get('big_aspect', np.full(nt, np.nan)),
                           dtype=float)
    asp_thr = float(getattr(r, 'geo_aspect_bot', 1.8))
    big_stretch = np.isfinite(big_asp_v) & (big_asp_v >= asp_thr)

    raw = (big_tilt | seeded_comp1 | high_comp1 |
           cent_wander | bc_drop | fast_jump | big_stretch)

    # co-occurrence with wind-disturbance gate
    # a geopotential disturbance only counts if there's
    # also a wind-disturbance signature on the same day at some level.
    peak_U = np.asarray(fl.get('peak_U', np.full(nt, np.nan)),  dtype=float)
    east_U = np.asarray(fl.get('east_U', np.full(nt, np.nan)),  dtype=float)
    east_intact_level = np.asarray(
        fl.get('east_intact_level', np.zeros(nt)), dtype=float)
    jet_intact = np.asarray(
        fl.get('jet_intact', np.ones(nt)), dtype=float)

    strong_U = float(getattr(r, 'strong_peak_U', 100.0))
    peak_s = uniform_filter1d(
        np.where(np.isfinite(peak_U), peak_U, strong_U), size=3, mode='nearest')
    east_s = uniform_filter1d(
        np.where(np.isfinite(east_U), np.abs(east_U), 0.0),
        size=3, mode='nearest')
    jet_s  = uniform_filter1d(
        np.where(np.isfinite(jet_intact), jet_intact, 1.0),
        size=3, mode='nearest')

    # ring-level loss: the circumpolar ring coming apart across altitudes,
    # i.e. n_ring_levels dropping well below its recent local maximum. This
    # is a primary "upper jet rings come apart" signature independent of peak
    nrl = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                     dtype=float)
    ring_loss = np.zeros(nt, dtype=bool)
    for k in range(nt):
        a = max(0, k - 5)
        seg = nrl[a:k + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2 and np.isfinite(nrl[k]):
            recent_max = float(np.max(seg))
            if recent_max >= 6 and nrl[k] <= 0.7 * recent_max:
                ring_loss[k] = True

    wind_disturb = ((peak_s < 0.70 * strong_U) |
                    (east_intact_level >= 0.5) |
                    (jet_s < 0.60) |
                    (east_s > 5.0) |
                    ring_loss)

    # Dilate the wind-disturbance gate by ±2 days so geo flags that
    # *immediately precede* a wind reversal (the tilt builds up first,
    # the easterly ring forms a few days later) still survive.
    if wind_disturb.any():
        wd = wind_disturb.copy()
        for sh in (1, 2):
            wd[sh:]   |= wind_disturb[:-sh]
            wd[:-sh]  |= wind_disturb[sh:]
        wind_disturb = wd

    raw = raw & wind_disturb

    # Dilate by 1 day on each side: a 2-day blip becomes 4 days,
    # surviving the 3-day despeckle pass downstream.
    if raw.any():
        out = raw.copy()
        out[1:] |= raw[:-1]
        out[:-1] |= raw[1:]
    else:
        out = raw

    # "returned to normal" suppression
    # a geopotential disturbance ends when the
    # centroid returns to its baseline and the winds re-establish.
    RETURN_TOL = 3.0   # degrees within baseline to count as "returned"
    fast_signal = big_tilt | seeded_comp1 | high_comp1 | fast_jump
    win_b = 21
    for k in range(nt):
        if not out[k]:
            continue
        if fast_signal[k]:
            continue
        # Require wind recovery: if the wind-disturbance gate is still
        # on, the vortex hasn't actually recovered even if the centroid
        # has drifted back to the baseline latitude.
        if wind_disturb[k]:
            continue
        a = max(0, k - win_b + 1); b = k + 1
        seg = bc[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 7 and np.isfinite(bc[k]):
            base75 = float(np.percentile(seg, 75))
            # returned: bc within tolerance of baseline or more poleward
            if bc[k] >= base75 - RETURN_TOL:
                out[k] = False

    return out


def detect_end_of_season_ev(fl, r, last_event_end, times=None):
    # End-of-season detection - winds altogether slowing down  (sustained low smoothed wind)
    nt = len(fl['peak_U'])
    if nt == 0:
        return -1

    peak  = np.asarray(fl['peak_U'], dtype=float)
    eastu = np.asarray(fl.get('east_U', np.full(nt, np.nan)), dtype=float)
    dT    = np.asarray(fl['dT_col'], dtype=float)

    w_mag = np.where(np.isfinite(peak),  np.abs(peak),  0.0)
    e_mag = np.where(np.isfinite(eastu), np.abs(eastu), 0.0)
    wind_mag = np.maximum(w_mag, e_mag)

    # Smoothing window: ~2 weeks (recovery_window_days) so transient
    # dips don't trigger end-of-season.
    win = max(7, int(getattr(r, 'recovery_window_days', 14)))
    wind_smooth = uniform_filter1d(wind_mag, size=win, mode='nearest')
    peak_smooth = uniform_filter1d(
        np.where(np.isfinite(peak), peak, 0.0), size=win, mode='nearest')
    dT0         = np.where(np.isfinite(dT), dT, 0.0)
    dT_smooth   = uniform_filter1d(dT0, size=win, mode='nearest')

    # Wind threshold: be permissive; the user's spec is "slowing down",
    # not "below a season percentile".  Anchor to strong_peak_U so a
    # uniformly strong season never qualifies, regardless of how
    wind_thr = 0.65 * float(r.strong_peak_U)
    wind_low = wind_smooth <= wind_thr

    # "winds slow" is a trend, not only an absolute
    # threshold.  Add an alternative: westerlies have been slowing
    # for a sustained period.  Compare smoothed peak against its
    peak_shift = np.full(nt, np.nan)
    if nt > win:
        peak_shift[win:] = peak_smooth[win:] - peak_smooth[:-win]
    wind_slowing = np.isfinite(peak_shift) & (peak_shift <= 0.0)
    wind_low = wind_low | wind_slowing

    # Stability + gradual warming: smoothed dT/dt is small and not
    # cooling.  We additionally require dT to be stable day-to-day
    # (low rolling stddev); the user spec specifically calls for
    dT_mag_ok = np.abs(dT_smooth) <= max(0.30,
                                          5.0 * float(r.end_dT_abs))
    dT_pos    = dT_smooth >= -float(r.end_dT_abs)

    # 7-day rolling stddev of dT_smooth.  Threshold scales with the
    # season's own dT_col scale via end_dT_abs (auto-calibrated), with
    # an absolute floor so it doesn't get unreasonably tight.
    dT_filled = np.where(np.isfinite(dT_smooth), dT_smooth, 0.0)
    dT_valid  = np.isfinite(dT_smooth).astype(float)
    win_s = 7
    n_v   = uniform_filter1d(dT_valid, size=win_s, mode='nearest') * win_s
    s_x   = uniform_filter1d(dT_filled, size=win_s, mode='nearest') * win_s
    s_xx  = uniform_filter1d(dT_filled**2, size=win_s, mode='nearest') * win_s
    with np.errstate(invalid='ignore', divide='ignore'):
        m  = np.where(n_v > 0, s_x / np.maximum(n_v, 1), np.nan)
        v  = np.where(n_v > 1,
                      (s_xx - n_v * m**2) / np.maximum(n_v - 1, 1),
                      np.nan)
    dT_rstd = np.sqrt(np.maximum(v, 0.0))
    dT_stable_thr = max(0.50, 4.0 * float(r.end_dT_abs))
    dT_stable = np.isfinite(dT_rstd) & (dT_rstd <= dT_stable_thr)

    settle = wind_low & dT_mag_ok

    # "winds slow, temperature change calms" is the
    # spec for EOS.  We do not require dT_smooth >= 0; the late-
    # season transition includes brief periods of mild cooling that

    # April-mode (relaxed) settle
    # when we reach April with the westerlies slowing
    # and weak easterlies forming, that is the end of season.  Don't
    nr_eos = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                        dtype=float)
    bc_eos = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                        dtype=float)
    nr_pos_eos = nr_eos[np.isfinite(nr_eos) & (nr_eos > 0)]
    half_ring_eos = (0.5 * float(np.percentile(nr_pos_eos, 95))
                     if nr_pos_eos.size else 8.0)
    strong_ring = (np.isfinite(nr_eos) & (nr_eos >= half_ring_eos) &
                   np.isfinite(bc_eos) & (bc_eos >= 70.0))
    recon_look = 14
    recon_ahead = np.zeros(nt, dtype=bool)
    for k in range(nt):
        seg = strong_ring[k:min(nt, k + recon_look + 1)].astype(float)
        recon_ahead[k] = seg.size > 0 and float(np.mean(seg)) >= 0.5
    apr_eos_day = ((peak_smooth <= 0.70 * float(r.strong_peak_U)) &
                   (dT_smooth >= -0.5) & ~recon_ahead & ~strong_ring)

    # A genuine >=25 K warming ahead means the season is not over. Build a
    # per-day "warming within the next ~14 days" mask so EOS does not start
    # just before a real warming (8081: ring coverage dropped late Feb but
    w7_eos = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)), dtype=float)
    thr_eos = float(getattr(r, 'ssw_warm25_K', 25.0))
    warm_ahead = np.zeros(nt, dtype=bool)
    look = 14
    for k in range(nt):
        hi = min(nt, k + look + 1)
        seg = w7_eos[k:hi]
        warm_ahead[k] = bool(np.any(np.isfinite(seg) &
                                     (np.round(seg, 2) >= thr_eos)))

    if last_event_end >= 0:
        scan_start = last_event_end + int(r.end_post_warming_days)
    else:
        scan_start = nt // 3
    if scan_start >= nt - int(r.min_end_run_days):
        scan_start = max(0, nt // 2)

    # Pass 1: relaxed detection. EOS begins once the westerlies have slowed
    # and the temperature change has calmed, past the last warming event,
    # in any month. A 3-day run is required (relaxed from min_end_run_days).
    i = last_event_end + 1 if last_event_end >= 0 else scan_start
    while i < nt:
        if apr_eos_day[i] and not warm_ahead[i]:
            j = i
            while j < nt and apr_eos_day[j] and not warm_ahead[j]:
                j += 1
            if (j - i) >= 3:
                return i
            i = j
        else:
            i += 1

    # Pass 2: regular (strict) settle detection
    i = scan_start
    while i < nt:
        if settle[i] and not warm_ahead[i]:
            j = i
            while j < nt and settle[j] and not warm_ahead[j]:
                j += 1
            if (j - i) >= int(r.min_end_run_days):
                return i
            i = j
        else:
            i += 1
    return -1


# Top-level entry: per-day states + events DataFrame + flag arrays.
# `rules` can be StateRules, a dict override, or None (full auto-calibration).
# `mode` selects the warming-event trigger:
def classify_season(ds, rules=None, smooth=3, mode='cum25',
                    t_bin_lat_group=2, t_bin_lon_group=1,
                    t_bin_alt_group=1, t_bin_alt_nlevels=None,
                    t_bin_alt_km=4.0):
    r = rules if isinstance(rules, StateRules) else \
        (StateRules(**rules) if isinstance(rules, dict) else StateRules())

    fl = compute_flags(ds, smooth=smooth,
                       warming_alt_lo=r.warming_alt_lo_km,
                       warming_alt_hi=r.warming_alt_hi_km,
                       t_bin_lat_group=t_bin_lat_group,
                       t_bin_lon_group=t_bin_lon_group,
                       t_bin_alt_group=t_bin_alt_group,
                       t_bin_alt_nlevels=t_bin_alt_nlevels,
                       t_bin_alt_km=t_bin_alt_km)
    r  = calibrate_rules(fl, r)
    fl['event_physics'] = train_event_physics(fl, r)

    nt = ds.sizes['time']
    times = ds['time'].values
    states = states_from_flags(fl, r, nt, mode=mode, times=times)
    states = finalize_classified_states(states, fl, r)

    return dict(
        states=states,
        times=times,
        flags=fl,
        rules=asdict(r),
        events=events_dataframe(states, times, fl=fl, rules=r),
        mode=mode,
        ds=ds,
        t_bin_lat_group=t_bin_lat_group,
        t_bin_lon_group=t_bin_lon_group,
        t_bin_alt_group=t_bin_alt_group,
        t_bin_alt_nlevels=t_bin_alt_nlevels,
        t_bin_alt_km=t_bin_alt_km,
        event_physics=fl['event_physics'],
    )


def classify_season_alt_grid(ds, alt_kms=None, **kwargs):
    # Run ``classify_season`` for each fixed altitude bin width (km).
    if alt_kms is None:
        alt_kms = alt_km_bin_options()
    out = {}
    for km in alt_kms:
        key = float(km)
        try:
            kw = dict(kwargs)
            kw['t_bin_alt_km'] = float(km)
            kw.pop('t_bin_alt_nlevels', None)
            out[key] = classify_season(ds, **kw)
        except Exception as exc:
            print(f"[classify_season_alt_grid] km={km} failed: {exc}")
    return out


def classify_season_tbin_grid(ds, **kwargs):
    # Run ``classify_season`` for each fixed 3-D bin grouping scheme.
    out = {}
    for spec, lbl in tbin_scheme_combinations():
        lg = int(spec['lat_group'])
        mg = int(spec['lon_group'])
        km = float(spec['alt_km'])
        key = (lg, mg, km)
        try:
            cs_kw = dict(t_bin_lat_group=lg, t_bin_lon_group=mg,
                         t_bin_alt_km=km, **kwargs)
            cs_kw.pop('t_bin_alt_nlevels', None)
            out[key] = classify_season(ds, **cs_kw)
        except Exception as exc:
            print(f"[classify_season_tbin_grid] {key} failed: {exc}")
    return out


def states_from_flags(fl, r, nt, mode='cum25', times=None):
    # Full state-painting pipeline starting from compute_flags
    states = np.full(nt, STATE_STRONG, dtype=np.int16)

    # 1. Detect warming events
    events = detect_warming_events_ev(fl, r, gap_tol=10, mode=mode, times=times)

    # post-pass EOS demotion: drop any event whose
    # entire envelope sits in the end-of-season regime (late winter
    # / early spring, gradual warming with no sharp single-day pulse).
    if events:
        kept = []
        for evt in events:
            ev_lo, ev_hi = evt
            all_eos = True
            for k in range(ev_lo, ev_hi + 1):
                if not is_eos_regime(k, times, fl, r):
                    all_eos = False
                    break
            if not all_eos:
                kept.append(evt)
        events = kept

    # mark which days fall inside any warming event window for later rules
    in_warming_evt = np.zeros(nt, dtype=bool)

    # Pre-compute classification for every event so we can enforce
    # gap-between-events: between any two warming
    # events there must be at least min_run_length days of
    cls_list = []
    for idx_e, e in enumerate(events):
        nxt = events[idx_e + 1][0] if (idx_e + 1) < len(events) else None
        cls_list.append(classify_event_ev(e, fl, r, nt, next_evt_start=nxt))

    # `evt_label_start[i]` is the first day to be painted with event
    # i's warming code.  We use this to cap the previous event's
    # morphology window.
    evt_label_start = []
    for evt, cls in zip(events, cls_list):
        evt_label_start.append(int(cls.get('label_start', evt[0])))

    # between any two warming events there must be a
    # period of weak recovery (or strong vortex / EOS); never a direct
    # warming-to-warming handoff.  We require at least
    min_gap = max(3, int(r.min_run_length) + 1)

    # Hard cap on how far any event can extend in time; past
    # April + a small buffer is EOS territory and cannot be a warming.
    april_cap = april_cap_index(times, r, nt)

    # Iterate events in chronological order, applying per-event labels
    # and capping each event's labeled span to leave a `min_gap` buffer
    # before the next event's label_start.  Records each event's
    painted_spans = []   # list of (lo, hi, code)
    for idx_evt, (evt, cls) in enumerate(zip(events, cls_list)):
        evt_start, evt_end = evt
        kind = cls['kind']; major = bool(cls['major'])
        label_start = int(cls.get('label_start', evt_start))

        # cap = last day this event is allowed to occupy.  The cap
        # is the tighter of:
        # • the next event's label_start minus min_gap (so a
        if idx_evt + 1 < len(events):
            cap = evt_label_start[idx_evt + 1] - min_gap - 1
        else:
            cap = nt - 1
        cap = min(cap, april_cap)
        # Don't let cap force the event to vanish; it must at least
        # cover label_start so the warming itself is labeled.
        cap = max(cap, label_start)

        if kind == 'split':
            code = STATE_SSW_SPLIT if major else STATE_SSW_SPLIT_MIN
        elif kind == 'partial_split':
            code = (STATE_SSW_PARTIAL_SPLIT if major
                    else STATE_SSW_PARTIAL_SPLIT_MIN)
        elif kind == 'displaced':
            code = STATE_SSW_DISPLACED if major else STATE_SSW_DISPLACED_MIN
        elif kind == 'mixed' or kind == 'ambiguous':
            # mixed has both major and minor variants
            # (state 13 / 14), decided by the same "intact easterly
            # ring during the event" rule as split / displaced.
            code = STATE_SSW_MIXED_MAJ if major else STATE_SSW_MIXED_MIN
        else:
            code = STATE_WARM_NO_GEO

        evt_end_painted = min(evt_end, cap)
        if cls['window'] is not None:
            g0, g1 = cls['window']
            span_lo = min(label_start, g0)
            span_hi = min(max(evt_end, g1), cap)
        else:
            span_lo = label_start
            span_hi = evt_end_painted

        # trim trailing days when wind+geo recovery fires; jet
        # strengthening and geopotential returning end the morphology
        # tail even if warm7 is still elevated residually.
        phys_trim = fl.get('event_physics') or {}
        peak_trim = np.asarray(fl.get('peak_U', np.full(nt, np.nan)), dtype=float)
        nr_trim = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                             dtype=float)
        evt_sl = slice(max(0, evt_start), min(nt, evt_end + 1))
        evt_peak_u = float(np.nanmax(peak_trim[evt_sl]))
        evt_nr_peak = float(np.nanmax(nr_trim[evt_sl]))
        ssw_thr = float(getattr(r, "ssw_warm25_K", 25.0))
        w7_trim = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)), dtype=float)
        trig_trim = sustained_warming_trigger_mask(w7_trim, ssw_thr)

        def is_recovering_back(k):
            if wind_geo_recovery(k, fl, phys_trim, ref_idx=None, full=True):
                return True
            if recovery_stable(k, fl, phys_trim, ref_idx=None,
                                thr=ssw_thr, trig=trig_trim):
                return True
            return jet_collapse_recovery(
                k, fl, phys_trim,
                evt_peak_u=evt_peak_u, evt_nr_peak=evt_nr_peak,
                evt_start=evt_start)

        # Don't trim the event below evt_end (the last 25 K day),
        # but also never extend past the cap (next event buffer +
        # April cap).  trim_floor = the latest day we are forced to
        trim_floor = max(label_start, min(evt_end, cap))
        # Walk back from span_hi while we keep seeing recovery
        # signatures (wind strengthening or cooling+rings).  Trim
        # those days off.  Stops the morphology tail from squatting
        k = span_hi
        last_kept = span_hi
        while k > trim_floor and is_recovering_back(k):
            last_kept = k - 1
            k -= 1
        span_hi = max(last_kept, trim_floor)
        # And never exceed the cap.
        span_hi = min(span_hi, cap)
        # And never extend the morphology window into the EOS regime:
        # the seasonal end has begun there, so the warming label cannot
        # paint over it (8384 03/10-16: vortex U~25-30, bc dropping
        if times is not None:
            for kk in range(trim_floor + 1, span_hi + 1):
                if is_eos_regime(kk, times, fl, r):
                    span_hi = max(trim_floor, kk - 1)
                    break

        # Paint label across the (possibly trimmed) span
        for k in range(label_start, min(evt_end_painted, span_hi) + 1):
            if 0 <= k < nt:
                states[k] = code
                in_warming_evt[k] = True
        for k in range(span_lo, span_hi + 1):
            if 0 <= k < nt:
                in_warming_evt[k] = True
                cur = int(states[k])
                if cur in (STATE_STRONG,
                           STATE_GEO_DISTURBED, STATE_WARM_NO_GEO):
                    states[k] = code
        painted_spans.append((span_lo, span_hi, code))

    # Latest day touched by any warming-related label.
    last_event_end = max([s[1] for s in painted_spans], default=-1)

    # 2. End of season detection (run before recovery so the
    # post-last-event gap knows where to stop)
    eos_start = detect_end_of_season_ev(fl, r, last_event_end, times=times)
    if eos_start >= 0:
        states[eos_start:] = STATE_END

    # 3. Recovery painting (between every pair of events, and
    # after the last event)
    # between two warming events there must be at least
    is_inter_event_gap = np.zeros(nt, dtype=bool)
    if len(painted_spans) > 0:
        peak = np.asarray(fl['peak_U'], dtype=float)
        n_ring = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                            dtype=float)
        w7_gap = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)),
                            dtype=float)
        phys_gap = fl.get('event_physics') or {}
        ssw_thr = float(getattr(r, "ssw_warm25_K", 25.0))
        trig_gap = sustained_warming_trigger_mask(w7_gap, ssw_thr)
        peak_smooth = uniform_filter1d(
            np.where(np.isfinite(peak), peak, 0.0), size=5, mode='nearest')
        n_ring_smooth = uniform_filter1d(
            np.where(np.isfinite(n_ring), n_ring, 0.0), size=5,
            mode='nearest')
        recover_strong_thr = 0.85 * float(r.strong_peak_U)
        recover_ring_thr   = 12.0

        min_floor_recovery = max(2, int(r.min_run_length))
        for idx_evt in range(len(painted_spans)):
            gap_lo = painted_spans[idx_evt][1] + 1
            if idx_evt + 1 < len(painted_spans):
                gap_hi = painted_spans[idx_evt + 1][0]  # exclusive
                is_inter = True
            else:
                gap_hi = eos_start if eos_start >= 0 else nt
                is_inter = False
            if gap_lo >= gap_hi:
                continue

            # A post-event day with an intact easterly ring and no westerly
            # ring is still disturbed; easterly rings are not recovery. Keep
            # such leading days as the prior event's disturbance state and
            eint = np.asarray(fl.get('east_intact_level', np.zeros(nt)),
                               dtype=float)
            nrw = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                              dtype=float)
            prev_state = int(states[painted_spans[idx_evt][1]])
            rec_start = gap_lo

            def easterly_clear_sustained(k0, need=3):
                # True if >= `need` consecutive days from k0 have no intact
                # easterly ring (the disturbance has genuinely cleared).
                cnt = 0
                for m in range(k0, min(gap_hi, k0 + need)):
                    if eint[m] >= 1.0:
                        return False
                    cnt += 1
                return cnt >= min(need, gap_hi - k0)

            while rec_start < gap_hi:
                still_easterly = (eint[rec_start] >= 1.0 and
                                  (not np.isfinite(nrw[rec_start]) or
                                   nrw[rec_start] <= 0.0))
                # a lone clear day amid ongoing easterly intermittency is not
                # recovery; require the easterly to clear for >= 3 days.
                if still_easterly or not easterly_clear_sustained(rec_start):
                    if int(states[rec_start]) in (
                            STATE_STRONG,
                            STATE_GEO_DISTURBED, STATE_RECOVERING):
                        states[rec_start] = prev_state
                    rec_start += 1
                else:
                    break
            gap_lo = rec_start
            if gap_lo >= gap_hi:
                continue

            # Walk forward from the end of the previous warming.
            # Weak recovery = winds rising from the warming low (a short
            # positive trend), even if they never regain full strength.
            strength = np.asarray(fl.get('region_speed',
                                         np.full(nt, np.nan)), dtype=float)
            if not np.isfinite(strength).any():
                strength = np.asarray(fl.get('mean_U', np.full(nt, np.nan)),
                                      dtype=float)
            strength_smooth = uniform_filter1d(
                np.where(np.isfinite(strength), strength, 0.0),
                size=5, mode='nearest')
            sf = strength[np.isfinite(strength)]
            strong_ref = float(np.percentile(sf, 75)) if sf.size else \
                float(r.strong_peak_U)
            strong_strength_thr = 0.70 * strong_ref
            # ring extent required for strong: at least half the altitudes
            # the vortex covers at full extent. Reference = the season's peak
            # ring coverage (robust 98th pct of the unsmoothed count), halved.
            nr_raw = np.asarray(fl.get('n_ring_levels',
                                       np.full(nt, np.nan)), dtype=float)
            nr_pos = nr_raw[np.isfinite(nr_raw) & (nr_raw > 0)]
            strong_ring_ref = (float(np.percentile(nr_pos, 98))
                               if nr_pos.size else recover_ring_thr)
            half_ring_thr = 0.5 * strong_ring_ref
            recover_end = gap_hi
            run = 0
            for k in range(gap_lo, gap_hi):
                ring_ok = ((not np.isfinite(n_ring_smooth[k])) or
                           n_ring_smooth[k] >= half_ring_thr)
                strong_now = (
                    np.isfinite(strength_smooth[k]) and
                    strength_smooth[k] >= strong_strength_thr and
                    ring_ok)
                # collapsed ring (below half the altitudes) is never strong,
                # regardless of strength.
                if (np.isfinite(n_ring_smooth[k]) and
                        n_ring_smooth[k] < half_ring_thr):
                    strong_now = False
                if strong_now:
                    run += 1
                    if run >= 3:
                        recover_end = k - run + 1
                        break
                else:
                    run = 0

            if is_inter:
                # between two warming events there
                # must be at least a minimal recovery buffer so the
                # output never shows two warmings adjacent.  Force
                recover_end = max(recover_end,
                                   min(gap_lo + min_floor_recovery + 1,
                                       gap_hi))
                # Mark the whole inter-event gap as "no canonical 25K"
                # so an isolated w7 echo in the strong-mid portion
                # can't accidentally become a WARM_NO_GEO day.
                is_inter_event_gap[gap_lo:gap_hi] = True

            for k in range(gap_lo, recover_end):
                cur = int(states[k])
                # Don't override warming events painted earlier or EOS;
                # everything else (strong / wind / geo / WARM_NO_GEO)
                # becomes recovering while jet+geo are reforming.
                if cur in (STATE_STRONG,
                           STATE_GEO_DISTURBED, STATE_WARM_NO_GEO):
                    states[k] = STATE_RECOVERING

            # Hierarchy rule: a warming must be followed by weak recovery,
            # a strong vortex, or end-of-season; never a geo disturbance.
            # Any geo-disturbed day remaining anywhere in the post-warming
            for k in range(gap_lo, gap_hi):
                if int(states[k]) == STATE_GEO_DISTURBED:
                    states[k] = (STATE_RECOVERING if k < recover_end
                                 else STATE_STRONG)

    # 4. Canonical 25 K day enforcement (run after recovery)
    # Warming is top of the hierarchy: a day reaching 25 K (warm7_max) that
    # the event detector did not already fold into a warming event is claimed
    w7_pre = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)),
                        dtype=float)
    thr_pre = float(getattr(r, "ssw_warm25_K", 25.0))
    warm_day_re = sustained_warming_trigger_mask(w7_pre, thr_pre)
    # vortex-reconvened gate (same as the detector): a re-formed ring across
    # half the column means the cold cap is back; not a warming day.
    nrt = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                      dtype=float)
    bct = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                      dtype=float)
    nrf = nrt[np.isfinite(nrt) & (nrt > 0)]
    if nrf.size:
        half = 0.5 * float(np.percentile(nrf, 98))
        # Use a smoothed ring for the reconvene block so a brief 1-2 day ring
        # dip inside an otherwise strongly-ringed vortex does not open the
        # reclaim (which bridging/despeckle would then smear into a long
        nr_sm_re = uniform_filter1d(
            np.where(np.isfinite(nrt), nrt, 0.0), size=5, mode='nearest')
        warm_day_re = warm_day_re & ~(nr_sm_re >= half)
        # Polar-cap-exists gate (same as the detector): a warming reclaim
        # cannot fire on a day with no recent reconvened vortex (no cold cap
        # to warm). 8485 02/04-16: warm7>=25 but n_ring=0, bc never
        ring_hi = np.isfinite(nrt) & (nrt >= half)
        ring_recent = np.zeros(nt, dtype=bool)
        for kkx in range(nt):
            lox = max(0, kkx - 3)
            ring_recent[kkx] = bool(np.any(ring_hi[lox:kkx + 1]))
        geo_ctr = np.isfinite(bct) & (bct >= 70.0)
        reconv_day = ring_recent & geo_ctr
        reconvened = np.zeros(nt, dtype=bool)
        runx = 0
        for kkx in range(nt):
            if reconv_day[kkx]:
                runx += 1
                if runx >= 2:
                    reconvened[kkx - 1:kkx + 1] = True
            else:
                runx = 0
        cap_lookback = 21
        cap_exists = np.zeros(nt, dtype=bool)
        for kkx in range(nt):
            lox = max(0, kkx - cap_lookback)
            cap_exists[kkx] = bool(np.any(reconvened[lox:kkx + 1]))
        warm_day_re = warm_day_re & cap_exists
        # EOS gate: do not reclaim a day as warming when the vortex is in
        # the late-season EOS regime (winds slowed below 0.70x strong and
        # temperature change calmed, from March onward). At that weakness
        if times is not None:
            for kkx in range(nt):
                if warm_day_re[kkx] and is_eos_regime(kkx, times, fl, r):
                    warm_day_re[kkx] = False
    for k in range(nt):
        if not warm_day_re[k]:
            continue
        cur = int(states[k])
        if cur in WARMING_STATES or cur == STATE_END:
            continue
        if not onset_month_ok(k, times, r):
            in_span = any(lo <= k <= hi for lo, hi, _ in painted_spans)
            if not in_span:
                continue
        states[k] = STATE_WARM_NO_GEO
        in_warming_evt[k] = True

    # 5. Geo disturbance without warming
    # Geo disturbance is detected only outside warming periods. Warming is at
    # the top of the hierarchy: any day inside a warming event span belongs to
    geo_dist = geo_disturbance_no_warming_mask_ev(fl, r)
    phys_geo = fl.get('event_physics') or {}
    w7_guard = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)),
                          dtype=float)
    warm_day = np.isfinite(w7_guard) & (np.round(w7_guard, 2) >=
                                        float(r.ssw_warm25_K))
    warming_codes = (STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
                     STATE_SSW_SPLIT,     STATE_SSW_SPLIT_MIN,
                     STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
                     STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN,
                     STATE_WARM_NO_GEO)
    # days inside any detected warming event span (warming periods are closed)
    in_warming_period = np.zeros(nt, dtype=bool)
    for lo, hi, flav in painted_spans:
        in_warming_period[max(0, lo):min(nt, hi + 1)] = True
    # half-column ring threshold for "vortex reconvened" check
    nr_geo = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                         dtype=float)
    bc_geo = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                         dtype=float)
    nrpos = nr_geo[np.isfinite(nr_geo) & (nr_geo > 0)]
    half_geo = (0.5 * float(np.percentile(nrpos, 98))
                 if nrpos.size else 8.0)
    full_geo = (float(np.percentile(nrpos, 98))
                 if nrpos.size else 16.0)

    def bc_rising(k, lookback=5):
        # bc has risen significantly over the recent days (geo centering)
        lo = max(0, k - lookback)
        recent = bc_geo[lo:k + 1]
        recent = recent[np.isfinite(recent)]
        if recent.size < 2:
            return False
        return bool(recent[-1] - np.min(recent) >= 5.0)

    for k in range(nt):
        if warm_day[k] or in_warming_period[k]:
            continue  # warming period; geo disturbance is subsumed
        if geo_dist[k] and (int(states[k]) not in warming_codes):
            if int(states[k]) == STATE_END:
                continue
            # Vortex has reconvened (do not paint geo) when rings are at
            # half-column or more and the geopotential is either centered
            # or actively centering (bc rising toward poleward). Three
            r_now = nr_geo[k] if np.isfinite(nr_geo[k]) else 0.0
            b_now = bc_geo[k] if np.isfinite(bc_geo[k]) else 0.0
            clean = (r_now >= half_geo) and (b_now >= 70.0)
            strong_rings = (r_now >= 0.8 * full_geo) and (b_now >= 60.0)
            centering = (r_now >= half_geo and b_now >= 60.0
                         and bc_rising(k))
            if clean or strong_rings or centering:
                states[k] = STATE_STRONG
                continue
            states[k] = STATE_GEO_DISTURBED

    # 6. Stable strong vs wind-disturbed for remaining strong days
    # "wind disturbances really aren't that interesting
    # on their own".  We only label a day as state 3 (wind disturbed)
    stable = stable_strong_mask_ev(fl, r)
    geo_hint = geo_morphology_anomaly_mask(fl, r)
    # half-of-full-coverage ring threshold on the smoothed ring count, so a
    # lone single-day ring spike during a displacement does not read strong.
    nr_all = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                         dtype=float)
    nr_sm = uniform_filter1d(np.where(np.isfinite(nr_all), nr_all, 0.0),
                              size=5, mode='nearest')
    nrposx = nr_sm[nr_sm > 0]
    strong_half_ring = (0.5 * float(np.percentile(nrposx, 95))
                         if nrposx.size else 8.0)
    strong_full_ring = (float(np.percentile(nrposx, 95))
                         if nrposx.size else 16.0)
    bc_all = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                         dtype=float)

    def bc_rising_s6(k, lookback=5):
        lo = max(0, k - lookback)
        recent = bc_all[lo:k + 1]
        recent = recent[np.isfinite(recent)]
        if recent.size < 2:
            return False
        return bool(recent[-1] - np.min(recent) >= 5.0)

    ea_all = np.asarray(fl.get('east_active', np.zeros(nt)), dtype=float)
    eint_all = np.asarray(fl.get('east_intact_level', np.zeros(nt)),
                           dtype=float)
    rev_all = np.asarray(fl.get('rev_lev_count', np.zeros(nt)), dtype=float)

    pk6 = np.asarray(fl['peak_U'], dtype=float)
    pk6_sm = uniform_filter1d(np.where(np.isfinite(pk6), pk6, 0.0),
                               size=5, mode='nearest')
    strong_wind6 = pk6_sm >= 0.55 * float(r.strong_peak_U)
    for k in range(nt):
        if int(states[k]) == STATE_STRONG:
            r_raw = nr_all[k] if np.isfinite(nr_all[k]) else 0.0
            b_raw = bc_all[k] if np.isfinite(bc_all[k]) else 0.0
            # hard block: a strong vortex requires an actual westerly ring
            # and no easterly reversal. No raw westerly ring, or an intact
            # easterly ring / reversed wind, disqualifies strong regardless
            easterly_present = (eint_all[k] >= 1.0)
            if r_raw <= 0.0 or easterly_present:
                states[k] = STATE_RECOVERING
                continue
            # wind floor: rings can be present but slow at recovery / end of
            # season; a strong vortex needs fast wind too (8384 03-08/09:
            # reformed 29-level centered ring but peak only ~37% of strong).
            if not strong_wind6[k]:
                states[k] = STATE_RECOVERING
                continue
            # ring coverage gate (matches step 5 promotion logic):
            # (a) smoothed n_ring exceeds the threshold (sustained), or
            # (b) raw rings strongly high (>= 0.8 x full) and bc >= 60, or
            smoothed_ok = nr_sm[k] >= strong_half_ring
            strong_rings_ok = (r_raw >= 0.8 * strong_full_ring and
                               b_raw >= 60.0)
            centering_ok = (r_raw >= strong_half_ring and b_raw >= 60.0
                            and bc_rising_s6(k))
            if not (smoothed_ok or strong_rings_ok or centering_ok):
                states[k] = STATE_RECOVERING
                continue
            # ring gate passed -> this is the strong vortex.

    # 7. Early-season prefix (bridge/separation run in classify_season)
    states = force_early_season_prefix(states, r)
    states = despeckle(states, r.min_run_length)

    # 7b. Surface warming-no-geo on strong-but-warming days
    # A strong day carrying a genuine >=25K warming (warm7, reset-clamped)
    # with an intact cap (rings + centered geo) is a "warming, no geo
    states = np.asarray(states)
    w7_s = np.asarray(fl.get('warm7_max', np.full(nt, np.nan)), dtype=float)
    nr_s = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                       dtype=float)
    bc_s = np.asarray(fl.get('geo_c0_bottom_cent', np.full(nt, np.nan)),
                       dtype=float)
    thr_s = float(getattr(r, 'ssw_warm25_K', 25.0))
    nrp_s = nr_s[np.isfinite(nr_s) & (nr_s > 0)]
    half_s = (0.5 * float(np.percentile(nrp_s, 98)) if nrp_s.size else 8.0)
    # Hierarchy: temperature triggers the warming first. A genuine >=25K
    # warming day surfaces as warm-no-geo even when a westerly ring is
    # present. The warming is ended by a sustained reconvene downstream (a
    for k in range(nt):
        if (int(states[k]) == STATE_STRONG and
                np.isfinite(w7_s[k]) and round(float(w7_s[k]), 2) >= thr_s and
                np.isfinite(nr_s[k]) and nr_s[k] > 0.0 and
                np.isfinite(bc_s[k]) and bc_s[k] >= 70.0):
            states[k] = STATE_WARM_NO_GEO
    states = despeckle(states, r.min_run_length)

    # End a warm-no-geo day once the vortex has sustainedly reconvened, but
    # only if the ring actually broke (dropped below half-col) during or just
    # before the warming; i.e. there was a real disruption to recover from.
    nr_sm_end = uniform_filter1d(
        np.where(np.isfinite(nr_s), nr_s, 0.0), size=5, mode='nearest')
    broke = nr_sm_end < half_s          # ring below half-col (disrupted)
    reconv_sustained = np.zeros(nt, dtype=bool)
    runx = 0
    for k in range(nt):
        day_reconv = (nr_sm_end[k] >= half_s and
                      np.isfinite(bc_s[k]) and bc_s[k] >= 70.0)
        if day_reconv:
            runx += 1
            if runx >= 3:
                reconv_sustained[k - 2:k + 1] = True
        else:
            runx = 0
    # mark, for each day, whether the ring broke within the trailing 10 days
    broke_recent = np.zeros(nt, dtype=bool)
    for k in range(nt):
        lo = max(0, k - 10)
        broke_recent[k] = bool(np.any(broke[lo:k + 1]))
    for k in range(nt):
        if (int(states[k]) == STATE_WARM_NO_GEO and reconv_sustained[k]
                and broke_recent[k]):
            states[k] = STATE_STRONG
    states = despeckle(states, r.min_run_length)

    # 8. Final hierarchy guard: a warming is followed by weak
    states = np.asarray(states)
    nr_g = np.asarray(fl.get('n_ring_levels', np.full(nt, np.nan)),
                       dtype=float)
    nr_gsm = uniform_filter1d(np.where(np.isfinite(nr_g), nr_g, 0.0),
                               size=5, mode='nearest')
    nr_gp = nr_gsm[nr_gsm > 0]
    g_half = (0.5 * float(np.percentile(nr_gp, 95))
               if nr_gp.size else 8.0)
    # A strong vortex also needs fast wind, not just reformed rings: rings can
    # be present but slow at recovery/end of season. Require the smoothed peak
    # to be a reasonable fraction of the season's strong reference.
    pk = np.asarray(fl['peak_U'], dtype=float)
    pk_sm = uniform_filter1d(np.where(np.isfinite(pk), pk, 0.0),
                              size=5, mode='nearest')
    strong_wind = pk_sm >= 0.55 * float(r.strong_peak_U)
    seen_warm = False
    for k in range(len(states)):
        sc = int(states[k])
        if sc == STATE_END:
            seen_warm = False          # EOS resets; new season segment
        elif sc == STATE_STRONG:
            seen_warm = False          # vortex fully recovered; prior
            #                            warming resolved. A later weak
            #                            recovery needs a new warming.
        elif sc in WARMING_STATES:
            seen_warm = True
        elif seen_warm and sc in (STATE_GEO_DISTURBED, STATE_EARLY):
            # After a warming: only weak recovery, strong vortex, or EOS.
            # Neither a geo disturbance nor early season can follow a warming;
            # demote to strong only if the ring has genuinely reconvened (no
            ring_ok2 = (nr_gsm[k] >= g_half and strong_wind[k] and
                         not (eint_all[k] >= 1.0) and
                         (nr_all[k] if np.isfinite(nr_all[k]) else 0.0) > 0)
            states[k] = (STATE_STRONG if ring_ok2
                         else STATE_RECOVERING)
        elif (not seen_warm) and sc == STATE_RECOVERING:
            # A weak recovery can only follow a warming. A recovering day
            # with no preceding unresolved warming (since the last EOS /
            # strong vortex / season start) is invalid; you don't recover
            ringokx = (nr_gsm[k] >= g_half and strong_wind[k] and
                        not (eint_all[k] >= 1.0) and
                        (nr_all[k] if np.isfinite(nr_all[k]) else 0.0) > 0)
            states[k] = (STATE_STRONG if ringokx
                         else STATE_GEO_DISTURBED)
    states = despeckle(states, r.min_run_length)
    return states


# Contiguous event table with integer indices (avoids cftime casting).
def events_dataframe(states, times, fl=None, rules=None):
    # Build the contiguous event table.  When `fl` is provided, also
    rows = []; n = len(states); i = 0
    warm_codes = (
        STATE_WARM_NO_GEO, STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
        STATE_SSW_SPLIT,   STATE_SSW_SPLIT_MIN,
        STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
        STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN,
    )
    if fl is not None:
        w7 = np.asarray(fl.get('warm7_max', []), dtype=float)
        w_alt = np.asarray(fl.get('warm7_alt_km', []), dtype=float)
        w_lat = np.asarray(fl.get('warm7_lat', []), dtype=float)
        w_d = np.asarray(fl.get('warm7_lookback_d', []), dtype=int)
    else:
        w7 = np.full(n, np.nan)
        w_alt = np.full(n, np.nan)
        w_lat = np.full(n, np.nan)
        w_d = np.full(n, -1, dtype=int)
    ssw_thr = float(getattr(rules, 'ssw_warm25_K', 25.0)) if rules else 25.0
    if fl is not None and rules is not None:
        try:
            gd_type = geo_disturbance_type(fl, rules)
        except Exception:
            gd_type = np.zeros(n, dtype=np.int16)
    else:
        gd_type = np.zeros(n, dtype=np.int16)
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        st = int(states[i])
        onset_idx = -1
        if st in warm_codes and w7.size == n:
            seg = w7[i:j]
            # first day that reaches 25 K
            ix = np.where(np.isfinite(seg) & (seg >= ssw_thr))[0]
            if ix.size:
                onset_idx = int(i + ix[0])
            else:
                # fallback: peak day inside the run
                if np.isfinite(seg).any():
                    onset_idx = int(i + int(np.nanargmax(seg)))
        elif st == STATE_GEO_DISTURBED:
            onset_idx = int(i)
        gd_label = ''
        if st == STATE_GEO_DISTURBED:
            seg_types = gd_type[i:j]
            # Every geo disturbance gets a label by PRIORITY (not a plurality
            # vote): filamentation (1) > stretching (3) > tilting (2). A single
            # day where a piece actually comes off (filamentation) outranks a
            if seg_types.size:
                present = set(int(v) for v in np.unique(seg_types))
                for code in (1, 3, 2):
                    if code in present:
                        gd_label = GEO_DIST_TYPE_NAME.get(code, '')
                        break
            if not gd_label:
                # incipient disturbance: nothing crossed threshold, but each
                # geo disturbance still needs a label, so pick the closest
                # morphology (filamentation if a 2nd component ever appears,
                sl = slice(i, j)
                c1 = np.asarray(fl.get('comp1_present', []), dtype=float)
                asp = np.asarray(fl.get('big_aspect', []), dtype=float)
                til = np.abs(np.asarray(fl.get('geo_tilt', []), dtype=float))
                if c1.size >= j and (c1[sl] >= 0.5).any():
                    gd_label = GEO_DIST_TYPE_NAME[1]
                else:
                    asp_thr = float(getattr(rules, 'geo_aspect_bot', 1.8))
                    tilt_thr = float(getattr(rules, 'geo_disturb_min_tilt',
                                             0.06) or 0.06)
                    asp_pk = (float(np.nanmax(asp[sl]))
                              if asp.size >= j and np.isfinite(asp[sl]).any()
                              else np.nan)
                    til_pk = (float(np.nanmax(til[sl]))
                              if til.size >= j and np.isfinite(til[sl]).any()
                              else np.nan)
                    asp_frac = asp_pk / asp_thr if np.isfinite(asp_pk) else -1.0
                    til_frac = til_pk / tilt_thr if np.isfinite(til_pk) else -1.0
                    gd_label = (GEO_DIST_TYPE_NAME[3] if asp_frac >= til_frac
                                else GEO_DIST_TYPE_NAME[2])
        rows.append(dict(
            state=st,
            name=STATE_NAMES.get(st, f'state {st}'),
            geo_dist_type=gd_label,
            start_idx=int(i), end_idx=int(j - 1),
            start=times[i], end=times[j - 1],
            n_days=int(j - i),
            onset_idx=onset_idx,
            onset=(times[onset_idx] if onset_idx >= 0 else pd.NaT),
            onset_dT_K=(float(w7[onset_idx]) if onset_idx >= 0
                        and w7.size == n else np.nan),
            onset_alt_km=(float(w_alt[onset_idx]) if onset_idx >= 0
                          and w_alt.size == n else np.nan),
            onset_lat=(float(w_lat[onset_idx]) if onset_idx >= 0
                       and w_lat.size == n else np.nan),
            onset_lookback_d=(int(w_d[onset_idx]) if onset_idx >= 0
                              and w_d.size == n else -1),
        ))
        i = j
    return pd.DataFrame(rows)


def detect_subwarmings(seg_w7, seg_alt, seg_d, thr=25.0, valley_drop=8.0):
    # Within a single warming-event run, find SEPARATE warming
    n = len(seg_w7)
    if n == 0:
        return []
    above = np.isfinite(seg_w7) & (seg_w7 >= thr)
    if not above.any():
        return []
    # Walk through, accumulating sub-pulses.  A new sub-pulse starts
    # whenever the signal first crosses thr after having dropped
    # by valley_drop below the running max since the last sub-pulse.
    pulses = []
    in_pulse = False
    peak_K = -np.inf; peak_idx = -1; onset_idx = -1
    running_max_since_pulse = -np.inf
    for i in range(n):
        v = seg_w7[i] if np.isfinite(seg_w7[i]) else -np.inf
        if v >= thr:
            if not in_pulse:
                # new sub-pulse; either first one, or we already
                # closed a previous one and this is a re-flare.
                in_pulse = True
                onset_idx = i
                peak_K = v; peak_idx = i
                running_max_since_pulse = v
            else:
                if v > peak_K:
                    peak_K = v; peak_idx = i
                if v > running_max_since_pulse:
                    running_max_since_pulse = v
        else:
            # below threshold; don't close the pulse yet, but watch
            # for a deep valley
            if in_pulse:
                if running_max_since_pulse - v >= valley_drop:
                    # close current pulse
                    rate = peak_K / max(1, int(seg_d[peak_idx]))  \
                        if 0 <= peak_idx < n and seg_d[peak_idx] > 0  \
                        else float('nan')
                    pulses.append(dict(
                        onset_idx=onset_idx, peak_idx=peak_idx,
                        peak_K=float(peak_K),
                        alt_km=float(seg_alt[peak_idx])
                                  if 0 <= peak_idx < n
                                  and np.isfinite(seg_alt[peak_idx])
                                  else float('nan'),
                        lookback_d=int(seg_d[peak_idx])
                                    if 0 <= peak_idx < n
                                    and seg_d[peak_idx] > 0 else -1,
                        rate_K_per_day=rate,
                    ))
                    # reset; we'll start fresh on next >=thr day
                    in_pulse = False
                    peak_K = -np.inf; peak_idx = -1; onset_idx = -1
                    running_max_since_pulse = v
    if in_pulse:
        rate = peak_K / max(1, int(seg_d[peak_idx]))  \
            if 0 <= peak_idx < n and seg_d[peak_idx] > 0  \
            else float('nan')
        pulses.append(dict(
            onset_idx=onset_idx, peak_idx=peak_idx,
            peak_K=float(peak_K),
            alt_km=float(seg_alt[peak_idx])
                      if 0 <= peak_idx < n
                      and np.isfinite(seg_alt[peak_idx])
                      else float('nan'),
            lookback_d=int(seg_d[peak_idx])
                        if 0 <= peak_idx < n
                        and seg_d[peak_idx] > 0 else -1,
            rate_K_per_day=rate,
        ))
    return pulses


def geo_alt_band_of_comp(fl, comp, i0, i1):
    # (lo, hi) km altitude band a geopotential component spans over the
    lo = np.asarray(fl.get('geo_alt_lo', np.full((0, 0), np.nan)), dtype=float)
    hi = np.asarray(fl.get('geo_alt_hi', np.full((0, 0), np.nan)), dtype=float)
    if lo.ndim != 2 or lo.shape[1] <= comp or hi.shape[1] <= comp:
        return (np.nan, np.nan)
    seg_lo = lo[i0:i1 + 1, comp]
    seg_hi = hi[i0:i1 + 1, comp]
    blo = float(np.nanmin(seg_lo)) if np.isfinite(seg_lo).any() else np.nan
    bhi = float(np.nanmax(seg_hi)) if np.isfinite(seg_hi).any() else np.nan
    return (blo, bhi)


def filament_location_str(fl, i0, i1):
    # Altitude span where a full second component (comp 1) is present in the
    c1 = np.asarray(fl.get('comp1_present', []), dtype=float)
    if c1.size <= i1:
        return ''
    seg = c1[i0:i1 + 1] >= 0.5
    if not seg.any():
        return ''
    days = np.where(seg)[0] + i0
    d0, d1 = int(days.min()), int(days.max())
    blo, bhi = geo_alt_band_of_comp(fl, 1, d0, d1)
    if not (np.isfinite(blo) and np.isfinite(bhi)):
        return 'second component present'
    where = f'{blo:.0f}-{bhi:.0f} km'
    c0lo, c0hi = geo_alt_band_of_comp(fl, 0, d0, d1)
    if np.isfinite(c0lo) and np.isfinite(c0hi):
        mid1 = 0.5 * (blo + bhi)
        mid0 = 0.5 * (c0lo + c0hi)
        if mid1 > mid0 + 3.0:
            where += ' (upper column)'
        elif mid1 < mid0 - 3.0:
            where += ' (lower column)'
    return where


def tilt_direction_str(fl, i0, i1):
    # Lean direction of the column tilt (poleward / equatorward) and whether
    t = np.asarray(fl.get('geo_tilt', []), dtype=float)
    bc = np.asarray(fl.get('geo_c0_bottom_cent', []), dtype=float)
    cc = np.asarray(fl.get('big_cent_lat', []), dtype=float)
    parts = []
    if t.size > i1:
        seg = t[i0:i1 + 1]
        if np.isfinite(seg).any():
            kk = int(np.nanargmax(np.where(np.isfinite(seg),
                                           np.abs(seg), -np.inf)))
            parts.append('poleward' if seg[kk] > 0 else 'equatorward')
    base_str = ''
    if bc.size > i1:
        seg_bc = bc[i0:i1 + 1]
        if np.isfinite(seg_bc).any():
            base = float(np.nanmean(seg_bc))
            toward = 'pole'
            if cc.size > i1:
                seg_cc = cc[i0:i1 + 1]
                if np.isfinite(seg_cc).any() and base < float(
                        np.nanmean(seg_cc)):
                    toward = 'equator'
            base_str = f'base toward {toward} ({base:.0f}°N)'
    lead = f'tilt {parts[0]}' if parts else 'tilt'
    return f'{lead}; {base_str}' if base_str else lead


def stretch_location_str(fl, i0, i1):
    # Whether the high-aspect stretching is at the base or through the
    basp = np.asarray(fl.get('base_aspect', np.full((0, 0), np.nan)),
                      dtype=float)
    bigasp = np.asarray(fl.get('big_aspect', []), dtype=float)
    base_pk = np.nan
    if basp.ndim == 2 and basp.shape[1] >= 1:
        seg = basp[i0:i1 + 1, 0]
        base_pk = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan
    col_pk = np.nan
    if bigasp.size > i1:
        seg = bigasp[i0:i1 + 1]
        col_pk = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan
    if np.isfinite(base_pk) and (not np.isfinite(col_pk)
                                 or base_pk >= col_pk):
        return f'at base (aspect {base_pk:.1f})'
    if np.isfinite(col_pk):
        return f'through column (aspect {col_pk:.1f})'
    return ''


def warming_run_is_major(fl, i0, i1):
    # MAJOR if any day in the run reaches the 7-day rise major threshold
    below = np.asarray(fl.get('warm7_max_below10', []), dtype=float)
    above = np.asarray(fl.get('warm7_max_above10', []), dtype=float)
    mb = (np.isfinite(below[i0:i1 + 1]) &
          (np.round(below[i0:i1 + 1], 2) >= MAJOR_DK_AT_OR_BELOW_10HPA)
          ) if below.size > i1 else np.array([False])
    ma = (np.isfinite(above[i0:i1 + 1]) &
          (np.round(above[i0:i1 + 1], 2) >= MAJOR_DK_ABOVE_10HPA)
          ) if above.size > i1 else np.array([False])
    return bool(mb.any() or ma.any())


def contiguous_runs(mask):
    # List of (lo, hi) inclusive index pairs for each True run in mask.
    runs = []
    m = np.asarray(mask, dtype=bool)
    i = 0
    n = m.size
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def print_onsets(result):
    # Print warming-event diagnostics, one event per line group:
    df = result['events']
    fl = result.get('flags', {})
    times = result.get('times', np.array([]))
    if 'onset' not in df.columns:
        df = events_dataframe(result['states'], times, fl=fl,
                              rules=type('R', (), result.get('rules', {}))())
    rows = df[df['onset'].notna()].copy()
    if rows.empty:
        print('(no warming or geo-disturbance events detected)')
        return
    def md(t):
        try:    return f'{t.month:02d}-{t.day:02d}'
        except: return str(t)
    warm_codes = (
        STATE_WARM_NO_GEO, STATE_SSW_DISPLACED, STATE_SSW_DISPLACED_MIN,
        STATE_SSW_SPLIT,   STATE_SSW_SPLIT_MIN,
        STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN,
        STATE_SSW_PARTIAL_SPLIT, STATE_SSW_PARTIAL_SPLIT_MIN,
    )

    w7    = np.asarray(fl.get('warm7_max',     []), dtype=float)
    w_alt = np.asarray(fl.get('warm7_alt_km',  []), dtype=float)
    w_lat = np.asarray(fl.get('warm7_lat',     []), dtype=float)
    w_d   = np.asarray(fl.get('warm7_lookback_d', []), dtype=int)
    w_eoff = np.asarray(fl.get('warm7_end_off', []), dtype=int)
    w1    = np.asarray(fl.get('warm1_max',     []), dtype=float)
    w1_alt = np.asarray(fl.get('warm1_alt_km', []), dtype=float)
    w1_lat = np.asarray(fl.get('warm1_lat',    []), dtype=float)
    rules = result.get('rules', {})
    thr25 = float(rules.get('ssw_warm25_K', 25.0))

    for _, r in rows.iterrows():
        st_int = int(r['state'])
        span = f'{md(r["start"])} → {md(r["end"])}'
        onset = md(r['onset']) if pd.notna(r['onset']) else '-'

        if st_int in warm_codes and w7.size:
            i0 = int(r['start_idx']); i1 = int(r['end_idx'])
            seg_w7  = w7   [i0:i1 + 1]
            seg_alt = w_alt[i0:i1 + 1]
            seg_d   = w_d  [i0:i1 + 1] if w_d.size else np.full(
                len(seg_w7), -1, dtype=int)
            seg_lat = w_lat[i0:i1 + 1] if w_lat.size else np.full(
                len(seg_w7), np.nan)
            seg_w1  = (w1[i0:i1 + 1] if w1.size == w7.size
                       else np.full(len(seg_w7), np.nan))
            seg_w1_alt = (w1_alt[i0:i1 + 1] if w1_alt.size == w7.size
                          else np.full(len(seg_w7), np.nan))
            seg_w1_lat = (w1_lat[i0:i1 + 1] if w1_lat.size == w7.size
                          else np.full(len(seg_w7), np.nan))

            # max ΔT and altitudes touched at >=25K
            max_K = (float(np.nanmax(seg_w7))
                     if np.isfinite(seg_w7).any() else float('nan'))
            above = np.isfinite(seg_w7) & (seg_w7 >= thr25)
            alts_hit = sorted({int(round(a)) for a in seg_alt[above]
                               if np.isfinite(a)})
            # a warming must reach 25K at some altitude; if none does, this is
            # not a warming to report
            if not alts_hit:
                continue
            alts_str = ', '.join(f'{a}km' for a in alts_hit)

            # duration since first 25K day
            first_25 = np.where(above)[0]
            dur_25 = (int(i1 - (i0 + first_25[0]) + 1)
                      if first_25.size else 0)

            # The accumulation span of the peak warming. The peak warm7 day
            # carries a lookback (warm7_lookback_d) = the number of days the
            # >=25K rise actually took to accumulate. That span may begin
            if np.isfinite(seg_w7).any():
                pk_rel = int(np.nanargmax(np.where(np.isfinite(seg_w7),
                                                   seg_w7, -np.inf)))
                pk_abs = i0 + pk_rel
                lb_pk = int(w_d[pk_abs]) if (w_d.size and
                                             w_d[pk_abs] > 0) else 0
                end_off = int(w_eoff[pk_abs]) if (w_eoff.size and
                                                  w_eoff[pk_abs] >= 0) else 0
                end_abs = max(0, pk_abs - end_off)        # day rise ended
                span_lo = max(0, end_abs - lb_pk)         # day rise began
                acc_days = int(end_abs - span_lo)         # day-span of rise
                acc_lo_day = md(times[span_lo])
                acc_hi_day = md(times[end_abs])
                # fastest single-day jump within the accumulation span
                if w1.size == w7.size:
                    w1_span = w1[span_lo:end_abs + 1]
                    if np.isfinite(w1_span).any():
                        kk = int(np.nanargmax(np.where(np.isfinite(w1_span),
                                                       w1_span, -np.inf)))
                        max_w1 = float(w1_span[kk])
                        kk_abs = span_lo + kk
                        max_w1_day = md(times[kk_abs])
                        max_w1_alt = (float(w1_alt[kk_abs])
                                      if (w1_alt.size == w7.size and
                                          np.isfinite(w1_alt[kk_abs]))
                                      else float('nan'))
                        max_w1_lat = (float(w1_lat[kk_abs])
                                      if (w1_lat.size == w7.size and
                                          np.isfinite(w1_lat[kk_abs]))
                                      else float('nan'))
                    else:
                        max_w1 = float('nan'); max_w1_day = '-'
                        max_w1_alt = float('nan'); max_w1_lat = float('nan')
                else:
                    max_w1 = float('nan'); max_w1_day = '-'
                    max_w1_alt = float('nan'); max_w1_lat = float('nan')
            else:
                acc_days = 0; acc_lo_day = '-'; acc_hi_day = '-'
                max_w1 = float('nan'); max_w1_day = '-'
                max_w1_alt = float('nan'); max_w1_lat = float('nan')

            # sub-pulses (separated by valley of >=8K below running max)
            pulses = detect_subwarmings(seg_w7, seg_alt, seg_d,
                                          thr=thr25, valley_drop=8.0)

            print()
            print(f'{r["name"]:<34} {span:<14}  '
                  f'duration={dur_25}d  max ΔT={max_K:.1f}K'
                  f' over {acc_lo_day}→{acc_hi_day} ({acc_days}d)')
            print(f'{"":34}  altitudes hit 25K: {alts_str}')
            if np.isfinite(max_w1):
                lat_s = f', {max_w1_lat:.0f}°N' if np.isfinite(max_w1_lat) else ''
                alt_s = f' @ {max_w1_alt:.0f}km' if np.isfinite(max_w1_alt) else ''
                print(f'{"":34}  fastest single-day jump: '
                      f'{max_w1:.1f}K on {max_w1_day}{alt_s}{lat_s}')

            if not pulses:
                pass
            elif len(pulses) == 1:
                p = pulses[0]
                t_peak  = md(times[i0 + p['peak_idx']]) \
                    if (i0 + p['peak_idx']) < len(times) else '?'
                print(f'{"":34}  warming pulse: peak '
                      f'{t_peak}, {p["peak_K"]:.1f}K @ '
                      f'{p["alt_km"]:.0f}km')
            else:
                print(f'{"":34}  {len(pulses)} sub-warmings:')
                for p in pulses:
                    t_peak = md(times[i0 + p['peak_idx']]) \
                        if (i0 + p['peak_idx']) < len(times) else '?'
                    print(f'{"":34}    peak {t_peak}, {p["peak_K"]:.1f}K @ '
                          f'{p["alt_km"]:.0f}km')

            # (1) no-geo warmings have no major/minor *state*, so report it;
            # also flag any sub-threshold filamentation / pinch and where.
            if st_int == STATE_WARM_NO_GEO:
                mm = 'major' if warming_run_is_major(fl, i0, i1) else 'minor'
                print(f'{"":34}  no geo disturbance: {mm}')
                fil = filament_location_str(fl, i0, i1)
                if fil:
                    print(f'{"":34}  sub-threshold filamentation: {fil}')
                fb = np.asarray(fl.get('b0_frac_bot_partial', []), dtype=float)
                ft = np.asarray(fl.get('b0_frac_top_partial', []), dtype=float)
                par = np.asarray(fl.get('b0_partial_split', []), dtype=float)
                top = np.asarray(fl.get('b0_top_pinch_split', []), dtype=float)
                has_pinch = ((par.size > i1 and (par[i0:i1 + 1] >= 0.5).any())
                             or (top.size > i1 and
                                 (top[i0:i1 + 1] >= 0.5).any()))
                if has_pinch:
                    where = []
                    if (fb.size > i1 and np.isfinite(fb[i0:i1 + 1]).any()
                            and np.nanmax(fb[i0:i1 + 1]) > 0):
                        where.append('lower')
                    if (ft.size > i1 and np.isfinite(ft[i0:i1 + 1]).any()
                            and np.nanmax(ft[i0:i1 + 1]) > 0):
                        where.append('upper')
                    wstr = (' (' + '/'.join(where) + ' levels)'
                            if where else '')
                    print(f'{"":34}  sub-threshold splitting/pinch{wstr}')

            # (2) mixed: displaced-phase vs split-phase timing/duration, and
            # whether a later warming pulse aligns with the phase change.
            if st_int in (STATE_SSW_MIXED_MAJ, STATE_SSW_MIXED_MIN):
                fs = np.asarray(fl.get('b0_full_split_prog',
                                       fl.get('b0_full_split', [])),
                                dtype=float)
                split_day = ((fs[i0:i1 + 1] >= 0.5) if fs.size > i1
                             else np.zeros(i1 - i0 + 1, bool))
                disp_day = ~split_day

                def phase_str(mask, name):
                    out = [f'{md(times[i0 + a])}->{md(times[i0 + b])} '
                           f'({b - a + 1}d)'
                           for (a, b) in contiguous_runs(mask)]
                    return f'{name}: ' + ('; '.join(out) if out else 'none')
                print(f'{"":34}  {phase_str(split_day, "split phase")}')
                print(f'{"":34}  {phase_str(disp_day, "displaced phase")}')
                trans = set()
                for (a, b) in contiguous_runs(split_day):
                    trans.add(a)
                    trans.add(b + 1)
                if len(pulses) >= 2:
                    aligned = any(abs(p['peak_idx'] - tt) <= 2
                                  for p in pulses[1:] for tt in trans)
                    note = ('a later pulse aligns with the displaced/split '
                            'change' if aligned
                            else 'pulses not aligned with the change')
                    print(f'{"":34}  {len(pulses)} pulses; {note}')
        else:
            # (3) geo disturbance: every one carries a priority label
            # (filamentation > stretching > tilting); report the specifics.
            i0g = int(r['start_idx']); i1g = int(r['end_idx'])
            gdt = r.get('geo_dist_type', '') if hasattr(r, 'get') else ''
            type_str = f'  type={gdt}' if gdt else ''
            print(f'{r["name"]:<34} {span:<14}  '
                  f'onset={onset}  duration={int(r["n_days"])}d{type_str}')
            if gdt == 'filamentation':
                loc = filament_location_str(fl, i0g, i1g)
                if loc:
                    print(f'{"":34}  filament at {loc}')
            elif gdt == 'stretching':
                loc = stretch_location_str(fl, i0g, i1g)
                if loc:
                    print(f'{"":34}  stretching {loc}')
            elif gdt == 'tilting':
                print(f'{"":34}  {tilt_direction_str(fl, i0g, i1g)}')


def hierarchical_states(ds, k=9, smooth=3, n_svd_T=5, n_svd_ringU=3):
    lat = ds['lat'].values
    lat_polar = lat >= 60.0
    # Use the geopotential-contour per-bin temperature; fall back to T_zonal.
    if ('T_in_geo_mean' in ds.data_vars and 'lat_bin_center' in ds.variables):
        Tc_mean = np.asarray(ds['T_in_geo_mean'].values, dtype=np.float32)
        latb    = np.asarray(ds['lat_bin_center'].values, dtype=float)
        polar_b = latb >= 60.0
        if polar_b.any():
            Tcm_lat = np.nanmean(Tc_mean[:, :, polar_b, :], axis=3)
            wlat = np.cos(np.deg2rad(latb[polar_b])).astype(np.float32)
            wlat = np.where(np.isfinite(wlat) & (wlat > 0.0), wlat, 0.0)
            wmat = wlat[None, None, :]
            fin  = np.isfinite(Tcm_lat).astype(np.float32)
            T_sum = (np.where(fin > 0, Tcm_lat, 0.0) * wmat).sum(axis=2)
            w_sum = (fin * wmat).sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                Tp = np.where(w_sum > 0,
                              T_sum / np.maximum(w_sum, 1e-12),
                              np.nan).astype(np.float32)
        else:
            Tp = np.nanmean(np.nanmean(Tc_mean, axis=3), axis=2).astype(np.float32)
    else:
        T  = ds['T_zonal'].values.astype(np.float32)
        Tp = np.nanmean(T[:, :, lat_polar], axis=2)
    if smooth and smooth > 1:
        Tp = uniform_filter1d(Tp, size=smooth, axis=0, mode='nearest')
    T_anom  = Tp - np.nanmean(Tp, axis=0, keepdims=True)
    # per-level daily T tendency; first day is edge-effect but acceptable
    # because we assume the season starts in a quiet state.
    dTp_lev = np.gradient(Tp, axis=0).astype(np.float32)

    # concatenate T anomaly and dT/dt so the thermal SVD sees both the
    # standing pattern and the rate of change.
    T_block = np.hstack([
        np.where(np.isfinite(T_anom),  T_anom,  0.0).astype(np.float32),
        np.where(np.isfinite(dTp_lev), dTp_lev, 0.0),
    ])

    n_svd_T = int(min(n_svd_T, T_block.shape[1] - 1, T_block.shape[0] - 1))
    svd = TruncatedSVD(n_components=max(n_svd_T, 1),
                       random_state=42).fit(T_block)
    T_scores = svd.transform(T_block)

    fl = compute_flags(ds, smooth=smooth)
    nt = ds.sizes['time']

    # ring U(t, alt) SVD: gives the clustering information about jet
    # vertical structure (bottom-only, upper-only, deep, broken, etc.)
    ringU = fl['ring_U_alt']
    ringU_flat = np.where(np.isfinite(ringU), ringU, 0.0)
    if ringU_flat.shape[1] > 1 and ringU_flat.shape[0] > 1:
        nsv_r = int(min(n_svd_ringU, ringU_flat.shape[1] - 1,
                        ringU_flat.shape[0] - 1))
        nsv_r = max(nsv_r, 1)
        ringU_svd    = TruncatedSVD(n_components=nsv_r,
                                    random_state=42).fit(ringU_flat)
        ringU_scores = ringU_svd.transform(ringU_flat)
        ringU_names  = [f'ringU_svd_{i+1}' for i in range(nsv_r)]
    else:
        ringU_svd    = None
        ringU_scores = np.zeros((nt, 0), dtype=np.float32)
        ringU_names  = []

    scalar_cols = np.stack([
        fl['peak_U'],     fl['mean_U'],
        fl['inner_lat'],  fl['outer_lat'],  fl['ring_alt'],
        fl['ring_area'],  fl['east_area'],
        fl['east_inner'], fl['east_outer'], fl['east_alt'], fl['east_U'],
        fl['core_lat_mean'], fl['core_lat_std'],
        fl['core_alt_mean'], fl['core_alt_std'],
        fl['geo_tilt'],
        fl['band_lo'],    fl['band_hi'],    fl['band_span'],
        fl['west_pct'],   fl['east_pct'],
        fl['Tp_col'],     fl['dT_col'],
        fl['warm_anom_max'], fl['warm_dT_max'],
        fl['geo_b0'],     fl['big_bot_lat'],
        fl['big_aspect'], fl['area_fracs'],
        fl['jet_intact'], fl['east_intact_level'],
        np.gradient(np.where(np.isfinite(fl['peak_U']),
                             fl['peak_U'], 0.0)),
        np.gradient(np.where(np.isfinite(fl['mean_U']),
                             fl['mean_U'], 0.0)),
    ], axis=1).astype(np.float32)
    scalar_names = ['peak_U', 'mean_U',
                    'inner_lat', 'outer_lat', 'ring_alt',
                    'ring_area', 'east_area',
                    'east_inner', 'east_outer', 'east_alt', 'east_U',
                    'core_lat_mean', 'core_lat_std',
                    'core_alt_mean', 'core_alt_std',
                    'geo_tilt',
                    'band_lo', 'band_hi', 'band_span',
                    'west_pct', 'east_pct',
                    'Tp_col', 'dT_col',
                    'warm_anom_max', 'warm_dT_max',
                    'geo_b0', 'big_bot_lat',
                    'big_aspect', 'area_fracs',
                    'jet_intact', 'east_intact_level',
                    'd_peak_U', 'd_mean_U']
    svd_names = [f'T_svd_{i+1}' for i in range(T_scores.shape[1])]
    names = svd_names + ringU_names + scalar_names

    X = np.hstack([T_scores, ringU_scores, scalar_cols])

    # clean NaN -> column median
    col_med = np.nanmedian(X, axis=0)
    col_med[~np.isfinite(col_med)] = 0.0
    mask = ~np.isfinite(X)
    X[mask] = np.take(col_med, np.where(mask)[1])

    # drop zero-variance cols so Ward doesn't choke
    var  = X.var(axis=0)
    keep = var > 1e-10
    X    = X[:, keep]
    names = [n for n, k_ in zip(names, keep) if k_]

    Xs = RobustScaler().fit_transform(X).astype(np.float64)

    n_pc = int(min(8, Xs.shape[1], Xs.shape[0] - 1))
    X_pca = PCA(n_components=n_pc, random_state=42).fit_transform(Xs)

    labels = AgglomerativeClustering(n_clusters=int(k),
                                     linkage='ward').fit_predict(X_pca)
    Z = (ward_linkage(X_pca, method='ward')
         if X_pca.shape[0] <= 8000 else None)

    return dict(
        labels=labels,
        times=ds['time'].values,
        flags=fl,
        feature_names=names,
        X=X, X_pca=X_pca,
        linkage_matrix=Z,
        svd=svd,
        T_scores=T_scores,
        ringU_svd=ringU_svd,
        ringU_scores=ringU_scores,
    )


# Physical signature per cluster: mean of each diagnostic for days
# assigned to that cluster. Use to decide which cluster is which state.
def cluster_physics_summary(result):
    labels = np.asarray(result['labels'])
    fl     = result['flags']
    uniq   = sorted(int(c) for c in np.unique(labels))
    rows = []
    for c in uniq:
        m = labels == c
        if not m.any():
            continue
        def mx(a):
            return float(np.nanmean(np.asarray(a, float)[m]))
        rows.append(dict(
            cluster=c, n_days=int(m.sum()),
            peak_U=mx(fl['peak_U']),
            mean_U=mx(fl['mean_U']),
            inner_lat=mx(fl['inner_lat']),
            outer_lat=mx(fl['outer_lat']),
            ring_alt=mx(fl['ring_alt']),
            ring_area=mx(fl['ring_area']),
            east_area=mx(fl['east_area']),
            band_lo=mx(fl['band_lo']),
            band_hi=mx(fl['band_hi']),
            band_span=mx(fl['band_span']),
            east_inner=mx(fl['east_inner']),
            east_outer=mx(fl['east_outer']),
            east_alt=mx(fl['east_alt']),
            east_U=mx(fl['east_U']),
            core_lat_mean=mx(fl['core_lat_mean']),
            core_lat_std=mx(fl['core_lat_std']),
            core_alt_mean=mx(fl['core_alt_mean']),
            core_alt_std=mx(fl['core_alt_std']),
            geo_tilt=mx(fl['geo_tilt']),
            west_pct=mx(fl['west_pct']),
            east_pct=mx(fl['east_pct']),
            Tp_col=mx(fl['Tp_col']),
            dT_col=mx(fl['dT_col']),
            warm_anom_max=mx(fl['warm_anom_max']),
            warm_dT_max=mx(fl['warm_dT_max']),
            geo_b0=mx(fl['geo_b0']),
            big_bot_lat=mx(fl['big_bot_lat']),
            big_aspect=mx(fl['big_aspect']),
            area_fracs=mx(fl['area_fracs']),
            jet_intact_frac=100.0 * mx(fl['jet_intact']),
            east_intact_frac=100.0 * mx(fl['east_intact_level']),
        ))
    return pd.DataFrame(rows).set_index('cluster')


# Turn a cluster result + {cluster_id: state_code} mapping into a
# classify_season-shaped result so plot_events works.
def apply_cluster_mapping(cluster_result, mapping,
                          default_state=STATE_STRONG, rules=None, smooth=3):
    # build a StateRules so the rest of the plotting machinery has the
    # same knobs available, but we don't use its decision tree here.
    fl     = cluster_result['flags']
    r      = calibrate_rules(fl,
                              rules if isinstance(rules, StateRules)
                              else StateRules())
    labels = np.asarray(cluster_result['labels'])
    states = np.array([int(mapping.get(int(c), default_state))
                       for c in labels], dtype=np.int16)
    states = despeckle(states, r.min_run_length)
    states = bridge_split_ssw_component_gaps(states, fl, r)
    states = merge_post_warming_geo(states, window=r.recovery_window_days)
    states = bridge_warming_near_geo(states, fl, r)
    states = align_warming_block_to_geo_start(states)
    states = promote_geo_with_local_warming(states, fl, r, lag_days=3)
    states = refine_major_minor(states, fl, r)
    states = end_warming_on_wind_restabilize(states, fl, r, stable_days=2)
    states = absorb_event_precursors(states, r)
    states = merge_trailing_warm_into_ssw(states, fl, r)
    states = validate_ssw_warming(states, fl, r)
    states = refine_major_minor(states, fl, r)
    states = apply_end_of_season(states, fl, r)
    states = apply_recovery(states, fl, r)
    states = collapse_non_ssw_states(states)
    states = despeckle(states, r.min_run_length)
    states = collapse_major_minor(states)
    states = force_early_season_prefix(states, r)
    times  = cluster_result['times']
    return dict(
        states=states, times=times, flags=fl,
        rules=asdict(r),
        events=events_dataframe(states, times),
        cluster_labels=labels,
        cluster_to_state=dict(mapping),
    )


# visualization

def to_pydates(times):
    import datetime as dt
    out = []
    for t in np.asarray(times).tolist():
        try:
            if hasattr(t, 'year') and hasattr(t, 'month'):
                y  = int(t.year); m = int(t.month)
                d  = int(getattr(t, 'day', 1))
                hh = int(getattr(t, 'hour', 0))
                mm = int(getattr(t, 'minute', 0))
                try:
                    out.append(dt.datetime(y, m, d, hh, mm))
                except ValueError:
                    out.append(dt.datetime(y, m, min(d, 28), hh, mm))
            else:
                out.append(pd.Timestamp(t).to_pydatetime())
        except Exception:
            out.append(None)
    return out


# Five-panel state/event timeline.
# 0: colored state strip
# 1: jet strength (peak/mean |U|) + easterly % at 10 hPa / 60N
FIG_W        = 14.0
# Left/right margins are reserved in ABSOLUTE inches and converted to a
# fraction of the (constant) figure width, so the axes-box left/right edges
# land at the same pixel in every plot (time axes stay aligned when stacked)
LEFT_IN      = 0.50    # y-axis label + tick numbers
RIGHT_IN     = 0.62    # inset colorbar + its tick numbers, or a twinx label
PLOT_LEFT    = LEFT_IN / FIG_W            # ~0.036
PLOT_RIGHT   = 1.0 - RIGHT_IN / FIG_W     # ~0.956
XLABEL_IN    = 0.46    # bottom space reserved for the two-tier month/date axis
PLOT_TOP     = 0.93    # legacy fractional margins, used by plot_events only
PLOT_BOTTOM  = 0.06
PANEL_HSPACE = 0.7       # room for per-panel legends above each panel
# Common y-axis label x-coordinate (in axes-relative coords).  Force
# every primary ylabel onto the same vertical strip so short labels
# like "altitude (km)" sit at the same fig_x as long labels like
YLABEL_X     = -0.03


# Month-only x-axis (no year header); reused by every plot so cross-
# season comparisons don't drag a year label around.  Major ticks at
# month boundaries with the abbreviated month name; minor ticks every
def pin_ylabels(*axes, x=None):
    # Force every primary axes' y-axis label to a common x coordinate
    if x is None:
        x = YLABEL_X
    for ax in axes:
        if ax is None:
            continue
        try:
            ax.yaxis.set_label_coords(float(x), 0.5)
        except Exception:
            pass


def lock_canvas_extent(fig):
    # Place invisible anchor texts at the figure's left and right
    fig.text(0.0, 0.5, ' ', alpha=0.0, fontsize=1, transform=fig.transFigure,
             clip_on=False)
    fig.text(1.0, 0.5, ' ', alpha=0.0, fontsize=1, transform=fig.transFigure,
             clip_on=False)


def month_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(
        mdates.DayLocator(bymonthday=[5, 9, 13, 17, 21, 25, 29]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    ax.tick_params(axis='x', which='major',
                   labelsize=10, pad=2, length=5)
    ax.tick_params(axis='x', which='minor',
                   labelsize=8, pad=13, length=3, colors='0.30')


def apply_margins(fig, top_in, hspace=None, bottom_in=None):
    # Set figure margins so the axes box keeps the shared (constant) left and
    fh = float(fig.get_figheight())
    if bottom_in is None:
        bottom_in = XLABEL_IN
    kw = dict(left=PLOT_LEFT, right=PLOT_RIGHT,
              top=1.0 - float(top_in) / fh,
              bottom=float(bottom_in) / fh)
    if hspace is not None:
        kw['hspace'] = hspace
    fig.subplots_adjust(**kw)


def apply_common_xlim(fig, times):
    # Force every Axes in *fig* (including twinx) to share the same
    import datetime as dt
    tseq = to_pydates(times)
    ok = [t for t in tseq if t is not None]
    if not ok:
        return
    t0, t1 = ok[0], ok[-1]
    if t0 == t1:
        t0 = t0 - dt.timedelta(days=1)
        t1 = t1 + dt.timedelta(days=1)
    for ax in fig.axes:
        ax.set_xlim(t0, t1)


# Consistent legend placement: on top of every panel, spanning its
# width so the axis itself stays full-length.
def top_legend(ax, ncol=4, fontsize=8):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels,
              loc='lower left', bbox_to_anchor=(0.0, 1.02),
              ncol=ncol, fontsize=fontsize, frameon=False,
              borderaxespad=0.0)


def default_tbin_scheme_specs(for_plot=True):
    # Combinatorial lat × lon × altitude grouping specs for ``plot_events``.
    return tbin_scheme_combinations(for_plot=for_plot)


def classify_for_tbin_scheme(ds, base_result, spec, mode='cum25'):
    # Run classify_season for one 3-D bin grouping *spec*.
    kw = dict(smooth=3, mode=mode,
              t_bin_lat_group=int(spec.get('lat_group', 1)),
              t_bin_lon_group=int(spec.get('lon_group', 1)))
    if spec.get('alt_km') is not None:
        kw['t_bin_alt_km'] = float(spec['alt_km'])
    elif spec.get('alt_nlevels') is not None:
        kw['t_bin_alt_nlevels'] = int(spec['alt_nlevels'])
    rules = base_result.get('rules')
    if rules is not None:
        kw['rules'] = rules
    return classify_season(ds, **kw)


def paint_state_strip(ax, result, times, ylabel=None):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel, rotation=0, fontsize=7,
                        ha='right', va='center', labelpad=28)
    evs = result.get('events')
    if evs is None:
        return
    rt = to_pydates(result['times'])
    for _, row in evs.iterrows():
        s = int(row['state'])
        i0, i1 = int(row['start_idx']), int(row['end_idx'])
        if i0 >= len(rt) or i1 >= len(rt):
            continue
        t0, t1 = rt[i0], rt[i1]
        if t0 is None or t1 is None:
            continue
        ax.axvspan(t0, t1, color=STATE_COLORS.get(s, '#888'),
                   alpha=0.9, linewidth=0)


def save_figure(fig, save_path, dpi=200):
    # Save *fig* to *save_path* with a tight bounding box. No-op if either is
    if save_path is not None and fig is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


def train_partial_split_params(big_b0, sec_b0, result=None):
    # Season-trained vertical run length and level fractions.
    fin = np.isfinite(big_b0)
    nt, nlev = big_b0.shape
    raw = fin & (big_b0 >= 2) & (~np.isfinite(sec_b0) | (sec_b0 <= 0))
    min_vert = 3
    if result is not None:
        evs = result.get('events')
        if evs is not None and len(evs):
            labels = evs.get('label', evs.get('kind', ''))
            if hasattr(labels, 'astype'):
                disp = evs[labels.astype(str).str.contains('displaced', na=False)]
                if len(disp):
                    runs = []
                    for _, row in disp.iterrows():
                        i0, i1 = int(row['start_idx']), int(row['end_idx'])
                        seg = raw[max(0, i0):min(nt, i1 + 1)]
                        for ti in range(seg.shape[0]):
                            run = 0
                            best = 0
                            for lev in range(nlev):
                                if seg[ti, lev]:
                                    run += 1
                                    best = max(best, run)
                                else:
                                    run = 0
                            if best >= 2:
                                runs.append(best)
                    if len(runs) >= 3:
                        min_vert = max(2, int(np.percentile(runs, 25)))
    pop_fracs = []
    sec_fracs = []
    for ti in range(nt):
        pop = float(fin[ti].sum())
        if pop <= 0:
            continue
        pop_fracs.append(float(raw[ti].sum()) / pop)
        sec_fracs.append(float((np.isfinite(sec_b0[ti]) &
                                (sec_b0[ti] > 0)).sum()) / pop)
    floor = (float(np.percentile(pop_fracs, 25))
             if len(pop_fracs) >= 5
             else float(min_vert) / max(float(nlev), 1.0))
    ceiling = (float(np.percentile(pop_fracs, 90))
               if len(pop_fracs) >= 5 else 0.65)
    full_thr = (float(np.percentile(sec_fracs, 75))
                if len(sec_fracs) >= 5 else 0.50)
    return dict(min_vert=min_vert, floor=floor, ceiling=ceiling,
                full_thr=full_thr)


def plot_state_timeseries(result, figsize=None, title=None, legend=True,
                          save_path=None):
    # Classification state colour strip over the season (the 'colorbar'
    times = to_pydates(result['times'])
    # The strip (axes) is a fixed height in both cases; only the figure grows
    # to make room for the legend, so the coloured bar is identical with or
    # without it.
    strip_in = 0.55
    top_in   = 0.72 if legend else 0.12
    if figsize is None:
        figsize = (FIG_W, strip_in + XLABEL_IN + top_in)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    paint_state_strip(ax, result, times)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[c])
                   for c in LEGEND_ORDER]
        labels  = [STATE_NAMES[c] for c in LEGEND_ORDER]
        ax.legend(handles, labels,
                  loc='lower left', bbox_to_anchor=(0.0, 1.02),
                  ncol=7, fontsize=8, frameon=False, borderaxespad=0.0)
    if title:
        ax.set_title(title, pad=22)
    month_xaxis(ax)
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True)
    apply_common_xlim(fig, result['times'])
    apply_margins(fig, top_in=top_in)
    pin_ylabels(ax)
    lock_canvas_extent(fig)
    save_figure(fig, save_path)
    return fig, ax