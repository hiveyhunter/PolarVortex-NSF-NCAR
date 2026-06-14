# vortexclass

Stratospheric polar vortex state classifier. `vortexclass.py` takes one winter
season of per-day diagnostics (the NetCDF written by `vortexstates.py`) and
assigns one state label to each day, with sudden stratospheric warming (SSW)
type and major/minor distinctions resolved from the vortex geometry. It is the
file you run to classify a season. Outputs are the state colour strip
and the text event report.

## Usage

```python
import xarray as xr
import vortexclass as vc

ds  = xr.open_dataset("vortex_full_timeseries_8485.nc")
res = vc.classify_season(ds)

res["states"]   # per-day integer state codes (1-14)
res["events"]   # DataFrame, one row per event run
res["times"]    # per-day time coordinate
res["flags"]    # the per-day diagnostic arrays used by the classifier

vc.plot_state_timeseries(res)   # state colour strip across the season
vc.print_onsets(res)            # per-event text report
```

The single topmost altitude level (~50 km model top) is always excluded from
warming detection, so a warming that registers only at the very top level is
not counted.

## States

| code | name |
|------|------|
| 1 | early season |
| 2 | strong |
| 3 | geo disturbance (no warming) |
| 4 | warming, no geo disturbance |
| 5 | warming (displaced, major) |
| 6 | warming (displaced, minor) |
| 7 | warming (split, major) |
| 8 | warming (split, minor) |
| 9 | weak recovering |
| 10 | end of season |
| 11 | warming (mixed, major) |
| 12 | warming (mixed, minor) |
| 13 | warming (partial split, major) |
| 14 | warming (partial split, minor) |

## How a day is classified

Temperature triggers a warming; wind reformation ends it; geometry only refines
the type. Each day resolves to the first match in the hierarchy:

1. **Warming** — a 7-day in-vortex temperature rise of at least 25 K while the
   geopotential lobe is still intact (enclosed by the jet). The event continues
   until the westerly ring reforms across levels or the season ends.
2. **Geo disturbance** — no warming, but the lobe is displaced, split,
   stretched, or tilted, so its temperature no longer measures trapped polar
   air.
3. **Strong / early / end** otherwise.

Within a warming the type is set by geometry: major vs minor from the 7-day
rise threshold (>= 30 K at or below 10 hPa, or >= 40 K above 10 hPa); morphology
from the lobe count and shape (split, displaced, partial split, or mixed when
both displaced and split phases occur). Warming onsets are gated to Nov-Mar; an
event already underway may extend at most a few days into April.

## Reporting

`print_onsets(res)` prints a per-event diagnostic report:

- For each warming: duration since the first 25 K day, peak ΔT and the window
  over which it accumulated, the altitudes that reached 25 K, the fastest
  single-day jump, and the warming pulse(s) (separate sub-warmings when the
  signal drops by at least 8 K between peaks).
- For warmings with no geo disturbance: the major/minor result (these share one
  state code, so it is reported here), plus any sub-threshold filamentation
  (with its altitude span) or pinch (lower/upper levels).
- For mixed events: the displaced-phase and split-phase spans and durations
  separately, and whether a later warming pulse aligns with the phase change.
- For each geo disturbance: a label by priority **filamentation > stretching >
  tilting**, with specifics — filament altitude span and upper/lower column,
  stretching at base or through column with peak aspect, or tilt direction
  (poleward/equatorward) and whether the lobe base sits toward the equator or
  the pole. Every disturbance receives a label; one with no threshold crossing
  is given the nearest incipient morphology with its value shown.

`plot_state_timeseries(res)` draws the state colour strip across the season
(the classification time series), with an optional state legend.

## Events DataFrame

One row per event run, with `state`, `name`, `geo_dist_type`, `start`/`end`,
`start_idx`/`end_idx`, `n_days`, and warming-onset columns (`onset`,
`onset_dT_K`, `onset_alt_km`, `onset_lat`).