# vortexstates

Polar vortex contour and geometry extraction from ERA5. The extraction turns
daily ERA5 fields into one record per day of topological and geometric
diagnostics and writes them to a NetCDF time series. That time series is the
sole input to the classifier (`vortexclass.py`).

- **`vortexstates.py`** — saves the diagnostics the
  classifier consumes.
   Use this to generate classifier input.

## Method

For each day the vortex is identified as connected components (persistent
homology, H0/H1) on two physically-informed submanifolds:

- the geopotential field, thresholded at a low percentile, giving the cold
  polar lobe(s);
- the zonal wind field, thresholded on westerly speed, giving the circumpolar
  jet (the westerly ring) and any easterly ring.

Component boundaries are refined with a watershed cut on the steepest gradient,
the longitude seam is stitched so rings close across 0/360, and merged
lower-latitude troughs are cut at the saddle. Components that are truncated at
the 40 N domain edge — touching that edge and elongated *along* it (a zonal
shelf, longitude extent far exceeding latitude extent in physical km) — are
dropped, while deep lobes that merely reach the edge and filaments away from
the edge are kept. Geometry (centroids, latitudes, altitudes, aspect ratios,
tilt, area) and the per-level Betti-0 profile of the largest lobe are then
measured directly from the contour voxels.

## Pipeline

```python
import vortexstates as vs 

# 1. load an ERA5 subset (geopotential phi, zonal wind U, temperature T, p)
phi, U, T, p, dates = vs.load_era5_subset(start, end)

# 2. write the per-day diagnostic time series for one season
vs.run_timeseries(phi_all, U_all, T_all, p_all, dates,
                  "vortex_full_timeseries_8485.nc")
```

Key functions:

- `load_era5_subset(start, end, ...)` — read and prepare ERA5 fields.
- `analyze_geopotential`, `analyze_wind`, `analyze_temperature` — the three
  per-field analyses for a single timestep.
- `analyze_vortex(phi, U, T, p)` — full single-timestep analysis; returns a
  dataset carrying `geopotential_edges` and `wind_edges` (the 3-D edge masks
  used by the plotters below).
- `collect_timestep(...)` — assemble one day's record.
- `run_timeseries(..., outpath)` — loop over a season and write the NetCDF.

## Saved outputs (vortexstates.py)

Coordinates are `time`, `lev`, `altitude_km`, `lat`, `lon`, and the coarse
temperature grid (`lat_bin_center`, `lon_bin_center`, and edges). The data
variables are:

- **Geopotential geometry** — `geo_b0`, `geo_largest_b0_profile`,
  `geo_second_b0_profile`, `geo_total_area_km2`, `geo_aspect_ratio`,
  `geo_bottom_lat`, `geo_lowest_lat`, `geo_alt_lo`, `geo_alt_hi`, and the
  per-level fields `geo_lev_centroid_lat`, `geo_lev_centroid_lon`,
  `geo_lev_lat_equatorward`, `geo_lev_lat_poleward`, `geo_lev_aspect_ratio`.
- **Wind ring** — `wind_sign`, `wind_is_h1`, `wind_mean_U`,
  `wind_greatest_mag_U`, `wind_mean_inner_lat`, `wind_mean_outer_lat`,
  `wind_mean_alt`, `wind_pct_10hPa_60lat`, `wind_ring_n_levels`,
  `wind_total_area_km2`, `wind_tilt_slope`, `wind_ring_alt_bands`,
  `wind_grad_refined_region_speed`, the per-level `wind_lev_mean_U` and
  `wind_lev_lon_span`, and the jet-core curves `wind_core_lat`, `wind_core_alt`.
- **Temperature and gradient** — the coarse lat/lon-binned in-vortex
  temperature `T_coarse_max` / `T_coarse_mean` / `T_coarse_n`, the
  geopotential-contour temperature `T_in_geo_max` / `T_in_geo_mean` /
  `T_in_geo_n`, the polar-cap temperature `T_cap_max` / `T_cap_mean` /
  `T_cap_n`, the meridional gradient profile `grad_prof`, and the flags
  `gradient_reversed`, `jet_intact`.

`vtxplt_wsd.py` writes all of the above plus the additional per-lobe and
per-level geometry it computes (areas, spreads, voxel counts, pole distances,
tilt intercept/r2, refined boundary speeds, wind centroids and per-level wind
geometry, and more).

## Plotting (vortexstates.py)

- `plot_vortex_edges_3d(edges_da, altitude_km, base_field_da=None, field_type='t'|'u'|'pv', ...)`
  — 3-D scatter of the vortex edges on a polar (radius, longitude, altitude)
  frame, coloured by an underlying field. Wind uses a symmetric `RdBu_r`
  scale; temperature/PV/geopotential use sequential maps.
- `plot_level_with_mask_outline(field_da, mask_da, lev_idx=..., ...)`
  — north polar stereographic map of one level with the vortex mask drawn as a
  red contour outline. Requires `cartopy`.

Both take the per-timestep edge/mask arrays from `analyze_vortex` and return a
Matplotlib figure (and save to `output_file` when given).