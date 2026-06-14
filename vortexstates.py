# Loads ERA5 U, T, surface pressure over a polar cap, reconstructs
# pressure and geopotential, detects vortex lobes and wind rings using
# persistent homology on the cubical grid, and provides 2D/3D plotting
# helpers on top of the resulting masks.

from __future__ import annotations

from collections import defaultdict

import time

import numpy as np
import xarray as xr

from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    generate_binary_structure,
    label,
)
from skimage import measure, morphology
from skimage.segmentation import watershed

try:
    import cripser
except ImportError as exc:
    raise RuntimeError(
        "cripser (CubicalRipser) is required by vortexstates; "
        "install with `pip install cripser`."
    ) from exc


# physical constants
G       = 9.80665     # standard gravity (m s-2)
R_DRY   = 287.05      # dry-air gas constant (J kg-1 K-1)
R_EARTH = 6371.0      # mean Earth radius (km)
H_SCALE = 7.0         # log-pressure scale height (km)
P0      = 101325.0    # reference surface pressure (Pa)

# pad widths used throughout the module: longitude wrap-padding to handle
# the 0/360 seam, and zero-pad rows beyond the pole to close polar topology
LON_PAD = 10
LAT_PAD = 3

# When True, collect_timestep prints a per-timestep breakdown of where the
# time goes (wind / geopotential / temperature analysis). Set False to silence.
PRINT_SECTION_TIMING = True

# Anisotropic Gaussian sigma (lat, lon) for smoothing the refined wind-jet
# boundary. Longitude cells are narrower than latitude cells, so smooth more
# in longitude. (The geopotential contour uses the physical-isotropic
# smooth_mask_polar instead; the wind boundary keeps this fixed ratio.)
WIND_SMOOTH_SIGMA = (2.0, 6.0)

# Per-level trough cut: a displaced vortex can merge at the threshold contour
# with a separate lower-latitude trough basin. Segment the mask into phi-basins
# and drop a non-vortex basin only when BOTH (a) it is a real separate feature
# (prominence >= BASIN_REAL_PROM_FRAC of the vortex depth, so a clean vortex is
# not trimmed by a dimple) and (b) much shallower than the vortex (its min sits
# >= BASIN_TROUGH_GAP_FRAC of the vortex depth above the vortex min; a genuine
# split lobe is deep, small gap, and kept). BASIN_TROUGH_GAP_FRAC is the main
# knob: lower to cut more aggressively, set >= 1.0 to disable.
BASIN_TROUGH_GAP_FRAC = 0.50
BASIN_REAL_PROM_FRAC  = 0.05
BASIN_MIN_H_FRAC      = 0.01


# Roll longitude so the occupied columns form one contiguous block (no 0/360
# wrap), returning the shift. Lets a seam-straddling region be cropped to a
# tight bounding box. Returns 0 if the region already spans every column.
def roll_to_contiguous(occ):
    n = len(occ)
    if occ.all():
        return 0
    e = np.where(~occ)[0]
    runs = np.split(e, np.where(np.diff(e) != 1)[0] + 1)
    if len(runs) > 1 and e[0] == 0 and e[-1] == n - 1:
        runs[0] = np.concatenate([runs[-1], runs[0]])
        runs = runs[:-1]
    longest = max(runs, key=len)
    return (-(int(longest[-1]) + 1)) % n


# Given the watershed basins of a region, decide which to keep: the deepest
# (vortex core) plus any basin that is NOT a merged trough. A trough is a real,
# separate, SHALLOW basin smaller than the vortex core; genuine split lobes are
def basin_keep_set(sub_fld, sub_reg, basins, nmk, gap_frac, prom_frac):
    bmin = {}
    for b in range(1, nmk + 1):
        sel = (basins == b)
        if sel.any():
            bmin[b] = float(sub_fld[sel].min())
    if not bmin:
        return None
    thr = float(sub_fld[sub_reg].max())
    vb = min(bmin, key=bmin.get)
    vmin = bmin[vb]
    vdepth = max(thr - vmin, 1e-9)
    struct2 = generate_binary_structure(2, 2)
    vsize = int((basins == vb).sum())
    keep = {vb}
    for b in bmin:
        if b == vb:
            continue
        sel = (basins == b)
        ring = binary_dilation(sel, structure=struct2) & (basins > 0) & (~sel)
        saddle = float(sub_fld[ring].min()) if ring.any() else thr
        prominence = (saddle - bmin[b]) / vdepth
        gap = (bmin[b] - vmin) / vdepth
        is_trough = (prominence >= prom_frac and gap >= gap_frac and
                     int(sel.sum()) < vsize)
        if not is_trough:
            keep.add(b)
    return keep


# Segment a single level's candidate mask into phi-basins and drop merged
# low-phi trough basins, cutting the boundary at the saddle. The detection
# (h_minima + watershed) is confined to the region's bounding box -- longitude
def cut_to_vortex_basin(fld, region, gap_frac=BASIN_TROUGH_GAP_FRAC,
                        prom_frac=BASIN_REAL_PROM_FRAC,
                        h_frac=BASIN_MIN_H_FRAC):
    if gap_frac >= 1.0 or not np.any(region):
        return region
    span = float(np.max(fld) - np.min(fld))
    if span <= 0.0:
        return region
    h = max(h_frac * span, 1e-6)
    ny, nx = region.shape
    occ = region.any(axis=1)
    rows = np.where(occ)[0]
    margin = 2
    r0 = max(0, int(rows[0]) - margin)
    r1 = min(ny, int(rows[-1]) + 1 + margin)
    wall = float(np.max(fld)) + span
    cols_occ = region.any(axis=0)
    struct2 = generate_binary_structure(2, 2)
    if cols_occ.all():
        # spans all longitudes: crop rows only, keep periodic longitude pad
        sub_f = fld[r0:r1].copy()
        sub_r = region[r0:r1]
        sub_f[~sub_r] = wall
        fld_pad = pad_lon(sub_f, LON_PAD)
        reg_pad = pad_lon(sub_r.astype(np.uint8), LON_PAD) > 0
        hmin = morphology.h_minima(fld_pad, h)
        mk, nmk = label((hmin & reg_pad).astype(np.uint8), structure=struct2)
        if nmk <= 1:
            return region
        basins = watershed(fld_pad, markers=mk, mask=reg_pad)
        keep = basin_keep_set(fld_pad, reg_pad, basins, nmk, gap_frac, prom_frac)
        if keep is None:
            return region
        keep_rows = unpad_lon(np.isin(basins, list(keep)).astype(np.uint8),
                              LON_PAD).astype(bool)
        out = np.zeros((ny, nx), dtype=bool)
        out[r0:r1] = keep_rows
        return region & out
    # otherwise crop both dimensions; roll longitude to a contiguous block and
    # wall the outside so no periodic pad is needed
    roll = roll_to_contiguous(cols_occ)
    fld_r = np.roll(fld, roll, axis=1) if roll else fld
    reg_r = np.roll(region, roll, axis=1) if roll else region
    cols = np.where(reg_r.any(axis=0))[0]
    c0 = max(0, int(cols[0]) - margin)
    c1 = min(nx, int(cols[-1]) + 1 + margin)
    sub_f = fld_r[r0:r1, c0:c1].copy()
    sub_r = reg_r[r0:r1, c0:c1]
    sub_f[~sub_r] = wall
    hmin = morphology.h_minima(sub_f, h)
    mk, nmk = label((hmin & sub_r).astype(np.uint8), structure=struct2)
    if nmk <= 1:
        return region
    basins = watershed(sub_f, markers=mk, mask=sub_r)
    keep = basin_keep_set(sub_f, sub_r, basins, nmk, gap_frac, prom_frac)
    if keep is None:
        return region
    full_r = np.zeros((ny, nx), dtype=bool)
    full_r[r0:r1, c0:c1] = np.isin(basins, list(keep))
    keep_full = np.roll(full_r, -roll, axis=1) if roll else full_r
    return region & keep_full


# ECMWF L137 hybrid sigma/pressure half-level coefficients
A_HALF = np.array([
    0, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424,
    18.608931, 24.985718, 32.98571, 42.879242, 54.955463, 69.520576,
    86.895882, 107.415741, 131.425507, 159.279404, 191.338562,
    227.968948, 269.539581, 316.420746, 368.982361, 427.592499,
    492.616028, 564.413452, 643.339905, 729.744141, 823.967834,
    926.34491, 1037.201172, 1156.853638, 1285.610352, 1423.770142,
    1571.622925, 1729.448975, 1897.519287, 2076.095947, 2265.431641,
    2465.770508, 2677.348145, 2900.391357, 3135.119385, 3381.743652,
    3640.468262, 3911.490479, 4194.930664, 4490.817383, 4799.149414,
    5119.89502, 5452.990723, 5798.344727, 6156.074219, 6526.946777,
    6911.870605, 7311.869141, 7727.412109, 8159.354004, 8608.525391,
    9076.400391, 9562.682617, 10065.97852, 10584.63184, 11116.66211,
    11660.06738, 12211.54785, 12766.87305, 13324.66895, 13881.33106,
    14432.13965, 14975.61523, 15508.25684, 16026.11523, 16527.32227,
    17008.78906, 17467.61328, 17901.62109, 18308.43359, 18685.71875,
    19031.28906, 19343.51172, 19620.04297, 19859.39063, 20059.93164,
    20219.66406, 20337.86328, 20412.30859, 20442.07813, 20425.71875,
    20361.81641, 20249.51172, 20087.08594, 19874.02539, 19608.57227,
    19290.22656, 18917.46094, 18489.70703, 18006.92578, 17471.83984,
    16888.6875, 16262.04688, 15596.69531, 14898.45313, 14173.32422,
    13427.76953, 12668.25781, 11901.33984, 11133.30469, 10370.17578,
    9617.515625, 8880.453125, 8163.375, 7470.34375, 6804.421875,
    6168.53125, 5564.382813, 4993.796875, 4457.375, 3955.960938,
    3489.234375, 3057.265625, 2659.140625, 2294.242188, 1961.5,
    1659.476563, 1387.546875, 1143.25, 926.507813, 734.992188,
    568.0625, 424.414063, 302.476563, 202.484375, 122.101563,
    62.78125, 22.835938, 3.757813, 0, 0,
])

B_HALF = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.000007, 0.000024, 0.000059, 0.000112, 0.000199, 0.00034, 0.000562,
    0.00089, 0.001353, 0.001992, 0.002857, 0.003971, 0.005378, 0.007133,
    0.009261, 0.011806, 0.014816, 0.018318, 0.022355, 0.026964, 0.032176,
    0.038026, 0.044548, 0.051773, 0.059728, 0.068448, 0.077958, 0.088286,
    0.099462, 0.111505, 0.124448, 0.138313, 0.153125, 0.16891, 0.185689,
    0.203491, 0.222333, 0.242244, 0.263242, 0.285354, 0.308598, 0.332939,
    0.358254, 0.384363, 0.411125, 0.438391, 0.466003, 0.4938, 0.521619,
    0.549301, 0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669,
    0.727739, 0.749797, 0.770798, 0.790717, 0.809536, 0.827256, 0.843881,
    0.859432, 0.873929, 0.887408, 0.8999, 0.911448, 0.922096, 0.931881,
    0.94086, 0.949064, 0.95655, 0.963352, 0.969513, 0.975078, 0.980072,
    0.984542, 0.9885, 0.991984, 0.995003, 0.99763, 1,
])


# data loading

# Load ERA5 U, V, T, SP over a polar cap for the requested date range,
# renaming dims to (lev, lat, lon). Requires era5functions.generate_era5_paths.
def load_era5_subset(start, end, lev_min=11, lev_max=60,
                     lat_min=40, lat_max=90):
    from pathlib import Path
    from era5functions import generate_era5_paths

    var_defs = {
        'u':  ('0_5_0_2_2', 'uv'),
        'v':  ('0_5_0_2_3', 'uv'),
        't':  ('0_5_0_0_0', 'sc'),
        'w':  ('0_5_0_2_8', 'sc'),
        'sp': ('128_134',   'sc'),
    }
    file_lists = {
        var: [f for f in generate_era5_paths(var, code, dom, start, end)
              if Path(f).stem.split('.')[-1].split('_')[0].endswith("00")]
        for var, (code, dom) in var_defs.items()
    }

    def make_sp_preproc(lmin, lmax, latmin, latmax):
        def pre(ds):
            ds = xr.decode_cf(ds)
            if 'level' in ds.dims:
                ds = ds.sel(level=slice(lmin + 1, lmax + 1))
            if 'half_level' in ds.dims:
                ds = ds.sel(half_level=slice(lmin + 1, lmax + 2))
            if 'latitude' in ds.dims:
                ds = ds.sel(latitude=slice(latmax, latmin))
            return ds.isel(time=0)
        return pre

    def make_var_preproc(lmin, lmax, latmin, latmax):
        def pre(ds):
            ds = xr.decode_cf(ds)
            if 'level' in ds.dims:
                ds = ds.sel(level=slice(lmin + 1, lmax + 1))
            if 'latitude' in ds.dims:
                ds = ds.sel(latitude=slice(latmax, latmin))
            return ds.isel(time=0)
        return pre

    pre_var = make_var_preproc(lev_min, lev_max, lat_min, lat_max)
    pre_sp  = make_sp_preproc(lev_min, lev_max, lat_min, lat_max)

    def open_var(key, pre):
        return xr.open_mfdataset(
            file_lists[key], combine='nested', concat_dim='time',
            parallel=True, preprocess=pre, engine='netcdf4',
        )

    ds = xr.merge([
        open_var('u',  pre_var),
        open_var('v',  pre_var),
        open_var('t',  pre_var),
        open_var('sp', pre_sp),
    ])
    return ds.rename({'level': 'lev', 'latitude': 'lat', 'longitude': 'lon'})


# Full-level pressure reconstructed from surface pressure and hybrid
# A/B coefficients. Returns an xarray DataArray on the (lev, lat, lon) grid.
def era5_pressure(ds, sp_name="SP"):
    levels  = ds['lev'].values
    hlevels = np.arange(levels.min() - 1, levels.max() + 1)
    A_da = xr.DataArray(A_HALF[hlevels], dims=['hlevel'],
                        coords={'hlevel': hlevels})
    B_da = xr.DataArray(B_HALF[hlevels], dims=['hlevel'],
                        coords={'hlevel': hlevels})
    sp = ds[sp_name].expand_dims(hlevel=hlevels)
    p_h = A_da + B_da * sp
    p_f = 0.5 * (p_h.isel(time=0, hlevel=slice(0,  -1)).values
                 + p_h.isel(time=0, hlevel=slice(1, None)).values)
    return xr.DataArray(
        p_f,
        dims=['lev', 'lat', 'lon'],
        coords={'lev': levels, 'lat': p_h.lat.values, 'lon': p_h.lon.values},
    )


# Integrate hydrostatic balance top-down to get geopotential phi from
# temperature T and pressure p. Returns an xarray DataArray matching T.
def compute_geopotential(T, p):
    lev = T['lev'].values
    lev_hPa = lev / 100.0 if lev[0] > 1e4 else lev
    phi = np.zeros_like(T.values)
    for i in range(len(lev) - 1, 0, -1):
        p_lower = lev_hPa[i]     * 100.0
        p_upper = lev_hPa[i - 1] * 100.0
        T_mean = 0.5 * (T.isel(lev=i).values + T.isel(lev=i - 1).values)
        d_phi = -R_DRY * T_mean * np.log(p_upper / p_lower)
        phi[i - 1, :, :] = phi[i, :, :] + d_phi
    return xr.DataArray(phi, coords=T.coords, dims=T.dims,
                        attrs={'units': 'm2/s2', 'long_name': 'Geopotential'})


# Add Pa pressure and scale-height altitude coordinates onto the dataset.
def attach_pressure_and_altitude(ds, p):
    pm = p.mean(dim=('lat', 'lon')).values
    ak = -H_SCALE * np.log(pm / P0)
    ds = ds.assign_coords(lev=('lev', pm))
    ds['altitude_km'] = ('lev', ak)
    ds['lev'].attrs['units'] = 'Pa'
    ds['altitude_km'].attrs['units'] = 'km'
    return ds


# Approximate altitude (km) for a pressure (Pa or hPa) using a log-pressure
# scale height. Accepts scalar or array-like input.
def altitude_from_pressure(p):
    arr = np.asarray(p, dtype=float)
    if np.nanmax(arr) < 2000.0:
        arr = arr * 100.0
    return -H_SCALE * np.log(arr / P0)


# periodicity helpers

# Wrap-pad the last axis (longitude) with pad cells from each side.
def pad_lon(arr, pad=LON_PAD):
    if arr.ndim == 2:
        return np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)
    if arr.ndim == 3:
        return np.concatenate([arr[:, :, -pad:], arr, arr[:, :, :pad]], axis=2)
    return arr


# Undo pad_lon.
def unpad_lon(arr, pad=LON_PAD):
    if arr.ndim == 2:
        return arr[:, pad:-pad]
    if arr.ndim == 3:
        return arr[:, :, pad:-pad]
    return arr


# Pad the latitude axis on both ends with zero-valued rows. This closes
# the polar boundary for 2D/3D persistent-homology computations so that
# a ring touching the pole row has a properly enclosed polar interior.
def pad_pole(arr, pad=LAT_PAD):
    pad_shape = list(arr.shape)
    pad_shape[-2] = pad
    zeros = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([zeros, arr, zeros], axis=-2)


# Gaussian smooth a 3D field with periodic longitude handling.
def gaussian_filter_periodic(arr, sigma):
    sigma = tuple(float(s) for s in sigma)
    trunc = 4.0
    pad = max(1, int(np.ceil(trunc * sigma[2])))
    ap = pad_lon(arr, pad)
    sm = gaussian_filter(ap, sigma=sigma,
                         mode=('nearest', 'nearest', 'nearest'))
    return unpad_lon(sm, pad)


# 2D contours of a longitude-periodic mask. Contours are found on the
# lon-padded array and mapped back to [0, nx) to avoid seam artifacts.
def periodic_contours(mask_2d, pad=LON_PAD, min_len=20):
    ny, nx = mask_2d.shape
    pad_mask = pad_lon(mask_2d.astype(float), pad)
    raw = measure.find_contours(pad_mask, 0.5)
    out = []
    for c in raw:
        if c.shape[0] < min_len:
            continue
        cj = c[:, 0]
        ci = (c[:, 1] - pad) % nx
        out.append(np.column_stack([cj, ci]))
    return out


# 3D binary edge map for a mask, computed with longitude periodicity.
def edges_3d_periodic(mask_3d, pad=LON_PAD):
    m = (mask_3d > 0).astype(np.uint8)
    mp = pad_lon(m, pad)
    s = generate_binary_structure(3, 1)
    d = binary_dilation(mp, structure=s).astype(bool)
    e = binary_erosion(mp, structure=s).astype(bool)
    edges = np.logical_xor(d, e)
    return unpad_lon(edges.astype(np.uint8), pad).astype(int)


# Merge connected-component labels that are split only by the 0/360 seam.
# Works on 2D and 3D integer-labelled arrays.
def merge_lon_seam_labels(labels):
    if labels.ndim not in (2, 3):
        return labels
    out = labels.copy()
    max_lab = int(out.max())
    if max_lab <= 0:
        return out
    parent = np.arange(max_lab + 1, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if out.ndim == 2:
        ny, _ = out.shape
        for j in range(ny):
            a = int(out[j, 0])
            if a > 0:
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if 0 <= jj < ny:
                        b = int(out[jj, -1])
                        if b > 0:
                            union(a, b)
            b0 = int(out[j, -1])
            if b0 > 0:
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if 0 <= jj < ny:
                        a2 = int(out[jj, 0])
                        if a2 > 0:
                            union(b0, a2)
    else:
        nz, ny, _ = out.shape
        for k in range(nz):
            for j in range(ny):
                a = int(out[k, j, 0])
                if a > 0:
                    for dk in (-1, 0, 1):
                        kk = k + dk
                        if not (0 <= kk < nz):
                            continue
                        for dj in (-1, 0, 1):
                            jj = j + dj
                            if 0 <= jj < ny:
                                b = int(out[kk, jj, -1])
                                if b > 0:
                                    union(a, b)
                b0 = int(out[k, j, -1])
                if b0 > 0:
                    for dk in (-1, 0, 1):
                        kk = k + dk
                        if not (0 <= kk < nz):
                            continue
                        for dj in (-1, 0, 1):
                            jj = j + dj
                            if 0 <= jj < ny:
                                a2 = int(out[kk, jj, 0])
                                if a2 > 0:
                                    union(b0, a2)

    roots = np.array([find(i) for i in range(max_lab + 1)], dtype=int)
    uniq = np.unique(roots[1:])
    remap = np.zeros(max_lab + 1, dtype=int)
    for new_id, old_id in enumerate(uniq, 1):
        remap[old_id] = new_id
    return remap[roots[out]]


# If a mask touches one or both seam columns and the opposite side has a
# similar field value, extend the mask across the seam. Prevents broken
# components at the prime meridian for features that straddle 0/360.
def enforce_lon_seam_continuity(mask_2d, field_2d, thr, tol):
    m = mask_2d.astype(bool).copy()
    f = np.asarray(field_2d, dtype=np.float64)
    ny, nx = m.shape
    if nx < 3:
        return m
    f_l = f[:, 0]
    f_r = f[:, -1]
    trigger = (
        (m[:, 0] | m[:, -1])
        & (f_l <= (thr + tol))
        & (f_r <= (thr + tol))
        & (np.abs(f_l - f_r) <= tol)
    )
    if np.any(trigger):
        m[trigger, 0] = True
        m[trigger, -1] = True
        m[trigger, 1] = True
        m[trigger, -2] = True
    return m


# geometry helpers

# (ny, nx) grid of cell areas (km^2) for uniform lat/lon spacing.
def grid_cell_areas(lat, lon):
    dlat_r = np.abs(lat[1] - lat[0]) * np.pi / 180.0
    dlon_r = np.abs(lon[1] - lon[0]) * np.pi / 180.0
    cos_lat = np.cos(lat * np.pi / 180.0)
    cell = (R_EARTH ** 2) * cos_lat * dlat_r * dlon_r
    return np.outer(cell, np.ones(len(lon)))


# Projected horizontal footprint area (unique lat/lon cells across all
# levels) and a dict of per-level areas for one component.
def component_areas(ji, ii, ki, lat, lon, cell_areas):
    area_per_level = {}
    for kk in np.unique(ki):
        jj_k = ji[ki == kk]
        ii_k = ii[ki == kk]
        area_per_level[int(kk)] = float(np.sum(cell_areas[jj_k, ii_k]))
    footprint = np.unique(np.stack([ji, ii], axis=1), axis=0)
    total = float(np.sum(cell_areas[footprint[:, 0], footprint[:, 1]]))
    return total, area_per_level


# Latitude smoothing scale (grid cells) for the post-watershed contour. The
# longitude scale is derived per row so the kernel is isotropic in PHYSICAL
# distance; see smooth_mask_polar.
SMOOTH_LAT_SIGMA = 3.5
# cos(lat) floor so the per-row longitude sigma does not blow up at the pole
# (cos -> 0). 0.05 ~ 87 deg; poleward of that all longitudes have collapsed.
COS_LAT_FLOOR = 0.05


# Smooth a binary watershed mask with a kernel that is ISOTROPIC in physical
# distance on the polar lat-lon grid. Latitude is smoothed uniformly
# (SMOOTH_LAT_SIGMA cells); each latitude row is smoothed in longitude with
def smooth_mask_polar(mask, lat, lon, lat_sigma=SMOOTH_LAT_SIGMA):
    if not np.any(mask):
        return mask
    f = mask.astype(float)
    f = gaussian_filter1d(f, lat_sigma, axis=0, mode='nearest')
    dphi = abs(float(np.mean(np.diff(lat)))) or 1.0
    dlam = abs(float(np.mean(np.diff(lon)))) or 1.0
    ratio = dphi / dlam
    coslat = np.clip(np.cos(np.deg2rad(np.asarray(lat))), COS_LAT_FLOOR, 1.0)
    nx = f.shape[1]
    for r in range(f.shape[0]):
        sig_lon = min(lat_sigma * ratio / coslat[r], nx / 4.0)
        if sig_lon > 0.3:
            f[r] = gaussian_filter1d(f[r], sig_lon, mode='wrap')
    return f >= 0.5


# Weighted circular mean of a set of longitudes (degrees).
def circular_mean_lon(lons, w=None):
    r = np.radians(lons)
    if w is None:
        w = np.ones_like(r)
    w = w / w.sum()
    return np.degrees(np.arctan2(
        np.average(np.sin(r), weights=w),
        np.average(np.cos(r), weights=w),
    )) % 360


# 3D weighted centroid of a boolean mask (lat, lon, altitude_km, voxel count).
def centroid_3d(mask, lat, lon, altitude_km, weights=None):
    ki, ji, ii = np.where(mask)
    if len(ki) == 0:
        return None
    w = np.abs(weights[ki, ji, ii]) if weights is not None else np.ones(len(ki))
    ws = w.sum()
    if ws < 1e-12:
        w = np.ones(len(ki)); ws = w.sum()
    w /= ws
    return {
        'lat':         float(np.average(lat[ji], weights=w)),
        'lon':         float(circular_mean_lon(lon[ii], w=w)),
        'altitude_km': float(np.average(altitude_km[ki], weights=w)),
        'n_voxels':    int(len(ki)),
        'n_levels':    int(len(np.unique(ki))),
    }


# Area-weighted geometric centre of a region in the pole-centred azimuthal-
# equidistant plane. Unlike a mean-latitude centroid (which spherical area
# pulls equatorward), a cap centred on the pole centres on the pole here, and
def region_center_polar(region, lat, lon, altitude_km):
    ki, ji, ii = np.where(region)
    if len(ki) == 0:
        return None
    w = np.clip(np.cos(np.deg2rad(np.asarray(lat)[ji])), 0.0, None)
    if w.sum() < 1e-12:
        w = np.ones(len(ki))
    colat = 90.0 - np.asarray(lat)[ji]
    lonr = np.deg2rad(np.asarray(lon)[ii])
    mx = float(np.average(colat * np.cos(lonr), weights=w))
    my = float(np.average(colat * np.sin(lonr), weights=w))
    cc = float(np.hypot(mx, my))
    return {
        'lat':         float(90.0 - cc),
        'lon':         float(np.degrees(np.arctan2(my, mx)) % 360.0),
        'altitude_km': float(np.average(np.asarray(altitude_km)[ki], weights=w)),
        'n_voxels':    int(len(ki)),
        'n_levels':    int(len(np.unique(ki))),
    }


# Per level, the region a (closed) ring ENCLOSES: background cells on the
# poleward side of the ring -- i.e. background that the ring blocks off from
# the equatorward (low-latitude) domain edge. For a closed annulus this is the
def enclosed_by_ring(mask, lat):
    nz, ny, nx = mask.shape
    eq_row = 0 if float(lat[0]) < float(lat[-1]) else ny - 1
    st = generate_binary_structure(2, 1)
    out = np.zeros_like(mask)
    for k in range(nz):
        ring = mask[k]
        if not ring.any():
            continue
        bg = ~ring
        bgp = pad_lon(bg.astype(np.uint8), LON_PAD).astype(bool)
        lbl, nlab = label(bgp, structure=st)
        if nlab == 0:
            continue
        # one labeling pass: background components touching the equatorward
        # edge row are "outside"; whatever the ring walls off is the interior.
        outside_labels = np.unique(lbl[eq_row, :])
        outside_labels = outside_labels[outside_labels > 0]
        outside = (np.isin(lbl, outside_labels) if outside_labels.size
                   else np.zeros_like(bgp))
        outside = unpad_lon(outside.astype(np.uint8), LON_PAD).astype(bool)
        out[k] = bg & ~outside
    return out


# topological helpers

# Drop persistence pairs whose persistence is below a fraction of the
# field range. Infinite deaths are capped at the field range.
def filter_persistence(pairs, field_range, min_persistence_frac=0.15):
    if len(pairs) == 0:
        return pairs
    pf = pairs.copy()
    pf[np.isinf(pf[:, 1]), 1] = field_range
    keep = (pf[:, 1] - pf[:, 0]) > (min_persistence_frac * field_range)
    return pairs[keep]


# Topological strength of the wind ring per level. The polar vortex jet is a
# closed annulus around the pole; on the raw lat-lon grid that annulus touches
# the pole edge and breaks at the longitude seam, so cubical H1 cannot see it.
def wind_ring_persistence_profile(speed, lat, lon, grid=64):
    # The vortex ring is detected by cubical persistence on a small polar grid
    # (pole at centre, no longitude seam). The reprojection is a resample of a
    # REGULAR (lat,lon) grid, so it is plain regular-grid bilinear
    from scipy.ndimage import map_coordinates
    nz = speed.shape[0]
    prof = np.zeros(nz, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ny = lat.size
    nx = lon.size
    rmax = 90.0 - float(lat.min())
    gx = np.linspace(-rmax, rmax, grid)
    GX, GY = np.meshgrid(gx, gx)
    inside = (GX ** 2 + GY ** 2) <= rmax ** 2
    qlat = 90.0 - np.hypot(GX, GY)
    qlon = np.rad2deg(np.arctan2(GY, GX)) % 360.0
    # fractional indices into the regular grid (lat axis is monotone; lon is
    # periodic, handled by grid-wrap). Built once, reused for every level.
    row = np.interp(qlat, lat[::-1], np.arange(ny)[::-1])
    col = (qlon / 360.0) * nx
    coords = np.vstack([row.ravel(), col.ravel()])
    for kk in range(nz):
        vals = speed[kk]
        if (not np.isfinite(vals).any()
                or float(np.nanmax(vals) - np.nanmin(vals)) < 1e-6):
            continue
        reg = map_coordinates(vals, coords, order=1,
                              mode='grid-wrap').reshape(grid, grid)
        valid = inside & np.isfinite(reg)
        if valid.sum() < 9:
            continue
        field = np.where(valid, reg, float(np.nanmax(reg[valid])))
        pers = persistence_2d_on_submanifold(
            -field, valid, fill_high=True, min_persistence_frac=0.0)
        h1 = pers['H1']
        if len(h1):
            prof[kk] = float(np.max(h1[:, 1] - h1[:, 0]))
    return prof


# 2D cubical persistence of a scalar field restricted to a mask.
# Cells outside the mask are filled with a value above (or below) the
# field range so they never participate. Returns {'H0', 'H1'} pair arrays.
def persistence_2d_on_submanifold(field_2d, mask_2d, fill_high=True,
                                  min_persistence_frac=0.15):
    vals = field_2d[mask_2d]
    if len(vals) == 0:
        return {'H0': np.zeros((0, 2)), 'H1': np.zeros((0, 2))}
    fv = float(np.max(vals) + 1) if fill_high else float(np.min(vals) - 1)
    sub = np.full_like(field_2d, fv, dtype=np.float64)
    sub[mask_2d] = field_2d[mask_2d]
    ph = cripser.computePH(sub, maxdim=1)
    H0 = ph[ph[:, 0] == 0, 1:3]
    H1 = ph[ph[:, 0] == 1, 1:3]
    fr = sub.max() - sub.min()
    if fr < 1e-12:
        return {'H0': H0, 'H1': H1}
    return {
        'H0': filter_persistence(H0, fr, min_persistence_frac),
        'H1': filter_persistence(H1, fr, min_persistence_frac),
    }


# Group a sorted set of level indices into contiguous altitude bands,
# tolerating gap_tol missing levels within a band. Returns a list of
# (low_alt_km, high_alt_km) tuples.
def bands_from_levels(lev_set, altitude_km, gap_tol=2):
    if not lev_set:
        return []
    sorted_lev = sorted(int(k) for k in lev_set)
    bands, lo, hi = [], sorted_lev[0], sorted_lev[0]
    for lv in sorted_lev[1:]:
        if lv - hi <= gap_tol + 1:
            hi = lv
        else:
            bands.append((lo, hi))
            lo = hi = lv
    bands.append((lo, hi))
    return [(float(altitude_km[lo]), float(altitude_km[hi]))
            for lo, hi in bands]


# geopotential analysis

# Per-level vortex mask, built in this order:
#   (1) 20th-percentile sublevel threshold of smoothed phi -> a seed
#       region (hole-filled, seam-continuous). Used only to seed the
def analyze_geopotential(phi, altitude_km, percentile=20,
                         broad_radius=16, closing_radius=8):
    lat, lon = phi.lat.values, phi.lon.values
    nz, ny, nx = phi.shape
    phi_v = phi.values.copy()
    phi_s = gaussian_filter_periodic(phi_v, sigma=(0.5, 1.0, 1.0))
    cell_areas = grid_cell_areas(lat, lon)

    mask_per_level     = []
    h0_per_level       = []
    contours_per_level = {}

    # Per-level component filters, applied in order:
    #   (1) min size: drop any component below MIN_COMP_VOXELS.
    #   (2) cap-touching, with companion: if any component reaches
    POLAR_CAP_LAT_DEG = 60.0
    MIN_COMP_VOXELS   = 500
    DEPTH_TOL_FRAC    = 0.50
    cap_lat_m = (lat >= POLAR_CAP_LAT_DEG)[:, None]

    # a component touching the equatorward (40 N) edge and elongated ALONG it
    # (wide in lon, shallow in lat) is truncation material; high aspect ratio
    # away from the edge, and deep lobes that merely reach it, are kept
    EDGE_ALONG_AR_MAX = 4.0
    edge_row = int(np.argmin(lat))      # the 40 N domain edge (lat is descending)

    def is_edge_truncation(m_cc):
        if not m_cc[edge_row].any():
            return False                # not on the truncated edge
        rows = np.where(m_cc.any(axis=1))[0]
        lat_deg = abs(float(lat[rows.max()]) - float(lat[rows.min()]))
        cols = m_cc.any(axis=0)
        if cols.all():
            lon_deg = 360.0
        else:
            cp = np.concatenate([cols, cols])
            longest = cur = 0
            for v in cp:
                if not v:
                    cur += 1
                    longest = max(longest, cur)
                else:
                    cur = 0
            lon_deg = 360.0 * (nx - min(longest, nx)) / nx
        mean_lat = np.deg2rad(
            0.5 * (float(lat[rows.max()]) + float(lat[rows.min()])))
        lat_km = max(lat_deg * 111.0, 1e-6)
        lon_km = lon_deg * 111.0 * max(np.cos(mean_lat), 1e-3)
        return (lon_km / lat_km) >= EDGE_ALONG_AR_MAX

    for k in range(nz):
        fld      = phi_s[k]
        thr_phys = float(np.percentile(fld, percentile))
        fld_pad  = pad_lon(fld, LON_PAD)

        # (1) 20th-percentile narrow mask (seed only)
        # Only the interior (watershed foreground seed) and extent (broad
        # envelope) matter; the boundary is redrawn by the watershed and
        bm_pad = (fld_pad <= thr_phys).astype(np.uint8)
        bm_pad = binary_fill_holes(bm_pad).astype(np.uint8)
        bm_narrow = unpad_lon(bm_pad, LON_PAD)
        local_span = float(np.percentile(fld, 95.0) - np.percentile(fld, 5.0))
        seam_tol = max(0.5, 0.02 * local_span)
        bm_narrow = enforce_lon_seam_continuity(
            bm_narrow, fld, thr_phys, seam_tol)

        if not np.any(bm_narrow):
            mask_per_level.append(np.zeros((ny, nx), dtype=bool))
            h0_per_level.append(0)
            continue

        slice_min_phi = float(np.min(fld))
        slice_max_phi = float(np.max(fld))
        slice_span    = max(slice_max_phi - slice_min_phi, 1e-9)
        depth_tol     = DEPTH_TOL_FRAC * slice_span

        # (2) expand to the broad submanifold
        bm_broad_pad = binary_dilation(
            pad_lon(bm_narrow, LON_PAD).astype(np.uint8),
            structure=morphology.disk(broad_radius), iterations=1).astype(np.uint8)
        broad_pad = bm_broad_pad > 0

        # (3) watershed: boundary on the steepest gradient
        # Within the envelope, place the boundary on the discrete-Morse
        # separatrix (ridge of |grad phi|), seeded by the deep narrow
        grad_pad   = np.hypot(*np.gradient(fld_pad))
        narrow_pad = pad_lon(bm_narrow.astype(np.uint8), LON_PAD) > 0
        fg = binary_erosion(narrow_pad, structure=morphology.disk(4))
        if not fg.any():
            fg = narrow_pad
        markers = np.zeros(fld_pad.shape, dtype=int)
        markers[~broad_pad] = 1
        markers[fg] = 2
        # The vortex (ws == 2) lies entirely within the broad envelope, and
        # everything outside it is marker 1, so the expensive flood only needs
        # to run inside the envelope's bounding box (+ a small margin so the
        rows_b = np.where(broad_pad.any(axis=1))[0]
        cols_b = np.where(broad_pad.any(axis=0))[0]
        wmar = 2
        wr0 = max(0, int(rows_b[0]) - wmar)
        wr1 = min(broad_pad.shape[0], int(rows_b[-1]) + 1 + wmar)
        wc0 = max(0, int(cols_b[0]) - wmar)
        wc1 = min(broad_pad.shape[1], int(cols_b[-1]) + 1 + wmar)
        ws_sub = watershed(grad_pad[wr0:wr1, wc0:wc1],
                           markers=markers[wr0:wr1, wc0:wc1])
        ws = np.ones(fld_pad.shape, dtype=ws_sub.dtype)   # outside = marker 1
        ws[wr0:wr1, wc0:wc1] = ws_sub
        bm = unpad_lon((ws == 2).astype(np.uint8), LON_PAD).astype(bool)

        # (3b) cut merged low-phi troughs at the saddle
        # The gradient watershed keeps everything inside the wrapped contour,
        # so a displaced vortex that merged with a lower-latitude trough keeps
        bm = cut_to_vortex_basin(fld, bm)

        # (4) refine: smooth edges, fill holes
        # The watershed divide traces |grad phi| cell-by-cell, so the raw
        # edge is jagged at the single-pixel scale. Round it with a kernel
        if np.any(bm):
            bm = smooth_mask_polar(bm, lat, lon)
            bp = pad_lon(bm.astype(np.uint8), LON_PAD)
            bp = binary_fill_holes(bp).astype(np.uint8)
            bm = unpad_lon(bp, LON_PAD).astype(bool)

        # (5) drop edge-truncation components: those touching the 40 N edge and
        # elongated along it (high lon/lat aspect there). filaments and deep
        # lobes that merely touch the edge are kept
        if np.any(bm):
            lbl0, n0 = label(bm.astype(np.uint8),
                             structure=generate_binary_structure(2, 2))
            keep = np.zeros_like(bm, dtype=bool)
            for cc in range(1, n0 + 1):
                m = (lbl0 == cc)
                if not is_edge_truncation(m):
                    keep |= m
            bm = keep

        # (6) remove small components + cap/depth-companion
        if np.any(bm):
            lbl2d = merge_lon_seam_labels(
                label(bm.astype(np.uint8),
                      structure=generate_binary_structure(2, 2))[0])
            cc_info = {}
            for cc in np.unique(lbl2d[lbl2d > 0]):
                m_cc = (lbl2d == cc)
                cc_info[cc] = {
                    'mask':        m_cc,
                    'size':        int(m_cc.sum()),
                    'min_phi':     float(np.min(fld[m_cc])),
                    'touches_cap': bool(np.any(m_cc & cap_lat_m)),
                }
            cc_info = {cc: d for cc, d in cc_info.items()
                       if d['size'] >= MIN_COMP_VOXELS}
            if cc_info:
                cap_ids = [cc for cc, d in cc_info.items() if d['touches_cap']]
                if cap_ids:
                    cap_min_phi = min(cc_info[cc]['min_phi'] for cc in cap_ids)
                    keep_ids = [cc for cc, d in cc_info.items()
                                if d['touches_cap']
                                or d['min_phi'] <= cap_min_phi + depth_tol]
                else:
                    keep_ids = list(cc_info.keys())
                keep = np.zeros_like(bm, dtype=bool)
                for cc in keep_ids:
                    keep |= cc_info[cc]['mask']
                bm = keep
            else:
                bm = np.zeros_like(bm, dtype=bool)

        mask_per_level.append(bm > 0)
        # Per-level H0 basin count, taken FROM the final mask itself:
        # the number of connected components in bm (seam-merged so a
        # piece straddling 0/360 counts once).  No separate diagnostic
        if np.any(bm):
            bl = merge_lon_seam_labels(
                label(bm.astype(np.uint8),
                      structure=generate_binary_structure(2, 2))[0])
            n_basins = int(np.unique(bl[bl > 0]).size)
        else:
            n_basins = 0
        h0_per_level.append(n_basins)

        # Draw the contour from the same mask `bm` used downstream.
        contours = periodic_contours(
            (bm > 0).astype(float), pad=LON_PAD, min_len=20)
        if contours:
            contours_per_level[k] = contours

    # 3D link with longitude wrapping
    mask3d     = np.stack(mask_per_level, axis=0)
    struct3    = generate_binary_structure(3, 3)
    mask3d_pad = pad_lon(mask3d, LON_PAD)
    mask3d_pad = binary_closing(mask3d_pad, structure=struct3, iterations=2)
    linked_pad, _ = label(mask3d_pad.astype(int), structure=struct3)
    linked         = linked_pad[:, :, LON_PAD:LON_PAD + nx]

    unique_labs = np.unique(linked[linked > 0])
    if len(unique_labs) > 0:
        remap = np.zeros(int(linked.max()) + 1, dtype=int)
        for new_id, old_id in enumerate(unique_labs, 1):
            remap[old_id] = new_id
        linked = remap[linked]

    # Filter candidate 3D components by size, vertical coherence, and a
    # short-edge reject. A component is a spurious mid-latitude feature
    # only if BOTH:
    MIN_MAX_SLICE_VOXELS = 2000
    SHORT_TOP_ALT_KM     = 33.0
    EDGE_LAT_DEG = 41.0
    candidates = []
    for c in range(1, int(linked.max()) + 1):
        m  = (linked == c)
        sz = int(np.sum(m))
        if sz < 500:
            continue
        ki, ji, ii = np.where(m)
        if len(np.unique(ki)) < 5:
            continue
        top_alt_km = float(altitude_km[int(ki.min())])
        touches_40 = bool(float(lat[ji].min()) <= EDGE_LAT_DEG)
        if touches_40 and top_alt_km <= SHORT_TOP_ALT_KM:
            continue
        # 3D pillar rejection: thickest slice must be substantial
        max_slice_vox = int(m.sum(axis=(1, 2)).max())
        if max_slice_vox < MIN_MAX_SLICE_VOXELS:
            continue
        candidates.append((c, sz, m))

    # persistence-based merge of adjacent candidates by saddle depth
    if len(candidates) > 1:
        merge_map = {c[0]: c[0] for c in candidates}
        for i in range(len(candidates)):
            ci, _, mi = candidates[i]
            mi_pad = pad_lon(mi, LON_PAD)
            mi_dil = unpad_lon(
                binary_dilation(
                    mi_pad.astype(np.uint8),
                    structure=generate_binary_structure(3, 3),
                    iterations=2,
                ).astype(bool),
                LON_PAD,
            )
            for j in range(i + 1, len(candidates)):
                cj, _, mj = candidates[j]
                boundary = mi_dil & mj
                if not np.any(boundary):
                    mj_pad = pad_lon(mj, LON_PAD)
                    mj_dil = unpad_lon(
                        binary_dilation(
                            mj_pad.astype(np.uint8),
                            structure=generate_binary_structure(3, 3),
                            iterations=2,
                        ).astype(bool),
                        LON_PAD,
                    )
                    boundary = mj_dil & mi
                if not np.any(boundary):
                    continue
                saddle = float(phi_s[boundary].min())
                if saddle <= max(float(phi_s[mi].max()),
                                 float(phi_s[mj].max())):
                    ri, rj = merge_map[ci], merge_map[cj]
                    if ri != rj:
                        for kk in merge_map:
                            if merge_map[kk] == rj:
                                merge_map[kk] = ri

        groups = defaultdict(list)
        for c, sz, m in candidates:
            groups[merge_map[c]].append((c, sz, m))
        candidates = []
        for root, members in groups.items():
            combined = np.zeros_like(linked, dtype=bool)
            total_sz = 0
            for c, sz, m in members:
                combined |= m
                total_sz += sz
            candidates.append((root, total_sz, combined))
        candidates.sort(key=lambda x: x[1], reverse=True)

    # build final labelled array and per-component metadata
    filt  = np.zeros_like(linked)
    comps = []
    nid   = 1
    for _, sz, m in candidates:
        ki, ji, ii = np.where(m)

        # Apply the touch-40 + short reject HERE, on the FINAL merged
        # component, not only pre-merge.  The merge step can assemble a
        # lobe out of pieces that individually passed the pre-merge
        top_alt_km = float(altitude_km[int(ki.min())])
        touches_40 = bool(float(lat[ji].min()) <= EDGE_LAT_DEG)
        if touches_40 and top_alt_km <= SHORT_TOP_ALT_KM:
            continue

        filt[m] = nid
        level_indices  = np.unique(ki)
        top_lev_idx    = int(level_indices.min())
        bottom_lev_idx = int(level_indices.max())
        jj_bot = ji[ki == bottom_lev_idx]
        ii_bot = ii[ki == bottom_lev_idx]

        total_area_km2, area_per_level = component_areas(
            ji, ii, ki, lat, lon, cell_areas)

        # per-level shape metrics
        ar_levels = []
        lev_records = []
        for kk in level_indices:
            jj_k = ji[ki == kk]
            ii_k = ii[ki == kk]
            n_vox_k = len(jj_k)
            if n_vox_k < 5:
                continue
            clat_k    = float(np.mean(lat[jj_k]))
            clon_k    = float(circular_mean_lon(lon[ii_k]))
            lat_min_k = float(lat[jj_k].min())
            lat_max_k = float(lat[jj_k].max())
            lons_k    = lon[ii_k]
            lr        = float(np.ptp(
                ((lons_k - lons_k.mean() + 180) % 360) - 180))

            # aspect ratio from area-weighted covariance eigenvalues in an
            # azimuthal-equidistant projection centred on the pole
            if len(jj_k) >= 4:
                colat_k = (90.0 - lat[jj_k]) * 111.0
                lon_r_k = np.radians(lon[ii_k])
                ax_k    = colat_k * np.sin(lon_r_k)
                ay_k    = colat_k * np.cos(lon_r_k)
                w_k     = colat_k
                w_k     = w_k / w_k.sum()
                mx_k    = float(np.sum(w_k * ax_k))
                my_k    = float(np.sum(w_k * ay_k))
                dx_k    = ax_k - mx_k;  dy_k = ay_k - my_k
                cxx = float(np.sum(w_k * dx_k * dx_k))
                cyy = float(np.sum(w_k * dy_k * dy_k))
                cxy = float(np.sum(w_k * dx_k * dy_k))
                eigs = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
                eigs = np.maximum(eigs, 0.0)
                ar_k = (float(np.sqrt(eigs[1] / eigs[0]))
                        if eigs[0] > 1e-6 else 1.0)
            else:
                ar_k = 1.0
            ar_levels.append(ar_k)

            phi_k = float(np.mean(phi_v[kk, jj_k, ii_k]))
            lev_records.append({
                'level':             int(kk),
                'altitude_km':       float(altitude_km[kk]),
                'centroid_lat':      clat_k,
                'centroid_lon':      clon_k,
                'pole_distance_deg': 90.0 - clat_k,
                'lat_equatorward':   lat_min_k,
                'lat_poleward':      lat_max_k,
                'lat_spread_deg':    lat_max_k - lat_min_k,
                'lon_spread_deg':    lr,
                'aspect_ratio':      ar_k,
                'n_voxels':          n_vox_k,
                'area_km2':          area_per_level.get(int(kk), 0.0),
                'mean_phi':          phi_k,
            })

        # Per-level beta_0 profile: at each level, count the 2D
        # connected components of THIS 3D component's slice. A real
        # single lobe has beta_0 = 1 at every level it occupies. A
        b0_profile = np.zeros(nz, dtype=np.int32)
        struct2d = generate_binary_structure(2, 2)
        for kk in range(nz):
            slc = m[kk]
            if not slc.any():
                continue
            slc_lbl, _ = label(slc.astype(np.uint8), structure=struct2d)
            slc_lbl = merge_lon_seam_labels(slc_lbl)
            b0_profile[kk] = int(np.unique(slc_lbl[slc_lbl > 0]).size)

        cent = centroid_3d(m, lat, lon, altitude_km, weights=-phi_v)
        cent.update({
            'mean_phi':           float(np.mean(phi_v[m])),
            'min_phi':            float(np.min(phi_v[m])),
            'component_id':       nid,
            'n_voxels':           int(sz),
            'n_levels':           int(len(level_indices)),
            'top_level_idx':      top_lev_idx,
            'top_altitude_km':    float(altitude_km[top_lev_idx]),
            'bottom_level_idx':   bottom_lev_idx,
            'bottom_altitude_km': float(altitude_km[bottom_lev_idx]),
            'lowest_lat':         float(lat[ji].min()),
            'centroid_bottom': {
                'lat':         float(np.mean(lat[jj_bot]))
                               if len(jj_bot) else cent['lat'],
                'lon':         float(circular_mean_lon(lon[ii_bot]))
                               if len(ii_bot) else cent['lon'],
                'altitude_km': float(altitude_km[bottom_lev_idx]),
            },
            'aspect_ratio_mean':   float(np.mean(ar_levels))
                                   if ar_levels else 1.0,
            'aspect_ratio_bottom': float(ar_levels[-1])
                                   if ar_levels else 1.0,
            'total_area_km2':      total_area_km2,
            'b0_profile':          b0_profile,
            'level_centroids':     lev_records,
        })
        comps.append(cent)
        nid += 1

    filt  = merge_lon_seam_labels(filt)

    # bridging (final step)
    # Close small within-level gaps and notches in each lobe AFTER the
    # watershed has placed the contour -- not on the seed, so the gradient
    if np.any(filt > 0):
        dk = morphology.disk(int(closing_radius))
        bridged = np.zeros_like(filt)
        for cid in np.unique(filt[filt > 0]):
            cm = (filt == cid)
            for k in range(nz):
                if not cm[k].any():
                    continue
                cp = pad_lon(cm[k].astype(np.uint8), LON_PAD)
                cp = morphology.closing(cp, dk)
                cp = binary_fill_holes(cp).astype(np.uint8)
                sl = unpad_lon(cp, LON_PAD).astype(bool)
                # only fill into cells not already claimed by another label
                bridged[k][sl & (bridged[k] == 0)] = cid
        filt = bridged

    b0    = len(comps)
    edges = edges_3d_periodic(filt, pad=LON_PAD)

    return {
        'labels':             filt,
        'edges':              edges,
        'betti0':             b0,
        'betti1':             0,
        'components':         comps,
        'contours_per_level': contours_per_level,
        'h0_per_level':       h0_per_level,
    }


# wind ring analysis

# Per-slice ring-altitude test for a confirmed 3D ring component.
# A level K is included as a ring altitude if the mask at K encircles
# the pole on its own (SOLO), or the union of the mask at K with the
def ring_altitude_ranges(mask, speed, altitude_km, lat=None,
                         speed_floor=0.0, gap_tol=2):
    nz, ny, nx = mask.shape

    def encircles_pole(sl_mask_2d):
        if not np.any(sl_mask_2d):
            return False
        mp = pad_lon(sl_mask_2d.astype(np.uint8), LON_PAD) > 0
        mp = pad_pole(mp.astype(np.uint8), LAT_PAD) > 0
        comp = ~mp
        if not np.any(comp):
            return False
        lbl, n = label(comp.astype(np.uint8),
                       structure=generate_binary_structure(2, 2))
        if n < 2:
            return False
        top_labels = set(int(v) for v in np.unique(lbl[0,  :]))
        bot_labels = set(int(v) for v in np.unique(lbl[-1, :]))
        top_labels.discard(0)
        bot_labels.discard(0)
        if not top_labels or not bot_labels:
            return False
        return len(top_labels & bot_labels) == 0

    def h1_on_slice(sl_mask_2d, spd_2d):
        if np.sum(sl_mask_2d) < 8:
            return False
        if speed_floor > 0.0:
            mean_sp = float(np.mean(spd_2d[sl_mask_2d]))
            if mean_sp < speed_floor:
                return False
        return encircles_pole(sl_mask_2d)

    # SOLO test: a level is a ring level if its mask alone encircles the pole
    solo = {}
    for kk in range(nz):
        sl = mask[kk]
        if not np.any(sl):
            continue
        solo[kk] = h1_on_slice(sl, speed[kk])

    # PAIR test: a non-SOLO level qualifies if pairing with K+/-1 creates
    # a pole-encircling union AND the neighbour is not SOLO on its own.
    # This rejects unicorn-horn spikes sitting on top of an annular neighbour.
    ring_levels = set()
    for kk, has_solo in solo.items():
        if has_solo:
            ring_levels.add(kk)
            continue
        for dk in (-1, +1):
            kn = kk + dk
            if kn not in solo or solo[kn]:
                continue
            sl_union = mask[kk] | mask[kn]
            if h1_on_slice(sl_union, np.maximum(speed[kk], speed[kn])):
                ring_levels.add(kk)
                break

    return bands_from_levels(ring_levels, altitude_km, gap_tol), ring_levels


# Smooth the stored ring mask per level with periodic-lon morphology,
# fill tiny holes, and keep only the largest 3D component (also with
# lon-wrap). Used before per-level diagnostics so seam fragments are
def smooth_ring_mask(mask, constraint, close_radius=3, open_radius=1):
    m = mask.astype(bool)
    cons = constraint.astype(bool)
    if not np.any(m):
        return m
    nz, ny, nx = m.shape
    out = np.zeros_like(m, dtype=bool)
    dk_close = morphology.disk(int(close_radius))
    dk_open  = morphology.disk(int(open_radius))
    s8  = generate_binary_structure(2, 2)
    s26 = generate_binary_structure(3, 3)
    for k in range(nz):
        sl = m[k] & cons[k]
        if not np.any(sl):
            continue
        sp = pad_lon(sl.astype(np.uint8), LON_PAD) > 0
        sp = morphology.binary_closing(sp, dk_close)
        sp = binary_fill_holes(sp)
        sp = morphology.binary_opening(sp, dk_open)
        lbl_pad, n = label(sp.astype(np.uint8), structure=s8)
        if n > 0:
            sizes = np.bincount(lbl_pad.ravel())
            sizes[0] = 0
            best = int(np.argmax(sizes))
            largest_pad = (lbl_pad == best)
            su = unpad_lon(largest_pad.astype(np.uint8), LON_PAD) > 0
        else:
            su = unpad_lon(sp.astype(np.uint8), LON_PAD) > 0
        out[k] = su & cons[k]

    out_pad = pad_lon(out.astype(np.uint8), LON_PAD)
    lbl3_pad, n3 = label(out_pad.astype(np.uint8), structure=s26)
    if n3 > 0:
        sizes3 = np.bincount(lbl3_pad.ravel())
        sizes3[0] = 0
        best3 = int(np.argmax(sizes3))
        out_pad = (lbl3_pad == best3).astype(np.uint8)
        out = unpad_lon(out_pad, LON_PAD) > 0
    else:
        out = np.zeros_like(out, dtype=bool)
    return out.astype(bool)


# Refine a jet mask per level by placing its boundary on the steepest
# wind-speed gradient -- the discrete-Morse separatrix -- instead of the
# raw speed threshold. For each level slice of the seed mask: dilate to a
def smooth_jet_mask(mask, sign_mask=None):
    # Stored wind contour: just anisotropically smooth the seed mask (more
    # in lon than lat), confined to same-sign flow. No dilation/watershed --
    # the expand-only watershed did not move the edge off the seed
    nz = mask.shape[0]
    out = np.zeros_like(mask, dtype=bool)
    for k in range(nz):
        mk = mask[k]
        if not np.any(mk):
            continue
        mp = pad_lon(mk.astype(float), LON_PAD)
        mp = (gaussian_filter(mp, WIND_SMOOTH_SIGMA) >= 0.5)
        if sign_mask is not None:
            mp &= pad_lon(sign_mask[k].astype(np.uint8), LON_PAD) > 0
        out[k] = unpad_lon(mp.astype(np.uint8), LON_PAD).astype(bool)
    return out


def refine_jet_boundary(speed, seed_mask, sign_mask=None,
                        return_boundary=False, grad_pad_stack=None,
                        sp_pad_stack=None):
    # Strength diagnostic only. The SEARCH REGION is the component's own
    # footprint confined to same-sign flow (no dilation). Within it, a
    # contract-capable watershed of |grad speed| places the divide on the
    nz, ny, nx = speed.shape
    out = np.zeros_like(seed_mask, dtype=bool)
    bnd = np.zeros_like(seed_mask, dtype=bool) if return_boundary else None
    for k in range(nz):
        sk = seed_mask[k]
        if not np.any(sk):
            continue
        sp_pad = (sp_pad_stack[k] if sp_pad_stack is not None
                  else pad_lon(speed[k], LON_PAD))
        env = pad_lon(sk.astype(np.uint8), LON_PAD) > 0
        if sign_mask is not None:
            # confine the search region to same-sign (coherent) flow.
            env &= pad_lon(sign_mask[k].astype(np.uint8), LON_PAD) > 0
        if not np.any(env):
            continue
        grad = (grad_pad_stack[k] if grad_pad_stack is not None
                else np.hypot(*np.gradient(sp_pad)))
        # contract-capable: divide free anywhere in the search region.
        sv = sp_pad[env]
        fg = env & (sp_pad >= float(np.percentile(sv, 90)))
        bg = env & (sp_pad <= float(np.percentile(sv, 10)))
        if not fg.any():
            fg = env & (sp_pad >= float(np.percentile(sv, 75)))
        if not bg.any():
            bg = ~env
        markers = np.zeros(sp_pad.shape, dtype=int)
        markers[bg] = 1
        markers[fg] = 2
        wl = return_boundary
        ws = watershed(grad, markers=markers, mask=env, watershed_line=wl)
        ref = (ws == 2)
        if return_boundary:
            # ws==0 is BOTH the divide line AND everything outside the
            # masked search region; keep only the divide inside env.
            bnd[k] = unpad_lon(((ws == 0) & env).astype(np.uint8),
                               LON_PAD).astype(bool)
        ref = (gaussian_filter(ref.astype(float), WIND_SMOOTH_SIGMA) >= 0.5)
        if sign_mask is not None:
            ref &= pad_lon(sign_mask[k].astype(np.uint8), LON_PAD) > 0
        out[k] = unpad_lon(ref.astype(np.uint8), LON_PAD).astype(bool)
    if return_boundary:
        return out, bnd
    return out


# Discrete Morse ring detection on a seed subcomplex. Labels 3D
# seed components, finds saddles between them, merges by decreasing
# saddle speed. On each circumpolar-candidate union the jet boundary is
def morse_ring_from_seeds(speed, seeds, sign_mask, nx, lat, altitude_km):
    nz, ny, _ = speed.shape
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, :] = True
    struct6[1, :, 1] = True
    struct6[:, 1, 1] = True

    seeds_pad = pad_lon(seeds.astype(np.uint8), LON_PAD)
    lp, n = label(seeds_pad.astype(int), structure=struct6)
    comp_labels = lp[:, :, LON_PAD:LON_PAD + nx]
    if n == 0:
        return None, 0.0, None

    # find saddles between every adjacent pair of seed components
    saddle = {}
    for axis, roll in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        c2 = np.roll(comp_labels, roll, axis=axis)
        if axis == 1:
            c2 = c2.copy()
            if roll == -1:
                c2[:, -1, :] = 0
            else:
                c2[:, 0, :] = 0
        s2 = np.roll(speed, roll, axis=axis)
        inter = seeds & (c2 > 0) & (comp_labels != c2) & (comp_labels > 0)
        if not np.any(inter):
            continue
        ki, ji, ii = np.where(inter)
        ca = comp_labels[ki, ji, ii]
        cb = c2[ki, ji, ii]
        sv = np.minimum(speed[ki, ji, ii], s2[ki, ji, ii])
        lo = np.minimum(ca, cb).astype(np.int64)
        hi = np.maximum(ca, cb).astype(np.int64)
        keys = lo * (n + 1) + hi
        order = np.argsort(keys)
        ks = keys[order]
        vs = sv[order]
        splits = np.where(np.diff(ks))[0] + 1
        for seg in np.split(np.arange(len(ks)), splits):
            k_val = int(ks[seg[0]])
            v_max = float(vs[seg].max())
            a, b  = int(k_val // (n + 1)), int(k_val % (n + 1))
            key   = (a, b)
            if key not in saddle or v_max > saddle[key]:
                saddle[key] = v_max

    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def is_circumpolar(root, parent):
        members = [c for c in range(1, n + 1) if find(c, parent) == root]
        m = np.zeros((nz, ny, nx), dtype=bool)
        for c in members:
            m |= (comp_labels == c)
        m &= sign_mask
        if not np.any(m):
            return False, m
        # Cheap geometry gates on the RAW union first (avoids the costly
        # watershed on every intermediate merge candidate).
        ki2 = np.where(m)[0]
        level_counts = np.bincount(ki2, minlength=nz)
        best_k = int(np.argmax(level_counts))
        if np.sum(m[best_k]) < 8:
            return False, m
        if len(np.unique(np.where(m[best_k])[1])) / nx < 0.50:
            return False, m
        # Gates passed -> run H1 on the REFINED (smoothed) best-level slice.
        # Only the one slice the persistence test uses is smoothed, so this
        # tests the refined contour without smoothing the whole 3D mask per
        ki2 = np.where(m)[0]
        level_counts = np.bincount(ki2, minlength=nz)
        best_k = int(np.argmax(level_counts))
        sl = m[best_k]
        if np.sum(sl) < 8:
            return False, m
        sl_sm = (gaussian_filter(pad_lon(sl.astype(float), LON_PAD),
                                 WIND_SMOOTH_SIGMA) >= 0.5)
        sl_sm = unpad_lon(sl_sm.astype(np.uint8), LON_PAD).astype(bool)
        sl_sm &= sign_mask[best_k]
        if np.sum(sl_sm) < 8:
            return False, m
        spd_pad = pad_pole(pad_lon(speed[best_k], LON_PAD), LAT_PAD)
        msk_pad = pad_pole(
            pad_lon(sl_sm.astype(np.uint8), LON_PAD), LAT_PAD) > 0
        pers = persistence_2d_on_submanifold(
            -spd_pad, msk_pad, fill_high=False, min_persistence_frac=0.08)
        return len(pers['H1']) > 0, m

    # single-component check before any merging
    parent = list(range(n + 1))
    for c in range(1, n + 1):
        ok, m = is_circumpolar(c, parent)
        if ok:
            floor = float(speed[m].min()) if np.any(m) else 0.0
            return m & sign_mask, floor, m

    if not saddle:
        return None, 0.0, None

    saddles = sorted([(v, a, b) for (a, b), v in saddle.items()], reverse=True)
    parent  = list(range(n + 1))
    for sv, ca, cb in saddles:
        ra, rb = find(ca, parent), find(cb, parent)
        if ra == rb:
            continue
        parent[rb] = ra
        ok, m = is_circumpolar(ra, parent)
        if ok:
            return m & sign_mask, float(sv), m

    return None, 0.0, None


# Detect westerly and easterly wind rings from the zonal wind U.
# Westerly seeds = U > 0 with speed above the `speed_percentile` percentile;
# easterly seeds = U < 0 with a smaller absolute-speed floor. Each sign is
def analyze_wind(U, altitude_km, speed_percentile=60):
    lat, lon = U.lat.values, U.lon.values
    Uv = U.values.copy()
    nz, ny, nx = Uv.shape
    cell_areas = grid_cell_areas(lat, lon)

    Us    = gaussian_filter_periodic(Uv, sigma=(1.0, 1.0, 1.0))
    speed = np.abs(Us)

    # Precompute the padded per-level speed and |grad speed| once; the
    # strength diagnostic reuses these across every saved component instead
    # of re-padding and re-differencing per call.
    sp_pad_stack = [pad_lon(speed[k], LON_PAD) for k in range(nz)]
    grad_pad_stack = [np.hypot(*np.gradient(sp)) for sp in sp_pad_stack]

    # Topological STRENGTH of the wind ring: persistence of the dominant H1
    # (loop) generator of the wind-speed super-level filtration, evaluated in
    # a polar (pole-at-centre) projection so the vortex annulus is a genuine
    ring_persistence_prof = wind_ring_persistence_profile(speed, lat, lon)
    ring_persistence_max = (float(np.max(ring_persistence_prof))
                            if nz else float('nan'))

    speed_thr   = float(np.percentile(speed, speed_percentile))
    east_floor  = max(2.5, 0.20 * speed_thr)

    west_seeds = (Uv > 0) & (speed >= speed_thr)
    east_seeds = (Uv < 0) & (speed >= east_floor)
    west_sign  = (Uv > 0)
    east_sign  = (Uv < 0)

    west_ring, west_floor, _ = morse_ring_from_seeds(
        speed, west_seeds, west_sign, nx, lat, altitude_km)
    east_ring, east_floor, _ = morse_ring_from_seeds(
        speed, east_seeds, east_sign, nx, lat, altitude_km)

    west_is_ring = (west_ring is not None
                    and np.sum(west_ring) >= 100)
    east_is_ring = (east_ring is not None
                    and np.sum(east_ring) >= 100)

    filt  = np.zeros((nz, ny, nx), dtype=np.int32)
    comps = []
    nid   = [1]

    # level nearest to 10 hPa (1000 Pa) and latitude nearest to 60 N,
    # for the pct_10hPa_60lat diagnostic
    lev_pa = P0 * np.exp(-altitude_km / H_SCALE)
    lev_10 = int(np.argmin(np.abs(lev_pa - 1000.0)))
    lat_60 = int(np.argmin(np.abs(lat - 60.0)))
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, :] = True
    struct6[1, :, 1] = True
    struct6[:, 1, 1] = True

    def add_comp(m, is_ring, floor_g):
        m = m.astype(bool)
        if np.sum(m) < 300:
            return
        ki, ji, ii = np.where(m)
        if len(np.unique(ki)) < 5:
            return

        total_area_km2, area_per_level = component_areas(
            ji, ii, ki, lat, lon, cell_areas)

        mean_U       = float(np.mean(Uv[m]))
        max_U        = float(np.max(Uv[m]))
        min_U        = float(np.min(Uv[m]))
        greatest_mag = max_U if abs(max_U) >= abs(min_U) else min_U
        sign         = 'westerly' if mean_U > 0 else 'easterly'
        nlev         = len(np.unique(ki))

        filt[m] = nid[0]

        # Per-component gradient-refined wind STRENGTH diagnostic. Run the
        # FULL (contract-capable) watershed on this component's footprint
        # to find the sharpest-gradient jet region, then average speed
        comp_sign = (Uv > 0) if sign == 'westerly' else (Uv < 0)
        gr_region, gr_bnd = refine_jet_boundary(
            speed, m, sign_mask=comp_sign, return_boundary=True,
            grad_pad_stack=grad_pad_stack, sp_pad_stack=sp_pad_stack)
        grad_refined_region_speed = (float(np.mean(speed[gr_region]))
                                     if np.any(gr_region) else np.nan)
        grad_refined_boundary_speed = (float(np.mean(speed[gr_bnd]))
                                       if np.any(gr_bnd) else np.nan)

        # per-longitude jet core curve: for each longitude we locate the
        # single (level, lat) voxel inside the component where the wind
        # speed is maximal. The resulting (core_lat, core_alt, core_speed)
        masked = np.where(m, speed, -np.inf)                  # (nz, ny, nx)
        flat   = masked.reshape(nz * ny, nx)
        flat_idx = np.argmax(flat, axis=0)                    # (nx,)
        kk_star  = flat_idx // ny
        jj_star  = flat_idx %  ny
        has_any  = m.any(axis=(0, 1))                         # (nx,)
        lon_idx  = np.arange(nx)

        core_lat_arr   = np.where(has_any, lat        [jj_star],
                                  np.nan).astype(np.float32)
        core_alt_arr   = np.where(has_any, altitude_km[kk_star],
                                  np.nan).astype(np.float32)
        core_speed_arr = np.where(has_any,
                                  speed[kk_star, jj_star, lon_idx],
                                  np.nan).astype(np.float32)
        core_U_arr     = np.where(has_any,
                                  Uv   [kk_star, jj_star, lon_idx],
                                  np.nan).astype(np.float32)

        # speed-weighted linear regression of core latitude on altitude.
        # slope > 0 = core tilts poleward with height (positive deg/km);
        # slope < 0 = core tilts equatorward with height.
        valid = (np.isfinite(core_lat_arr)
                 & np.isfinite(core_alt_arr)
                 & np.isfinite(core_speed_arr))
        if valid.sum() >= 3:
            x = core_alt_arr  [valid].astype(np.float64)
            y = core_lat_arr  [valid].astype(np.float64)
            w = core_speed_arr[valid].astype(np.float64)
            W = w.sum()
            if W > 1e-9:
                x_mean = float((w * x).sum() / W)
                y_mean = float((w * y).sum() / W)
                sxx = float((w * (x - x_mean) ** 2).sum())
                sxy = float((w * (x - x_mean) * (y - y_mean)).sum())
                if sxx > 1e-9:
                    tilt_slope     = sxy / sxx
                    tilt_intercept = y_mean - tilt_slope * x_mean
                    y_pred = tilt_slope * x + tilt_intercept
                    ss_res = float((w * (y - y_pred) ** 2).sum())
                    ss_tot = float((w * (y - y_mean) ** 2).sum())
                    tilt_r2 = (1.0 - ss_res / ss_tot
                               if ss_tot > 1e-9 else np.nan)
                else:
                    tilt_slope = tilt_intercept = tilt_r2 = np.nan
            else:
                tilt_slope = tilt_intercept = tilt_r2 = np.nan
        else:
            tilt_slope = tilt_intercept = tilt_r2 = np.nan

        # per-level shape diagnostics (no jet-core fields here; the core
        # is tracked per longitude only)
        inner_lats = []
        outer_lats = []
        per_level_lats = []
        for kk in np.unique(ki):
            jj_k = ji[ki == kk]
            ii_k = ii[ki == kk]
            n_vox_k = len(jj_k)
            lat_in  = float(lat[jj_k].max())
            lat_out = float(lat[jj_k].min())
            inner_lats.append(lat_in)
            outer_lats.append(lat_out)
            lons_k   = lon[ii_k]
            clon_k   = float(circular_mean_lon(lons_k))
            lon_sp_k = float(len(np.unique(ii_k))) / nx
            mean_U_k = float(np.mean(Uv[kk, jj_k, ii_k]))

            # per-level WATERSHED-REFINED wind metrics: size of the refined
            # jet region at this level, and whether that refined region
            # closes into a ring (spans essentially all longitudes).
            refined_k = gr_region[kk]
            if np.any(refined_k):
                refined_area_k  = float(cell_areas[refined_k].sum())
                refined_lon_cov = (float(np.count_nonzero(refined_k.any(axis=0)))
                                   / nx)
            else:
                refined_area_k  = 0.0
                refined_lon_cov = 0.0
            refined_is_ring_k = 1.0 if refined_lon_cov >= 0.95 else 0.0

            per_level_lats.append({
                'level':         int(kk),
                'altitude_km':   float(altitude_km[kk]),
                'inner_lat':     lat_in,
                'outer_lat':     lat_out,
                'lat_width':     lat_in - lat_out,
                'centroid_lat':  float(np.mean(lat[jj_k])),
                'centroid_lon':  clon_k,
                'lon_span_frac': lon_sp_k,
                'mean_U':        mean_U_k,
                'n_voxels':      n_vox_k,
                'area_km2':      area_per_level.get(int(kk), 0.0),
                'refined_area_km2': refined_area_k,
                'refined_is_ring':  refined_is_ring_k,
            })

        if is_ring:
            ring_alt_bands, ring_lev_set = ring_altitude_ranges(
                m, speed, altitude_km, lat, speed_floor=floor_g)
        else:
            ring_alt_bands, ring_lev_set = [], set()

        if ring_alt_bands:
            ring_alt_range = max(ring_alt_bands, key=lambda ab: ab[1] - ab[0])
            ring_n_levels  = len(ring_lev_set)
        else:
            ring_alt_range = (float(altitude_km[ki.min()]),
                              float(altitude_km[ki.max()]))
            ring_n_levels  = nlev

        pct = 100.0 * float(np.sum(m[lev_10, lat_60, :])) / nx
        # "Strongest pull": the wind-speed-weighted centroid of the ring cells
        # (where the fast flow concentrates). Also drives the annular-width
        # sanity check, since for a real annulus it sits equatorward of the
        cent_pull = centroid_3d(m, lat, lon, altitude_km, weights=Uv)
        # CENTROID: the lat/lon the ring ENCLOSES -- the centre of the region
        # the winds encompass -- as the area-weighted geometric centre of that
        # interior in the pole-centred plane (so a pole-encircling ring centres
        interior = enclosed_by_ring(m, lat)
        if int(interior.sum()) >= max(20, int(0.10 * m.sum())):
            cent = region_center_polar(interior, lat, lon, altitude_km)
        else:
            cent = region_center_polar(m, lat, lon, altitude_km)
        cent = cent or cent_pull
        cent['pull_lat'] = cent_pull['lat']
        cent['pull_lon'] = cent_pull['lon']
        cent['pull_altitude_km'] = cent_pull['altitude_km']

        # annular-width sanity: a real ring has its (pull) centroid at least
        # 5 deg equatorward of its mean poleward edge
        mean_inner = float(np.mean(inner_lats))
        annular_width = mean_inner - cent_pull['lat']
        if is_ring and annular_width < 5.0:
            is_ring = False
        if is_ring:
            min_ann_levels = max(6, int(0.20 * nlev))
            if len(ring_lev_set) < min_ann_levels:
                is_ring = False

        cent.update({
            'mean_U':          mean_U,
            'greatest_mag_U':  greatest_mag,
            'sign':            sign,
            'component_id':    nid[0],
            'is_h1':           is_ring,
            'topology':        'H1' if is_ring else 'H0',
            'mean_inner_lat':  mean_inner,
            'mean_outer_lat':  float(np.mean(outer_lats)),
            'lat_range':       (float(np.min(outer_lats)),
                                float(np.max(inner_lats))),
            'alt_range':       (float(altitude_km[ki.min()]),
                                float(altitude_km[ki.max()])),
            'mean_alt':        float(np.mean(altitude_km[ki])),
            'ring_alt_range':  ring_alt_range,
            'ring_alt_bands':  ring_alt_bands,
            'ring_n_levels':   ring_n_levels,
            'ring_lev_set':    set(int(k) for k in ring_lev_set),
            'pct_10hPa_60lat': pct,
            'total_area_km2':  total_area_km2,
            'per_level_lats':  per_level_lats,
            'core_lat':        core_lat_arr,
            'core_alt':        core_alt_arr,
            'core_speed':      core_speed_arr,
            'core_U':          core_U_arr,
            'tilt_slope':      float(tilt_slope),
            'tilt_intercept':  float(tilt_intercept),
            'tilt_r2':         float(tilt_r2),
            'grad_refined_region_speed':   grad_refined_region_speed,
            'grad_refined_boundary_speed': grad_refined_boundary_speed,
        })
        comps.append(cent)
        nid[0] += 1

    if west_is_ring:
        west_ring = smooth_ring_mask(west_ring, west_sign)
        add_comp(west_ring, True, west_floor)
    if east_is_ring:
        east_ring = smooth_ring_mask(east_ring, east_sign)
        add_comp(east_ring, True, east_floor)

    # per-level mutual exclusion: at each altitude at most one sign can
    # own the polar ring. When both the westerly and easterly comps pass
    # the encircles-pole test at the same level, the sign whose mask
    wc = next((c for c in comps
               if c.get('sign') == 'westerly' and c.get('is_h1')), None)
    ec = next((c for c in comps
               if c.get('sign') == 'easterly' and c.get('is_h1')), None)
    if (wc is not None and ec is not None
            and west_ring is not None and east_ring is not None):
        shared = wc['ring_lev_set'] & ec['ring_lev_set']
        for K in sorted(shared):
            w_rows = np.any(west_ring[K], axis=1)
            e_rows = np.any(east_ring[K], axis=1)
            if not np.any(w_rows) or not np.any(e_rows):
                continue
            w_pole_lat = float(lat[w_rows].max())
            e_pole_lat = float(lat[e_rows].max())
            if w_pole_lat > e_pole_lat:
                ec['ring_lev_set'].discard(K)
            elif e_pole_lat > w_pole_lat:
                wc['ring_lev_set'].discard(K)
            else:
                ec['ring_lev_set'].discard(K)
        for c, ring_m in ((wc, west_ring), (ec, east_ring)):
            levs = c['ring_lev_set']
            c['ring_alt_bands'] = bands_from_levels(levs, altitude_km)
            c['ring_n_levels']  = len(levs)
            if c['ring_alt_bands']:
                c['ring_alt_range'] = max(
                    c['ring_alt_bands'], key=lambda ab: ab[1] - ab[0])
            else:
                ki_c = np.where(np.any(ring_m, axis=(1, 2)))[0]
                if ki_c.size:
                    c['ring_alt_range'] = (float(altitude_km[ki_c.min()]),
                                           float(altitude_km[ki_c.max()]))
            nlev_c = int(np.sum(np.any(ring_m, axis=(1, 2))))
            min_ann_c = max(6, int(0.20 * nlev_c))
            if len(levs) < min_ann_c:
                c['is_h1']    = False
                c['topology'] = 'H0'

    # non-ring H0 seed components (kept only if large and vertically coherent)
    for seeds_arr, ring_m, sgn in [(west_seeds, west_ring, (Uv > 0)),
                                   (east_seeds, east_ring, (Uv < 0))]:
        remaining = seeds_arr.copy()
        if ring_m is not None:
            remaining &= ~ring_m
        if not np.any(remaining):
            continue
        lp_pad, _ = label(
            pad_lon(remaining.astype(np.uint8), LON_PAD).astype(int),
            structure=struct6,
        )
        lp = lp_pad[:, :, LON_PAD:LON_PAD + nx]
        for c in range(1, lp.max() + 1):
            blob = (lp == c)
            if np.any(blob):
                add_comp(blob, False, 0.0)

    filt = merge_lon_seam_labels(filt)

    # Smooth the stored wind contours once, on the final label field: each
    # labeled component is anisotropically smoothed (more in lon than lat)
    # and written back. Done once here rather than per component during
    if np.any(filt > 0):
        smoothed = np.zeros_like(filt)
        for cid in np.unique(filt[filt > 0]):
            comp_mask = smooth_jet_mask(filt == cid)
            smoothed[comp_mask & (smoothed == 0)] = cid
        filt = smoothed

    h1_comps = [c for c in comps if c['is_h1']]
    h0_comps = [c for c in comps if not c['is_h1']]
    jet_intact   = any(c['sign'] == 'westerly'
                       and c.get('pct_10hPa_60lat', 0) > 50
                       for c in h1_comps)
    has_easterly = any(c['sign'] == 'easterly' for c in h1_comps)

    edges_viz = edges_3d_periodic(filt, pad=LON_PAD)

    return {
        'labels':        filt,
        'edges':         edges_viz,
        'betti0':        len(h0_comps),
        'betti1':        len(h1_comps),
        'jet_intact':    jet_intact,
        'has_easterly':  has_easterly,
        'h1_components': h1_comps,
        'h0_components': h0_comps,
        'components':    comps,
        'ring_persistence_prof': ring_persistence_prof,
        'ring_persistence_max':  ring_persistence_max,
    }


# Coarse longitude-latitude binning of temperature restricted to a
# given in-vortex mask, per level. Supports localised 7-day warming
# detection without storing the full T field: a zonal mean would throw
LAT_BIN_DEG = 5.0
LON_BIN_DEG = 30.0
LAT_BIN_LO  = 40.0
LAT_BIN_HI  = 90.0
LON_BIN_LO  = 0.0
LON_BIN_HI  = 360.0


def coarse_T_inside_ring(Tv, inside, lat, lon):
    # Bin temperature voxels inside the polar interior onto a coarse
    nz, ny, nx = Tv.shape
    lat_edges = np.arange(LAT_BIN_LO, LAT_BIN_HI + 1e-6, LAT_BIN_DEG,
                          dtype=np.float64)
    lon_edges = np.arange(LON_BIN_LO, LON_BIN_HI + 1e-6, LON_BIN_DEG,
                          dtype=np.float64)
    nlat = len(lat_edges) - 1
    nlon = len(lon_edges) - 1

    lat_idx = np.digitize(lat, lat_edges) - 1
    lat_idx = np.where(lat == lat_edges[-1], nlat - 1, lat_idx)
    lat_idx = np.where((lat_idx >= 0) & (lat_idx < nlat), lat_idx, -1)

    lon_wrap = np.mod(lon, 360.0)
    lon_idx  = np.digitize(lon_wrap, lon_edges) - 1
    lon_idx  = np.where(lon_wrap == lon_edges[-1], nlon - 1, lon_idx)
    lon_idx  = np.where((lon_idx >= 0) & (lon_idx < nlon), lon_idx, -1)

    valid_j = (lat_idx >= 0)[:, None]
    valid_i = (lon_idx >= 0)[None, :]
    flat_idx = np.full((ny, nx), -1, dtype=np.int64)
    ok = valid_j & valid_i
    flat_idx[ok] = (lat_idx[:, None] * nlon + lon_idx[None, :])[ok]

    # cos-lat area weight per native cell (regular lat-lon grid: cell area
    # scales with cos of latitude). Broadcast to (ny, nx).
    area_w = np.repeat(np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
                       [:, None], nx, axis=1)

    T_max  = np.full((nz, nlat, nlon), np.nan, dtype=np.float32)
    T_mean = np.full((nz, nlat, nlon), np.nan, dtype=np.float32)
    n_vox  = np.zeros((nz, nlat, nlon), dtype=np.float32)

    n_bins = nlat * nlon
    for k in range(nz):
        msk = inside[k] & (flat_idx >= 0)
        if not np.any(msk):
            continue
        idx = flat_idx[msk]
        Tk  = Tv[k][msk].astype(np.float64)
        wk  = area_w[msk]
        area  = np.bincount(idx, weights=wk, minlength=n_bins)
        sums  = np.bincount(idx, weights=Tk * wk, minlength=n_bins)
        max_flat = np.full(n_bins, -np.inf, dtype=np.float64)
        np.maximum.at(max_flat, idx, Tk)

        area2 = area.reshape(nlat, nlon)
        sums2 = sums.reshape(nlat, nlon)
        maxs2 = max_flat.reshape(nlat, nlon)
        nonemp = area2 > 0

        n_vox[k] = area2.astype(np.float32)
        T_mean[k][nonemp] = (sums2[nonemp]
                             / area2[nonemp]).astype(np.float32)
        T_max[k][nonemp]  = maxs2[nonemp].astype(np.float32)

    return {
        'T_max':    T_max,
        'T_mean':   T_mean,
        'n_vox':    n_vox,
        'lat_edge': lat_edges.astype(np.float32),
        'lon_edge': lon_edges.astype(np.float32),
    }


# In-vortex temperature aggregation on coarse per-level grids.
# Produces three grids, each binned per altitude level x 5 deg lat x
# 30 deg lon with per-bin voxel counts (n) for later area weighting:
def analyze_temperature(T, altitude_km, U=None, ring_mask_3d=None,
                        wind_comp0_mask=None, geo_comp0_mask=None):
    lat = T.lat.values
    lon = T.lon.values
    Tv = T.values
    nz, ny, nx = Tv.shape

    # In-vortex temperature regions = the comp-0 contour voxels: the
    # largest wind component, and separately the largest geopotential
    # component. These give the exact (lat, lon) the vortex occupies at
    def as_bool_mask(m):
        if m is None:
            return None
        a = np.asarray(m)
        if a.dtype == bool:
            return a if a.any() else None
        b = a > 0
        return b if b.any() else None

    wind_mask = as_bool_mask(wind_comp0_mask)
    geo_mask  = as_bool_mask(geo_comp0_mask)
    # If no comp-0 wind mask is supplied, fall back to the ring mask,
    # then to U > 0.
    if wind_mask is None:
        wind_mask = as_bool_mask(ring_mask_3d)
    if wind_mask is None and U is not None and hasattr(U, 'values'):
        wind_mask = as_bool_mask(np.asarray(U.values, dtype=np.float32) > 0)

    # Coarse in-vortex temperature grids on the comp-0 footprints.
    cw = coarse_T_inside_ring(Tv, wind_mask, lat, lon) if wind_mask is not None else None
    cg = coarse_T_inside_ring(Tv, geo_mask, lat, lon) if geo_mask is not None else None

    # Per-level inside/outside gradient on the wind comp-0 footprint.
    grad_per_level = {}
    if wind_mask is not None:
        for k in range(nz):
            ins = wind_mask[k]
            out = (~wind_mask[k])
            T_in  = float(np.mean(Tv[k][ins])) if ins.any() else np.nan
            T_out = float(np.mean(Tv[k][out])) if out.any() else np.nan
            grad_per_level[float(altitude_km[k])] = (
                T_in - T_out if np.isfinite(T_in) and np.isfinite(T_out)
                else np.nan)
        if wind_mask.any():
            grad = (float(np.mean(Tv[wind_mask]))
                    - float(np.mean(Tv[~wind_mask])))
        else:
            grad = np.nan
    else:
        grad = np.nan
    reversed_ = bool(np.isfinite(grad) and grad > 0)

    # Fixed polar-cap reference grid (60-90 N), wind-independent.
    cap_mask = np.zeros((nz, ny, nx), dtype=bool)
    cap_mask[:, (lat >= 60.0) & (lat <= 90.0), :] = True
    coarse_cap = coarse_T_inside_ring(Tv, cap_mask, lat, lon)

    out = {
        'altitude_km':         altitude_km,
        'lat':                 lat,
        'gradient':            grad,
        'gradient_reversed':   reversed_,
        'gradient_per_level':  grad_per_level,
        'gradient_source':     'comp0_wind_footprint',
        # fixed polar cap (60-90 N), wind-independent reference
        'T_cap_max':           coarse_cap['T_max'],
        'T_cap_mean':          coarse_cap['T_mean'],
        'T_cap_n':             coarse_cap['n_vox'],
    }
    # comp-0 WIND footprint (native alt x 5 lat x 30 lon, with area)
    if cw is not None:
        out.update({
            'T_in_wind_max':      cw['T_max'],
            'T_in_wind_mean':     cw['T_mean'],
            'T_in_wind_n':        cw['n_vox'],
            'T_in_wind_lat_edge': cw['lat_edge'],
            'T_in_wind_lon_edge': cw['lon_edge'],
            # primary coarse grid alias = wind comp-0 footprint
            'T_coarse_max':       cw['T_max'],
            'T_coarse_mean':      cw['T_mean'],
            'T_coarse_n':         cw['n_vox'],
            'T_coarse_lat_edge':  cw['lat_edge'],
            'T_coarse_lon_edge':  cw['lon_edge'],
        })
    # comp-0 GEOPOTENTIAL footprint (native alt x 5 lat x 30 lon)
    if cg is not None:
        out.update({
            'T_in_geo_max':      cg['T_max'],
            'T_in_geo_mean':     cg['T_mean'],
            'T_in_geo_n':        cg['n_vox'],
            'T_in_geo_lat_edge': cg['lat_edge'],
            'T_in_geo_lon_edge': cg['lon_edge'],
        })
    return out


# Full analysis: geopotential lobes, wind rings, temperature diagnostic.
# Returns an xarray Dataset with the labelled masks and edge arrays as
# variables and the diagnostic scalars as attributes.
def analyze_vortex(phi, U, T, p):
    phi, U, T, p = phi.squeeze(), U.squeeze(), T.squeeze(), p.squeeze()
    Z  = phi / G
    ds = xr.Dataset(coords={'lev': phi.lev, 'lat': phi.lat, 'lon': phi.lon})
    ds['phi'], ds['Z'], ds['U'], ds['T'], ds['p'] = phi, Z, U, T, p
    ds = attach_pressure_and_altitude(ds, p)
    ak = ds['altitude_km'].values

    geo  = analyze_geopotential(phi, ak)
    ds['geopotential_mask']  = (('lev', 'lat', 'lon'), geo['labels'])
    ds['geopotential_edges'] = (('lev', 'lat', 'lon'), geo['edges'])

    wind = analyze_wind(U, ak)
    ds['wind_mask']  = (('lev', 'lat', 'lon'), wind['labels'])
    ds['wind_edges'] = (('lev', 'lat', 'lon'), wind['edges'])

    west_h1 = next((c for c in wind.get('h1_components', [])
                    if c.get('sign') == 'westerly'), None)
    ring_mask_3d = ((wind['labels'] == west_h1['component_id'])
                    if west_h1 is not None else None)
    # comp-0 = largest component footprint (largest by voxel count).
    def largest_label_mask(labels):
        lab = np.asarray(labels)
        vals = lab[lab > 0]
        if vals.size == 0:
            return None
        ids, counts = np.unique(vals, return_counts=True)
        return lab == int(ids[int(np.argmax(counts))])

    wind_comp0 = largest_label_mask(wind['labels'])
    geo_comp0  = largest_label_mask(geo['labels'])
    temp = analyze_temperature(T, ak, U=U, ring_mask_3d=ring_mask_3d,
                               wind_comp0_mask=wind_comp0,
                               geo_comp0_mask=geo_comp0)

    for k, v in [
        ('betti0_phi',                 geo['betti0']),
        ('betti1_phi',                 geo['betti1']),
        ('betti0_wind',                wind['betti0']),
        ('betti1_wind',                wind['betti1']),
        ('jet_intact',                 wind['jet_intact']),
        ('has_easterly',               wind['has_easterly']),
        ('gradient_pole_minus_midlat', temp['gradient']),
        ('gradient_reversed',          temp['gradient_reversed']),
    ]:
        ds.attrs[k] = v

    ds.attrs['phi_components']  = str([
        {k: v for k, v in c.items()
         if k not in ('level_centroids', 'b0_profile')}
        for c in geo['components']
    ])
    ds.attrs['wind_components'] = str(wind['components'])

    return ds


# time-series extraction

# Run all three analyses on a single timestep and return a flat dict of
# diagnostics and per-component metadata. No 3D masks or edge arrays are
# kept in the returned record, so this is safe to call in a long loop
def collect_timestep(phi, U, T, p, prev_ring_mask=None):
    phi, U, T, p = phi.squeeze(), U.squeeze(), T.squeeze(), p.squeeze()
    ds_tmp = xr.Dataset(
        coords={'lev': phi.lev, 'lat': phi.lat, 'lon': phi.lon})
    ds_tmp['p'] = p
    ds_tmp = attach_pressure_and_altitude(ds_tmp, p)
    ak  = ds_tmp['altitude_km'].values
    lat = phi.lat.values
    lon = phi.lon.values

    clk0 = time.perf_counter()
    wind = analyze_wind(U, ak)
    clk1 = time.perf_counter()
    geo  = analyze_geopotential(phi, ak)
    clk2 = time.perf_counter()

    west_h1 = next((c for c in wind.get('h1_components', [])
                    if c.get('sign') == 'westerly'), None)
    current_ring_mask = ((wind['labels'] == west_h1['component_id'])
                         if west_h1 is not None else None)
    mask_for_grad = (current_ring_mask
                     if current_ring_mask is not None else prev_ring_mask)

    # comp-0 = largest component footprint (wind labels are in
    # detection order, so pick largest by voxel count for both).
    def largest_label_mask(labels):
        lab = np.asarray(labels)
        vals = lab[lab > 0]
        if vals.size == 0:
            return None
        ids, counts = np.unique(vals, return_counts=True)
        return lab == int(ids[int(np.argmax(counts))])

    wind_comp0 = largest_label_mask(wind['labels'])
    geo_comp0  = largest_label_mask(geo['labels'])

    clk3 = time.perf_counter()
    temp = analyze_temperature(T, ak, U=U, ring_mask_3d=mask_for_grad,
                               wind_comp0_mask=wind_comp0,
                               geo_comp0_mask=geo_comp0)
    clk4 = time.perf_counter()
    if PRINT_SECTION_TIMING:
        print(f"[timing] wind={clk1 - clk0:6.1f}s  "
              f"geo={clk2 - clk1:6.1f}s  temp={clk4 - clk3:6.1f}s",
              flush=True)

    nz = len(ak)
    grad_prof = np.array(
        [temp['gradient_per_level'].get(float(ak[k]), np.nan)
         for k in range(nz)],
        dtype=np.float32,
    )
    lev_hPa = ds_tmp['lev'].values / 100.0

    rec = {
        'ak':                ak,
        'lev_hPa':           lev_hPa,
        'lat':               lat,
        'lon':               lon,
        'T_coarse_max':      temp['T_coarse_max'].astype(np.float32),
        'T_coarse_mean':     temp['T_coarse_mean'].astype(np.float32),
        'T_coarse_n':        temp['T_coarse_n'].astype(np.int32),
        'T_coarse_lat_edge': temp['T_coarse_lat_edge'].astype(np.float32),
        'T_coarse_lon_edge': temp['T_coarse_lon_edge'].astype(np.float32),
        'T_cap_max':         temp['T_cap_max'].astype(np.float32),
        'T_cap_mean':        temp['T_cap_mean'].astype(np.float32),
        'T_cap_n':           temp['T_cap_n'].astype(np.int32),
        'grad_prof':         grad_prof,
        'gradient':          float(temp['gradient']),
        'gradient_reversed': bool(temp['gradient_reversed']),
        'gradient_source':   temp['gradient_source'],
        'geo_b0':            int(geo['betti0']),
        'geo_components':    geo['components'],
        'wind_b0':           int(wind['betti0']),
        'wind_b1':           int(wind['betti1']),
        'jet_intact':        bool(wind['jet_intact']),
        'has_easterly':      bool(wind['has_easterly']),
        'mean_U_subspace':   float(np.mean(U.values)),
        'wind_components':   wind['components'],
        'wind_ring_persistence_prof': np.asarray(
            wind['ring_persistence_prof'], dtype=np.float32),
        'wind_ring_persistence_max':  float(wind['ring_persistence_max']),
        '_current_ring_mask': current_ring_mask,
    }
    # comp-0 GEOPOTENTIAL-footprint temperature grid (parallel to the
    # wind-footprint grid stored as T_coarse_*).  Stored so classification
    # can be tested against either region.
    if 'T_in_geo_mean' in temp:
        rec['T_in_geo_max']  = temp['T_in_geo_max'].astype(np.float32)
        rec['T_in_geo_mean'] = temp['T_in_geo_mean'].astype(np.float32)
        rec['T_in_geo_n']    = temp['T_in_geo_n'].astype(np.int32)
    return rec


# Write a list of collect_timestep records to a single NetCDF file.
# Each record contributes one entry along the time axis. Per-component
# arrays are NaN-padded out to MAX_GEO_COMP / MAX_WIND_COMP slots so the
def write_timeseries_nc(records, dates, outpath):
    import netCDF4 as nc4

    n  = len(records)
    nz = len(records[0]['ak'])
    ny = len(records[0]['lat'])
    nx = len(records[0]['lon'])

    MAX_GEO_COMP  = 4
    MAX_WIND_COMP = 4
    MAX_ALT_BANDS = 4
    n_lat_bin = len(records[0]['T_coarse_lat_edge']) - 1
    n_lon_bin = len(records[0]['T_coarse_lon_edge']) - 1

    calendar = getattr(dates[0], 'calendar', 'standard')

    with nc4.Dataset(outpath, 'w', format='NETCDF4') as ds:
        ds.description = 'Polar vortex TDA diagnostics - time series'

        # dimensions
        ds.createDimension('time',         n)
        ds.createDimension('lev',          nz)
        ds.createDimension('lat',          ny)
        ds.createDimension('lon',          nx)
        ds.createDimension('lat_bin',      n_lat_bin)
        ds.createDimension('lon_bin',      n_lon_bin)
        ds.createDimension('lat_bin_edge', n_lat_bin + 1)
        ds.createDimension('lon_bin_edge', n_lon_bin + 1)
        ds.createDimension('geo_comp',  MAX_GEO_COMP)
        ds.createDimension('wind_comp', MAX_WIND_COMP)
        ds.createDimension('alt_band',  MAX_ALT_BANDS)
        ds.createDimension('two',       2)

        # time, lev, altitude_km, lat, lon coordinates
        tv          = ds.createVariable('time', 'f8', ('time',))
        tv.units    = 'hours since 1900-01-01 00:00:00'
        tv.calendar = calendar
        tv[:]       = nc4.date2num(list(dates),
                                   units=tv.units, calendar=tv.calendar)

        lv           = ds.createVariable('lev', 'f4', ('lev',))
        lv.units     = 'hPa'
        lv.long_name = 'pressure level'
        lv.axis      = 'Z'
        lv.positive  = 'down'
        lv[:]        = records[0]['lev_hPa'].astype(np.float32)

        akv           = ds.createVariable('altitude_km', 'f4', ('lev',))
        akv.units     = 'km'
        akv.long_name = 'approximate altitude (scale-height estimate)'
        akv[:]        = records[0]['ak'].astype(np.float32)

        ltv        = ds.createVariable('lat', 'f4', ('lat',))
        ltv.units  = 'degrees_north'
        ltv[:]     = records[0]['lat'].astype(np.float32)

        lnv        = ds.createVariable('lon', 'f4', ('lon',))
        lnv.units  = 'degrees_east'
        lnv[:]     = records[0]['lon'].astype(np.float32)

        # coarse-bin coordinates for T_coarse_* arrays
        lat_edges = records[0]['T_coarse_lat_edge']
        lon_edges = records[0]['T_coarse_lon_edge']
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

        bc = ds.createVariable('lat_bin_center', 'f4', ('lat_bin',))
        bc.units = 'degrees_north'
        bc[:] = lat_centers.astype(np.float32)

        bc = ds.createVariable('lon_bin_center', 'f4', ('lon_bin',))
        bc.units = 'degrees_east'
        bc[:] = lon_centers.astype(np.float32)

        be = ds.createVariable('lat_bin_edges', 'f4', ('lat_bin_edge',))
        be.units = 'degrees_north'
        be[:] = lat_edges.astype(np.float32)

        be = ds.createVariable('lon_bin_edges', 'f4', ('lon_bin_edge',))
        be.units = 'degrees_east'
        be[:] = lon_edges.astype(np.float32)

        def mkvar(name, dims, units='', long_name=''):
            v = ds.createVariable(name, 'f4', dims,
                                  fill_value=np.float32(np.nan))
            if units:
                v.units = units
            if long_name:
                v.long_name = long_name
            return v

        # (time, lev, lat_bin, lon_bin): coarse-binned T inside the
        # outer perimeter of the polar westerly regime (every voxel
        # from the pole through and including the outermost
        v = mkvar('T_coarse_max',
                  ('time', 'lev', 'lat_bin', 'lon_bin'),
                  'K', 'Max T inside polar westerly perimeter, per coarse bin')
        v[:] = np.array([r['T_coarse_max'] for r in records],
                        dtype=np.float32)

        v = mkvar('T_coarse_mean',
                  ('time', 'lev', 'lat_bin', 'lon_bin'),
                  'K', 'Mean T inside polar westerly perimeter, per coarse bin')
        v[:] = np.array([r['T_coarse_mean'] for r in records],
                        dtype=np.float32)

        v = ds.createVariable(
            'T_coarse_n', 'i2',
            ('time', 'lev', 'lat_bin', 'lon_bin'),
            fill_value=np.int16(0))
        v.units = '1'
        v[:] = np.array([r['T_coarse_n'] for r in records],
                        dtype=np.int16)

        # comp-0 GEOPOTENTIAL-footprint coarse-T grid (parallel to the
        # wind footprint above).  Same 5 deg lat x 30 deg lon bins per
        # level.  Only written when present in every record.
        if all('T_in_geo_mean' in r for r in records):
            v = mkvar('T_in_geo_max',
                      ('time', 'lev', 'lat_bin', 'lon_bin'),
                      'K', 'Max T inside comp-0 geopotential footprint, per coarse bin')
            v[:] = np.array([r['T_in_geo_max'] for r in records],
                            dtype=np.float32)

            v = mkvar('T_in_geo_mean',
                      ('time', 'lev', 'lat_bin', 'lon_bin'),
                      'K', 'Mean T inside comp-0 geopotential footprint, per coarse bin')
            v[:] = np.array([r['T_in_geo_mean'] for r in records],
                            dtype=np.float32)

            v = ds.createVariable(
                'T_in_geo_n', 'i2',
                ('time', 'lev', 'lat_bin', 'lon_bin'),
                fill_value=np.int16(0))
            v.units = '1'
            v[:] = np.array([r['T_in_geo_n'] for r in records],
                            dtype=np.int16)

        # (time, lev, lat_bin, lon_bin): coarse-binned T over a FIXED
        # polar cap 60-90 N at every level, wind-independent.  This
        # is a second, unconditional reference grid alongside the
        v = mkvar('T_cap_max',
                  ('time', 'lev', 'lat_bin', 'lon_bin'),
                  'K', 'Max T over fixed polar cap (60-90 N), per coarse bin')
        v[:] = np.array([r['T_cap_max'] for r in records],
                        dtype=np.float32)

        v = mkvar('T_cap_mean',
                  ('time', 'lev', 'lat_bin', 'lon_bin'),
                  'K', 'Mean T over fixed polar cap (60-90 N), per coarse bin')
        v[:] = np.array([r['T_cap_mean'] for r in records],
                        dtype=np.float32)

        v = ds.createVariable(
            'T_cap_n', 'i2',
            ('time', 'lev', 'lat_bin', 'lon_bin'),
            fill_value=np.int16(0))
        v.units = '1'
        v[:] = np.array([r['T_cap_n'] for r in records],
                        dtype=np.int16)

        # (time, lev): inside-minus-outside T gradient per level
        v = mkvar('grad_prof', ('time', 'lev'),
                  'K',
                  'Inside-minus-outside wind-ring T gradient per level')
        v[:] = np.array([r['grad_prof'] for r in records], dtype=np.float32)

        # (time,): scalar diagnostics per timestep
        for name, units, long_name in [
            ('gradient',          'K',   'Column-mean inside-minus-outside T gradient'),
            ('gradient_reversed', '1',   'Gradient reversed flag (1=reversed)'),
            ('geo_b0',            '1',   'Geopotential Betti-0'),
            ('wind_b0',           '1',   'Wind Betti-0'),
            ('wind_b1',           '1',   'Wind Betti-1'),
            ('jet_intact',        '1',   'Westerly jet intact flag'),
            ('has_easterly',      '1',   'Easterly H1 ring present'),
            ('mean_U_subspace',   'm/s', 'Mean U over the loaded polar cap'),
        ]:
            v = mkvar(name, ('time',), units, long_name)
            v[:] = np.array([float(r[name]) for r in records],
                            dtype=np.float32)

        # gradient source (0=lat60 fallback, 1=wind ring mask)
        v = mkvar('gradient_source', ('time',), '1',
                  'Source of gradient mask (0=lat60, 1=wind ring)')
        v[:] = np.array(
            [1.0 if r.get('gradient_source') == 'ring' else 0.0
             for r in records], dtype=np.float32)

        # (time, geo_comp): per-lobe scalars
        geo_scalar_vars = {
            'geo_bottom_lat':      ('centroid_bottom.lat',   'degrees_north'),
            'geo_lowest_lat':      ('lowest_lat',            'degrees_north'),
            'geo_alt_lo':          ('bottom_altitude_km',    'km'),
            'geo_alt_hi':          ('top_altitude_km',       'km'),
            'geo_aspect_ratio':    ('aspect_ratio_mean',     '1'),
            'geo_total_area_km2':  ('total_area_km2',        'km2'),
        }
        geo_arrs = {name: np.full((n, MAX_GEO_COMP), np.nan,
                                  dtype=np.float32)
                    for name in geo_scalar_vars}
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['geo_components'][:MAX_GEO_COMP]):
                for nc_name, (field, _) in geo_scalar_vars.items():
                    if '.' in field:
                        a, b = field.split('.')
                        val = comp.get(a, {}).get(b, np.nan)
                    else:
                        val = comp.get(field, np.nan)
                    geo_arrs[nc_name][i, ci] = (float(val)
                                                if val is not None
                                                else np.nan)
        for name, (_, units) in geo_scalar_vars.items():
            v = mkvar(name, ('time', 'geo_comp'), units)
            v[:] = geo_arrs[name]

        # (time, geo_comp, lev): per-lobe per-level metrics
        geo_lev_fields = {
            'geo_lev_centroid_lat':    ('centroid_lat',      'degrees_north',
                                        'Geo lobe centroid latitude per level'),
            'geo_lev_centroid_lon':    ('centroid_lon',      'degrees_east',
                                        'Geo lobe centroid longitude per level'),
            'geo_lev_lat_equatorward': ('lat_equatorward',   'degrees_north',
                                        'Geo lobe equatorward edge latitude per level'),
            'geo_lev_lat_poleward':    ('lat_poleward',      'degrees_north',
                                        'Geo lobe poleward edge latitude per level'),
            'geo_lev_aspect_ratio':    ('aspect_ratio',      '1',
                                        'Geo lobe aspect ratio per level'),
        }
        geo_lev_arrs = {
            nc: np.full((n, MAX_GEO_COMP, nz), np.nan, dtype=np.float32)
            for nc in geo_lev_fields
        }
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['geo_components'][:MAX_GEO_COMP]):
                for lc in comp.get('level_centroids', []):
                    k = int(lc['level'])
                    if 0 <= k < nz:
                        for nc, (field, _, _) in geo_lev_fields.items():
                            val = lc.get(field, np.nan)
                            geo_lev_arrs[nc][i, ci, k] = (
                                float(val) if val is not None else np.nan)
        for nc, (_, units, long_name) in geo_lev_fields.items():
            v = mkvar(nc, ('time', 'geo_comp', 'lev'), units, long_name)
            v[:] = geo_lev_arrs[nc]

        # (time, lev): beta_0 profile of the largest geopotential lobe.
        # b0_profile[k] is the number of 2D connected pieces of the
        # largest 3D component's slice at level k. A single intact
        v = ds.createVariable(
            'geo_largest_b0_profile', 'i2', ('time', 'lev'),
            fill_value=np.int16(0))
        v.units = '1'
        v.long_name = ('Per-level 2D Betti-0 of largest geopotential '
                       'lobe (1=intact slice, >=2=split slice)')
        b0_arr = np.zeros((n, nz), dtype=np.int16)
        for i, r in enumerate(records):
            gcs = r.get('geo_components', [])
            if gcs:
                prof = gcs[0].get('b0_profile')
                if prof is not None:
                    b0_arr[i, :] = np.asarray(prof, dtype=np.int16)
        v[:] = b0_arr

        # (time, lev): beta_0 profile of the SECOND-largest
        # geopotential lobe, stored only when it spans a comparable
        # fraction of the vertical column as the largest (n_levels
        SPLIT_NLEV_FRAC = 0.50
        v = ds.createVariable(
            'geo_second_b0_profile', 'i2', ('time', 'lev'),
            fill_value=np.int16(0))
        v.units = '1'
        v.long_name = ('Per-level 2D Betti-0 of second-largest '
                       'geopotential lobe, populated only when its '
                       'vertical extent is at least '
                       f'{SPLIT_NLEV_FRAC:.2f} that of the largest '
                       '(split indicator); zeros otherwise')
        b0_arr2 = np.zeros((n, nz), dtype=np.int16)
        for i, r in enumerate(records):
            gcs = r.get('geo_components', [])
            if len(gcs) < 2:
                continue
            n1 = int(gcs[0].get('n_levels', 0))
            n2 = int(gcs[1].get('n_levels', 0))
            if n1 <= 0:
                continue
            if n2 / n1 < SPLIT_NLEV_FRAC:
                continue
            prof = gcs[1].get('b0_profile')
            if prof is not None:
                b0_arr2[i, :] = np.asarray(prof, dtype=np.int16)
        v[:] = b0_arr2

        # (time, wind_comp): per-wind-component scalars
        wind_scalar_vars = {
            'wind_sign':            ('sign_int',        '1'),
            'wind_is_h1':           ('is_h1_int',       '1'),
            'wind_mean_U':          ('mean_U',          'm/s'),
            'wind_greatest_mag_U':  ('greatest_mag_U',  'm/s'),
            'wind_mean_inner_lat':  ('mean_inner_lat',  'degrees_north'),
            'wind_mean_outer_lat':  ('mean_outer_lat',  'degrees_north'),
            'wind_mean_alt':        ('mean_alt',        'km'),
            'wind_pct_10hPa_60lat': ('pct_10hPa_60lat', '%'),
            'wind_ring_n_levels':   ('ring_n_levels',   '1'),
            'wind_total_area_km2':  ('total_area_km2',  'km2'),
            'wind_tilt_slope':      ('tilt_slope',      'degrees/km'),
            'wind_grad_refined_region_speed':
                ('grad_refined_region_speed',   'm/s'),
        }
        wind_arrs = {name: np.full((n, MAX_WIND_COMP), np.nan,
                                   dtype=np.float32)
                     for name in wind_scalar_vars}
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['wind_components'][:MAX_WIND_COMP]):
                comp['sign_int']  = (0.0 if comp.get('sign') == 'westerly'
                                     else 1.0)
                comp['is_h1_int'] = 1.0 if comp.get('is_h1') else 0.0
                for nc_name, (field, _) in wind_scalar_vars.items():
                    val = comp.get(field, np.nan)
                    wind_arrs[nc_name][i, ci] = (float(val)
                                                 if val is not None
                                                 else np.nan)
        for name, (_, units) in wind_scalar_vars.items():
            v = mkvar(name, ('time', 'wind_comp'), units)
            v[:] = wind_arrs[name]

        # (time, wind_comp, lev): per-wind-component per-level metrics
        # (jet core is NOT per level -- see wind_core_* (time, wind_comp, lon))
        wind_lev_fields = {
            'wind_lev_lon_span':     ('lon_span_frac', '1',
                                      'Ring longitude coverage fraction per level'),
            'wind_lev_mean_U':       ('mean_U',        'm/s',
                                      'Mean zonal wind across the ring per level'),
        }
        wind_lev_arrs = {
            nc: np.full((n, MAX_WIND_COMP, nz), np.nan, dtype=np.float32)
            for nc in wind_lev_fields
        }
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['wind_components'][:MAX_WIND_COMP]):
                for entry in comp.get('per_level_lats', []):
                    k = int(entry['level'])
                    if 0 <= k < nz:
                        for nc, (field, _, _) in wind_lev_fields.items():
                            val = entry.get(field, np.nan)
                            wind_lev_arrs[nc][i, ci, k] = (
                                float(val) if val is not None else np.nan)
        for nc, (_, units, long_name) in wind_lev_fields.items():
            v = mkvar(nc, ('time', 'wind_comp', 'lev'), units, long_name)
            v[:] = wind_lev_arrs[nc]

        # (time, lev) and (time): topological strength of the wind ring --
        # persistence of the dominant H1 (loop) generator of the wind-speed
        # super-level filtration per level, and its column maximum.
        ring_pers_lev = np.full((n, nz), np.nan, dtype=np.float32)
        ring_pers_max = np.full((n,), np.nan, dtype=np.float32)
        for i, r in enumerate(records):
            prof = r.get('wind_ring_persistence_prof')
            if prof is not None and len(prof) == nz:
                ring_pers_lev[i] = np.asarray(prof, dtype=np.float32)
            ring_pers_max[i] = np.float32(
                r.get('wind_ring_persistence_max', np.nan))
        v = mkvar('wind_lev_ring_persistence', ('time', 'lev'), 'm/s',
                  'H1 persistence of the wind ring per level (super-level '
                  'filtration of wind speed)')
        v[:] = ring_pers_lev
        v = mkvar('wind_ring_persistence_max', ('time',), 'm/s',
                  'Column-maximum H1 wind-ring persistence')
        v[:] = ring_pers_max

        # (time, wind_comp, lon): jet core curve -- one (lat, alt, speed)
        # per longitude. Conceptually a "wire through the donut".
        core_lat_arr   = np.full((n, MAX_WIND_COMP, nx), np.nan, dtype=np.float32)
        core_alt_arr   = np.full((n, MAX_WIND_COMP, nx), np.nan, dtype=np.float32)
        core_speed_arr = np.full((n, MAX_WIND_COMP, nx), np.nan, dtype=np.float32)
        core_U_arr     = np.full((n, MAX_WIND_COMP, nx), np.nan, dtype=np.float32)
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['wind_components'][:MAX_WIND_COMP]):
                cl = comp.get('core_lat')
                ca = comp.get('core_alt')
                cs = comp.get('core_speed')
                cu = comp.get('core_U')
                if cl is not None and len(cl) == nx:
                    core_lat_arr  [i, ci] = np.asarray(cl, dtype=np.float32)
                    core_alt_arr  [i, ci] = np.asarray(ca, dtype=np.float32)
                    core_speed_arr[i, ci] = np.asarray(cs, dtype=np.float32)
                    core_U_arr    [i, ci] = np.asarray(cu, dtype=np.float32)

        v = mkvar('wind_core_lat', ('time', 'wind_comp', 'lon'),
                  'degrees_north',
                  'Jet core latitude at each longitude (peak |U| per lon column)')
        v[:] = core_lat_arr
        v = mkvar('wind_core_alt', ('time', 'wind_comp', 'lon'),
                  'km',
                  'Jet core altitude at each longitude (peak |U| per lon column)')
        v[:] = core_alt_arr
        v = mkvar('wind_core_speed', ('time', 'wind_comp', 'lon'),
                  'm/s',
                  'Jet core peak wind speed at each longitude')
        v[:] = core_speed_arr
        v = mkvar('wind_core_U', ('time', 'wind_comp', 'lon'),
                  'm/s',
                  'Jet core signed zonal wind at each longitude')
        v[:] = core_U_arr

        # (time, wind_comp, alt_band, 2): separated ring altitude bands
        wind_bands = np.full((n, MAX_WIND_COMP, MAX_ALT_BANDS, 2),
                             np.nan, dtype=np.float32)
        for i, r in enumerate(records):
            for ci, comp in enumerate(
                    r['wind_components'][:MAX_WIND_COMP]):
                for bi, (lo, hi) in enumerate(
                        comp.get('ring_alt_bands', [])[:MAX_ALT_BANDS]):
                    wind_bands[i, ci, bi, 0] = float(lo)
                    wind_bands[i, ci, bi, 1] = float(hi)
        v = mkvar('wind_ring_alt_bands',
                  ('time', 'wind_comp', 'alt_band', 'two'),
                  'km',
                  'Wind ring altitude band ranges (lo, hi) per component')
        v[:] = wind_bands


# Convenience driver: iterate over timesteps, collect records with
# collect_timestep, and write them to outpath via write_timeseries_nc.
# phi_all, U_all, T_all, p_all are 4D DataArrays with a leading `time`
def run_timeseries(phi_all, U_all, T_all, p_all, dates, outpath):
    records = []
    last_ring_mask = None
    for i, date in enumerate(dates):
        rec = collect_timestep(
            phi_all.isel(time=i), U_all.isel(time=i),
            T_all.isel(time=i),   p_all.isel(time=i),
            prev_ring_mask=last_ring_mask,
        )
        cur = rec.pop('_current_ring_mask', None)
        if cur is not None:
            last_ring_mask = cur
        records.append(rec)
    write_timeseries_nc(records, list(dates), outpath)
    return records


def plot_vortex_edges_3d(edges_da, altitude_km, base_field_da=None,
                         field_type='temperature', title=None, cmap=None,
                         cbar_label=None, vmin=None, vmax=None,
                         output_file=None):
    # 3D vortex-edge scatter on a polar (r, theta, altitude) frame, coloured
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    lev = edges_da[edges_da.dims[0]].values
    lat = edges_da['lat'].values
    lon = edges_da['lon'].values

    lev_idx, lat_idx, lon_idx = np.where(~np.isnan(edges_da.values))
    edge_vals = edges_da.values[lev_idx, lat_idx, lon_idx]

    z = altitude_km[lev_idx]
    theta = np.deg2rad(lon[lon_idx])
    r = 90 - lat[lat_idx]

    # colours come from the base field if given, else the edge values
    if base_field_da is not None:
        colors = base_field_da.values[lev_idx, lat_idx, lon_idx]
    else:
        colors = edge_vals

    is_wind = field_type.lower() == 'u'

    # colormap
    if cmap is None:
        if is_wind:
            cmap = 'RdBu_r'
        elif field_type.lower() in ['pv', 't']:
            cmap = 'bone'
        else:
            cmap = 'cividis'

    # colorbar label
    if cbar_label is None:
        if field_type.lower() == 'pv':
            cbar_label = 'Potential Vorticity (PVU)'
        elif field_type.lower() == 't':
            cbar_label = 'Temperature (K)'
        elif is_wind:
            cbar_label = 'Zonal Wind (m/s)'
        else:
            cbar_label = 'Geopotential (m² s⁻²)'

    # normalization: symmetric about zero for wind, data range otherwise
    if is_wind:
        if vmin is None or vmax is None:
            absmax = np.nanmax(np.abs(colors))
            vmin = -absmax
            vmax = absmax
    else:
        if vmin is None:
            vmin = np.nanmin(colors)
        if vmax is None:
            vmax = np.nanmax(colors)

    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # fixed domain
    r_max = 50
    z_min = np.min(altitude_km)
    z_max = np.max(altitude_km)

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])

    sc = ax.scatter(r * np.cos(theta), r * np.sin(theta), z,
                    c=colors, cmap=cmap, norm=norm, s=5, alpha=0.6)

    # latitude circles
    for ref_lat in [45, 60, 75]:
        r_ref = 90 - ref_lat
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r_ref * np.cos(theta_circle), r_ref * np.sin(theta_circle),
                np.full_like(theta_circle, z_min),
                color=(0, 0, 0, 0.15), linewidth=1, zorder=1)
        ax.text(r_ref, 0, z_min, f'{ref_lat}°N', fontsize=8, zorder=1)

    # longitude lines
    for ref_lon in [0, 45, 90, 135, 180, 225, 270, 315]:
        theta_ref = np.deg2rad(ref_lon)
        r_line = np.linspace(0, r_max, 80)
        ax.plot(r_line * np.cos(theta_ref), r_line * np.sin(theta_ref),
                np.full_like(r_line, z_min),
                color=(0, 0, 0, 0.08), linewidth=0.6, zorder=1)
        if ref_lon == 0:
            lon_label = "0°"
        elif ref_lon < 180:
            lon_label = f"{ref_lon}°E"
        elif ref_lon == 180:
            lon_label = "180°W"
        else:
            lon_label = f"{360 - ref_lon}°W"
        ax.text((r_max + 5) * np.cos(theta_ref),
                (r_max + 5) * np.sin(theta_ref), z_min,
                lon_label, fontsize=8, ha='center', va='center', zorder=1)

    ax.autoscale(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('Altitude (km)', labelpad=1)
    ax.set_title(title, fontsize=14, pad=0.01)
    ax.view_init(elev=20, azim=155)

    fig.colorbar(sc, ax=ax, label=cbar_label, pad=0.01, shrink=0.6)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.95)

    if output_file:
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
    return fig


def plot_level_with_mask_outline(field_da, mask_da, lev_idx=30, title=None,
                                 cmap='bone_r', cbar_label='Temperature (K)',
                                 vmin=None, vmax=None, nlevels=30,
                                 output_file=None, pressure=None,
                                 central_longitude=150, draw_60_circle=True,
                                 lat_label_lon=45, draw_lon_labels=True,
                                 lon_label_lats=(40.5,)):
    # North polar stereographic map of a single level with the vortex mask
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    field = field_da.isel(lev=lev_idx).values
    mask = mask_da.isel(lev=lev_idx).values.astype(float)

    lat = field_da['lat'].values
    lon = field_da['lon'].values
    lon = np.mod(lon, 360)

    # wind fields get a symmetric diverging colormap
    is_wind = (field_da.name == 'U')
    if is_wind:
        absmax = np.nanmax(np.abs(field))
        vmin, vmax = -absmax, absmax
        cmap = 'RdBu_r'
    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)
    levels = np.linspace(vmin, vmax, nlevels)

    # wrap the seam so the contour closes across 0/360
    lon_ext = np.append(lon, lon[0] + 360.0)
    field_ext = np.concatenate([field, field[:, :1]], axis=1)
    wrap_col = np.maximum(mask[:, :1], mask[:, -1:])
    mask_ext = np.concatenate([mask, wrap_col], axis=1)

    # figure and projection
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(
        central_longitude=central_longitude))
    ax.set_extent([0, 360, 40, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="k",
                   zorder=0)
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    # field contour
    cf = ax.contourf(lon_ext, lat, field_ext, levels=levels, cmap=cmap,
                     transform=ccrs.PlateCarree(), zorder=1)

    # mask outline
    ax.contour(lon_ext, lat, mask_ext, levels=[0.5], colors='red',
               linewidths=2.0, transform=ccrs.PlateCarree(), zorder=4)

    # gridlines (no labels)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.6, color='black', alpha=0.18,
                      linestyle='-', zorder=10)
    gl.xlocator = mticker.FixedLocator(np.arange(0, 361, 45))

    # longitude meridians
    for lon0 in np.arange(0, 360, 45):
        ax.plot([lon0, lon0], [40, 90], transform=ccrs.PlateCarree(),
                color='black', linewidth=0.6, alpha=0.25, zorder=2)

    # latitude line at 60 N
    x = np.linspace(0, 360, 600)
    ax.plot(x, np.full_like(x, 60), transform=ccrs.PlateCarree(),
            color='black', linewidth=1.1,
            linestyle='--' if draw_60_circle else '-', alpha=0.9, zorder=2)
    if draw_60_circle:
        theta = np.linspace(0, 2 * np.pi, 300)
        r = 90 - 60
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                transform=ccrs.PlateCarree(), color='black', linewidth=1.1,
                linestyle='--', zorder=5)

    # latitude labels
    for y in [50, 60, 70, 80]:
        ax.text(lat_label_lon, y, f"{y}°N", transform=ccrs.PlateCarree(),
                ha='right', va='center', fontsize=9, color='black', zorder=11)

    # longitude labels
    if draw_lon_labels:
        for lon0 in np.arange(0, 360, 45):
            for lat0 in lon_label_lats:
                ax.text(lon0, lat0, f"{lon0}°", transform=ccrs.PlateCarree(),
                        ha='center', va='center', fontsize=9, rotation=0,
                        zorder=12)

    # colorbar and title
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)

    base_title = (f"{field_da.name} with Geopotential Mask Outline\n"
                  f"{field_da.lev.values[lev_idx]:.1f} hPa")
    if pressure is not None:
        base_title += f"\nPressure: {pressure}"
    if title is not None:
        base_title = title
    ax.set_title(base_title)
    if pressure is not None:
        ax.text(0.5, 1.02, f"Pressure level: {pressure}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=9)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    print("vortexstates - polar vortex contour and geometry extraction on ERA5.")
    print("See vortexstates_README.md for usage.")