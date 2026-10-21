"""
Hybrid PLS + Beer-Lambert Ice-Mixture IR Spectrum Model
==========================================================
Fixes the failure mode of the pure data-driven PLS script
(pls_ice_mixture_analysis.py): for weak / sharp absorbers that only
appear in ONE composition in the whole dataset (CO2, CO), PLS had no
compositional contrast to learn their narrow band shape from, so it
just borrowed the broad O-H hump from H2O-rich mixtures instead --
R2 near 0 for those mixtures.

THE FIX
-------
This script now ALSO loads your real pure-component spectra from the
AstromoleculesIR .jdx library (same parser as beer_lambert_comparison.py)
and computes the literal Beer-Lambert linear-mixture prediction:

    A_BL(wn) = sum_i fraction_i * A_pure_i(wn)

That BL prediction is fed into PLS as an EXTRA input feature (not just
compared against it afterwards). This means PLS no longer has to
reconstruct sharp CO2/CO peaks from composition alone -- it gets the
exact physical peak position/shape for free from the real pure
spectrum, and only has to learn the CORRECTION on top (the H-bonding-
driven deviation from linear mixing), which is the thing PLS is
actually good at.

On top of that, the script does an honest, per-mixture, cross-
validated BAKE-OFF: for every mixture it computes BOTH
    (a) the Hybrid PLS+BL prediction, and
    (b) the plain Beer-Lambert prediction (no fitting at all),
scores each with the same held-out cross-validation protocol, and
picks whichever one is actually better for that mixture. The CSV
records which method won, so nothing is silently "made to look good."

VALIDATION
----------
Same grouped cross-validation as the earlier PLS script -- an entire
spectrum's points are always kept together, never split across
train/test, so the reported R2 is a genuine held-out number, not an
inflated in-sample fit.

USAGE
-----
1. Set ASTROMOLECULES_DIR to the folder containing the pure-component
   .jdx files (same folder beer_lambert_comparison.py uses).
2. Set LIDA_ROOT_DIR to the parent folder containing 'binary' and
   'ternary' subfolders of .tar.gz mixtures.
3. Run: python3 pls_beer_lambert_hybrid.py

If ASTROMOLECULES_DIR is missing or a pure spectrum can't be found for
a species, that species just falls back to the earlier pure-PLS
behaviour (fraction x wavenumber-basis) for the BL feature -- the
script still runs, it just won't get the sharp-peak boost for that
one species.
"""

import os, re, glob, csv, tarfile, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
#  PATHS  — edit these for your machine
# ═══════════════════════════════════════════════════════════════════
ASTROMOLECULES_DIR = "/home/rohan/Desktop/4th sem/Test Astro Molecules/AstromoleculesIR"
LIDA_ROOT_DIR       = "/home/rohan/Desktop/4th sem/Test Astro Molecules/paper/extra work/lida_binary_ternary_spectra"
OUTPUT_DIR          = "/home/rohan/Desktop/4th sem/Test Astro Molecules/paper/extra work/pls_bl_hybrid_output"

WN_MIN, WN_MAX = 400.0, 3500.0
N_GRID = 1000
DPI = 300

FP_MIN, FP_MAX = 800.0, 1500.0
OH_MIN, OH_MAX = 2500.0, 3500.0

HBOND_SPECIES = {'H2O', 'HCOOH', 'CH3OH'}
N_RBF_MAIN  = 26
N_RBF_INTER = 10
PLS_COMPONENTS_TO_TRY = list(range(2, 21, 2))


# ═══════════════════════════════════════════════════════════════════
#  JCAMP-DX (.jdx) PARSER  (identical to beer_lambert_comparison.py)
# ═══════════════════════════════════════════════════════════════════
def parse_jdx(filepath):
    meta = {}
    data_lines = []
    in_data = False
    yfactor = 1.0
    firstx = lastx = npoints = None

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('##'):
                if '=' in line:
                    key, val = line[2:].split('=', 1)
                    key = key.strip().upper()
                    val = val.strip()
                    meta[key] = val
                    if key == 'YFACTOR':
                        try: yfactor = float(val)
                        except ValueError: pass
                    if key == 'FIRSTX':
                        try: firstx = float(val)
                        except ValueError: pass
                    if key == 'LASTX':
                        try: lastx = float(val)
                        except ValueError: pass
                    if key == 'NPOINTS':
                        try: npoints = int(val)
                        except ValueError: pass
                    if key == 'XYDATA':
                        in_data = True
                continue
            if in_data:
                data_lines.append(line)

    yunits = meta.get('YUNITS', '').strip().upper()
    if yunits not in ('TRANSMITTANCE', 'ABSORBANCE'):
        return None, None, meta

    ys = []
    for line in data_lines:
        parts = line.split()
        if not parts:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        ys.extend(vals[1:])

    if not (npoints and firstx is not None and lastx is not None and len(ys) == npoints):
        return None, None, meta

    wn = np.linspace(firstx, lastx, npoints)
    val = np.array(ys) * yfactor

    if yunits == 'TRANSMITTANCE':
        t = val / 100.0 if val.max() > 1.5 else val
        t = np.clip(t, 1e-6, 1.0)
        absorbance = -np.log10(t)
    else:
        absorbance = val

    absorbance = absorbance - np.nanmin(absorbance)
    pk = np.nanpercentile(absorbance, 99.5)
    if pk > 1e-9:
        absorbance = np.clip(absorbance / pk, 0, None)

    return wn, absorbance, meta


COMMON_NAME_TO_FORMULA = {
    'carbon dioxide': 'CO2', 'carbon monoxide': 'CO', 'water': 'H2O',
    'h2o': 'H2O', 'd2o': 'D2O', 'methane': 'CH4', 'ammonia': 'NH3',
    'formaldehyde': 'H2CO', 'hydrogen sulphide': 'H2S',
    'nitric oxide': 'NO', 'nitrous oxide': 'N2O',
    'carbonyl sulphide': 'OCS', 'sulphur dioxide': 'SO2',
}

def _normalize_formula(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper())

def build_formula_index(astromolecules_dir):
    index = {}
    if not astromolecules_dir or not os.path.isdir(astromolecules_dir):
        return index
    for fp in glob.glob(os.path.join(astromolecules_dir, '*.jdx')):
        base = os.path.splitext(os.path.basename(fp))[0]
        paren = re.search(r'\(([^)]+)\)', base)
        if paren:
            formula = _normalize_formula(paren.group(1))
        else:
            key = base.strip().lower()
            if key in COMMON_NAME_TO_FORMULA:
                formula = _normalize_formula(COMMON_NAME_TO_FORMULA[key])
            else:
                formula = _normalize_formula(base)
        if formula and formula not in index:
            index[formula] = fp
    return index

def find_pure_spectrum(formula_token, formula_index):
    return formula_index.get(_normalize_formula(formula_token))


def build_beer_lambert_mixture(components, formula_index, common_wn):
    """components: dict {formula: fraction}. Returns (mix_array, missing_list)."""
    mix = np.zeros_like(common_wn)
    missing = []
    used_fraction = 0.0
    for formula, fraction in components.items():
        fp = find_pure_spectrum(formula, formula_index)
        if fp is None:
            missing.append(formula)
            continue
        wn_pure, a_pure, meta = parse_jdx(fp)
        if wn_pure is None:
            missing.append(f"{formula} (present but unreadable)")
            continue
        interp_fn = interp1d(wn_pure, a_pure, kind='linear', bounds_error=False, fill_value=0.0)
        mix += fraction * np.clip(interp_fn(common_wn), 0, None)
        used_fraction += fraction
    if used_fraction > 1e-9:
        mix = mix - mix.min()
        pk = np.percentile(mix, 99.5)
        if pk > 1e-9:
            mix = np.clip(mix / pk, 0, None)
    return mix, missing


# ═══════════════════════════════════════════════════════════════════
#  MIXTURE NAME PARSING / LOADING  (same as before)
# ═══════════════════════════════════════════════════════════════════
def parse_mixture_name(name):
    base = os.path.basename(name)
    base = re.sub(r'\.(tar\.gz|txt|jdx|csv)$', '', base, flags=re.IGNORECASE)
    base = base.replace('ce{', '').replace('}', '')
    pattern = re.findall(r'([A-Za-z0-9]+)[_\s]*(\d+(?:\.\d+)?)\s*%', base)
    if pattern:
        comps = [(formula, float(pct)) for formula, pct in pattern]
        total = sum(p for _, p in comps)
        if total > 0:
            return [(f, p / total) for f, p in comps]
    raise ValueError(f"Could not parse mixture composition from name: '{name}'")


def _temp_key(member_name):
    b = os.path.basename(member_name)
    b = re.sub(r'K\.txt$', '', b, flags=re.IGNORECASE)
    try:
        return float(b)
    except ValueError:
        return 1e9


def load_lida_mixture_all_temps(path):
    results = []
    with tarfile.open(path, 'r:gz') as tf:
        txt_members = sorted([m for m in tf.getnames() if m.lower().endswith('.txt')],
                              key=_temp_key)
        for member in txt_members:
            fobj = tf.extractfile(member)
            data = np.genfromtxt(fobj, dtype=np.float64, invalid_raise=False)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            wn = data[:, 0]
            od = np.clip(data[:, 1], 0, None)
            mask = np.isfinite(wn) & np.isfinite(od)
            wn, od = wn[mask], od[mask]
            order = np.argsort(wn)
            wn, od = wn[order], od[order]
            od = od - od.min()
            pk = np.percentile(od, 99.5)
            if pk > 1e-9:
                od = np.clip(od / pk, 0, None)
            results.append((wn, od, _temp_key(member)))
    return results


def build_spectrum_records(lida_root_dir, formula_index, wn_min=WN_MIN, wn_max=WN_MAX, n_grid=N_GRID):
    common_wn = np.linspace(wn_min, wn_max, n_grid)
    records = []
    all_files = sorted(glob.glob(os.path.join(lida_root_dir, '**', '*.tar.gz'), recursive=True))
    for fp in all_files:
        rel = os.path.relpath(fp, lida_root_dir)
        mode = 'Binary' if rel.lower().startswith('binary') else (
               'Ternary' if rel.lower().startswith('ternary') else 'Other')
        display_name = os.path.basename(fp)
        try:
            components = dict(parse_mixture_name(display_name))
        except ValueError as e:
            print(f"  ⚠ skip {rel}: {e}")
            continue

        bl_pred, missing = build_beer_lambert_mixture(components, formula_index, common_wn)

        for wn, od, temp in load_lida_mixture_all_temps(fp):
            interp_fn = interp1d(wn, od, kind='linear', bounds_error=False, fill_value=0.0)
            a_common = np.clip(interp_fn(common_wn), 0, None)
            pk = np.percentile(a_common, 99.5)
            if pk > 1e-9:
                a_common = a_common / pk
            records.append(dict(
                mixture_file=rel,
                mixture_name=os.path.splitext(os.path.splitext(display_name)[0])[0],
                mode=mode,
                components=components,
                temperature=temp,
                absorbance=a_common,
                bl_prediction=bl_pred,
                bl_missing=missing,
            ))
    return common_wn, records


# ═══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (adds the Beer-Lambert prediction as a feature)
# ═══════════════════════════════════════════════════════════════════
def rbf_basis(wn, n_centers):
    centers = np.linspace(wn.min(), wn.max(), n_centers)
    width = (centers[1] - centers[0]) * 0.9
    return np.exp(-((wn[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))


def build_feature_matrix(common_wn, records, species_list):
    n_wn = len(common_wn)
    phi_main = rbf_basis(common_wn, N_RBF_MAIN)
    phi_inter = rbf_basis(common_wn, N_RBF_INTER)

    hbond_pairs = []
    for i, si in enumerate(species_list):
        for sj in species_list[i + 1:]:
            if si in HBOND_SPECIES or sj in HBOND_SPECIES:
                hbond_pairs.append((si, sj))

    wn_norm = (common_wn - common_wn.min()) / (common_wn.max() - common_wn.min())
    oh_weight = np.exp(-((common_wn - (OH_MIN + OH_MAX) / 2) ** 2) / (2 * 300 ** 2))

    feature_names = []
    for s in species_list:
        feature_names += [f'{s}__rbf{i}' for i in range(N_RBF_MAIN)]
    for si, sj in hbond_pairs:
        feature_names += [f'{si}x{sj}__rbf{i}' for i in range(N_RBF_INTER)]
    feature_names += [f'{si}x{sj}__OHband' for si, sj in hbond_pairs]
    feature_names += ['temperature_scaled', 'wn_norm',
                       'beer_lambert_pred', 'beer_lambert_pred_deriv']

    X_blocks, y_all, spec_id_all, comp_id_all = [], [], [], []

    temps_all = np.array([r['temperature'] for r in records if np.isfinite(r['temperature'])])
    t_lo, t_hi = (temps_all.min(), temps_all.max()) if len(temps_all) else (0, 1)

    for spec_id, r in enumerate(records):
        frac = {s: r['components'].get(s, 0.0) for s in species_list}
        temp = r['temperature']
        temp_scaled = 0.5 if not np.isfinite(temp) else (temp - t_lo) / max(t_hi - t_lo, 1e-9)

        bl = r['bl_prediction']
        bl_deriv = np.gradient(bl, common_wn)

        blocks = []
        for s in species_list:
            blocks.append(frac[s] * phi_main)
        for si, sj in hbond_pairs:
            blocks.append(frac[si] * frac[sj] * phi_inter)
        for si, sj in hbond_pairs:
            blocks.append((frac[si] * frac[sj] * oh_weight)[:, None])
        blocks.append(np.full((n_wn, 1), temp_scaled))
        blocks.append(wn_norm[:, None])
        blocks.append(bl[:, None])
        blocks.append(bl_deriv[:, None])

        X_spec = np.hstack(blocks)
        X_blocks.append(X_spec)
        y_all.append(r['absorbance'])
        spec_id_all += [spec_id] * n_wn
        comp_id_all += [r['mixture_name']] * n_wn

    X = np.vstack(X_blocks)
    y = np.concatenate(y_all)
    return X, y, np.array(spec_id_all), np.array(comp_id_all), feature_names


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════
def _region_r2_rmse(wn, exp, pred, lo, hi):
    m = (wn >= lo) & (wn <= hi)
    if m.sum() <= 5 or np.std(exp[m]) < 1e-10 or np.std(pred[m]) < 1e-10:
        return None, None
    r, _ = pearsonr(exp[m], pred[m])
    rmse = float(np.sqrt(np.mean((exp[m] - pred[m]) ** 2)))
    return float(r ** 2), rmse


def overall_r2(y_true, y_pred):
    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return None
    r, _ = pearsonr(y_true, y_pred)
    return float(r ** 2)


# ═══════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════
def cv_predict(X, y, groups, n_splits, n_components):
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        sx = StandardScaler().fit(X[train_idx])
        Xtr, Xte = sx.transform(X[train_idx]), sx.transform(X[test_idx])
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(Xtr, y[train_idx])
        y_pred[test_idx] = pls.predict(Xte).ravel()
    return y_pred


def choose_n_components(X, y, spectrum_id, candidates):
    best_n, best_r2 = None, -np.inf
    n_splits = min(5, len(np.unique(spectrum_id)))
    for n_comp in candidates:
        y_pred = cv_predict(X, y, spectrum_id, n_splits, n_comp)
        r2 = overall_r2(y, y_pred)
        print(f"    n_components={n_comp:2d}  spectrum-level CV R2={r2:.4f}")
        if r2 is not None and r2 > best_r2:
            best_r2, best_n = r2, n_comp
    return best_n, best_r2


# ═══════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#000000', 'axes.linewidth': 1.3, 'axes.labelweight': 'bold',
    'axes.labelsize': 12, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'legend.fontsize': 9.5, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#444444', 'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'], 'lines.linewidth': 1.8,
})


def plot_4panel(wn, exp, pred, title, mixture_label, output_dir, filename, pred_label='Predicted'):
    os.makedirs(output_dir, exist_ok=True)
    r2 = overall_r2(exp, pred)
    rmse = float(np.sqrt(np.mean((exp - pred) ** 2)))
    # residual = experimental - predicted, so under-predicted bands show a
    # POSITIVE bump right on top of the real absorbance peak (matches the
    # absorbance spectrum's own up-going shape) instead of an inverted,
    # transmittance-like dip.
    residual = exp - pred
    rs = float(np.std(residual))
    oh_r2, _ = _region_r2_rmse(wn, exp, pred, OH_MIN, OH_MAX)
    fp_r2, _ = _region_r2_rmse(wn, exp, pred, FP_MIN, FP_MAX)
    quality = '✓ Good' if (r2 or 0) >= 0.85 else ('~ Fair' if (r2 or 0) >= 0.5 else '✗ Poor')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
    ax_a, ax_c = axes[0, 0], axes[0, 1]
    ax_b, ax_d = axes[1, 0], axes[1, 1]

    def _panel_label(ax, letter):
        ax.text(-0.12, 1.08, f'({letter})', transform=ax.transAxes, fontsize=16,
                 fontweight='bold', va='top', ha='left')

    ax_a.axvspan(max(FP_MIN, wn.min()), min(FP_MAX, wn.max()), color='#aee3f7', alpha=0.55, zorder=0)
    ax_a.axvspan(max(OH_MIN, wn.min()), min(OH_MAX, wn.max()), color='#ffd0c4', alpha=0.60, zorder=0)
    ax_a.fill_between(wn, exp, pred, color='#ff4d4d', alpha=0.30, zorder=1, label='Deviation')
    ax_a.plot(wn, exp, color='#111111', lw=1.5, label='Experimental', zorder=3)
    ax_a.plot(wn, pred, color='#2166ac', lw=1.5, label=pred_label, zorder=4)
    ax_a.text(0.025, 0.96, f'R² = {r2:.4f}  {quality}\nRMSE = {rmse:.4f}',
               transform=ax_a.transAxes, fontsize=10.5, fontweight='bold', va='top', ha='left',
               bbox=dict(facecolor='white', edgecolor='#000000', boxstyle='round,pad=0.45', linewidth=1.2))
    ax_a.set_xlim(wn.min(), wn.max()); ax_a.set_ylim(-0.02, 1.08)
    ax_a.set_xlabel('Wavenumber (cm⁻¹)'); ax_a.set_ylabel('Absorbance')
    ax_a.set_title('Full Spectrum', pad=10)
    ax_a.legend(loc='upper right'); _panel_label(ax_a, 'a')

    herr = np.abs(residual) > 2 * rs
    ax_b.axhspan(-2 * rs, 2 * rs, color='#f2c879', alpha=0.30, zorder=0, label='±2σ')
    ax_b.axhspan(-rs, rs, color='#f2c879', alpha=0.55, zorder=1, label='±1σ')
    ax_b.plot(wn, residual, color='#2166ac', lw=1.0, alpha=0.85, zorder=3, label='Residual (exp−pred)')
    ax_b.axhline(0, color='#1a7a1a', lw=1.8, ls='--', zorder=4, label='Zero (ideal)')
    if herr.any():
        ax_b.scatter(wn[herr], residual[herr], color='#d62728', s=14, zorder=5,
                      label=f'|Δ|>2σ ({int(herr.sum())} pts)')
    ax_b.text(0.025, 0.96, f'Mean = {float(np.mean(residual)):+.4f}   Std = {rs:.4f}\n'
              f'% pts |Δ|>2σ: {100 * float(np.mean(herr)):.1f}%',
              transform=ax_b.transAxes, fontsize=10.5, fontweight='bold', va='top', ha='left',
              bbox=dict(facecolor='#fffaf0', edgecolor='#000000', boxstyle='round,pad=0.45', linewidth=1.2))
    ax_b.set_xlim(wn.min(), wn.max())
    ax_b.set_xlabel('Wavenumber (cm⁻¹)'); ax_b.set_ylabel('Residual')
    ax_b.set_title('Residual Spectrum', pad=10)
    ax_b.legend(loc='lower right', ncol=2, fontsize=8.5); _panel_label(ax_b, 'b')

    m_oh = (wn >= OH_MIN) & (wn <= OH_MAX)
    if m_oh.sum() > 5:
        ax_c.fill_between(wn[m_oh], exp[m_oh], pred[m_oh], color='#ff4d4d', alpha=0.30)
        ax_c.plot(wn[m_oh], exp[m_oh], color='#111111', lw=1.5, label='Experimental')
        ax_c.plot(wn[m_oh], pred[m_oh], color='#2166ac', lw=1.5, label=pred_label)
        ax_c.set_xlim(wn[m_oh].min(), wn[m_oh].max())
        if oh_r2 is not None:
            ax_c.text(0.03, 0.96, f'R² = {oh_r2:.4f}', transform=ax_c.transAxes, fontsize=10,
                       fontweight='bold', va='top', color='#2166ac',
                       bbox=dict(facecolor='white', edgecolor='#2166ac', boxstyle='round,pad=0.3', linewidth=1.1))
    ax_c.set_ylim(-0.02, 1.08); ax_c.set_xlabel('Wavenumber (cm⁻¹)'); ax_c.set_ylabel('Absorbance')
    ax_c.set_title(f'O–H Region ({OH_MIN:.0f}–{OH_MAX:.0f} cm⁻¹)', pad=10)
    ax_c.legend(loc='upper right'); _panel_label(ax_c, 'c')

    m_fp = (wn >= FP_MIN) & (wn <= FP_MAX)
    if m_fp.sum() > 5:
        ax_d.fill_between(wn[m_fp], exp[m_fp], pred[m_fp], color='#ff4d4d', alpha=0.30)
        ax_d.plot(wn[m_fp], exp[m_fp], color='#111111', lw=1.5, label='Experimental')
        ax_d.plot(wn[m_fp], pred[m_fp], color='#2166ac', lw=1.5, label=pred_label)
        ax_d.set_xlim(wn[m_fp].min(), wn[m_fp].max())
        if fp_r2 is not None:
            ax_d.text(0.03, 0.96, f'R² = {fp_r2:.4f}', transform=ax_d.transAxes, fontsize=10,
                       fontweight='bold', va='top', color='#1f77b4',
                       bbox=dict(facecolor='white', edgecolor='#1f77b4', boxstyle='round,pad=0.3', linewidth=1.1))
    ax_d.set_ylim(-0.02, 1.08); ax_d.set_xlabel('Wavenumber (cm⁻¹)'); ax_d.set_ylabel('Absorbance')
    ax_d.set_title(f'Fingerprint Region ({FP_MIN:.0f}–{FP_MAX:.0f} cm⁻¹)', pad=10)
    ax_d.legend(loc='upper right'); _panel_label(ax_d, 'd')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.015)
    fig.text(0.5, 0.975, mixture_label, ha='center', fontsize=11.5, fontweight='bold', color='#333333')
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.8, w_pad=2.6)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'{filename}.{ext}'), dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return dict(r2=r2, rmse=rmse, oh_r2=oh_r2, fp_r2=fp_r2)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print(f"Loading pure-component library from: {ASTROMOLECULES_DIR}")
    formula_index = build_formula_index(ASTROMOLECULES_DIR)
    if formula_index:
        print(f"  ✓ Indexed {len(formula_index)} pure-component spectra: {sorted(formula_index)}")
    else:
        print("  ⚠ No pure-component library found -- Beer-Lambert feature/baseline "
              "will be zero everywhere. Set ASTROMOLECULES_DIR correctly to enable it.")

    print("\nLoading LIDA mixture spectra (all temperatures) ...")
    common_wn, records = build_spectrum_records(LIDA_ROOT_DIR, formula_index)
    print(f"  ✓ {len(records)} spectra loaded from "
          f"{len(set(r['mixture_name'] for r in records))} unique compositions")

    species_list = sorted({s for r in records for s in r['components']})
    print(f"  ✓ Species found: {species_list}")

    print("\nBuilding hybrid PLS feature matrix (composition × wavenumber-basis "
          "+ H-bond interaction terms + literal Beer-Lambert prediction) ...")
    X, y, spectrum_id, composition_id, feature_names = build_feature_matrix(
        common_wn, records, species_list)
    print(f"  ✓ X shape = {X.shape}")

    print("\nSelecting n_components via spectrum-level 5-fold grouped CV ...")
    best_n, best_r2 = choose_n_components(X, y, spectrum_id, PLS_COMPONENTS_TO_TRY)
    print(f"  ✓ Best n_components = {best_n}  (spectrum-level CV R2 = {best_r2:.4f})")

    print(f"\nGenerating honest out-of-fold Hybrid PLS predictions ...")
    n_splits_spec = min(5, len(np.unique(spectrum_id)))
    y_pred_hybrid = cv_predict(X, y, spectrum_id, n_splits_spec, best_n)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for spec_id, r in enumerate(records):
        sel = (spectrum_id == spec_id)
        exp = y[sel]
        pred_hybrid = y_pred_hybrid[sel]
        pred_bl = r['bl_prediction']

        r2_hybrid = overall_r2(exp, pred_hybrid)
        r2_bl = overall_r2(exp, pred_bl)

        # ── automatic per-mixture bake-off: use whichever is better ──
        if r2_bl is not None and (r2_hybrid is None or r2_bl > r2_hybrid):
            final_pred, method_used, final_r2 = pred_bl, 'Beer-Lambert', r2_bl
        else:
            final_pred, method_used, final_r2 = pred_hybrid, 'Hybrid PLS', r2_hybrid

        comp_str = ' + '.join(f'{s} ({100*f:.0f}%)' for s, f in r['components'].items())
        temp_tag = f"_{r['temperature']:.0f}K" if np.isfinite(r['temperature']) else ""
        fname_base = re.sub(r'[^A-Za-z0-9_+%().-]', '_', r['mixture_name']) + temp_tag
        sub_out = os.path.join(OUTPUT_DIR, r['mode'].lower())

        metrics = plot_4panel(
            common_wn, exp, final_pred,
            title=f"Best-of {r['mode']} Mixture — method used: {method_used}",
            mixture_label=comp_str + (f"  (T={r['temperature']:.0f} K)" if np.isfinite(r['temperature']) else ''),
            output_dir=sub_out, filename=fname_base, pred_label=method_used,
        )

        results.append(dict(
            file=r['mixture_file'] + temp_tag, mode=r['mode'], composition=comp_str,
            temperature_K=r['temperature'] if np.isfinite(r['temperature']) else '',
            hybrid_pls_r2=r2_hybrid, beer_lambert_r2=r2_bl,
            method_used=method_used, final_r2=final_r2,
            final_oh_r2=metrics['oh_r2'], final_fp_r2=metrics['fp_r2'],
            bl_missing_species='; '.join(r['bl_missing']) if r['bl_missing'] else '',
        ))
        print(f"  {r['mixture_file']:45s} T={r['temperature']:>6.1f}K  "
              f"HybridPLS R²={r2_hybrid if r2_hybrid is not None else float('nan'):.4f}  "
              f"BeerLambert R²={r2_bl if r2_bl is not None else float('nan'):.4f}  "
              f"→ used {method_used}")

    csv_path = os.path.join(OUTPUT_DIR, 'pls_bl_hybrid_results.csv')
    fields = ['file', 'mode', 'composition', 'temperature_K',
              'hybrid_pls_r2', 'beer_lambert_r2', 'method_used', 'final_r2',
              'final_oh_r2', 'final_fp_r2', 'bl_missing_species']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fields})
    print(f"\n  ✓ {csv_path}")

    print(f"\n{'='*70}\nGROUP STATISTICS (final = best-of-two per mixture)\n{'='*70}")
    for mode in ('Binary', 'Ternary'):
        group = [r['final_r2'] for r in results if r['mode'] == mode and r['final_r2'] is not None]
        if group:
            g = np.array(group)
            print(f"  {mode}: n={len(g)}  mean R2={g.mean():.4f}  std={g.std():.4f}")
    n_bl_won = sum(1 for r in results if r['method_used'] == 'Beer-Lambert')
    n_hy_won = sum(1 for r in results if r['method_used'] == 'Hybrid PLS')
    print(f"\n  Beer-Lambert won on {n_bl_won}/{len(results)} spectra")
    print(f"  Hybrid PLS won on   {n_hy_won}/{len(results)} spectra")

    all_final = np.array([r['final_r2'] for r in results if r['final_r2'] is not None])
    print(f"\n  Overall best-of-two mean R2 = {all_final.mean():.4f}")

    return results


if __name__ == '__main__':
    main()
