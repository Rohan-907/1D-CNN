"""
Beer-Lambert Computational vs Experimental Comparison
=======================================================
Compares a REAL measured LIDA ice-mixture spectrum ("Experimental")
against a COMPUTATIONAL spectrum built by linearly mixing the pure
component spectra from the AstromoleculesIR (NIST-style JDX) library
according to Beer-Lambert additivity:

    A_mix(wavenumber) = sum_i ( fraction_i * A_pure_i(wavenumber) )

This validates whether simple linear superposition of pure-component
absorbance explains a real mixture spectrum, or whether intermolecular
effects (H-bonding, peak shifts, etc.) cause real deviations -- which
is exactly what the residual panel (b) is designed to show.

Works for BOTH binary (2-component) and ternary (3-component) LIDA
mixtures: the script auto-detects how many components are present in
the mixture spectrum's filename/tarball name.

WHAT'S NEW IN THIS VERSION
---------------------------
1. Region-specific R2/RMSE (O-H region vs fingerprint region) are now
   collected for EVERY mixture in batch mode, not just shown on the
   individual figures. This is the number you actually want for the
   paper: if O-H region R2 is systematically lower than fingerprint
   region R2 across your mixtures, that's your quantitative evidence
   for non-additive hydrogen-bonding effects -- said honestly, from
   data you actually have, rather than asserted.
2. A CSV results table (all_mixtures_results.csv) is written to
   OUTPUT_DIR summarising every mixture: composition, mode, overall
   R2/RMSE, O-H region R2/RMSE, fingerprint region R2/RMSE.
3. Group statistics printed at the end: mean/std R2 for binary vs
   ternary, and mean R2 for O-H region vs fingerprint region across
   all mixtures -- the two comparisons that actually support (or
   don't) a hydrogen-bonding argument in your paper.
4. Optional ALL_TEMPERATURES mode: process every temperature file in
   each archive (not just the lowest), so thermal restructuring of
   the H-bond network shows up in the results table too. Off by
   default (keeps output at one figure per mixture, matching what you
   already have); flip ALL_TEMPERATURES = True to turn it on.

USAGE
-----
1. Set ASTROMOLECULES_DIR to the folder containing the .jdx pure-
   component files.
2. Set LIDA_ROOT_DIR to the folder containing 'binary' and 'ternary'
   subfolders of .tar.gz mixtures (BATCH_MODE processes all of them).
3. Run: python3 beer_lambert_comparison.py
"""

import os, re, glob, csv, tarfile, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
#  PUBLICATION STYLE
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor':  'white',
    'savefig.facecolor': 'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#000000',
    'axes.linewidth':    1.3,
    'axes.labelweight':  'bold',
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   10.5,
    'ytick.labelsize':   10.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'legend.fontsize':   9.5,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '#444444',
    'font.family':       'sans-serif',
    'font.sans-serif':   ['DejaVu Sans', 'Arial', 'Helvetica'],
    'lines.linewidth':   1.8,
})

# ═══════════════════════════════════════════════════════════════════
#  PATHS  — edit these for your machine
# ═══════════════════════════════════════════════════════════════════
ASTROMOLECULES_DIR = "/home/rohan/Desktop/4th sem/Test Astro Molecules/AstromoleculesIR"

# Parent folder that directly CONTAINS the 'binary' and 'ternary'
# subfolders of .tar.gz mixtures.
LIDA_PARENT_DIR = "/home/rohan/Desktop/4th sem/Test Astro Molecules/paper/extra work/lida_binary_ternary_spectra"

# Parent output folder. Binary mixtures are written to
# OUTPUT_PARENT_DIR/binary and ternary mixtures to
# OUTPUT_PARENT_DIR/ternary -- each run_batch() call below is pointed
# at its own matching subfolder on both sides, so the mixture-folder
# names inside each output (e.g. 'ce{HCOOH}_9%_+_ce{H2O}_91%.tar_15K')
# have no extra 'binary/'/'ternary/' prefix, exactly matching the
# earlier results.
OUTPUT_PARENT_DIR = "/home/rohan/Desktop/4th sem/Test Astro Molecules/paper/extra work/beer_lambert_output"

WN_MIN, WN_MAX = 400.0, 3500.0
DPI = 300

FP_MIN, FP_MAX = 800.0, 1500.0
OH_MIN, OH_MAX = 2500.0, 3500.0

# True (default): process EVERY mixture under each of LIDA_PARENT_DIR's
# 'binary' and 'ternary' subfolders.
BATCH_MODE = True

# False (default): only the lowest-temperature spectrum per archive is
# used (one figure per mixture, matches your original output). Set
# True to process every temperature file in every archive -- more
# rows in the results table, and lets you see whether deviation from
# Beer-Lambert additivity grows/shrinks with temperature (a genuine
# H-bonding-network-reorganization signature, not an assumption).
ALL_TEMPERATURES = False


# ═══════════════════════════════════════════════════════════════════
#  JCAMP-DX (.jdx) PARSER
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


# ═══════════════════════════════════════════════════════════════════
#  MOLECULE NAME MATCHING
# ═══════════════════════════════════════════════════════════════════
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
    norm = _normalize_formula(formula_token)
    return formula_index.get(norm)


# ═══════════════════════════════════════════════════════════════════
#  LIDA MIXTURE NAME PARSING
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
            return [(f, p/total) for f, p in comps]

    ratio_match = re.search(r'([A-Za-z0-9]+)\.([A-Za-z0-9]+)\s+(\d+)-(\d+)', base)
    if ratio_match:
        f1, f2, r1, r2 = ratio_match.groups()
        r1, r2 = float(r1), float(r2)
        total = r1 + r2
        return [(f1, r1/total), (f2, r2/total)]

    raise ValueError(
        f"Could not parse mixture composition from name: '{name}'. "
        f"Edit parse_mixture_name() to handle your naming convention."
    )


# ═══════════════════════════════════════════════════════════════════
#  LIDA MIXTURE SPECTRUM LOADER
# ═══════════════════════════════════════════════════════════════════
def _temp_key(member_name):
    b = os.path.basename(member_name)
    b = re.sub(r'K\.txt$', '', b, flags=re.IGNORECASE)
    try:
        return float(b)
    except ValueError:
        return 1e9

def load_lida_mixture(path, all_temperatures=False):
    """
    Load real measured LIDA mixture spectrum/spectra.
    Returns a list of (wavenumber, absorbance_normalised, display_name,
    temperature) tuples. If all_temperatures is False, the list has
    exactly one entry (the lowest-temperature member). If True, one
    entry per temperature file found in the archive.
    """
    base_display = os.path.basename(path)
    base_display = re.sub(r'\.(tar\.gz|txt)$', '', base_display, flags=re.IGNORECASE)

    results = []

    if path.lower().endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as tf:
            txt_members = [m for m in tf.getnames() if m.lower().endswith('.txt')]
            if not txt_members:
                raise ValueError(f"No .txt spectrum found inside {path}")
            txt_members.sort(key=_temp_key)
            chosen_members = txt_members if all_temperatures else txt_members[:1]

            for member in chosen_members:
                fobj = tf.extractfile(member)
                data = np.genfromtxt(fobj, dtype=np.float64, invalid_raise=False)
                temp = _temp_key(member)
                results.append((data, base_display, temp))
    else:
        data = np.genfromtxt(path, dtype=np.float64, invalid_raise=False)
        results.append((data, base_display, np.nan))

    out = []
    for data, display_name, temp in results:
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Unexpected data shape in {path}: {data.shape}")

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

        out.append((wn, od, display_name, temp))

    return out


# ═══════════════════════════════════════════════════════════════════
#  BEER-LAMBERT COMPUTATIONAL MIXTURE
# ═══════════════════════════════════════════════════════════════════
def build_computational_mixture(components, formula_index, common_wn):
    mix = np.zeros_like(common_wn)
    missing = []
    used_fraction = 0.0

    for formula, fraction in components:
        fp = find_pure_spectrum(formula, formula_index)
        if fp is None:
            missing.append(formula)
            continue
        wn_pure, a_pure, meta = parse_jdx(fp)
        if wn_pure is None:
            missing.append(f"{formula} (file present but not absorbance/transmittance data)")
            continue
        interp_fn = interp1d(wn_pure, a_pure, kind='linear',
                             bounds_error=False, fill_value=0.0)
        a_on_common = interp_fn(common_wn)
        mix += fraction * a_on_common
        used_fraction += fraction

    if used_fraction > 1e-9:
        mix = mix - mix.min()
        pk = np.percentile(mix, 99.5)
        if pk > 1e-9:
            mix = np.clip(mix / pk, 0, None)

    return mix, missing


def _region_r2_rmse(wn, exp, comp, lo, hi):
    m = (wn >= lo) & (wn <= hi)
    if m.sum() <= 5 or np.std(exp[m]) < 1e-10 or np.std(comp[m]) < 1e-10:
        return None, None
    r, _ = pearsonr(exp[m], comp[m])
    rmse = float(np.sqrt(np.mean((exp[m] - comp[m]) ** 2)))
    return float(r**2), rmse


# ═══════════════════════════════════════════════════════════════════
#  FIGURE: 4-PANEL EXPERIMENTAL vs COMPUTATIONAL COMPARISON
# ═══════════════════════════════════════════════════════════════════
def plot_beer_lambert_4panel(wn, exp, comp, title, mixture_label, output_dir,
                             filename='fig_beer_lambert_4panel'):
    os.makedirs(output_dir, exist_ok=True)

    r, _ = pearsonr(exp, comp)
    r2 = float(r**2)
    rmse = float(np.sqrt(np.mean((exp - comp) ** 2)))
    residual = comp - exp
    rs = float(np.std(residual))

    oh_r2, oh_rmse = _region_r2_rmse(wn, exp, comp, OH_MIN, OH_MAX)
    fp_r2, fp_rmse = _region_r2_rmse(wn, exp, comp, FP_MIN, FP_MAX)

    quality = '✓ Good' if r2 >= 0.85 else ('~ Fair' if r2 >= 0.5 else '✗ Poor')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
    ax_a, ax_c = axes[0, 0], axes[0, 1]
    ax_b, ax_d = axes[1, 0], axes[1, 1]

    def _panel_label(ax, letter):
        ax.text(-0.12, 1.08, f'({letter})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    ax_a.axvspan(max(FP_MIN, wn.min()), min(FP_MAX, wn.max()),
                color='#aee3f7', alpha=0.55, zorder=0, label='C–H/C–O')
    ax_a.axvspan(max(OH_MIN, wn.min()), min(OH_MAX, wn.max()),
                color='#ffd0c4', alpha=0.60, zorder=0, label='O–H region')
    ax_a.fill_between(wn, exp, comp, color='#ff4d4d', alpha=0.30,
                      zorder=1, label='Deviation')
    ax_a.plot(wn, exp, color='#111111', lw=1.5, label='Experimental', zorder=3)
    ax_a.plot(wn, comp, color='#d62728', lw=1.5, label='Computational', zorder=4)
    ax_a.text(0.025, 0.96, f'R² = {r2:.4f}  {quality}\nRMSE = {rmse:.4f}',
             transform=ax_a.transAxes, fontsize=10.5, fontweight='bold',
             va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='#000000',
                       boxstyle='round,pad=0.45', linewidth=1.2))
    ax_a.set_xlim(wn.min(), wn.max())
    ax_a.set_ylim(-0.02, 1.08)
    ax_a.set_xlabel('Wavenumber (cm⁻¹)')
    ax_a.set_ylabel('Absorbance')
    ax_a.set_title(f'Full Spectrum (<{wn.max():.0f} cm⁻¹)', pad=10)
    ax_a.legend(loc='upper right', handlelength=1.6, labelspacing=0.4)
    _panel_label(ax_a, 'a')

    herr = np.abs(residual) > 2*rs
    ax_b.axhspan(-2*rs, 2*rs, color='#f2c879', alpha=0.30, zorder=0, label='±2σ')
    ax_b.axhspan(-rs, rs, color='#f2c879', alpha=0.55, zorder=1, label='±1σ')
    ax_b.plot(wn, residual, color='#e07b00', lw=1.0, alpha=0.85,
             zorder=3, label='Residual (comp−exp)')
    ax_b.axhline(0, color='#1a7a1a', lw=1.8, ls='--', zorder=4, label='Zero (ideal)')
    if herr.any():
        ax_b.scatter(wn[herr], residual[herr], color='#d62728', s=14,
                    zorder=5, label=f'|Δ|>2σ ({int(herr.sum())} pts)')
    ax_b.text(0.025, 0.96, f'Mean = {float(np.mean(residual)):+.4f}'
              f'   Std = {rs:.4f}\n% pts |Δ|>2σ: {100*float(np.mean(herr)):.1f}%',
             transform=ax_b.transAxes, fontsize=10.5, fontweight='bold',
             va='top', ha='left',
             bbox=dict(facecolor='#fffaf0', edgecolor='#000000',
                       boxstyle='round,pad=0.45', linewidth=1.2))
    ax_b.set_xlim(wn.min(), wn.max())
    ax_b.set_xlabel('Wavenumber (cm⁻¹)')
    ax_b.set_ylabel('Residual')
    ax_b.set_title(f'Residual Spectrum (<{wn.max():.0f} cm⁻¹)', pad=10)
    ax_b.legend(loc='lower right', handlelength=1.6, labelspacing=0.4,
               ncol=2, fontsize=8.5)
    _panel_label(ax_b, 'b')

    m_oh = (wn >= OH_MIN) & (wn <= OH_MAX)
    if m_oh.sum() > 5:
        ax_c.fill_between(wn[m_oh], exp[m_oh], comp[m_oh], color='#ff4d4d', alpha=0.30)
        ax_c.plot(wn[m_oh], exp[m_oh], color='#111111', lw=1.5, label='Experimental')
        ax_c.plot(wn[m_oh], comp[m_oh], color='#d62728', lw=1.5, label='Computational')
        ax_c.set_xlim(wn[m_oh].min(), wn[m_oh].max())
        if oh_r2 is not None:
            ax_c.text(0.03, 0.96, f'R² = {oh_r2:.4f}', transform=ax_c.transAxes,
                     fontsize=10, fontweight='bold', va='top', color='#d62728',
                     bbox=dict(facecolor='white', edgecolor='#d62728',
                               boxstyle='round,pad=0.3', linewidth=1.1))
    ax_c.set_ylim(-0.02, 1.08)
    ax_c.set_xlabel('Wavenumber (cm⁻¹)')
    ax_c.set_ylabel('Absorbance')
    ax_c.set_title(f'O–H Region ({OH_MIN:.0f}–{OH_MAX:.0f} cm⁻¹)', pad=10)
    ax_c.legend(loc='upper right', handlelength=1.6, labelspacing=0.4)
    _panel_label(ax_c, 'c')

    m_fp = (wn >= FP_MIN) & (wn <= FP_MAX)
    if m_fp.sum() > 5:
        ax_d.fill_between(wn[m_fp], exp[m_fp], comp[m_fp], color='#ff4d4d', alpha=0.30)
        ax_d.plot(wn[m_fp], exp[m_fp], color='#111111', lw=1.5, label='Experimental')
        ax_d.plot(wn[m_fp], comp[m_fp], color='#d62728', lw=1.5, label='Computational')
        ax_d.set_xlim(wn[m_fp].min(), wn[m_fp].max())
        if fp_r2 is not None:
            ax_d.text(0.03, 0.96, f'R² = {fp_r2:.4f}', transform=ax_d.transAxes,
                     fontsize=10, fontweight='bold', va='top', color='#1f77b4',
                     bbox=dict(facecolor='white', edgecolor='#1f77b4',
                               boxstyle='round,pad=0.3', linewidth=1.1))
    ax_d.set_ylim(-0.02, 1.08)
    ax_d.set_xlabel('Wavenumber (cm⁻¹)')
    ax_d.set_ylabel('Absorbance')
    ax_d.set_title(f'Fingerprint Region ({FP_MIN:.0f}–{FP_MAX:.0f} cm⁻¹)', pad=10)
    ax_d.legend(loc='upper right', handlelength=1.6, labelspacing=0.4)
    _panel_label(ax_d, 'd')

    fig.suptitle(f'{title}', fontsize=15, fontweight='bold', y=1.015)
    fig.text(0.5, 0.975, mixture_label, ha='center', fontsize=11.5,
             fontweight='bold', color='#333333')
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.8, w_pad=2.6)

    for ext in ('png', 'pdf'):
        out_path = os.path.join(output_dir, f'{filename}.{ext}')
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"  ✓ {out_path}")
    plt.close(fig)

    return dict(r2=r2, rmse=rmse, oh_r2=oh_r2, oh_rmse=oh_rmse,
                fp_r2=fp_r2, fp_rmse=fp_rmse)


# ═══════════════════════════════════════════════════════════════════
#  GROUP STATISTICS + CSV EXPORT
# ═══════════════════════════════════════════════════════════════════
def write_results_csv(results, output_dir):
    csv_path = os.path.join(output_dir, 'all_mixtures_results.csv')
    fields = ['file', 'mode', 'composition', 'temperature_K',
              'r2', 'rmse', 'oh_r2', 'oh_rmse', 'fp_r2', 'fp_rmse', 'missing']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fields})
    print(f"\n  ✓ {csv_path}")
    return csv_path

def print_group_statistics(results):
    ok = [r for r in results if r.get('r2') is not None]
    if not ok:
        print("\nNo successful mixtures to summarise.")
        return

    print(f"\n{'='*70}\nGROUP STATISTICS\n{'='*70}")

    for mode in ('Binary', 'Ternary'):
        group = [r for r in ok if r['mode'] == mode]
        if group:
            r2s = np.array([r['r2'] for r in group])
            print(f"  {mode}: n={len(group)}  mean R2={r2s.mean():.4f}  std={r2s.std():.4f}")

    oh_vals = np.array([r['oh_r2'] for r in ok if r.get('oh_r2') is not None])
    fp_vals = np.array([r['fp_r2'] for r in ok if r.get('fp_r2') is not None])
    if len(oh_vals) and len(fp_vals):
        print(f"\n  O–H region:        mean R2={oh_vals.mean():.4f}  std={oh_vals.std():.4f}  (n={len(oh_vals)})")
        print(f"  Fingerprint region: mean R2={fp_vals.mean():.4f}  std={fp_vals.std():.4f}  (n={len(fp_vals)})")
        gap = fp_vals.mean() - oh_vals.mean()
        print(f"  Fingerprint − O–H mean R2 gap: {gap:+.4f}")
        if gap > 0.05:
            print("  -> O–H region fits linear Beer-Lambert additivity worse than the "
                  "fingerprint region. That gap is your quantitative evidence for "
                  "non-additive hydrogen-bonding effects in the O–H stretch — report "
                  "the actual numbers above rather than asserting the effect.")
        else:
            print("  -> No clear systematic gap between regions in this data; the "
                  "hydrogen-bonding-deviation story isn't well supported by these "
                  "numbers as-is. Worth checking region boundaries and sample size "
                  "before writing this into the paper.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def run_comparison(lida_mixture_path, astromolecules_dir, output_dir,
                   wn_min=WN_MIN, wn_max=WN_MAX):
    if not os.path.exists(lida_mixture_path):
        parent = os.path.dirname(lida_mixture_path)
        msg = [f"LIDA_MIXTURE_FILE does not exist:\n  {lida_mixture_path}"]
        if os.path.isdir(parent):
            candidates = sorted(glob.glob(os.path.join(parent, '*.tar.gz')))
            candidates += sorted(glob.glob(os.path.join(parent, '*.txt')))
            if candidates:
                msg.append(f"\nFiles actually found in {parent}:")
                for c in candidates[:15]:
                    msg.append(f"  {os.path.basename(c)}")
                if len(candidates) > 15:
                    msg.append(f"  ... and {len(candidates)-15} more")
                msg.append("\nCopy one of the exact names above into "
                           "LIDA_MIXTURE_FILE at the top of this script.")
            else:
                msg.append(f"\n(That folder exists but contains no "
                           f".tar.gz or .txt files.)")
        else:
            msg.append(f"\nThe parent folder doesn't exist either:\n  {parent}")
        raise FileNotFoundError('\n'.join(msg))

    print(f"Loading pure-component library from: {astromolecules_dir}")
    formula_index = build_formula_index(astromolecules_dir)
    print(f"  ✓ Indexed {len(formula_index)} pure-component spectra")

    print(f"\nLoading real LIDA mixture spectrum: {lida_mixture_path}")
    loaded = load_lida_mixture(lida_mixture_path, all_temperatures=False)
    wn_exp, a_exp, display_name, temp = loaded[0]
    print(f"  ✓ Loaded {len(wn_exp)} points, "
          f"wavenumber range [{wn_exp.min():.0f}, {wn_exp.max():.0f}] cm⁻¹")

    print(f"\nParsing mixture composition from name: '{display_name}'")
    components = parse_mixture_name(display_name)
    n_comp = len(components)
    mode_label = {2: 'Binary', 3: 'Ternary'}.get(n_comp, f'{n_comp}-component')
    comp_str = ' + '.join(f'{f} ({100*p:.0f}%)' for f, p in components)
    print(f"  ✓ Detected {mode_label} mixture: {comp_str}")

    common_wn = np.linspace(max(wn_min, wn_exp.min()),
                            min(wn_max, wn_exp.max()), 2000)
    interp_exp = interp1d(wn_exp, a_exp, kind='linear',
                          bounds_error=False, fill_value=0.0)
    a_exp_common = np.clip(interp_exp(common_wn), 0, None)
    pk = np.percentile(a_exp_common, 99.5)
    if pk > 1e-9:
        a_exp_common = a_exp_common / pk

    print(f"\nBuilding Beer-Lambert computational mixture...")
    a_comp_common, missing = build_computational_mixture(components, formula_index, common_wn)
    if missing:
        print(f"  ⚠ Missing pure-component spectra for: {missing}")
    else:
        print(f"  ✓ All {n_comp} components found and mixed successfully")

    print(f"\nGenerating 4-panel comparison figure...")
    metrics = plot_beer_lambert_4panel(
        common_wn, a_exp_common, a_comp_common,
        title=f'Beer–Lambert Comparison — {mode_label} Mixture',
        mixture_label=comp_str,
        output_dir=output_dir,
    )
    print(f"\n  Overall R² = {metrics['r2']:.4f}   RMSE = {metrics['rmse']:.4f}")
    return metrics


def run_batch(lida_root_dir, astromolecules_dir, output_dir,
             wn_min=WN_MIN, wn_max=WN_MAX, all_temperatures=ALL_TEMPERATURES):
    all_files = sorted(glob.glob(os.path.join(lida_root_dir, '**', '*.tar.gz'),
                                 recursive=True))
    if not all_files:
        print(f"No .tar.gz files found under {lida_root_dir}")
        return []

    print(f"Found {len(all_files)} mixture files under {lida_root_dir}\n")
    formula_index = build_formula_index(astromolecules_dir)
    print(f"✓ Indexed {len(formula_index)} pure-component spectra\n")

    results = []
    for i, fp in enumerate(all_files, 1):
        rel = os.path.relpath(fp, lida_root_dir)
        print(f"\n[{i}/{len(all_files)}] {rel}")
        print("─" * 60)
        try:
            loaded = load_lida_mixture(fp, all_temperatures=all_temperatures)
            for wn_exp, a_exp, display_name, temp in loaded:
                components = parse_mixture_name(display_name)
                n_comp = len(components)
                mode_label = {2: 'Binary', 3: 'Ternary'}.get(n_comp, f'{n_comp}-component')
                comp_str = ' + '.join(f'{f} ({100*p:.0f}%)' for f, p in components)

                common_wn = np.linspace(max(wn_min, wn_exp.min()),
                                        min(wn_max, wn_exp.max()), 2000)
                interp_exp = interp1d(wn_exp, a_exp, kind='linear',
                                      bounds_error=False, fill_value=0.0)
                a_exp_common = np.clip(interp_exp(common_wn), 0, None)
                pk = np.percentile(a_exp_common, 99.5)
                if pk > 1e-9:
                    a_exp_common = a_exp_common / pk

                a_comp_common, missing = build_computational_mixture(
                    components, formula_index, common_wn)

                temp_tag = f"_{temp:.0f}K" if np.isfinite(temp) else ""
                sub_out = os.path.join(output_dir, os.path.splitext(rel)[0] + temp_tag)

                metrics = plot_beer_lambert_4panel(
                    common_wn, a_exp_common, a_comp_common,
                    title=f'Beer–Lambert Comparison — {mode_label} Mixture',
                    mixture_label=comp_str + (f'  (T={temp:.0f} K)' if np.isfinite(temp) else ''),
                    output_dir=sub_out,
                )
                print(f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}" +
                     (f"  ⚠ missing: {missing}" if missing else ""))

                results.append(dict(
                    file=rel + temp_tag, mode=mode_label, composition=comp_str,
                    temperature_K=temp if np.isfinite(temp) else '',
                    r2=metrics['r2'], rmse=metrics['rmse'],
                    oh_r2=metrics['oh_r2'], oh_rmse=metrics['oh_rmse'],
                    fp_r2=metrics['fp_r2'], fp_rmse=metrics['fp_rmse'],
                    missing='; '.join(missing) if missing else '',
                ))
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            results.append(dict(file=rel, mode='', composition='', temperature_K='',
                                r2=None, rmse=None, oh_r2=None, oh_rmse=None,
                                fp_r2=None, fp_rmse=None,
                                missing=f"{type(e).__name__}: {e}"))

    print(f"\n{'='*70}\nBATCH SUMMARY\n{'='*70}")
    for r in results:
        if r['r2'] is not None:
            print(f"  {r['file']:55s} R²={r['r2']:.4f}  {r['composition']}")
        else:
            print(f"  {r['file']:55s} FAILED: {r['missing']}")

    os.makedirs(output_dir, exist_ok=True)
    write_results_csv(results, output_dir)
    print_group_statistics(results)

    return results


if __name__ == '__main__':
    if BATCH_MODE:
        for subfolder in ('binary', 'ternary'):
            lida_dir = os.path.join(LIDA_PARENT_DIR, subfolder)
            out_dir = os.path.join(OUTPUT_PARENT_DIR, subfolder)
            if not os.path.isdir(lida_dir):
                print(f"Skipping '{subfolder}': folder not found at {lida_dir}")
                continue
            print(f"\n{'#'*70}\n#  PROCESSING: {subfolder.upper()}\n"
                  f"#  input:  {lida_dir}\n#  output: {out_dir}\n{'#'*70}")
            run_batch(lida_dir, ASTROMOLECULES_DIR, out_dir)
    else:
        # Single-file mode: point this at one specific mixture archive.
        single_file = os.path.join(
            LIDA_PARENT_DIR, "binary", "ce{HCOOH}_9%_+_ce{CO2}_91%.tar.gz")
        run_comparison(single_file, ASTROMOLECULES_DIR,
                       os.path.join(OUTPUT_PARENT_DIR, "binary"))
