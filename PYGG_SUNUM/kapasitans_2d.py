"""
PYGG_SUNUM/kapasitans_2d.py — Kapasitans eğrisi karşılaştırması (2D COMSOL veri seti)

Sensör kombinasyonu (PINN konv.):
    t1 = 4 µm  (alt parilen, t_parylene2)
    t2 = 0.2 µm (altın)
    t3 = 1 µm  (üst parilen, MATLAB t_1)
    ag = 10 µm  (hava aralığı)
    T_INS = 1 µm (sabit yalıtkan)

Her figür bir yarıçap değeri için C(P) eğrilerini gösterir:
    a ∈ {300, 350, 400, 450, 500} µm

a = 400 µm → Nominal sensör kombinasyonu olarak işaretlenir.

Kullanım:
    python PYGG_SUNUM/kapasitans_2d.py
    python PYGG_SUNUM/kapasitans_2d.py --checkpoint checkpoints/ckpt_final.pt
"""

import argparse, os, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad
from scipy.optimize import brentq

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _ROOT)

import torch
from model import DiaphragmPINN, InputNormaliser

# ── Çıktı ve veri yolları ─────────────────────────────────────────────────────
OUT_DIR   = _SCRIPT_DIR
EXCEL_PATH = os.path.join(_ROOT, "Sensor_2D_Dataset", "400um_4um.xlsx")

# ── Malzeme sabitleri ─────────────────────────────────────────────────────────
E_P       = 3.2e9
E_AU      = 70e9
NU_P      = 0.33
NU_AU     = 0.44
EPS_0     = 8.854e-12
EPS_P_REL = 3.15
T_INS     = 1.0e-6

# ── Sensör geometrisi (t_parylene2 = 4 µm bölümü) ────────────────────────────
T1_PINN = 4.0e-6   # alt parilen
T2_PINN = 0.2e-6   # altın
T3_PINN = 1.0e-6   # üst parilen
AG      = 10.0e-6  # hava aralığı

RADII_UM  = [300.0, 350.0, 400.0, 450.0, 500.0]   # µm
NOMINAL_A = 400.0   # nominal yarıçap (µm)

# ── Grafik stilleri ───────────────────────────────────────────────────────────
RENKLER = {"comsol": "#000000", "analitik": "#1565C0", "pinn": "#C62828"}
ETIKET  = {
    "comsol":   "Numerik Model (COMSOL)",
    "analitik": "Analitik Model",
    "pinn":     "PINN Modeli",
}
CIZGI   = {"comsol": "-", "analitik": "--", "pinn": ":"}


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL OKUMA
# ══════════════════════════════════════════════════════════════════════════════

def parse_excel(fpath):
    """
    400um_4um.xlsx dosyasını oku.

    Yapı: 5 bölüm (t_parylene2 = 4, 2, 6, 8, 10 µm)
    Her bölüm:
        satır 0  : başlık  "t_parylene2= X um"
        satır 1  : kolon adları  [p0, 300, 350, 400, 450, 500]
        satır 2+ : veri  P (Pa), C_300 (fF), C_350 (fF), ...

    Dönüş: dict { t_par2_um (float) -> {'P': ndarray, 'C': {r_um -> ndarray}} }
    """
    df_raw = pd.read_excel(fpath, header=None, engine="openpyxl")

    # Bölüm başlangıç satırlarını bul
    section_starts = []
    for i, val in enumerate(df_raw.iloc[:, 0]):
        if isinstance(val, str) and "t_parylene2" in val.lower():
            section_starts.append(i)

    if not section_starts:
        raise RuntimeError("Excel dosyasında bölüm başlığı bulunamadı.")

    sections = {}
    for idx, start in enumerate(section_starts):
        # Kalınlık değerini başlıktan çıkar
        title = str(df_raw.iloc[start, 0])
        m = re.search(r"([\d.]+)\s*um", title, re.IGNORECASE)
        t_um = float(m.group(1)) if m else float(idx)

        # Kolon adları (satır start+1)
        col_row = df_raw.iloc[start + 1, :].tolist()

        # Veri satırları
        data_start = start + 2
        data_end   = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(df_raw)
        block = df_raw.iloc[data_start:data_end].copy().reset_index(drop=True)

        # Sayısal olmayan satırları at (boş satır / sonraki başlık)
        p_col_numeric = pd.to_numeric(block.iloc[:, 0], errors="coerce")
        block = block[p_col_numeric.notna()].reset_index(drop=True)

        P_arr = block.iloc[:, 0].astype(float).values

        C_dict = {}
        for j, hdr in enumerate(col_row[1:], 1):
            if pd.isna(hdr):
                continue
            try:
                r_um = float(hdr)
            except (ValueError, TypeError):
                continue
            C_dict[r_um] = pd.to_numeric(block.iloc[:, j], errors="coerce").astype(float).values

        sections[t_um] = {"P": P_arr, "C": C_dict}
        print(f"  Bölüm t_parylene2={t_um} µm: {len(P_arr)} basınç noktası, "
              f"yarıçaplar: {sorted(C_dict.keys())} µm")

    return sections


# ══════════════════════════════════════════════════════════════════════════════
# ANALİTİK MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _analytical_stiffness(t1, t2, t3, a):
    """CLT yay katsayıları. t1=alt parilen, t3=üst parilen (PINN konv.)"""
    tm1, tm2, tm3 = t3, t2, t1   # MATLAB: üstten alta
    Ep  = E_P  / (1.0 - NU_P**2)
    Eau = E_AU / (1.0 - NU_AU**2)
    c1, c2, c3 = Ep * tm1, Eau * tm2, Ep * tm3
    ht = tm1 + tm2 + tm3
    d  = abs(-(c2 * tm1 + c3 * (tm1 + tm2)) / (2.0 * (c1 + c2 + c3)))
    b1 = ht / 2.0 - d
    b2 = tm3 - b1
    b4 = ht / 2.0 + d
    b3 = b4 - tm1
    k1 = (Ep  * (b2**3 - b1**3) +
          Eau * (b3**3 - b2**3) +
          Ep  * (b4**3 - b3**3))
    k_lin = 64.0 * np.pi * k1 / a**2

    t_arr  = np.array([tm1, tm2, tm3])
    E_arr  = np.array([E_P, E_AU, E_P])
    nu_arr = np.array([NU_P, NU_AU, NU_P])
    D_arr  = E_arr * t_arr**3 / (12.0 * (1.0 - nu_arr**2))
    D_t2   = D_arr / t_arr**2
    v_m    = np.dot(nu_arr, D_t2) / D_t2.sum()
    const  = 81.0 * np.pi * (-2109.0 * v_m**2 + 3210.0 * v_m + 5679.0) / (625.0 * a**2)
    k_cubic = const * D_t2.sum()
    return k_lin, k_cubic


def _solve_w0(P, t1, t2, t3, a):
    if P <= 0.0:
        return 0.0
    k_lin, k_cubic = _analytical_stiffness(t1, t2, t3, a)
    def eq(w):
        return (k_lin * w + k_cubic * w**3) / (np.pi * a**2) - P
    hi = max(a, 200e-6)
    for _ in range(60):
        if eq(hi) > 0.0:
            break
        hi *= 2.0
    try:
        w_avg = brentq(eq, 1e-18, hi, xtol=1e-15, maxiter=200)
    except Exception:
        return 0.0
    return 3.0 * w_avg


def capacitance_analytical(P, t1, t2, t3, a, ag):
    h_diel = (t1 + T_INS) / EPS_P_REL
    he     = ag + h_diel
    w0     = _solve_w0(P, t1, t2, t3, a)

    if w0 <= 0.0:
        return np.pi * EPS_0 * a**2 / he

    if w0 > ag:
        ratio   = np.clip(ag / w0, 0.0, 1.0)
        r_touch = a * np.sqrt(max(0.0, 1.0 - np.sqrt(ratio)))
        C_t = np.pi * r_touch**2 * EPS_0 / h_diel
        def ig(r):
            return 2.0 * np.pi * EPS_0 * r / (he - w0 * (1.0 - (r / a)**2)**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C_nt, _ = quad(ig, r_touch, a, limit=300)
        return C_t + C_nt
    else:
        def ig(r):
            return 2.0 * np.pi * EPS_0 * r / (he - w0 * (1.0 - (r / a)**2)**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C, _ = quad(ig, 0.0, a, limit=300)
        return C


# ══════════════════════════════════════════════════════════════════════════════
# PINN YARDIMCILARI
# ══════════════════════════════════════════════════════════════════════════════

def load_pinn(ckpt_path, device):
    model = DiaphragmPINN(n_hidden=5, hidden_dim=128).to(device).to(torch.float32)
    norm  = InputNormaliser()
    if not os.path.isfile(ckpt_path):
        print(f"  [!] Checkpoint bulunamadı: {ckpt_path}  → PINN atlanacak.")
        return None, norm
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  PINN yüklendi (adım {ckpt.get('step', '?')}): {ckpt_path}")
        return model, norm
    except Exception as e:
        print(f"  [!] PINN yüklenemedi: {e}")
        return None, norm


def pinn_capacitance(model, norm, P_arr, t1, t2, t3, a, ag, device, n_r=200):
    """C(P) dizisini PINN defleksiyon profili üzerinden hesapla."""
    if model is None:
        return None
    dtype = torch.float32
    r_arr = np.linspace(0.0, a, n_r + 1)[1:]   # r=0 hariç
    N_r, N_p = len(r_arr), len(P_arr)

    xi_b  = np.tile(r_arr / a, N_p)
    P_b   = np.repeat(P_arr, N_r)

    xi_t  = torch.tensor(xi_b, dtype=dtype, device=device)
    P_t   = torch.tensor(P_b,  dtype=dtype, device=device)
    t1_t  = torch.full_like(P_t, t1)
    t2_t  = torch.full_like(P_t, t2)
    t3_t  = torch.full_like(P_t, t3)
    a_t   = torch.full_like(P_t, a)
    ag_t  = torch.full_like(P_t, ag)

    x_in = norm.normalise(xi_t, P_t, t1_t, t2_t, t3_t, a_t, ag_t)
    with torch.no_grad():
        w, _ = model(x_in)
    w_np = w.squeeze(-1).cpu().numpy().reshape(N_p, N_r)

    h_diel = (t1 + T_INS) / EPS_P_REL
    he     = ag + h_diel

    C_out = np.empty(N_p)
    for i in range(N_p):
        gap = np.maximum(he + w_np[i], h_diel)
        C_out[i] = np.trapezoid(2.0 * np.pi * r_arr * EPS_0 / gap, r_arr)
    return C_out * 1e15   # fF


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════

def kapasitans_grafigi_2d(fig_no, r_um, P_arr, C_comsol_fF,
                          C_anal_fF, C_pinn_fF, is_nominal=False):
    """
    Tek yarıçap için C(P) karşılaştırma grafiği.

    fig_no       : çıktı dosya numarası (1..5)
    r_um         : yarıçap (µm)
    P_arr        : basınç dizisi (Pa)
    C_comsol_fF  : COMSOL kapasitansı (fF)
    C_anal_fF    : Analitik kapasitans (fF)
    C_pinn_fF    : PINN kapasitansı (fF veya None)
    is_nominal   : True → başlığa nominal sensör bilgisi ekle
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.plot(P_arr / 1e3, C_comsol_fF,
            CIZGI["comsol"], color=RENKLER["comsol"],
            lw=2.2, label=ETIKET["comsol"], zorder=5)

    ax.plot(P_arr / 1e3, C_anal_fF,
            CIZGI["analitik"], color=RENKLER["analitik"],
            lw=2.0, label=ETIKET["analitik"])

    if C_pinn_fF is not None:
        ax.plot(P_arr / 1e3, C_pinn_fF,
                CIZGI["pinn"], color=RENKLER["pinn"],
                lw=2.0, label=ETIKET["pinn"])

    a_m = r_um * 1e-6
    title_base = (
        f"$a$ = {r_um:.0f} µm  |  "
        f"$t_1$ = {T1_PINN*1e6:.0f} µm  |  "
        f"$t_2$ = {T2_PINN*1e6:.1f} µm  |  "
        f"$t_3$ = {T3_PINN*1e6:.0f} µm  |  "
        f"$ag$ = {AG*1e6:.0f} µm"
    )
    if is_nominal:
        title = (
            "Nominal Sensör: "
            "$t_1$=1 µm, $t_2$=0.2 µm, $t_3$=4 µm, $t_4$=1 µm\n"
            + title_base
        )
    else:
        title = title_base

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Basınç (kPa)", fontsize=11)
    ax.set_ylabel("Kapasitans (fF)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"kapasitans_2d_{fig_no}.pdf")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA PROGRAM
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kapasitans 2D veri seti grafikleri")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_ROOT, "checkpoints", "ckpt_final.pt"),
        help="PINN checkpoint dosyası"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")
    print(f"Excel dosyası: {EXCEL_PATH}\n")

    # ── Excel veri setini yükle ───────────────────────────────────────────────
    print("Excel dosyası okunuyor…")
    sections = parse_excel(EXCEL_PATH)

    # t_parylene2 = 4 µm bölümünü seç (nominal sensör kombinasyonu)
    target_t = 4.0
    if target_t not in sections:
        available = sorted(sections.keys())
        raise RuntimeError(
            f"t_parylene2={target_t} µm bölümü bulunamadı. "
            f"Mevcut: {available}"
        )
    sec = sections[target_t]
    P_arr = sec["P"]    # (N_p,)
    print(f"\nt_parylene2={target_t} µm bölümü seçildi.")
    print(f"Basınç aralığı: {P_arr.min():.0f} – {P_arr.max():.0f} Pa  "
          f"({len(P_arr)} nokta)\n")

    # ── PINN yükle ────────────────────────────────────────────────────────────
    model, norm = load_pinn(args.checkpoint, device)

    # ── Her yarıçap için figür üret ───────────────────────────────────────────
    print("=" * 55)
    print("  KAPASİTANS GRAFİKLERİ (2D COMSOL veri seti)")
    print("=" * 55)

    for fig_no, r_um in enumerate(RADII_UM, 1):
        a_m = r_um * 1e-6
        is_nominal = (r_um == NOMINAL_A)

        print(f"\n  Figür {fig_no}: a = {r_um:.0f} µm"
              + ("  [Nominal Sensör]" if is_nominal else ""))

        if r_um not in sec["C"]:
            print(f"    [!] {r_um} µm sütunu bulunamadı, atlanıyor.")
            continue
        C_comsol = sec["C"][r_um]   # fF

        # ── Analitik kapasitans ───────────────────────────────────────────────
        print(f"    Analitik model hesaplanıyor ({len(P_arr)} nokta)…")
        C_anal = np.empty(len(P_arr))
        for i, P in enumerate(P_arr):
            try:
                C_anal[i] = capacitance_analytical(
                    P, T1_PINN, T2_PINN, T3_PINN, a_m, AG)
            except Exception as exc:
                print(f"      [!] P={P:.0f} Pa: {exc}")
                C_anal[i] = np.nan
        C_anal_fF = C_anal * 1e15

        # ── PINN kapasitansı ──────────────────────────────────────────────────
        C_pinn_fF = None
        if model is not None:
            print(f"    PINN kapasitansı hesaplanıyor…")
            C_pinn_fF = pinn_capacitance(
                model, norm, P_arr, T1_PINN, T2_PINN, T3_PINN, a_m, AG, device)

        # ── Grafik ───────────────────────────────────────────────────────────
        kapasitans_grafigi_2d(
            fig_no, r_um, P_arr,
            C_comsol, C_anal_fF, C_pinn_fF,
            is_nominal=is_nominal,
        )

    print("\nTüm grafikler tamamlandı.")


if __name__ == "__main__":
    main()
