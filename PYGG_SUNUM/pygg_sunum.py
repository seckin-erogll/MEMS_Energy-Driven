"""
PYGG_SUNUM/pygg_sunum.py — Üç model karşılaştırması

Defleksiyon profili ve kapasitans eğrilerinde:
  - Analitik Model  : MATLAB yay modelinin Python uyarlaması
                      (Föppl-von Kármán + CLT yay katsayıları)
  - PINN Modeli     : Eğitilmiş fizik-bilgili sinir ağı
  - Numerik Model   : COMSOL FEM simülasyonu

Fiziksel yapı (alttan üste):
  Altın alt elektrot
  ── 1 µm parilen (yalıtkan, sabit, T_INS)
  ── ag µm hava aralığı (değişken)
  ── t1 µm parilen (diyafram alt katmanı)
  ── t2 µm altın (üst elektrot)
  ── t3 µm parilen (diyafram üst katmanı)

Kapasitans boşluğu:  h_eff = ag + (t1 + T_INS) / ε_p
Boşluk(r) = h_eff + w(r)        [w < 0 : PINN/COMSOL konv.]

Kullanım:
    python PYGG_SUNUM/pygg_sunum.py
    python PYGG_SUNUM/pygg_sunum.py --checkpoint checkpoints/ckpt_final.pt
"""

import argparse, os, sys, re, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import quad

# ── Üst dizinden PINN modüllerini yükle ──────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _ROOT)

import torch
from model import DiaphragmPINN, InputNormaliser

# ── Çıktı klasörü ─────────────────────────────────────────────────────────────
OUT_DIR    = _SCRIPT_DIR
COMSOL_DIR = os.path.join(_ROOT, "COMSOL_Deflection_Outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Malzeme sabitleri ─────────────────────────────────────────────────────────
E_P       = 3.2e9       # Pa   Parilen-C elastisite modülü
E_AU      = 70e9        # Pa   Altın elastisite modülü
NU_P      = 0.33        # Parilen-C Poisson oranı
NU_AU     = 0.44        # Altın Poisson oranı
EPS_0     = 8.854e-12   # F/m  Vakum geçirgenliği
EPS_P_REL = 3.15        # -    Parilen-C bağıl geçirgenlik
T_INS     = 1.0e-6      # m    Alt elektrot üzeri sabit yalıtkan parilen (1 µm)

# ── Grafik renkleri ve etiketleri ─────────────────────────────────────────────
RENKLER = {
    "comsol":   "#000000",   # siyah
    "analitik": "#1565C0",   # koyu mavi
    "pinn":     "#C62828",   # koyu kırmızı
}
ETIKET = {
    "comsol":   "Numerik Model (COMSOL)",
    "analitik": "Analitik Model",
    "pinn":     "PINN Modeli",
}
CIZGI = {
    "comsol":   "-",
    "analitik": "--",
    "pinn":     ":",
}


# ══════════════════════════════════════════════════════════════════════════════
# DOSYA YÖNETİMİ
# ══════════════════════════════════════════════════════════════════════════════

def parse_geo(fpath):
    """Dosya adından geometri parametrelerini (SI biriminde) çıkar."""
    name = os.path.basename(fpath).replace(".csv", "")
    def _v(k):
        m = re.search(rf"_{k}([\d.]+)", name)
        if m is None:
            raise ValueError(f"'{k}' bulunamadı: {name}")
        return float(m.group(1)) * 1e-6   # µm → m
    return dict(a=_v("rad"), t1=_v("t1"), t2=_v("t2"), t3=_v("t3"), ag=_v("ag"))


def load_sample_files(n=5):
    """Çeşitlilik için COMSOL dosyalarından eşit aralıklı n adet seç."""
    files = sorted(glob.glob(os.path.join(COMSOL_DIR, "*.csv")))
    if len(files) < n:
        raise RuntimeError(f"Yeterli COMSOL dosyası yok ({len(files)} < {n})")
    idx = np.round(np.linspace(0, len(files) - 1, n)).astype(int)
    return [files[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# ANALİTİK MODEL  (MATLAB Pressure_Sensor_Analytical_Model.m → Python)
# ══════════════════════════════════════════════════════════════════════════════

def _analytical_stiffness(t1, t2, t3, a):
    """
    MATLAB CLT yay katsayılarını hesapla.

    PINN katman sırası (alttan üste): t1=alt parilen, t2=altın, t3=üst parilen
    MATLAB katman sırası (üstten alta): tm1=üst parilen, tm2=altın, tm3=alt parilen

    Dönüş: (k_lin [N/m], k_cubic [N/m³])
    k_lin = 64π·k1 / a²  ←→  192π·D* / a²  (kırışma plağı, w_ort = w0/3)
    """
    # MATLAB katman eşleşmesi
    tm1, tm2, tm3 = t3, t2, t1   # üst parilen, altın, alt parilen

    # Redüklü modüller  Ē = E / (1−ν²)
    Ep  = E_P  / (1.0 - NU_P**2)
    Eau = E_AU / (1.0 - NU_AU**2)

    # Eksenel sertlik bileşenleri
    c1 = Ep  * tm1
    c2 = Eau * tm2
    c3 = Ep  * tm3
    ht = tm1 + tm2 + tm3   # toplam kalınlık

    # Geometrik orta düzlemden nötr eksen ofseti (MATLAB eq.)
    d = abs(-(c2 * tm1 + c3 * (tm1 + tm2)) / (2.0 * (c1 + c2 + c3)))

    # b değerleri (MATLAB konvansiyonu, Atik 2020 ek denklemler)
    b1 = ht / 2.0 - d
    b2 = tm3 - b1            # = tm3 − (ht/2 − d)
    b4 = ht / 2.0 + d
    b3 = b4 - tm1            # = (ht/2 + d) − tm1

    # Eğilme sertliği bileşenleri  k1 = Σ Ēk (bk_üst³ − bk_alt³)  ≈ 3·D*
    k1 = (Ep  * (b2**3 - b1**3) +
          Eau * (b3**3 - b2**3) +
          Ep  * (b4**3 - b3**3))

    k_lin = 64.0 * np.pi * k1 / a**2

    # Kübik yay katsayısı (von Kármán membran gerilimi)
    t_arr  = np.array([tm1, tm2, tm3])
    E_arr  = np.array([E_P, E_AU, E_P])
    nu_arr = np.array([NU_P, NU_AU, NU_P])
    D_arr  = E_arr * t_arr**3 / (12.0 * (1.0 - nu_arr**2))

    D_t2   = D_arr / t_arr**2                    # D_k / t_k²
    v_m    = np.dot(nu_arr, D_t2) / D_t2.sum()  # ağırlıklı ortalama Poisson oranı
    const  = 81.0 * np.pi * (-2109.0 * v_m**2 + 3210.0 * v_m + 5679.0) / (625.0 * a**2)
    k_cubic = const * D_t2.sum()

    return k_lin, k_cubic


def _solve_w_avg(P, k_lin, k_cubic, a):
    """Basınç denklemini çöz: (k_lin·w + k_cubic·w³) / (π·a²) = P."""
    if P <= 0.0:
        return 0.0
    def eq(w):
        return (k_lin * w + k_cubic * w**3) / (np.pi * a**2) - P
    lo = 1e-18
    hi = max(a, 200e-6)          # geniş başlangıç aralığı
    for _ in range(60):          # hi'yi büyüt ta ki f(hi) > 0
        if eq(hi) > 0.0:
            break
        hi *= 2.0
    try:
        return brentq(eq, lo, hi, xtol=1e-15, maxiter=200)
    except Exception:
        return 0.0


def analytical_w0(P, t1, t2, t3, a):
    """Analitik merkez sehimi w0 [m], MATLAB konvansiyonu (w0 > 0 aşağıya)."""
    k_lin, k_cubic = _analytical_stiffness(t1, t2, t3, a)
    w_avg = _solve_w_avg(P, k_lin, k_cubic, a)
    return 3.0 * w_avg   # w0 = 3·w_ort


def analytical_profile(P, t1, t2, t3, a, r_arr, ag):
    """
    w(r) defleksiyon profili [m], PINN/COMSOL konvansiyonu (aşağıya < 0).
    Temas sınırında fiziksel olarak −ag'da kırpılır.
    """
    w0 = analytical_w0(P, t1, t2, t3, a)
    w  = -w0 * (1.0 - (r_arr / a)**2)**2   # PINN konvansiyonuna çevir
    return np.maximum(w, -ag)


# ══════════════════════════════════════════════════════════════════════════════
# KAPASİTANS HESABIx
# ══════════════════════════════════════════════════════════════════════════════

def _h_eff(ag, t1):
    """
    Efektif kapasitans boşluğu [m].

    h_eff = ag  +  (t1 + T_INS) / ε_p
    t1   : diyafram alt parilen katmanı [m]
    T_INS: alt elektrot üzeri sabit yalıtkan (1 µm)
    """
    return ag + (t1 + T_INS) / EPS_P_REL


def capacitance_numerical(r, w_pinn, ag, t1):
    """
    Sayısal kapasitans (PINN veya COMSOL defleksiyon profili için).

    w_pinn : w(r) < 0  (aşağıya, PINN/COMSOL konvansiyonu)
    Boşluk : gap(r) = h_eff + w_pinn(r)   [azalır]
    Zemin  : h_diel = (t1 + T_INS) / ε_p   (tam temas durumu)
    C = ∫ 2π r ε₀ / gap(r) dr
    """
    h_diel = (t1 + T_INS) / EPS_P_REL
    he     = _h_eff(ag, t1)
    gap    = np.maximum(he + w_pinn, h_diel)   # temas zemini
    return np.trapezoid(2.0 * np.pi * r * EPS_0 / gap, r)


def capacitance_analytical(P, t1, t2, t3, a, ag):
    """
    Analitik kapasitans (MATLAB profili + doğru h_eff formülü).

    Temas modu: xtm = a·√(1 − √(ag/w0))
      C_temas   = π·xtm²·ε₀ / h_diel
      C_temas_dışı = ∫_{xtm}^{a} 2π r ε₀ / (h_eff − w0·(1−r²/a²)²) dr
    """
    h_diel = (t1 + T_INS) / EPS_P_REL
    he     = _h_eff(ag, t1)
    w0     = analytical_w0(P, t1, t2, t3, a)

    if w0 <= 0.0:
        return np.pi * EPS_0 * a**2 / he   # düz plaka (P=0)

    if w0 > ag:
        # Temas yarıçapı: w0·(1−xtm²/a²)² = ag  →  1−xtm²/a² = √(ag/w0)
        ratio   = np.clip(ag / w0, 0.0, 1.0)
        r_touch = a * np.sqrt(max(0.0, 1.0 - np.sqrt(ratio)))
        # Temas bölgesi kapasitansı (sabit boşluk = h_diel)
        C_t = np.pi * r_touch**2 * EPS_0 / h_diel
        # Temas dışı halka
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
    """PINN modelini ve normalizer'ı yükle; başarısız olursa None döndür."""
    model = DiaphragmPINN(n_hidden=5, hidden_dim=128).to(device).to(torch.float32)
    norm  = InputNormaliser()
    if not os.path.isfile(ckpt_path):
        print(f"  [!] Checkpoint bulunamadı: {ckpt_path}")
        print(      "      PINN eğrileri grafiklerden atlanacak.")
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


def pinn_batch(model, norm, r_arr, P_arr, geo, device):
    """
    Tüm (P, r) kombinasyonlarını tek ileri geçişte hesapla.

    Dönüş: w_np, şekil (len(P_arr), len(r_arr))  [m, PINN konv.]
    """
    if model is None:
        return None
    a, t1, t2, t3, ag = geo["a"], geo["t1"], geo["t2"], geo["t3"], geo["ag"]
    N_r, N_p = len(r_arr), len(P_arr)
    dtype = torch.float32

    xi_b = np.tile(r_arr / a, N_p)
    P_b  = np.repeat(P_arr, N_r)

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
    return w.squeeze(-1).cpu().numpy().reshape(N_p, N_r)


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def _geo_title(geo, fig_no):
    return (
        f"Geometri {fig_no} — "
        f"$a$ = {geo['a']*1e6:.0f} µm | "
        f"$t_1$ = {geo['t1']*1e6:.2f} µm | "
        f"$t_2$ = {geo['t2']*1e6:.3f} µm | "
        f"$t_3$ = {geo['t3']*1e6:.2f} µm | "
        f"$ag$ = {geo['ag']*1e6:.2f} µm"
    )


def defleksiyon_grafigi(fig_no, fpath, pressures_Pa, model, norm, device):
    """
    Üç model defleksiyon profil karşılaştırması.
    Her sütun bir basınç noktasına karşılık gelir.
    """
    geo  = parse_geo(fpath)
    df   = pd.read_csv(fpath)
    r_um = df["r_um"].values
    r_m  = r_um * 1e-6

    # Seçilen basınçlarda PINN'i toplu hesapla
    w_pinn_batch = pinn_batch(model, norm, r_m, np.array(pressures_Pa, dtype=float), geo, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    fig.suptitle(_geo_title(geo, fig_no), fontsize=9.5, fontweight="bold")

    for k, (ax, P) in enumerate(zip(axes, pressures_Pa)):
        col = f"w_{int(P)}Pa_um"
        if col not in df.columns:
            ax.set_visible(False)
            continue

        # COMSOL verisi
        w_cos = df[col].values * 1e-6   # m, negatif aşağıya
        ax.plot(r_um, w_cos * 1e6,
                CIZGI["comsol"], color=RENKLER["comsol"],
                lw=2.2, label=ETIKET["comsol"], zorder=5)

        # Analitik model
        w_an = analytical_profile(P, geo["t1"], geo["t2"], geo["t3"], geo["a"], r_m, geo["ag"])
        ax.plot(r_um, w_an * 1e6,
                CIZGI["analitik"], color=RENKLER["analitik"],
                lw=2.0, label=ETIKET["analitik"])

        # PINN modeli
        if w_pinn_batch is not None:
            w_p = w_pinn_batch[k]
            ax.plot(r_um, w_p * 1e6,
                    CIZGI["pinn"], color=RENKLER["pinn"],
                    lw=2.0, label=ETIKET["pinn"])

        # Hava aralığı sınırı
        ax.axhline(-geo["ag"] * 1e6, color="gray", ls=":", lw=1.0,
                   label=f"Hava aralığı (−{geo['ag']*1e6:.1f} µm)")

        ax.set_title(f"Basınç = {P/1e3:.1f} kPa", fontsize=9.5)
        ax.set_xlabel("Yarıçap $r$ (µm)", fontsize=9)
        ax.set_ylabel("Sehim $w$ (µm)", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, ls="--", alpha=0.4)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, f"defleksiyon_{fig_no}.pdf")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {out}")


def kapasitans_grafigi(fig_no, fpath, model, norm, device):
    """
    C(P) karşılaştırma grafiği — üç model.
    """
    geo  = parse_geo(fpath)
    df   = pd.read_csv(fpath)
    r_um = df["r_um"].values
    r_m  = r_um * 1e-6

    # Basınç dizisi (COMSOL kolonlarından çıkar)
    p_cols = []
    for c in df.columns[1:]:
        m = re.match(r"w_(\d+(?:\.\d+)?)Pa_um", c)
        if m:
            p_cols.append((c, float(m.group(1))))
    P_arr = np.array([p for _, p in p_cols])

    print(f"    Kapasitans hesaplanıyor ({len(P_arr)} basınç noktası)…")

    # ── COMSOL kapasitansı ────────────────────────────────────────────────────
    C_cos = np.array([
        capacitance_numerical(r_m, df[c].values * 1e-6, geo["ag"], geo["t1"])
        for c, _ in p_cols
    ]) * 1e15   # fF

    # ── Analitik kapasitans ───────────────────────────────────────────────────
    C_anal = np.zeros(len(P_arr))
    for i, P in enumerate(P_arr):
        try:
            C_anal[i] = capacitance_analytical(P, geo["t1"], geo["t2"],
                                               geo["t3"], geo["a"], geo["ag"])
        except Exception as exc:
            print(f"      [!] Analitik C hatası P={P:.0f} Pa: {exc}")
            C_anal[i] = np.nan
    C_anal *= 1e15  # fF

    # ── PINN kapasitansı ──────────────────────────────────────────────────────
    C_pinn = None
    if model is not None:
        w_all = pinn_batch(model, norm, r_m, P_arr, geo, device)  # (N_p, N_r)
        C_pinn = np.array([
            capacitance_numerical(r_m, w_all[i], geo["ag"], geo["t1"])
            for i in range(len(P_arr))
        ]) * 1e15

    # ── Grafik ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.plot(P_arr / 1e3, C_cos,
            CIZGI["comsol"], color=RENKLER["comsol"],
            lw=2.2, label=ETIKET["comsol"], zorder=5)

    ax.plot(P_arr / 1e3, C_anal,
            CIZGI["analitik"], color=RENKLER["analitik"],
            lw=2.0, label=ETIKET["analitik"])

    if C_pinn is not None:
        ax.plot(P_arr / 1e3, C_pinn,
                CIZGI["pinn"], color=RENKLER["pinn"],
                lw=2.0, label=ETIKET["pinn"])

    ax.set_xlabel("Basınç (kPa)", fontsize=11)
    ax.set_ylabel("Kapasitans (fF)", fontsize=11)
    ax.set_title(_geo_title(geo, fig_no), fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"kapasitans_{fig_no}.pdf")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV DIŞA AKTARMA
# ══════════════════════════════════════════════════════════════════════════════

def export_deflection_csv(fig_no, fpath, pressures_Pa, model, norm, device):
    """
    Defleksiyon profil verilerini CSV olarak kaydeder.

    Çıktı: defleksiyon_<fig_no>.csv
    Kolonlar: r_um, w_comsol_<P>Pa_um, w_analitik_<P>Pa_um, w_pinn_<P>Pa_um, ...
    """
    geo  = parse_geo(fpath)
    df   = pd.read_csv(fpath)
    r_um = df["r_um"].values
    r_m  = r_um * 1e-6

    P_arr = np.array(pressures_Pa, dtype=float)
    w_pinn_batch = pinn_batch(model, norm, r_m, P_arr, geo, device)

    out_df = pd.DataFrame({"r_um": r_um})

    for k, P in enumerate(pressures_Pa):
        P_int = int(P)
        col   = f"w_{P_int}Pa_um"

        if col in df.columns:
            out_df[f"w_comsol_{P_int}Pa_um"] = df[col].values
        else:
            out_df[f"w_comsol_{P_int}Pa_um"] = np.nan

        w_an = analytical_profile(P, geo["t1"], geo["t2"], geo["t3"], geo["a"], r_m, geo["ag"])
        out_df[f"w_analitik_{P_int}Pa_um"] = w_an * 1e6

        if w_pinn_batch is not None:
            out_df[f"w_pinn_{P_int}Pa_um"] = w_pinn_batch[k] * 1e6
        else:
            out_df[f"w_pinn_{P_int}Pa_um"] = np.nan

    out_path = os.path.join(OUT_DIR, f"defleksiyon_{fig_no}.csv")
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"    ✓ {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA PROGRAM
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PYGG Sunum grafikleri")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_ROOT, "checkpoints", "ckpt_final.pt"),
        help="PINN checkpoint dosyası"
    )
    parser.add_argument(
        "--pressures", nargs=3, type=float,
        default=[2000.0, 8000.0, 16000.0],
        metavar=("P1", "P2", "P3"),
        help="Defleksiyon grafikleri için 3 basınç değeri (Pa)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")
    print(f"Çıktı klasörü: {OUT_DIR}\n")

    # PINN yükle
    model, norm = load_pinn(args.checkpoint, device)

    # 5 geometri örneği seç
    sample_files = load_sample_files(5)

    pressures = [float(p) for p in args.pressures]
    print(f"Basınç noktaları: {[p/1e3 for p in pressures]} kPa\n")

    # ── 5 Defleksiyon Grafiği + CSV ───────────────────────────────────────────
    print("=" * 55)
    print("  DEFLEKSİYON GRAFİKLERİ + CSV")
    print("=" * 55)
    for i, fpath in enumerate(sample_files, 1):
        print(f"\n  Geometri {i}: {os.path.basename(fpath)}")
        defleksiyon_grafigi(i, fpath, pressures, model, norm, device)
        export_deflection_csv(i, fpath, pressures, model, norm, device)

    # ── 5 Kapasitans Grafiği ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  KAPASİTANS GRAFİKLERİ")
    print("=" * 55)
    for i, fpath in enumerate(sample_files, 1):
        print(f"\n  Geometri {i}: {os.path.basename(fpath)}")
        kapasitans_grafigi(i, fpath, model, norm, device)

    print("\nTüm grafikler tamamlandı.")


if __name__ == "__main__":
    main()
