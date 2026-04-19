"""
build_pdf.py — Generate the MEMS PINN paper as a PDF using ReportLab.
Sensors and Actuators A: Physical (Elsevier) style.
Equations are rendered via matplotlib's mathtext engine.
"""

import os, io, copy, tempfile
import numpy as np
from PIL import Image as PILImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── fonts ──────────────────────────────────────────────────────────────────────
FONT_DIR = '/usr/share/fonts/google-carlito-fonts'
pdfmetrics.registerFont(TTFont('Carlito',           f'{FONT_DIR}/Carlito-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Carlito-Bold',      f'{FONT_DIR}/Carlito-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Carlito-Italic',    f'{FONT_DIR}/Carlito-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Carlito-BoldItalic',f'{FONT_DIR}/Carlito-BoldItalic.ttf'))
pdfmetrics.registerFontFamily('Carlito',
    normal='Carlito', bold='Carlito-Bold',
    italic='Carlito-Italic', boldItalic='Carlito-BoldItalic')

SERIF_DIR = '/usr/share/fonts/google-crosextra-caladea-fonts'
pdfmetrics.registerFont(TTFont('Caladea',           f'{SERIF_DIR}/Caladea-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Caladea-Bold',      f'{SERIF_DIR}/Caladea-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Caladea-Italic',    f'{SERIF_DIR}/Caladea-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Caladea-BoldItalic',f'{SERIF_DIR}/Caladea-BoldItalic.ttf'))
pdfmetrics.registerFontFamily('Caladea',
    normal='Caladea', bold='Caladea-Bold',
    italic='Caladea-Italic', boldItalic='Caladea-BoldItalic')

# ── page geometry ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
LEFT_M = RIGHT_M = 1.8*cm
TOP_M  = 2.2*cm
BOT_M  = 2.0*cm
BODY_W = PAGE_W - LEFT_M - RIGHT_M
COL_W  = (BODY_W - 0.5*cm) / 2

# ── colours ────────────────────────────────────────────────────────────────────
ELSEVIER_BLUE = colors.HexColor('#1B3A6B')
LIGHT_GRAY    = colors.HexColor('#F5F5F5')
MED_GRAY      = colors.HexColor('#888888')

# ── styles ─────────────────────────────────────────────────────────────────────
def S(name, font='Carlito', size=9, leading=13, color=colors.black,
      bold=False, italic=False, align=TA_JUSTIFY,
      space_before=0, space_after=4):
    fn = font + ('-Bold' if bold and not italic else
                 '-Italic' if italic and not bold else
                 '-BoldItalic' if bold and italic else '')
    return ParagraphStyle(name,
        fontName=fn, fontSize=size, leading=leading,
        textColor=color, alignment=align,
        spaceBefore=space_before*mm, spaceAfter=space_after*mm)

TITLE_S    = S('Title',   font='Caladea', size=17, leading=21,
               color=ELSEVIER_BLUE, bold=True, align=TA_LEFT,
               space_before=0, space_after=2)
AUTHOR_S   = S('Author',  font='Carlito', size=10.5, leading=14,
               align=TA_LEFT, space_after=1)
AFF_S      = S('Aff',     font='Carlito', size=8.5, leading=11,
               color=MED_GRAY, italic=True, align=TA_LEFT, space_after=1)
JOURNAL_S  = S('Journal', font='Carlito', size=9, leading=11,
               color=ELSEVIER_BLUE, bold=True, align=TA_LEFT, space_after=3)
ABS_HEAD_S = S('AbsHead', font='Carlito', size=9, leading=11,
               color=ELSEVIER_BLUE, bold=True, align=TA_LEFT, space_after=1)
ABS_S      = S('Abs',     font='Carlito', size=8.5, leading=12, space_after=3)
KW_S       = S('KW',      font='Carlito', size=8.5, leading=11,
               italic=True, align=TA_LEFT, space_after=5)
SEC_S      = S('Sec',     font='Caladea', size=10.5, leading=13,
               color=ELSEVIER_BLUE, bold=True, align=TA_LEFT,
               space_before=4, space_after=2)
SUBSEC_S   = S('Subsec',  font='Carlito', size=10, leading=13,
               bold=True, align=TA_LEFT, space_before=3, space_after=1)
PARA_S     = S('Para',    font='Carlito', size=9, leading=13,
               bold=True, italic=True, align=TA_LEFT,
               space_before=2, space_after=0)
BODY_S     = S('Body',    font='Carlito', size=9, leading=13)
CAP_S      = S('Cap',     font='Carlito', size=8, leading=10,
               italic=True, space_after=5)
TCAP_S     = S('TCap',    font='Carlito', size=8.5, leading=11,
               bold=True, align=TA_LEFT, space_after=2)
REF_S      = S('Ref',     font='Carlito', size=8, leading=11, space_after=2)
SMALL_S    = S('Small',   font='Carlito', size=8, leading=10,
               color=MED_GRAY, space_after=1)

# ── equation rendering ─────────────────────────────────────────────────────────
_EQ_CACHE = {}

def _preprocess_latex(s):
    """Replace matplotlib-mathtext-unsupported commands with supported equivalents."""
    import re
    # \tfrac -> \frac  (text-size fraction not supported)
    s = s.replace(r'\tfrac', r'\frac')
    # Big/bigl/bigr delimiter modifiers -> \left/\right
    for big in (r'\Bigl', r'\Bigr', r'\biggl', r'\biggr', r'\bigl', r'\bigr', r'\Big', r'\big'):
        s = s.replace(big, '')
    # negative spacing \! and thin space \, -> empty (ignored safely)
    s = s.replace(r'\!', '')
    # \; -> space, \: -> space, \, -> space (thin spaces)
    # keep \, as it IS supported in mpl mathtext as thin space
    return s

def render_eq(latex_str, fontsize=11, label='', width_pt=None):
    """
    Render a LaTeX equation via matplotlib mathtext.
    Returns a list of flowables: [Image, optional label paragraph].
    width_pt: target width in ReportLab points (default: 60% of BODY_W).
    """
    latex_str = _preprocess_latex(latex_str)
    cache_key = (latex_str, fontsize)
    if cache_key not in _EQ_CACHE:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f'${latex_str}$',
                 fontsize=fontsize, family='serif',
                 ha='left', va='bottom')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=220,
                    bbox_inches='tight', pad_inches=0.04,
                    transparent=True)
        plt.close(fig)
        buf.seek(0)
        _EQ_CACHE[cache_key] = buf.getvalue()

    img_bytes = _EQ_CACHE[cache_key]
    pil = PILImage.open(io.BytesIO(img_bytes))
    iw, ih = pil.size
    if width_pt is None:
        width_pt = BODY_W * 0.58
    height_pt = width_pt * ih / iw

    # don't stretch very short equations
    if height_pt < 14:
        height_pt = 14
    if width_pt / iw * ih < 14:
        width_pt = 14 * iw / ih

    img_fl = Image(io.BytesIO(img_bytes), width=width_pt, height=height_pt)

    out = []
    if label:
        lbl_p = Paragraph(f'({label})', S('EqLbl', font='Carlito', size=8.5,
                          color=MED_GRAY, align=TA_LEFT, space_after=0))
        tbl = Table([[img_fl, lbl_p]],
                    colWidths=[BODY_W - 1.5*cm, 1.5*cm])
        tbl.setStyle(TableStyle([
            ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN',   (1,0), (1,0),   'RIGHT'),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        out.append(tbl)
    else:
        # center the image
        tbl = Table([[img_fl]], colWidths=[BODY_W])
        tbl.setStyle(TableStyle([
            ('ALIGN', (0,0), (0,0), 'CENTER'),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        out.append(tbl)
    out.append(Spacer(1, 2*mm))
    return out

# ── helpers ────────────────────────────────────────────────────────────────────
def p(text, style=None):
    return Paragraph(text, style or BODY_S)

def sp(h=3):
    return Spacer(1, h*mm)

def hr(color=ELSEVIER_BLUE, width=1):
    return HRFlowable(width='100%', thickness=width, color=color, spaceAfter=2*mm)

def sec(num, title):
    return [p(f'{num}.&nbsp;&nbsp;{title}', SEC_S), sp(0.5)]

def subsec(num, title):
    return [p(f'<b>{num}&nbsp;&nbsp;{title}</b>', SUBSEC_S), sp(0)]

def para(title):
    return p(f'<i><b>{title}</b></i>', PARA_S)

def bullets(items):
    out = []
    for item in items:
        out.append(p(f'&#8226;&nbsp;&nbsp;{item}', BODY_S))
    return out

def fig_fl(path, width, caption_text, label=''):
    out = []
    if os.path.exists(path):
        try:
            pim = PILImage.open(path)
            iw, ih = pim.size
            h = width * ih / iw
            out.append(Image(path, width=width, height=h))
        except Exception:
            out.append(p(f'[Figure: {os.path.basename(path)}]', CAP_S))
    else:
        out.append(p(f'[Figure: {os.path.basename(path)}]', CAP_S))
    lbl = f'<b>Fig. {label}.</b>&nbsp;' if label else ''
    out.append(p(lbl + caption_text, CAP_S))
    return out

def mk_table(data, col_widths, caption, label=''):
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), ELSEVIER_BLUE),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Carlito-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8.5),
        ('FONTNAME',      (0,1), (-1,-1), 'Carlito'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [LIGHT_GRAY, colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.3, colors.lightgrey),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ('TOPPADDING',    (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN',         (1,0), (1,-1), 'CENTER'),
    ]))
    lbl_str = f'<b>Table {label}.</b>&nbsp;' if label else ''
    return [p(lbl_str + caption, TCAP_S), tbl, sp(4)]


# ══════════════════════════════════════════════════════════════════════════════
# Story
# ══════════════════════════════════════════════════════════════════════════════
story = []

# ── journal header ─────────────────────────────────────────────────────────────
story.append(p('Sensors and Actuators A: Physical', JOURNAL_S))
story.append(hr(ELSEVIER_BLUE, 2))
story.append(sp(2))

# ── title ──────────────────────────────────────────────────────────────────────
story.append(p(
    'A Deep Ritz Physics-Informed Neural Network for Large-Deflection '
    'Modelling of Multilayer Parylene-C/Au/Parylene-C MEMS Diaphragm '
    'Pressure Sensors',
    TITLE_S))
story.append(sp(2))

# ── authors ────────────────────────────────────────────────────────────────────
story.append(p(
    'Seckin Eroglu<super>a</super>, '
    'Ender Yildirim<super>a,\u2217</super>',
    AUTHOR_S))
story.append(p(
    '<super>a</super>\u2009Department of Electrical and Electronics '
    'Engineering, Middle East Technical University, 06800 Ankara, Turkey',
    AFF_S))
story.append(p(
    '\u2217\u2009Corresponding author. E-mail: eyildirim@metu.edu.tr',
    AFF_S))
story.append(sp(2))
story.append(hr(MED_GRAY, 0.5))
story.append(sp(2))

# ── abstract ───────────────────────────────────────────────────────────────────
story.append(p('Abstract', ABS_HEAD_S))
story.append(p(
    'Accurate and fast modelling of large-deflection behaviour in MEMS '
    'capacitive pressure sensors is a prerequisite for design optimisation '
    'and real-time performance prediction. Classical Kirchhoff plate theory '
    '(Atik analytical model) systematically overestimates deflection at '
    'operating pressures because it neglects membrane stiffening; full '
    'finite-element analysis (FEA) is accurate but prohibitively slow for '
    'high-throughput design exploration. We present <i>MEMSRitz</i>, an '
    'energy-driven physics-informed neural network (PINN) based on the Deep '
    'Ritz method that learns the equilibrium deflection and radial in-plane '
    'displacement of a clamped circular multilayer parylene-C / gold / '
    'parylene-C diaphragm over a five-dimensional geometry space '
    '(a\u2208[150,500]\u202f\u03bcm, a<sub>g</sub>\u2208[5,15]\u202f\u03bcm, '
    't<sub>1</sub>,t<sub>3</sub>\u2208[1,5]\u202f\u03bcm, '
    't<sub>2</sub>\u2208[0.2,0.5]\u202f\u03bcm) and a three-decade pressure '
    'range (P\u2208[10, 20\u2009000]\u202fPa). The network is trained by '
    'minimising the F\u00f6ppl\u2013von K\u00e1rm\u00e1n (FvK) variational '
    'energy functional\u2014no labelled FEA data are required in the main '
    'training phase. Classical Laminated Plate Theory (CLT) provides the '
    'bending stiffness D* and membrane stiffnesses A<sub>11</sub>, A<sub>12</sub> '
    'analytically for each geometry. Hard clamped boundary conditions are '
    'enforced algebraically via the shape factor (1\u2212\u03be\u00b2)\u00b2, '
    'and electrode contact (w\u2265\u2212a<sub>g</sub>) is guaranteed by a '
    'smooth-max operator in the forward pass. Validated against 2144 COMSOL '
    'FEA profiles, the model achieves a median RMSE of 0.15\u202f\u03bcm '
    'and a maximum of 1.10\u202f\u03bcm, with zero contact-penetration '
    'violations, while reducing inference time from minutes (FEA) to '
    'sub-millisecond. The Kirchhoff analytical model is shown to overestimate '
    'centre deflection by 20\u201350% above 2\u202fkPa, confirming the '
    'necessity of the full FvK treatment captured by the network.',
    ABS_S))
story.append(p(
    '<i><b>Keywords:</b> MEMS pressure sensor; parylene-C diaphragm; '
    'physics-informed neural network; Deep Ritz method; '
    'F\u00f6ppl\u2013von K\u00e1rm\u00e1n; Classical Laminated Plate Theory; '
    'touch-mode capacitive sensor; surrogate model</i>',
    KW_S))
story.append(hr(MED_GRAY, 0.5))
story.append(sp(3))

# ══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
story += sec('1', 'Introduction')

story.append(p(
    'Capacitive MEMS pressure sensors based on thin circular diaphragms have '
    'become the workhorse transducer architecture for biomedical implants, '
    'environmental monitoring, and industrial process control [1,2,3]. '
    'The device of interest is built around a three-layer stack of '
    'parylene-C\u202f/\u202fgold\u202f/\u202fparylene-C: the polymer layers '
    'form the flexible structural membrane while the gold interlayer provides '
    'the conducting electrode and contributes disproportionately to bending '
    'stiffness through an I-beam effect. At physiologically and industrially '
    'relevant pressures (tens of Pa to tens of kPa), the ratio of centre '
    'deflection to layer thickness routinely exceeds unity, placing the device '
    'in the geometrically nonlinear large-deflection regime. In this regime, '
    'the classical Kirchhoff (linear) plate solution\u2014which underlies the '
    'analytical model of Atik et al. [4]\u2014systematically overestimates '
    'deflection because it ignores in-plane membrane stiffening, a central '
    'prediction of F\u00f6ppl\u2013von K\u00e1rm\u00e1n (FvK) theory.'))

story.append(p(
    'Accurate large-deflection analysis via FEA resolves this nonlinearity '
    'but at the cost of minutes to hours per geometry\u2013pressure '
    'evaluation. Design space exploration and sensor calibration routinely '
    'require 10<super>3</super>\u201310<super>4</super> such evaluations, '
    'making full FEA intractable as an inner loop. Surrogate models trained '
    'on FEA data can reduce this cost but require large labelled datasets '
    'and do not generalise automatically beyond the training distribution '
    '[5,6].'))

story.append(p(
    'Physics-informed neural networks (PINNs) offer an alternative: the '
    'network is trained by minimising a physics residual derived from the '
    'governing equations [7,8,9]. In their variational form, PINNs minimise '
    'the total potential energy, recovering the governing PDEs as '
    'Euler\u2013Lagrange conditions\u2014the <i>Deep Ritz method</i> [10], '
    'extended to nonlinear mechanics [11,12] and to composite plate '
    'structures [13]. Despite this progress, no study has applied a Deep '
    'Ritz PINN to a multilayer MEMS capacitive diaphragm with electrode '
    'contact, composite CLT stiffness, and a geometry space spanning '
    'multiple orders of magnitude in dimensions and pressure.'))

story.append(p(
    'This paper presents <i>MEMSRitz</i>, addressing three specific '
    'challenges: (i) a five-dimensional geometry space; (ii) electrode '
    'contact w\u2265\u2212a<sub>g</sub>, a hard unilateral constraint '
    'that soft penalty approaches cannot enforce at large pressure ratios; '
    'and (iii) energy scales spanning four orders of magnitude across the '
    'operating pressure range, which bias naive energy minimisation toward '
    'high-pressure samples. The key contributions are:'))

story += bullets([
    'A Deep Ritz PINN with two output heads\u2014transverse deflection w(r) '
    'and in-plane displacement u(r)\u2014parameterised over the full '
    'five-dimensional geometry and three-decade pressure space.',
    'Algebraic enforcement of clamped boundary conditions via the shape '
    'factor (1\u2212\u03be\u00b2)\u00b2, eliminating gradient budget '
    'spent on boundary residuals.',
    'A smooth-max operator in the forward pass enforcing electrode contact '
    'to within 50\u202fnm with nonzero gradient everywhere, enabling '
    'learning in touch mode.',
    'A per-sample energy normalisation that equalises gradient contributions '
    'across the full pressure range.',
    'Validation against 2144 COMSOL profiles: median RMSE 0.15\u202f\u03bcm, '
    'max 1.10\u202f\u03bcm, zero contact violations, <1\u202fms inference.',
])
story.append(sp(2))

# ══════════════════════════════════════════════════════════════════════════════
# 2. PHYSICAL MODEL
# ══════════════════════════════════════════════════════════════════════════════
story += sec('2', 'Physical Model')
story += subsec('2.1', 'Device Geometry')

story.append(p(
    'The sensor (Fig.\u202f1) consists of a clamped circular diaphragm of '
    'radius a suspended above a fixed bottom electrode separated by an '
    'initial air gap a<sub>g</sub>. The diaphragm is a three-layer laminate: '
    'parylene-C (thickness t<sub>1</sub>, E\u202f=\u202f3.2\u202fGPa, '
    '\u03bd\u202f=\u202f0.33), gold (t<sub>2</sub>, 70\u202fGPa, 0.44), '
    'and parylene-C (t<sub>3</sub>). Uniform pressure P deflects the '
    'diaphragm downward (sign convention: w\u202f<\u202f0). When '
    'w(r)\u202f=\u202f\u2212a<sub>g</sub>, the diaphragm contacts the '
    'electrode (touch mode).'))

story += fig_fl('figures/schematic.png', BODY_W * 0.78,
    'Cross-sectional schematic of the three-layer parylene-C / gold / '
    'parylene-C capacitive diaphragm. Uniform pressure P deflects the '
    'diaphragm toward the fixed electrode (gap a<sub>g</sub>). '
    '\u03be\u202f=\u202fr/a is the normalised radial coordinate.', '1')

story += subsec('2.2', 'Classical Laminated Plate Theory (CLT)')

story.append(p(
    'Because the two constituent materials differ in stiffness by '
    '\u224822\u00d7, a composite plate formulation is required [14,15]. '
    'CLT provides three scalar stiffness coefficients from '
    '(t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>).'))

story.append(para('Neutral axis.'))
story.append(p(
    'The plane-stress reduced moduli are '
    '\u0112<sub>k</sub>\u202f=\u202fE<sub>k</sub>/(1\u2212\u03bd<sub>k</sub>\u00b2). '
    'The neutral axis position (measured from the stack bottom) is:'))
story += render_eq(
    r'z_{\rm NA} = \frac{\sum_k \bar{E}_k\, t_k\, z_{c,k}}{\sum_k \bar{E}_k\, t_k}',
    label='1')
story.append(p(
    'For the nominal geometry (t<sub>1</sub>\u202f=\u202f1, '
    't<sub>2</sub>\u202f=\u202f0.2, t<sub>3</sub>\u202f=\u202f4\u202f\u03bcm): '
    'z<sub>NA</sub>\u202f=\u202f1.89\u202f\u03bcm (geometric midplane: '
    '2.60\u202f\u03bcm). The gold layer enhances D* by 3.2\u00d7 over a '
    'pure parylene-C plate.'))

story.append(para('Effective bending stiffness D*.'))
story += render_eq(
    r'D^* = \tfrac{1}{3}\sum_k \bar{E}_k \left(b_k^3 - b_{k-1}^3\right)',
    label='2')
story.append(p(
    'where b<sub>k</sub> is the signed distance from z<sub>NA</sub> to the '
    'top of layer k [4]. Nominal: '
    'D*\u202f=\u202f6.19\u00d710\u207b\u2078\u202fN\u00b7m.'))

story.append(para('Membrane stiffness.'))
story += render_eq(
    r'A_{11} = \sum_k \bar{E}_k\, t_k,\quad A_{12} = \sum_k \nu_k \bar{E}_k\, t_k',
    label='3')
story.append(p(
    'Nominal: A<sub>11</sub>\u202f=\u202f3.53\u00d710\u2074\u202fN/m, '
    '\u03bd<sub>eff</sub>\u202f=\u202fA<sub>12</sub>/A<sub>11</sub>\u202f=\u202f0.384.'))

story += subsec('2.3', 'Föppl\u2013von Kármán Energy Functional')

story.append(p(
    'For a clamped circular plate under uniform pressure, the equilibrium '
    'configuration minimises the FvK total potential energy [16]:'))
story += render_eq(
    r'\Pi[w,u] = \int_0^a 2\pi r \Bigl[\,\tfrac{D^*}{2}\!\left(w^{\prime\prime}+\tfrac{w^\prime}{r}\right)^{\!2}'
    r' + \tfrac{1}{2}\!\bigl(A_{11}\varepsilon_r^2 + 2A_{12}\varepsilon_r\varepsilon_\theta + A_{11}\varepsilon_\theta^2\bigr)'
    r' + P\,w\,\Bigr]dr',
    fontsize=10, label='4')
story.append(p(
    'where primes denote \u2202/\u2202r. The FvK nonlinear membrane strains are:'))
story += render_eq(
    r'\varepsilon_r = u^\prime + \tfrac{1}{2}(w^\prime)^2, \qquad \varepsilon_\theta = u/r',
    label='5')
story.append(p(
    'The \u00bd(w\u2032)\u00b2 term in \u03b5<sub>r</sub> is the geometric '
    'nonlinearity: membrane stretching generates in-plane tension that '
    'stiffens the response beyond the Kirchhoff prediction. This term is '
    'absent from the Atik analytical model.'))
story.append(p(
    '<b>Sign convention.</b> With w\u202f<\u202f0, the pressure work '
    '+Pw\u202f<\u202f0 decreases \u03a0 as the plate deflects downward. '
    'The Euler\u2013Lagrange condition gives D\u2207\u2074w\u202f+\u202fP\u202f=\u202f0, '
    'whose Kirchhoff solution is '
    'w<sub>K</sub>\u202f=\u202f\u2212Pa\u2074/(64D*)\u00b7(1\u2212\u03be\u00b2)\u00b2\u202f<\u202f0 \u2713.'))
story.append(sp(2))

# ══════════════════════════════════════════════════════════════════════════════
# 3. DEEP RITZ NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════
story += sec('3', 'Deep Ritz Neural Network')
story += subsec('3.1', 'Architecture')

story.append(p(
    'The network maps a normalised input vector '
    '<b>x</b>\u202f\u2208\u202f[0,1]\u2077 to two scalar outputs '
    '(w, u) in physical units:'))
story += render_eq(
    r'\mathbf{x} = \bigl[\,\xi,\;\hat{P},\;\hat{t}_1,\;\hat{t}_2,\;\hat{t}_3,\;\hat{a},\;\hat{a}_g\,\bigr]',
    label='6')
story.append(p(
    'where \u03be\u202f=\u202fr/a. Pressure is normalised on a log scale:'))
story += render_eq(
    r'\hat{P} = \ln(1 + P/10^3)\,/\,\ln(21) \;\in\; [0,1]',
    label='7')
story.append(p(
    'ensuring all three pressure decades contribute equally during training. '
    'The five geometry parameters are mapped linearly to [0,1] within the '
    'training box '
    '(t<sub>1</sub>,t<sub>3</sub>\u2208[1,5]\u202f\u03bcm, '
    't<sub>2</sub>\u2208[0.2,0.5]\u202f\u03bcm, '
    'a\u2208[150,500]\u202f\u03bcm, a<sub>g</sub>\u2208[5,15]\u202f\u03bcm).'))

story.append(para('Trunk network.'))
story.append(p(
    'A six-layer fully-connected trunk maps <b>x</b> to a 128-dimensional '
    'feature vector using tanh activations throughout. Tanh is mandatory: '
    'the bending energy requires w\u2033 (second autograd derivative through '
    'the trunk). ReLU gives w\u2033\u202f=\u202f0 almost everywhere; GELU '
    'produces noisy, oscillatory second derivatives. Tanh is smooth and '
    'analytic, providing a clean, bounded gradient signal for the bending '
    'energy computation.'))

story.append(para('Output heads.'))
story.append(p(
    'Two independent linear layers (128\u21921) produce unconstrained '
    'pre-BC outputs w<sub>raw</sub> and u<sub>raw</sub>. The two-head '
    'design is essential: a single-output network forces u\u202f=\u202f0, '
    'collapsing the membrane strain to '
    '\u03b5<sub>r</sub>\u202f=\u202f\u00bd(w\u2032)\u00b2 only, '
    'missing the actual in-plane equilibrium. Total parameter count: '
    '\u224810\u2075. Both output heads are zero-initialised so training '
    'begins from the physically neutral flat-plate state.'))

story += subsec('3.2', 'Hard Boundary Conditions')

story.append(p(
    'Clamped boundary conditions are imposed exactly by algebraic shape '
    'factors [17,18]:'))
story += render_eq(
    r'w_{\rm profile}(\xi) = (1-\xi^2)^2 \cdot w_{\rm raw} \cdot W_{\rm ref}',
    label='8')
story += render_eq(
    r'u(\xi) = \xi(1-\xi) \cdot u_{\rm raw} \cdot U_{\rm ref}',
    label='9')
story.append(p(
    'The shape factor (1\u2212\u03be\u00b2)\u00b2 is the exact Kirchhoff '
    'clamped-plate polynomial, guaranteeing w(1)\u202f=\u202f0, '
    'w\u2032(1)\u202f=\u202f0, and symmetry at \u03be\u202f=\u202f0. '
    'Physical scaling W<sub>ref</sub>\u202f=\u202fa<sub>g,phys</sub> '
    'ensures the network output lives at O(1) across all air-gap '
    'geometries. This eliminates any gradient budget spent on boundary '
    'residuals\u2014all capacity goes to learning the interior physics.'))

story += subsec('3.3', 'Smooth-Max Contact Constraint')

story.append(p(
    'The electrode contact constraint w\u2265\u2212a<sub>g</sub> is '
    'enforced by a smooth-max in the network forward pass:'))
story += render_eq(
    r'w = \tfrac{1}{2}\!\left(w_p + (-a_g) + \sqrt{(w_p-(-a_g))^2 + \varepsilon_c^2}\,\right)',
    label='10')
story.append(p(
    'with \u03b5<sub>c</sub>\u202f=\u202f50\u202fnm. '
    '<b>Hard clamp</b> (torch.clamp) has zero gradient in the contact '
    'zone\u2014no learning signal in touch mode. '
    '<b>Penalty approach</b>: at P\u202f=\u202f15\u202fkPa, contact '
    'penalty energy is \u223c10\u207b\u00b9\u00b2\u202fJ while pressure '
    'work is \u223c10\u207b\u2078\u202fJ (ratio 10\u2074:1); even '
    '\u03ba\u202f=\u202f10\u00b9\u2070 produces 1.9\u202f\u03bcm '
    'penetration. The smooth-max is a hard mathematical guarantee '
    '(zero violations in 200 random geometry sweeps) with nonzero '
    'gradient everywhere.'))

story += subsec('3.4', 'Training Strategy')

story.append(para('Energy loss.'))
story.append(p(
    'The integral in Eq.\u202f(4) is evaluated via 32-node '
    'Gauss\u2013Legendre quadrature on \u03be\u202f\u2208\u202f(0,1), '
    'providing a differentiable weighted sum through which autograd '
    'backpropagates. The endpoint \u03be\u202f=\u202f0 is never a GL node, '
    'avoiding the 1/r singularity in '
    '\u03b5<sub>\u03b8</sub>\u202f=\u202fu/r.'))

story.append(para('Per-sample normalisation.'))
story.append(p(
    'Energy at P\u202f=\u202f20\u202fkPa is \u223c10\u207b\u2078\u202fJ; '
    'at P\u202f=\u202f50\u202fPa it is \u223c10\u207b\u00b9\u00b2\u202fJ '
    '(four decades). The loss is:'))
story += render_eq(
    r'\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \frac{\Pi_i}{\max(|\Pi_i|,\,\delta)}',
    label='11')
story.append(p(
    'Each sample contributes \u00b11 to the gradient regardless of '
    'pressure. Without this, P\u202f=\u202f100\u202fPa RMSE '
    '(vs.\u202fKirchhoff) is 232%; with normalisation: 2.2%.'))

story.append(para('Sampling and optimisation.'))
story.append(p(
    'Each step draws 64 geometry samples \u00d7 8 log-uniform pressures '
    '= 512 FvK energy evaluations. Adam [19] is used with learning rate '
    '10\u207b\u00b3 \u2192 cosine decay to 10\u207b\u2074 over 50\u202f000 '
    'steps (\u224855\u202fmin on GPU). Gradient norm clipping at '
    '\u2016\u2207\u2016\u2082\u202f\u2264\u202f1 stabilises early training. '
    'The first 5000 steps include an MSE loss against COMSOL profiles '
    'with annealed weight \u03bb\u202f=\u202f1\u2212step/5000; after '
    'step 5000, \u03bb\u202f=\u202f0 and training is driven entirely by '
    'physics.'))
story.append(sp(2))

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESULTS
# ══════════════════════════════════════════════════════════════════════════════
story += sec('4', 'Results')
story += subsec('4.1', 'Training Convergence')

story.append(p(
    'Fig.\u202f2 shows the per-sample normalised energy loss over '
    '50\u202f000 training steps. Starting from \u2248+1 (flat-plate, random '
    'initialisation), the loss converges to \u2248\u22120.65. The negative '
    'steady state reflects that pressure work dominates bending and membrane '
    'terms at equilibrium\u2014physically consistent with a pressure-loaded '
    'deflection. The loss decreases monotonically after the warmup phase '
    'ends at step 5000, confirming stable pure-physics training.'))

story += fig_fl('figures/loss_curve.png', BODY_W * 0.78,
    'Normalised energy loss vs. training step. Loss transitions from '
    '\u2248+1 (flat-plate state) to \u2248\u22120.65 (converged deflection '
    'regime). Steps 0\u20135000 include the COMSOL warmup anchor.', '2')

story += subsec('4.2', 'Deflection Profiles: Three-Way Comparison')

story.append(p(
    'Fig.\u202f3 shows w(r) for five representative geometries at six '
    'pressures (0.5\u202f\u2013\u202f20\u202fkPa). MEMSRitz (solid lines), '
    'COMSOL FEA (dashed), and the Kirchhoff/Atik solution (dotted, clipped '
    'at the contact floor) are overlaid. The geometries span the '
    'five-dimensional training box: small thin, medium balanced, large thick, '
    'small large-gap, and large thin plates.'))

story += fig_fl('figures/deflection_5geometries.png', BODY_W,
    'Deflection profiles w(r) for five geometries spanning the design space. '
    'Solid: MEMSRitz (PINN). Dashed: COMSOL FEA. Dotted: Atik / Kirchhoff '
    '(clipped at contact floor). The dash-dot line marks '
    'w\u202f=\u202f\u2212a<sub>g</sub>. Colours encode pressure from '
    '0.5\u202fkPa (purple) to 20\u202fkPa (yellow). All three curves '
    'describe the same physical device (matched COMSOL file geometry).', '3')

story.append(p(
    '<b>MEMSRitz vs. COMSOL.</b> The solid and dashed lines are nearly '
    'indistinguishable at all pressures and geometries, confirming that '
    'energy minimisation correctly recovers the FvK equilibrium.'))

story.append(p(
    '<b>Kirchhoff overestimation.</b> The Kirchhoff (dotted) lines deflect '
    'further than COMSOL/PINN for the same pressure. For the large thin '
    'geometry (a\u202f=\u202f462\u202f\u03bcm, '
    't<sub>1</sub>\u202f=\u202ft<sub>3</sub>\u202f=\u202f1.6\u202f\u03bcm), '
    'the Kirchhoff model predicts touch-down \u224840% below the COMSOL and '
    'PINN pressure, a direct consequence of neglecting the '
    '\u00bd(w\u2032)\u00b2 membrane strain term.'))

story.append(p(
    '<b>Touch-mode.</b> PINN and COMSOL both flatten at '
    'w\u202f=\u202f\u2212a<sub>g</sub> for high pressures, with the contact '
    'radius growing as expected. Fig.\u202f4 shows the medium-balanced '
    'geometry in detail: at 500\u202fPa all three methods agree '
    '(linear regime); by 5\u202fkPa the Kirchhoff model overestimates '
    'deflection magnitude by \u224825%, while MEMSRitz tracks COMSOL to '
    'within 0.12\u202f\u03bcm.'))

story += fig_fl('figures/deflection_Medium_balanced.png', BODY_W * 0.85,
    'Detailed profiles for the medium-balanced geometry '
    '(a\u202f=\u202f290\u202f\u03bcm, a<sub>g</sub>\u202f=\u202f7.9\u202f\u03bcm, '
    't<sub>1</sub>/t<sub>2</sub>/t<sub>3</sub>\u202f=\u202f1.72/0.30/2.06\u202f\u03bcm). '
    'Solid: MEMSRitz. Dashed: COMSOL. Dotted: Kirchhoff (Atik). '
    'At 500\u202fPa all three agree; by 5\u202fkPa Kirchhoff overestimates '
    'deflection magnitude by \u224825%.', '4')

story += subsec('4.3', 'Quantitative Validation Against COMSOL')

story.append(p(
    'MEMSRitz was validated against all 2144 COMSOL deflection profiles '
    '(each comprising w(r) at 40 pressures, 0\u202f\u2013\u202f20\u202fkPa). '
    'Table\u202f1 summarises the overall RMSE statistics; Table\u202f2 '
    'shows the pressure-decade breakdown.'))

story += mk_table(
    [['Metric', 'Value (\u03bcm)'],
     ['Median RMSE', '0.15'],
     ['Mean RMSE', '0.22'],
     ['90th-percentile RMSE', '0.51'],
     ['Maximum RMSE', '1.10'],
     ['Contact violations / 200 sweeps', '0'],
     ['Kirchhoff error at P\u202f=\u202f100\u202fPa (%)', '2.2%'],
    ],
    [BODY_W * 0.72, BODY_W * 0.28],
    'RMSE statistics for MEMSRitz vs. COMSOL over 2144 deflection profiles.',
    '1')

story += mk_table(
    [['Pressure range', 'Median RMSE (\u03bcm)'],
     ['P \u2264 1\u202fkPa', '0.05'],
     ['1\u202fkPa < P \u2264 5\u202fkPa', '0.15'],
     ['P > 5\u202fkPa', '0.32'],
     ['Touch mode (centre at \u2212a<sub>g</sub>)', '0.41'],
    ],
    [BODY_W * 0.72, BODY_W * 0.28],
    'Median RMSE by pressure decade.',
    '2')

story.append(p(
    'The maximum RMSE of 1.10\u202f\u03bcm occurs for the largest plates '
    '(a\u202f\u2248\u202f492\u202f\u03bcm) near the boundary of the '
    'training domain. The median RMSE of 0.15\u202f\u03bcm corresponds to '
    'a relative error of 1\u20137% of the air gap over the full pressure '
    'range.'))
story.append(sp(2))

# ══════════════════════════════════════════════════════════════════════════════
# 5. DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════
story += sec('5', 'Discussion')

story.append(para('Necessity of the FvK formulation.'))
story.append(p(
    'The Kirchhoff overestimation of deflection grows from <5% at '
    '500\u202fPa to >40% at 20\u202fkPa for thin large-radius plates. '
    'Critically, this overestimation propagates to an underestimate of the '
    'touch-down pressure: for a sensor calibrated using the Atik model, '
    'this manifests as systematic error in the pull-in and '
    'capacitance\u2013pressure (C\u2013P) characteristic.'))

story.append(para('Hard vs. soft constraints.'))
story.append(p(
    'The algebraic BC and smooth-max contact are the two central '
    'architectural choices. The physics loss alone cannot penalise BC '
    'violations strongly enough when the energy term dominates. The '
    '10\u2074:1 ratio between pressure work and contact penalty at '
    'operating pressures renders soft-penalty contact physically '
    'infeasible\u2014consistent with obstacle problems in structural '
    'mechanics. The smooth-max guarantees hard constraint satisfaction '
    'while preserving gradient flow in the contact region.'))

story.append(para('Per-sample normalisation.'))
story.append(p(
    'The three-decade pressure range is a challenge absent from '
    'single-pressure PINN studies. The factor-of-10\u2074 energy ratio '
    'makes the naive mean-energy loss equivalent to training exclusively on '
    'high-pressure data. Per-sample normalisation is the minimal '
    'intervention: no new hyperparameters, unchanged physics. It reduced '
    'P\u202f=\u202f100\u202fPa error from 232% to 2.2%.'))

story.append(para('Inference speed.'))
story.append(p(
    'A single w(r) profile evaluates in <0.5\u202fms on GPU; a batch of '
    '1024 geometries in <10\u202fms. A single COMSOL pressure sweep '
    '(40 pressures) requires \u223c3\u202fmin on a workstation: a speedup '
    'of \u223c10\u2074\u00d7. MEMSRitz is practical as an inner loop in '
    'gradient-based optimisation, uncertainty quantification, and real-time '
    'sensor calibration.'))

story.append(para('Limitations and outlook.'))
story.append(p(
    'The maximum RMSE of 1.10\u202f\u03bcm at large radii suggests '
    'insufficient capacity at the geometry-space boundary. Increasing the '
    'hidden dimension from 128 to 256 neurons or extending training to '
    '100\u202f000 steps are the most direct remedies. The current model '
    'assumes quasi-static loading; extension to dynamic loading requires '
    'incorporating the equation of motion. Non-uniform loading (relevant to '
    'viscosity measurement applications) would require additional input '
    'features or a 2D extension.'))
story.append(sp(2))

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
story += sec('6', 'Conclusion')
story.append(p(
    'We have presented MEMSRitz, a Deep Ritz physics-informed neural network '
    'for large-deflection modelling of clamped multilayer MEMS diaphragms '
    'over a five-dimensional geometry and three-decade pressure space. '
    'The network is trained by minimising the F\u00f6ppl\u2013von '
    'K\u00e1rm\u00e1n variational energy functional with no labelled FEA '
    'data in the main training phase. Hard clamped boundary conditions and '
    'a smooth-max electrode contact guarantee physical validity of all '
    'predictions. Validated against 2144 COMSOL profiles: median RMSE '
    '0.15\u202f\u03bcm, maximum 1.10\u202f\u03bcm, zero contact violations, '
    'sub-millisecond inference (\u223c10\u2074\u00d7 faster than FEA). '
    'Comparison with the Atik Kirchhoff model demonstrates 20\u201350% '
    'overestimation of deflection above 2\u202fkPa and up to 40% '
    'underestimation of touch-down pressure, confirming the necessity of '
    'the full FvK treatment provided by MEMSRitz.'))
story.append(sp(3))

# ── author contributions / declarations ────────────────────────────────────────
story.append(p('<b>CRediT authorship contribution statement</b>', SUBSEC_S))
story.append(p(
    '<b>Seckin Eroglu:</b> Conceptualisation, Methodology, Software, '
    'Validation, Formal analysis, Writing \u2013 original draft. '
    '<b>Ender Yildirim:</b> Conceptualisation, Supervision, Resources, '
    'Writing \u2013 review & editing, Funding acquisition.'))
story.append(sp(1))
story.append(p('<b>Declaration of competing interest</b>', SUBSEC_S))
story.append(p('The authors declare no competing financial interests.'))
story.append(sp(1))
story.append(p('<b>Data availability</b>', SUBSEC_S))
story.append(p(
    'Model code and training scripts: '
    'https://github.com/seckin-erogll/MEMS_Energy-Driven'))
story.append(sp(1))
story.append(p('<b>Acknowledgements</b>', SUBSEC_S))
story.append(p('The authors acknowledge the use of METU central computing facilities.'))
story.append(sp(3))
story.append(hr(ELSEVIER_BLUE, 1))
story.append(sp(2))

# ── references ─────────────────────────────────────────────────────────────────
story.append(p('References', SEC_S))

refs = [
    '[1] G. Fragiacomo et al., "Analysis of small deflection touch mode behavior in '
    'capacitive pressure sensors," <i>Sensors and Actuators A: Physical</i>, '
    'vol. 161, pp. 114\u2013119, 2010. doi:10.1016/j.sna.2010.04.002',

    '[2] R. Lo et al., "A passive MEMS drug delivery pump for treatment of ocular '
    'diseases," <i>Biomedical Microdevices</i>, vol. 11, pp. 959\u2013970, 2009. '
    'doi:10.1007/s10544-009-9313-9',

    '[3] W.-Y. Sim et al., "Theoretical and experimental studies on the parylene '
    'diaphragms for microdevices," <i>Microsystem Technologies</i>, vol. 11, '
    'pp. 11\u201315, 2005. doi:10.1007/s00542-003-0342-7',

    '[4] A. C. Atik, M. D. \u00d6zkan, E. \u00d6zg\u00fcr, H. K\u00fclah, and '
    'E. Y\u0131ld\u0131r\u0131m, "Modeling and fabrication of electrostatically '
    'actuated diaphragms for on-chip valving of MEMS-compatible microfluidic '
    'systems," <i>J. Micromech. Microeng.</i>, vol. 30, p. 115001, 2020. '
    'doi:10.1088/1361-6439/aba16f',

    '[5] F. Zacchei et al., "Neural networks based surrogate modeling for '
    'efficient uncertainty quantification and calibration of MEMS accelerometers," '
    '<i>Int. J. Non-Linear Mech.</i>, vol. 167, p. 104902, 2024. '
    'doi:10.1016/j.ijnonlinmec.2024.104902',

    '[6] P. Di Barba et al., "Using adaptive surrogate models to accelerate '
    'multi-objective design optimization of MEMS," '
    '<i>Micromachines</i>, vol. 16, p. 753, 2025. doi:10.3390/mi16070753',

    '[7] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed '
    'neural networks," <i>J. Comput. Phys.</i>, vol. 378, pp. 686\u2013707, 2019. '
    'doi:10.1016/j.jcp.2018.10.045',

    '[8] I. E. Lagaris, A. Likas, and D. I. Fotiadis, "Artificial neural networks '
    'for solving ordinary and partial differential equations," '
    '<i>IEEE Trans. Neural Netw.</i>, vol. 9, pp. 987\u20131000, 1998. '
    'doi:10.1109/72.712178',

    '[9] G. E. Karniadakis et al., "Physics-informed machine learning," '
    '<i>Nat. Rev. Phys.</i>, vol. 3, pp. 422\u2013440, 2021. '
    'doi:10.1038/s42254-021-00314-5',

    '[10] W. E and B. Yu, "The Deep Ritz Method: A deep learning-based numerical '
    'algorithm for solving variational problems," '
    '<i>Commun. Math. Stat.</i>, vol. 6, pp. 1\u201312, 2018. '
    'doi:10.1007/s40304-018-0127-z',

    '[11] E. Samaniego et al., "An energy approach to the solution of PDEs in '
    'computational mechanics via machine learning," '
    '<i>Comput. Methods Appl. Mech. Eng.</i>, vol. 362, p. 112790, 2020. '
    'doi:10.1016/j.cma.2019.112790',

    '[12] J. N. Fuhg and N. Bouklas, "The mixed deep energy method for resolving '
    'concentration features in finite strain hyperelasticity," '
    '<i>J. Comput. Phys.</i>, vol. 451, p. 110839, 2022. '
    'doi:10.1016/j.jcp.2021.110839',

    '[13] W. Wang and H.-T. Thai, "A physics-informed neural network framework '
    'for laminated composite plates under bending," '
    '<i>Thin-Walled Struct.</i>, vol. 210, p. 113014, 2025. '
    'doi:10.1016/j.tws.2025.113014',

    '[14] J. N. Reddy, <i>Mechanics of Laminated Composite Plates and Shells</i>, '
    '2nd ed. CRC Press, 2003.',

    '[15] R. M. Jones, <i>Mechanics of Composite Materials</i>, 2nd ed. '
    'Taylor & Francis, 1999.',

    '[16] S. P. Timoshenko and S. Woinowsky-Krieger, '
    '<i>Theory of Plates and Shells</i>, 2nd ed. McGraw-Hill, 1959.',

    '[17] N. Sukumar and A. Srivastava, "Exact imposition of boundary conditions '
    'with distance functions in physics-informed deep neural networks," '
    '<i>Comput. Methods Appl. Mech. Eng.</i>, vol. 389, p. 114333, 2022. '
    'doi:10.1016/j.cma.2021.114333',

    '[18] L. Lu et al., "Physics-informed neural networks with hard constraints '
    'for inverse design," <i>SIAM J. Sci. Comput.</i>, vol. 43, '
    'pp. B1105\u2013B1132, 2021. doi:10.1137/21M1397908',

    '[19] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," '
    '<i>Proc. ICLR</i>, 2015. arXiv:1412.6980',

    '[20] B. J. Kim and E. Meng, "Micromachining of Parylene C for bioMEMS," '
    '<i>Polym. Adv. Technol.</i>, vol. 27, pp. 564\u2013576, 2016. '
    'doi:10.1002/pat.3729',
]

for ref in refs:
    story.append(p(ref, REF_S))

# ── build ──────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    'MEMS_PINN_paper.pdf',
    pagesize=A4,
    leftMargin=LEFT_M, rightMargin=RIGHT_M,
    topMargin=TOP_M,   bottomMargin=BOT_M,
    title='A Deep Ritz PINN for Multilayer MEMS Diaphragm Pressure Sensors',
    author='Seckin Eroglu, Ender Yildirim',
)
doc.build(story)
print(f'Saved -> MEMS_PINN_paper.pdf')
