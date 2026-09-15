"""Can the forward peak close the elastic transport cross section?

The transport code treats xs_transport as the first moment of the total
elastic cross section, so

    xs_transport = sigma_525 * <1-mu>_tabulated  +  sigma_peak * <1-mu>_peak

with sigma_peak = MT526 - MT525 and <1-mu>_peak bounded above by the 1e-6
cutoff where the tabulated distribution stops. The peak can only ADD, so
sigma_525 * <1-mu> must not exceed xs_transport. Where it does, the angular
tables and the transport cross section cannot both be right, and
compute_mean_deflection responds by rescaling the sampled deflection down by
whatever it takes.

Moments are taken from EEDL2023.ALL itself with the lin-lin law the evaluation
declares (LANG=12), not from the histogram the ACE route reconstructs, so a
1.4 to 2.1% interpolation bias cannot be mistaken for a data defect. Note the
transport cross section is NOT in EEDL -- it comes from JXS(27) of eprdata14 --
so what this measures is a mutual inconsistency between the evaluation's
angular data and MCNP's transport cross section, not an internal contradiction
in EEDL alone.

    peak_closure.py <EEDL2023.ALL> <electron-library-dir>
"""
import os, sys
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_eedl import fields, ints, read_tab1

def load(path):
    want = {(23, 525), (23, 526), (26, 525)}
    b = {}
    for line in open(path, errors="ignore"):
        if len(line) < 76:
            continue
        m = line[66:70].strip()
        if not m or not m.isdigit():
            continue
        key = (int(line[70:72]), int(line[72:75]))
        if key in want:
            b.setdefault((int(m), *key), []).append(line.rstrip("\n"))
    return b

def angular(L):
    k = 1
    c = ints(L[k]); nr, np_ = c[4], c[5]; k += 1
    k += (nr*2 + 5)//6; k += (2*np_ + 5)//6
    c = ints(L[k]); nr2, ne = c[4], c[5]; k += 1; k += (nr2*2 + 5)//6
    out = []
    for _ in range(ne):
        h = ints(L[k]); E = fields(L[k])[1]; nw = h[4]; k += 1
        v = []
        while len(v) < nw:
            v += fields(L[k]); k += 1
        out.append((E, np.array(v[:nw][0::2]), np.array(v[:nw][1::2])))
    return out

def main():
    eedl, libdir = sys.argv[1], sys.argv[2]
    sym = {}
    for p in os.listdir(libdir):
        if not p.endswith(".h5"):
            continue
        el = p[:-3]
        with h5py.File(os.path.join(libdir, p), "r") as f:
            sym[int(f[el].attrs["Z"])] = el
    b = load(eedl)
    rows = []
    for mat in sorted({k[0] for k in b}):
        Z = mat//100
        if Z not in sym:
            continue
        try:
            E5, x5 = read_tab1(b[(mat, 23, 525)], 1)[:2]
            E6, x6 = read_tab1(b[(mat, 23, 526)], 1)[:2]
            ang = angular(b[(mat, 26, 525)])
        except Exception:
            continue
        with h5py.File(os.path.join(libdir, f"{sym[Z]}.h5"), "r") as f:
            g = f[sym[Z]]
            Eg = g["energy"][()]; xtr = g["elastic"]["xs_transport"][()]
        for E, mu, fp in ang:
            if not (E5[0] <= E <= E5[-1] and Eg[0] <= E <= Eg[-1]):
                continue
            m1 = np.trapezoid(fp*(1.0 - mu), mu)
            ip = lambda EE, xx: np.exp(np.interp(
                np.log(E), np.log(EE), np.log(np.maximum(xx, 1e-300))))
            s5, s6, tr = ip(E5, x5), ip(E6, x6), ip(Eg, xtr)
            rows.append((sym[Z], Z, E, s5*m1/tr, (s6 - s5)/max(s6, 1e-300)))
    over = [r for r in rows if r[3] > 1.005]
    print(f"{len(rows)} (element, energy) points with a tabulated distribution")
    print(f"{len(over)} where sigma_525*<1-mu> exceeds xs_transport by >0.5%\n")
    ten = sorted([r for r in over if abs(r[2] - 1e7) < 1], key=lambda r: -r[3])
    oth = sorted([r for r in over if abs(r[2] - 1e7) >= 1], key=lambda r: -r[3])
    print(f"--- the 10 MeV table: {len(ten)} elements ---")
    for el, Z, E, r, pk in ten[:12]:
        print(f"  {el:>4s} Z={Z:3d}  overshoot {100*(r-1):6.1f}%  "
              f"peak/total {pk:.4f}")
    print(f"\n--- elsewhere: {len(oth)} points ---")
    for el, Z, E, r, pk in oth[:12]:
        print(f"  {el:>4s} Z={Z:3d}  {E:11.4e} eV  overshoot {100*(r-1):6.1f}%  "
              f"peak/total {pk:.4f}")

if __name__ == "__main__":
    main()
