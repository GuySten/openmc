"""Build an electron library whose elastic angular tables come from PENELOPE.

Diagnostic only -- see README.md. Everything except elastic/distribution is
copied unchanged, so elastic/xs, xs_total and xs_transport stay as EPICS gives
them and electron.cpp still rescales the sampled deflection onto the EPICS
transport cross section. The only thing that changes is the SHAPE of the
angular distribution and the energy grid it is tabulated on: 96 PENELOPE
energies from 50 eV to 100 MeV against EEDL's 16, which for carbon has nothing
at all between 0.256 and 10 MeV.

    build_library.py <source.h5> <graphite.mat> <out.h5> [--roundtrip]

--roundtrip rewrites the EEDL tables through the same code path instead of
substituting PENELOPE's. That is the control: it must reproduce the source
results exactly, or a null result from the substitution means nothing.
"""
import shutil, sys
import numpy as np
import h5py
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from penelope_elastic import read_mat, angular_grid

MU_CUT = 0.999999          # where EEDL stops and the analytic peak takes over
LIN_LIN = 2                # ENDF interpolation flag

def table_from_dcs(mu_desc, dcs):
    """(mu, pdf, cdf) over [-1, MU_CUT] from a DCS on PENELOPE's grid.

    PENELOPE tabulates against theta ascending, so mu descends from 1 to -1.
    Flip to ascending mu, drop the forward peak beyond the cut, and add the
    cut itself as an endpoint so every table ends in the same place the EEDL
    ones do.
    """
    mu = mu_desc[::-1].copy()
    p = 2.0*np.pi*dcs[::-1].copy()            # dsigma/dmu
    keep = mu <= MU_CUT
    mu_k, p_k = mu[keep], p[keep]
    p_end = np.interp(MU_CUT, mu, p)
    mu_k = np.append(mu_k, MU_CUT)
    p_k = np.append(p_k, p_end)
    # Collapse any duplicate abscissae the cut may have produced
    keep2 = np.concatenate(([True], np.diff(mu_k) > 0))
    mu_k, p_k = mu_k[keep2], p_k[keep2]
    c = np.concatenate(([0.0], np.cumsum(0.5*(p_k[1:]+p_k[:-1])*np.diff(mu_k))))
    if c[-1] <= 0:
        raise ValueError("degenerate angular table")
    p_k = p_k/c[-1]
    c = c/c[-1]
    c[-1] = 1.0
    return mu_k, p_k, c

def read_existing(group):
    E = group["energy"][()]
    mu = group["mu"]
    off = mu.attrs["offsets"]; interp = mu.attrs["interpolation"]; raw = mu[()]
    out = []
    for j in range(len(E)):
        a = off[j]; b = off[j+1] if j+1 < len(E) else raw.shape[1]
        out.append((raw[0, a:b].copy(), raw[1, a:b].copy(), raw[2, a:b].copy()))
    return E, out, interp

def write(group, energies, tables, interps):
    n = sum(len(t[0]) for t in tables)
    raw = np.empty((3, n)); off = np.empty(len(tables), dtype=int)
    k = 0
    for j, (x, p, c) in enumerate(tables):
        off[j] = k
        raw[0, k:k+len(x)] = x; raw[1, k:k+len(x)] = p; raw[2, k:k+len(x)] = c
        k += len(x)
    for name in ("energy", "mu"):
        if name in group:
            del group[name]
    group.create_dataset("energy", data=energies)
    d = group.create_dataset("mu", data=raw)
    d.attrs["offsets"] = off
    d.attrs["interpolation"] = np.asarray(interps, dtype=int)

def main():
    src, mat, out = sys.argv[1], sys.argv[2], sys.argv[3]
    roundtrip = "--roundtrip" in sys.argv
    shutil.copy(src, out)
    with h5py.File(out, "r+") as f:
        el = f[list(f.keys())[0]]["elastic"]["distribution"]
        E_old, tabs_old, interp_old = read_existing(el)
        if roundtrip:
            write(el, E_old, tabs_old, interp_old)
            print(f"round-trip: {len(E_old)} EEDL tables rewritten unchanged")
            return
        E_pen, CS, CS1, CS2, DCS = read_mat(mat)
        _, mu_desc = angular_grid()
        energies, tables, interps = [], [], []
        # Keep the EEDL tables outside PENELOPE's range so the grid still spans
        # the whole transport energy range.
        for j, e in enumerate(E_old):
            if e < E_pen[0]:
                energies.append(e); tables.append(tabs_old[j])
                interps.append(int(interp_old[j]))
        for j, e in enumerate(E_pen):
            energies.append(e)
            tables.append(table_from_dcs(mu_desc, DCS[j]))
            interps.append(LIN_LIN)
        for j, e in enumerate(E_old):
            if e > E_pen[-1]:
                energies.append(e); tables.append(tabs_old[j])
                interps.append(int(interp_old[j]))
        order = np.argsort(energies)
        energies = list(np.asarray(energies)[order])
        tables = [tables[i] for i in order]
        interps = [interps[i] for i in order]
        write(el, energies, tables, interps)
        print(f"PENELOPE tables: {len(energies)} energies "
              f"({energies[0]:.4g} to {energies[-1]:.4g} eV), "
              f"of which {len(E_pen)} from the .mat")

if __name__ == "__main__":
    main()
