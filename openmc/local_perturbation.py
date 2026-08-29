"""Branched Exact Perturbation (BEP): input generation and result analysis.

Reactivity worth of one or more local perturbations from a single eigenvalue
run. A perturbation is a *set* of cell->material substitutions applied
together, so the same machinery covers a sample swap, a sample displacement
(two thin slivers) and multi-region changes.

The C++ side writes only raw sums, including the full cross-product matrix.
The fit, the linearity check and every uncertainty live here so the analysis
can change without a rebuild.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import h5py
import numpy as np

__all__ = [
    "LocalPerturbation",
    "LocalPerturbationResults",
    "displacement",
    "export_local_perturbations",
    "read_local_perturbations",
    "sample_at",
]


@dataclass
class LocalPerturbation:
    """A set of cell->material substitutions applied together.

    Parameters
    ----------
    substitutions
        ``{cell_id: material_id}`` mapping, or a list of ``(cell, material)``
        pairs. The geometry must hold the REFERENCE material in every one of
        these cells; the substitution is applied only inside this
        perturbation's own shadow tree.
    id
        Identifier used in the statepoint. Assigned automatically if omitted.

    Branch sites are shared by every perturbation touching the same cell, and
    all trees spawned at a site share one random seed. Worths therefore come
    out strongly correlated, so differences and derivatives between them are
    far better determined than the individual values -- use
    :meth:`LocalPerturbationResults.combine`, which carries the covariance.
    """

    substitutions: dict[int, int] | list[tuple[int, int]]
    id: int | None = None

    def items(self) -> list[tuple[int, int]]:
        if isinstance(self.substitutions, dict):
            return list(self.substitutions.items())
        return [tuple(x) for x in self.substitutions]

    @classmethod
    def swap(cls, cell: int, material: int, id: int | None = None):
        """One cell, one material. The ordinary sample-worth case."""
        return cls({cell: material}, id=id)

    def to_xml_element(self, id: int) -> ET.Element:
        e = ET.Element("local_perturbation")
        e.set("id", str(id))
        for cell, mat in self.items():
            s = ET.SubElement(e, "substitution")
            ET.SubElement(s, "cell").text = str(cell)
            ET.SubElement(s, "material").text = str(mat)
        return e


def sample_at(
    slices: list[int],
    displaced: int,
    sample: int,
    start: int,
    length: int,
    id: int | None = None,
) -> LocalPerturbation:
    """Sample occupying ``slices[start:start+length]``, rest displaced.

    ``slices`` is the ordered list of cell ids of an axially sliced
    irradiation channel. The geometry should hold ``displaced`` in all of
    them, so this perturbation is 'insert the sample at this position'.
    """
    subs = {c: displaced for c in slices}
    for c in slices[start:start + length]:
        subs[c] = sample
    return LocalPerturbation(subs, id=id)


def displacement(
    slices: list[int],
    displaced: int,
    sample: int,
    start: int,
    length: int,
    shift: int,
    id: int | None = None,
) -> LocalPerturbation:
    """Move a sample by ``shift`` slices, as a single perturbation.

    Assumes the GEOMETRY already holds the sample at ``slices[start:...]`` and
    ``displaced`` elsewhere in the channel. Only the symmetric difference of
    the two positions is substituted: the trailing slivers revert to
    ``displaced`` and the leading slivers take ``sample``. The overlap is
    untouched, so BEP returns ``rho(z + shift) - rho(z)`` directly as one
    small number rather than as a difference of two larger worths.

    Divide by the slice thickness for ``d(rho)/dz``. Thinner slivers reduce
    the finite-difference bias but raise the variance as ``1/sqrt(delta)`` --
    vary ``shift`` and watch the derivative converge.
    """
    if shift == 0:
        raise ValueError("shift must be non-zero")
    old = set(range(start, start + length))
    new = set(range(start + shift, start + shift + length))
    if min(new) < 0 or max(new) >= len(slices):
        raise ValueError("shifted sample falls outside the slice list")

    subs = {}
    for i in old - new:          # trailing: sample leaves
        subs[slices[i]] = displaced
    for i in new - old:          # leading: sample arrives
        subs[slices[i]] = sample
    return LocalPerturbation(subs, id=id)


def export_local_perturbations(
    perturbations: list[LocalPerturbation],
    n_generation: int = 10,
    path: str = "settings.xml",
) -> None:
    """Append the perturbation blocks to an already-written settings.xml.

    ``n_generation`` (L) is shared by every perturbation and must be at least
    2, since the estimator is a finite difference in depth. 10-12 is the usual
    starting point; confirm with the linearity plot rather than assuming.
    """
    if n_generation < 2:
        raise ValueError("n_generation must be at least 2")
    if not perturbations:
        raise ValueError("no perturbations given")

    tree = ET.parse(path)
    root = tree.getroot()
    for tag in ("local_perturbation", "local_perturbation_n_generation"):
        for old in root.findall(tag):
            root.remove(old)

    ET.SubElement(root, "local_perturbation_n_generation").text = \
        str(n_generation)

    used = {p.id for p in perturbations if p.id is not None}
    next_id = 1
    for p in perturbations:
        if p.id is None:
            while next_id in used:
                next_id += 1
            pid = next_id
            used.add(pid)
        else:
            pid = p.id
        root.append(p.to_xml_element(pid))

    tree.write(path)


@dataclass
class LocalPerturbationResults:
    """All perturbations from one statepoint, with their covariance.

    ``rho`` is in pcm and indexed the same way as ``ids``. ``cov`` is the
    covariance of the mean, also in pcm^2, and is what makes differences
    between perturbations meaningful: they share branch sites and seeds, so
    treating them as independent badly overstates the error on a difference.
    """

    ids: list[int]
    rho: np.ndarray
    cov: np.ndarray
    n_generation: int
    d_half: int
    n_generations: int
    n_branch: int
    l: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    tau: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    tau_ref: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    # -------------------------------------------------------------- lookup
    def index(self, id: int) -> int:
        return self.ids.index(id)

    def value(self, id: int) -> tuple[float, float]:
        """(rho, sigma) in pcm for one perturbation."""
        i = self.index(id)
        return float(self.rho[i]), float(np.sqrt(max(self.cov[i, i], 0.0)))

    # --------------------------------------------------------- combinations
    def combine(self, weights: dict[int, float]) -> tuple[float, float]:
        """Value and 1-sigma of ``sum_i w_i * rho_i``, in pcm.

        Uses the full covariance, so a difference between two nearby sample
        positions gets the small error it deserves rather than the quadrature
        sum of two large ones.
        """
        w = np.zeros(len(self.ids))
        for pid, wi in weights.items():
            w[self.index(pid)] = wi
        val = float(w @ self.rho)
        var = float(w @ self.cov @ w)
        return val, float(np.sqrt(max(var, 0.0)))

    def difference(self, id_a: int, id_b: int) -> tuple[float, float]:
        """rho(a) - rho(b) with the correlation carried through."""
        return self.combine({id_a: 1.0, id_b: -1.0})

    def derivative(
        self, id_a: int, id_b: int, dz: float
    ) -> tuple[float, float]:
        """[rho(a) - rho(b)] / dz, in pcm per unit of dz."""
        v, s = self.difference(id_a, id_b)
        return v / dz, s / abs(dz)

    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(self.cov), 1e-300, None))
        return self.cov / np.outer(d, d)

    # ---------------------------------------------------------- diagnostics
    def linearity(self, id: int, d_min: int | None = None) -> float:
        """Reduced chi-square of a straight-line fit to l(d).

        Near 1 means the asymptotic regime has been reached over the fitted
        range. Much above 1 means L is too small, or d_min too low, and the
        finite difference is contaminated by the transient. There is no
        plateau to look for -- l(d) is linear, and its slope is the answer.
        """
        y = self.l[id]
        d = np.arange(len(y))
        if d_min is None:
            d_min = self.d_half
        sel = d >= d_min
        if sel.sum() < 3:
            return float("nan")
        p = np.polyfit(d[sel], y[sel], 1)
        resid = y[sel] - np.polyval(p, d[sel])
        scale = np.abs(y[sel]).max() or 1.0
        return float((resid**2).sum() / (sel.sum() - 2)) / scale**2

    def plot(self, id: int, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        y = 1e5 * self.l[id]
        d = np.arange(len(y))
        ax.plot(d, y, marker="o")
        sel = d >= self.d_half
        p = np.polyfit(d[sel], y[sel], 1)
        ax.plot(d, np.polyval(p, d), ls="--", color="k")
        ax.axvline(self.d_half, ls=":", color="grey")
        v, s = self.value(id)
        ax.set_xlabel("shadow tree depth $d$ (generations)")
        ax.set_ylabel(r"$\ell(d)$ (pcm)")
        ax.set_title(f"perturbation {id}: {v:.3f} $\\pm$ {s:.3f} pcm")
        return ax


def read_local_perturbations(statepoint: str) -> LocalPerturbationResults:
    with h5py.File(statepoint, "r") as f:
        if "local_perturbation" not in f:
            raise KeyError(
                "no local_perturbation group: was <local_perturbation> "
                "in settings.xml?"
            )
        g = f["local_perturbation"]
        L = int(g["n_generation"][()])
        dh = int(g["d_half"][()])
        ngen = int(g["n_generations"][()])
        n_branch = int(g["n_branch"][()])
        ids = [int(x) for x in np.asarray(g["ids"][()])]
        np_ = len(ids)

        s_rho = np.asarray(g["sum_rho"][()], dtype=float)
        s_cross = np.asarray(g["sum_cross"][()], dtype=float).reshape(np_, np_)
        n_pair = np.asarray(g["n_pair"][()], dtype=float).reshape(np_, np_)

        n_diag = np.diag(n_pair).copy()
        if (n_diag < 2).any():
            bad = [ids[i] for i in np.where(n_diag < 2)[0]]
            raise ValueError(
                f"perturbations {bad} have fewer than 2 usable generations; "
                "no branch site reached full depth. Increase particles per "
                "generation or enlarge the perturbed region."
            )
        mean = s_rho / n_diag

        # Covariance of the mean. n_pair is per pair because a starved
        # perturbation must not invalidate the others, so a pair with no
        # common usable generation is left at zero covariance.
        n_eff = np.where(n_pair > 1, n_pair, np.nan)
        cov = (s_cross / n_eff - np.outer(mean, mean)) / (n_eff - 1)
        cov = np.nan_to_num(cov)

        l, tau, tau_ref = {}, {}, {}
        for name in g:
            if not name.startswith("perturbation "):
                continue
            pid = int(name.split()[1])
            pg = g[name]
            i = ids.index(pid)
            l[pid] = np.asarray(pg["sum_l"][()], dtype=float) / n_diag[i]
            tau[pid] = np.asarray(pg["sum_tau"][()], dtype=float) / ngen
            tau_ref[pid] = np.asarray(
                pg["sum_tau_ref"][()], dtype=float) / ngen

    return LocalPerturbationResults(
        ids=ids,
        rho=1e5 * mean,
        cov=1e10 * cov,
        n_generation=L,
        d_half=dh,
        n_generations=ngen,
        n_branch=n_branch,
        l=l,
        tau=tau,
        tau_ref=tau_ref,
    )
