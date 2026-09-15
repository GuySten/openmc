# PENELOPE cross-check

**Diagnostic branch. Do not merge into `electron-transport`.**

The aim is to separate a *data* problem from an *implementation* problem. All
day the two have been confounded: when our depth-dose disagrees with EGSnrc and
PenRed we cannot tell whether the tables are inadequate or the sampler mistreats
them. Feeding the same binary different elastic numbers settles it.

Most of this needs no code change at all. Our angular tables are just
(energy, mu, pdf, cdf) in HDF5, so PENELOPE-derived tables drop into the
existing format and the transport is untouched.

## The ladder

Each rung changes one thing.

| rung | change | tests |
|------|--------|-------|
| 0 | EEDL tables rewritten through the same writer | the substitution machinery |
| 1 | `xs_transport` <- PENELOPE sigma_tr1 | the first-moment anchor |
| 2 | `elastic/distribution` <- PENELOPE DCS | **the 0.256-10 MeV gap** |
| 3 | both of the above | any interaction between them |
| 4 | constrain sigma_tr2 as well | the second-moment deficit (needs code) |

Rung 0 is not optional. Without it a null result at rung 2 is uninterpretable.

## Rung 1 is already answered, and it is a null

EPICS `xs_transport` agrees with PENELOPE's sigma_tr1 to 0.5-1% from 0.1 to
21 MeV, and to 2.5% at 50 MeV:

    E (MeV)    EPICS tr     PEN tr1   ratio
      0.256  2.1565e-22  2.1614e-22  0.9977
      1.000  2.6114e-23  2.5938e-23  1.0068
      4.270  2.5778e-24  2.5644e-24  1.0052
     21.130  1.5356e-25  1.5204e-25  1.0099

Same convention, same values. The anchor is right and swapping it changes
nothing, which is what a control should show. Worth recording separately: the
*total* elastic cross sections do not agree at all, EPICS falling from 1.01 of
PENELOPE's at 0.1 MeV to 0.585 at 21 MeV. That is the extreme forward peak,
which barely matters for transport -- sigma_tr1 is the quantity that does, and
it agrees.

So the shape is the only thing left, which is what rung 2 isolates.

## What rung 2 changes, and what it deliberately does not

Only `elastic/distribution`. `elastic/xs`, `xs_total` and `xs_transport` stay
as EPICS gives them, so `electron.cpp` still rescales the sampled deflection
onto the EPICS transport cross section and the first moment is unchanged by
construction. Any difference in the result is then purely the angular *shape*
and the energy grid it is tabulated on:

* EEDL: 16 energies, and for carbon nothing between 0.256 and 10 MeV, a factor
  of 39 with 4.27 MeV sitting inside it.
* PENELOPE: 96 energies from 50 eV to 100 MeV, 606 angles each.

Where both have a table the two differ anyway -- at 10 MeV `<1-mu>` is
1.723e-5 from EEDL against 1.879e-5 from PENELOPE, some 8% -- so a difference
at 4.27 MeV is not by itself proof that the gap is the cause. The 21.13 MeV
case, which EEDL brackets within a factor of two, is the control for that.

## Usage

    python3 penelope_elastic.py graphite.mat          # parse and self-check
    python3 build_library.py src.h5 graphite.mat out.h5 --roundtrip   # rung 0
    python3 build_library.py src.h5 graphite.mat out.h5               # rung 2

`penelope_elastic.py` reproduces PENELOPE's own tabulated sigma and sigma_tr1
from the DCS to 0.4% or better, which is the check that the angular grid has
been reconstructed correctly -- the .mat does not store it, so it is rebuilt
from the construction in PenRed's `pen_material.cpp`.
