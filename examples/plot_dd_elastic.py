#!/usr/bin/env python3
"""Plot d-d reaction cross sections vs energy."""

import matplotlib.pyplot as plt
import numpy as np
import openmc.data

# Load deuteron-on-deuterium data
data = openmc.data.IncidentDeuteron.from_endf(
    '/Users/gavin/Code/ENDF-B-VIII.0/deuterons/d-001_H_002.endf'
)

# Create energy grid for plotting
E = np.logspace(2, 7, 500)  # 100 eV to 10 MeV

# Plot all reactions
fig, ax = plt.subplots(figsize=(10, 6))

reactions = {
    2: ('d,d elastic', 'b-'),
    50: ('d,d → n + ³He (Q=+3.27 MeV)', 'r-'),
    600: ('d,d → p + t (Q=+4.03 MeV)', 'g-'),
}

for mt, (label, style) in reactions.items():
    xs = data[mt].xs['0K']
    sigma = xs(E)
    ax.loglog(E / 1e6, sigma, style, linewidth=2, label=label)

ax.set_xlabel('Deuteron Energy (MeV)', fontsize=12)
ax.set_ylabel('Cross Section (barns)', fontsize=12)
ax.set_title('d + d Reaction Cross Sections (ENDF/B-VIII.0)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-4, 10)
ax.set_ylim(1e-10, 10)

plt.tight_layout()
plt.savefig('dd_reactions_xs.png', dpi=150)
plt.show()

# Print some values
print("\nCross sections at selected energies:")
print(f"{'Energy':<12} {'Elastic':<12} {'n+³He':<12} {'p+t':<12}")
for E_val in [1e4, 1e5, 1e6, 5e6]:
    vals = [data[mt].xs['0K'](E_val) for mt in [2, 50, 600]]
    print(f"{E_val/1e6:<12.3f} {vals[0]:<12.3e} {vals[1]:<12.3e} {vals[2]:<12.3e}")
