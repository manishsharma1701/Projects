#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

fname = "mcfd.rhsav"

# -------------------------------------------------------
# Usage:
#   python3 residue.py
#       -> plots all iterations
#
#   python3 residue.py 50000
#       -> plots only up to iteration 50000
# -------------------------------------------------------

max_iteration = None
if len(sys.argv) > 1:
    try:
        max_iteration = int(sys.argv[1])
    except ValueError:
        print("Usage: python3 residue.py [last_iteration]")
        sys.exit(1)

names = [
    "energy",
    "mass",
    "x-momentum",
    "y-momentum",
    "z-momentum",
    "N",
    "O2",
    "NO",
    "O",
    "vib_energy"
]

data = []
iterations = []

with open(fname, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):

    line = lines[i].strip()
    parts = line.split()

    # Detect iteration line
    if len(parts) == 2:
        try:
            iteration = int(parts[0])

            vals1 = [float(x) for x in lines[i + 1].split()]
            vals2 = [float(x) for x in lines[i + 2].split()]

            if len(vals1) == 5 and len(vals2) == 5:

                if max_iteration is None or iteration <= max_iteration:
                    iterations.append(iteration)
                    data.append(vals1 + vals2)
                else:
                    break

            i += 3
            continue

        except ValueError:
            pass

    i += 1

if len(data) == 0:
    print("No residual data found.")
    sys.exit(1)

data = np.array(data)

# Normalize each residual by its own maximum value
maxvals = np.max(np.abs(data), axis=0)
maxvals[maxvals == 0] = 1.0

norm = data / maxvals

# Avoid log(0)
norm = np.clip(norm, 1e-30, None)

plt.figure(figsize=(10, 6))

exclude = {"z-momentum"}

for j, name in enumerate(names):
    if name in exclude:
        continue

    plt.plot(
        iterations,
        np.log10(norm[:, j]),
        linewidth=1.5,
        label=name
    )

plt.xlabel("Iteration")
plt.ylabel(r"$\log_{10}(R/R_{max})$")
plt.title("Normalized Residuals")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(ncol=2)
plt.tight_layout()

outfile = "normalized_residuals"

if max_iteration is not None:
    outfile += f"_to_{max_iteration}"

plt.savefig(outfile + ".png", dpi=300)
#plt.savefig(outfile + ".pdf")

plt.close()
