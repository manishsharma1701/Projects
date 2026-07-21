from __future__ import print_function

import sys
import os
import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import PolyCollection
#!/usr/bin/env python3

import re


def read_mcfd_info(fname="mcfd.inp"):

    params = {}

    species = set()
    reactions = set()

    keys = {
        "ifmdis",
        "disson",
        "ifmdps",
        "ifmdex",
        "mdisps",
        "mdtype",
        "mdpscf",
        "tnoneq_numeqns"
    }

    with open(fname, "r") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 2 and parts[0] in keys:
                params[parts[0]] = parts[1]

            # Species detection
            m = re.search(r"species_\d+_Mwt1_(\S+)", line)
            if m:
                species.add(m.group(1))

            # Reaction detection
            m = re.search(r"reaction_(\d+)_specification", line)
            if m:
                reactions.add(int(m.group(1)))

    nspecies = len(species)
    nreactions = len(reactions)

    # =====================================================
    # DISSIPATION
    # =====================================================

    print("\n" + "="*60)
    print("DISSIPATION SETTINGS")
    print("="*60)

    print(
        "Matrix Dissipation       : {}".format(
            "ON" if params.get("ifmdis", "0") == "1" else "OFF"
        )
    )

    if "disson" in params:
        print(
            "Pressure Switch Value    : {}".format(
                params["disson"]
            )
        )

    print(
        "Pressure Switch          : {}".format(
            "ON" if params.get("ifmdps", "0") == "1" else "OFF"
        )
    )

    if "ifmdex" in params:

        if params["ifmdex"] == "1":
            print("Expansion Detection      : OFF")
        else:
            print("Expansion Detection      : ON")

    if "mdisps" in params:
        print(
            "Pressure Dissipation     : {}".format(
                params["mdisps"]
            )
        )

    if "mdtype" in params:
        print(
            "Dissipation Type         : {}".format(
                params["mdtype"]
            )
        )

    if "mdpscf" in params:

        blend = 100.0 * float(params["mdpscf"])

        print(
            "Max 1st Order Blend      : {} ({:.0f}%)".format(
                params["mdpscf"],
                blend
            )
        )

    # =====================================================
    # CHEMISTRY
    # =====================================================

    print("\n" + "="*60)
    print("CHEMICAL KINETICS")
    print("="*60)

    print("Species Found            : {}".format(nspecies))
    print("Species List             : {}".format(
        ", ".join(sorted(species))
    ))

    print("Reactions Found          : {}".format(nreactions))

    tvib = int(params.get("tnoneq_numeqns", "0"))

    print("Vibrational Equations    : {}".format(tvib))

    model = "Unknown"

    # ---- Park 1985 ----
    if nspecies == 5 and nreactions == 8:
        model = "Park 1985"
        desc = "5 Species / 8 Reactions"

    # ---- Park 1993 ----
    elif nspecies == 5 and nreactions == 17:
        model = "Park 1993"
        desc = "5 Species / 17 Reactions"

    # ---- Gupta ----
    elif nspecies == 7 and nreactions >= 17:
        model = "Gupta 1990"
        desc = "7 Species Air Chemistry"

    # ---- Ionized Air ----
    elif nspecies >= 10:
        model = "Ionized Air Chemistry"
        desc = "High Temperature Air"

    else:
        desc = "Custom Mechanism"

    print("Chemistry Model          : {}".format(model))
    print("Mechanism                : {}".format(desc))

    print("\n" + "="*60)

if __name__ == "__main__":
    read_mcfd_info("mcfd.inp")
    
    
if len(sys.argv) < 4:
    print("Usage: python analyze_heatflux.py <mcfdsol_bc1.vtk> <output_folder> <vtk_volume>")
    sys.exit(1)

vtk_current  = sys.argv[1]
cycle_folder = sys.argv[2]
vtk_volume   = sys.argv[3]

if not os.path.exists(cycle_folder):
    os.makedirs(cycle_folder)

cycle_label = os.path.basename(os.path.normpath(cycle_folder))

# ============================================================
# 1. Load VTK  (each file read exactly once)
# ============================================================
mesh        = pv.read(vtk_current)
volume_mesh = pv.read(vtk_volume)

# ============================================================
# 2. Extract surface coordinates and fields
# ============================================================
coords = mesh.points
Qdot   = mesh.point_data["Qdot"]
P      = mesh.point_data["P"]
Yplus  = mesh.point_data["Y_plus"]

theta     = np.degrees(np.arctan2(coords[:, 1], -coords[:, 0]))
idx       = np.argsort(theta)
theta_s   = theta[idx]          # sorted once, reused everywhere

# ============================================================
# 3. Derived scalar quantities (computed once)
# ============================================================
max_qdot_idx      = np.argmax(Qdot)
max_qdot          = Qdot[max_qdot_idx]
theta_max_qdot    = theta[max_qdot_idx]

max_p_idx         = np.argmax(P)
max_pressure      = P[max_p_idx]
theta_max_pressure = theta[max_p_idx]

yplus_at_0  = np.interp(0,  theta_s, Yplus[idx])
yplus_at_90 = np.interp(90, theta_s, Yplus[idx])

print("--------------------------------------------------")
print("Cycle                : {}".format(cycle_label))
print("Max Heat Flux (Qdot) : {:.4e} W/m^2".format(max_qdot))
print("  at theta           : {:.2f} deg".format(theta_max_qdot))
print("Max Y+               : {:.4f}".format(np.max(Yplus)))
print("Min Y+               : {:.4f}".format(np.min(Yplus)))
print("--------------------------------------------------")

# ============================================================
# 4. Load experimental data
# ============================================================
experimental_Qdot = pd.read_csv("Exp_Qdot.csv")
experimental_P    = pd.read_csv("Experimental_Pressure_ratio.csv")

exp_q_x = experimental_Qdot.iloc[:, 0]
exp_q_y = experimental_Qdot.iloc[:, 2] * 4.83e6
q_error  = exp_q_y * 0.05

exp_p_x = experimental_P.iloc[:, 0]
exp_p_y = experimental_P.iloc[:, 2] * 68e3

# ============================================================
# 5. Surface plots  (heat flux, pressure, Y+)
#    Using a helper to avoid repetition
# ============================================================
def save_surface_plot(x_sim, y_sim, label_sim,
                      x_exp, y_exp, yerr,
                      xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_sim, y_sim, '-k', label=label_sim)
    if yerr is not None:
        ax.errorbar(x_exp, y_exp, yerr=yerr, fmt='^',
                    ecolor='blue', capsize=3, elinewidth=1,
                    markeredgecolor='blue', label="experimental (+-5%)")
    else:
        ax.plot(x_exp, y_exp, '^', label="experimental")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(path))

save_surface_plot(
    theta_s, Qdot[idx], cycle_label,
    exp_q_x, exp_q_y, q_error,
    "Theta (deg)", "Qdot (W/m^2)", "Heat Flux vs Theta",
    os.path.join(cycle_folder, "heat_flux_comparison.png")
)

save_surface_plot(
    theta_s, P[idx], cycle_label,
    exp_p_x, exp_p_y, None,
    "Theta (deg)", "Pressure (Pa)", "Pressure vs Theta",
    os.path.join(cycle_folder, "pressure_comparison.png")
)

save_surface_plot(
    theta_s, Yplus[idx], cycle_label,
    [], [], None,
    "Theta (deg)", "Y+", "Y+ vs Theta",
    os.path.join(cycle_folder, "yplus_comparison.png")
)

# ============================================================
# 6. Volume mesh  — shared geometry computed ONCE
# ============================================================
vol_coords = volume_mesh.points
x = vol_coords[:, 0]
y = vol_coords[:, 1]

all_r_vol  = np.sqrt(x**2 + y**2)
cyl_radius = all_r_vol.min() * 1.05   # used by contour plots

# Triangulate once; reuse for every contour field
tri_mesh  = volume_mesh.triangulate()
cells_raw = tri_mesh.cells.reshape(-1, 4)
triangles = cells_raw[:, 1:4]          # shape (N, 3)  — numpy array

tri_x = x[triangles].mean(axis=1)
tri_y = y[triangles].mean(axis=1)
tri_r = np.sqrt(tri_x**2 + tri_y**2)
inner_mask = tri_r < cyl_radius        # boolean mask reused for every field

def save_contour_plot(field_data, cbar_label, title, filename):
    """Render a tricontourf and save; shares the pre-built triangulation."""
    triang = tri.Triangulation(x, y, triangles)
    triang.set_mask(inner_mask)

    fig, ax = plt.subplots(figsize=(10, 12))
    cf = ax.tricontourf(triang, field_data, levels=100, cmap="turbo")
    fig.colorbar(cf, ax=ax, label=cbar_label)
    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_xlim(x.min(), x.max());  ax.set_ylim(y.min(), y.max())
    ax.set_title(title)
    ax.add_patch(plt.Circle((0, 0), cyl_radius,
                             color='black', fill=False, linewidth=1.5))
    out = os.path.join(cycle_folder, filename)
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(out))

# PLOT 4a: Mach contour
if "M" in volume_mesh.point_data:
    save_contour_plot(tri_mesh.point_data["M"],
                      "Mach", "Mach Contour", "mach_contour.png")
else:
    print("WARNING: Mach field 'M' not found in volume mesh.")
    print("Available fields: {}".format(list(volume_mesh.point_data.keys())))

# PLOT 4b: Temperature contour
if "T" in volume_mesh.point_data:
    save_contour_plot(tri_mesh.point_data["T"],
                      "Temperature", "Temp Contour", "temp_contour.png")
else:
    print("WARNING: Temp field 'T' not found in volume mesh.")
    print("Available fields: {}".format(list(volume_mesh.point_data.keys())))


# ============================================================
# 4c: Species mass-fraction contours  (one plot per species)
#     Auto-discovers any field whose name starts with "Y_"
# ============================================================
SPECIES_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999'
]

# Known MCFD species names (no Y_ prefix in this solver)
MCFD_SPECIES = ["N2", "O2", "NO", "N", "O", "NO+", "N2+", "O2+", "e-"]
species_keys = [k for k in MCFD_SPECIES if k in volume_mesh.point_data]

if species_keys:
    print("Species fields found: {}".format(species_keys))
    for sp_key in species_keys:
        sp_data = tri_mesh.point_data[sp_key]
        save_contour_plot(
            sp_data,
            "Mass fraction  {}  (-)".format(sp_key),
            "Species Contour  {}".format(sp_key),
            "species_{}_contour.png".format(sp_key)
        )
else:
    print("INFO: No species fields (Y_*) found in volume mesh.")
    print("      Available fields: {}".format(list(volume_mesh.point_data.keys())))

# ============================================================
# 7. Stagnation line  T and Tvib
# ============================================================
x_vol = x;  y_vol = y
y_tol     = (y_vol.max() - y_vol.min()) * 0.01
stag_mask = (np.abs(y_vol) < y_tol) & (x_vol < 0)

if np.sum(stag_mask) < 2:
    print("WARNING: Too few points on stagnation line (tol={:.4f}).".format(y_tol))
else:
    x_stag   = x_vol[stag_mask]
    cyl_r    = all_r_vol.min()
    dist     = np.abs(x_stag) - cyl_r
    sort_idx = np.argsort(dist)
    dist_s   = dist[sort_idx]

    print("Stagnation line points: {}".format(np.sum(stag_mask)))
    print("Distance range: {:.4f} to {:.4f} m".format(dist_s[0], dist_s[-1]))

    if "T" in volume_mesh.point_data and "Tvib" in volume_mesh.point_data:
        T_s    = volume_mesh.point_data["T"][stag_mask][sort_idx]
        Tvib_s = volume_mesh.point_data["Tvib"][stag_mask][sort_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dist_s, T_s,    '-k', label="T")
        ax.plot(dist_s, Tvib_s, '-r', label="Tvib")
        ax.set_xlabel("Distance from wall (m)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("T and Tvib along Stagnation Line")
        ax.legend();  ax.grid(True, linestyle='--', alpha=0.6)
        out = os.path.join(cycle_folder, "stagnation_T_Tvib.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: {}".format(out))

    # --------------------------------------------------------
    # PLOT: Species mass fractions along stagnation line
    # --------------------------------------------------------
    stag_species = [k for k in MCFD_SPECIES
                    if k in volume_mesh.point_data]

    if stag_species:
        fig, ax = plt.subplots(figsize=(9, 6))

        for i, sp_key in enumerate(stag_species):
            sp_vals = volume_mesh.point_data[sp_key][stag_mask][sort_idx]
            color   = SPECIES_COLORS[i % len(SPECIES_COLORS)]
            ax.plot(dist_s, sp_vals, color=color,
                    linewidth=1.4, label=sp_key)

        ax.set_xlabel("Distance from wall (m)")
        ax.set_ylabel("Mass fraction  Y  (-)")
        ax.set_title("Species along Stagnation Line")
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)

        out = os.path.join(cycle_folder, "stagnation_species.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: {}".format(out))
    else:
        print("INFO: No species fields (Y_*) — skipping stagnation species plot.")

# ============================================================
# 7b. First cell height for EVERY surface cell
#     Wall-normal spacing = distance from each surface-cell centroid
#     to the nearest off-wall volume node, projected onto the local
#     outward normal. Robust to unstructured / adapted meshes.
# ============================================================
def compute_first_cell_heights(surf_mesh, vol_pts, k_neighbors=8):
    """
    First cell height evaluated at each surface NODE, then averaged onto each
    surface cell. Returns (theta_per_cell, height_per_cell).

    Method: for a wall node the near-wall normal spacing is far smaller than
    the tangential spacing, so the k NEAREST off-wall volume nodes are its
    first-layer neighbours. The wall-normal distance is the smallest positive
    projection of those neighbours onto the local outward normal. Restricting
    to the k nearest neighbours (instead of the whole domain) is what makes
    this robust and anisotropy-proof; the projection removes any tangential
    component so the value is a true wall-normal spacing.
    """
    surf_xy = surf_mesh.points[:, :2]
    sr      = np.sqrt(surf_xy[:, 0]**2 + surf_xy[:, 1]**2)
    sr      = np.where(sr == 0.0, 1e-30, sr)
    n_node  = surf_xy / sr[:, None]                  # outward radial normals

    vol_xy = vol_pts[:, :2]
    r_vol  = np.sqrt(vol_xy[:, 0]**2 + vol_xy[:, 1]**2)
    r_wall = r_vol.min()
    off    = r_vol > r_wall * (1.0 + 1e-6)           # exclude on-wall nodes
    off_xy = vol_xy[off]

    def first_height(vec, n):
        nd = vec @ n                                 # wall-normal projections
        nd = nd[nd > 0.0]
        return nd.min() if nd.size else np.nan

    h_node = np.full(surf_xy.shape[0], np.nan)
    try:
        from scipy.spatial import cKDTree
        tree   = cKDTree(off_xy)
        k      = min(k_neighbors, off_xy.shape[0])
        _, nbr = tree.query(surf_xy, k=k)
        nbr    = np.atleast_2d(nbr)
        for i in range(surf_xy.shape[0]):
            h_node[i] = first_height(off_xy[nbr[i]] - surf_xy[i], n_node[i])
    except ImportError:
        # Pure-numpy fallback (no scipy): manual k-nearest per node
        for i in range(surf_xy.shape[0]):
            d2  = ((off_xy - surf_xy[i])**2).sum(axis=1)
            sel = np.argpartition(d2, min(k_neighbors, off_xy.shape[0]-1))[:k_neighbors]
            h_node[i] = first_height(off_xy[sel] - surf_xy[i], n_node[i])

    # Average node heights onto each surface cell
    n_cells = surf_mesh.n_cells
    h_cell  = np.full(n_cells, np.nan)
    ctr_xy  = surf_mesh.cell_centers().points[:, :2]
    for ci in range(n_cells):
        ids  = np.asarray(surf_mesh.get_cell(ci).point_ids)
        vals = h_node[ids]
        if np.any(np.isfinite(vals)):
            h_cell[ci] = np.nanmean(vals)

    theta_cell = np.degrees(np.arctan2(ctr_xy[:, 1], -ctr_xy[:, 0]))
    return theta_cell, h_cell

theta_cells, first_cell_h_per_cell = compute_first_cell_heights(mesh, vol_coords)

valid = np.isfinite(first_cell_h_per_cell)
if np.any(valid):
    first_cell_h_min  = float(np.nanmin(first_cell_h_per_cell))
    first_cell_h_max  = float(np.nanmax(first_cell_h_per_cell))
    first_cell_h_mean = float(np.nanmean(first_cell_h_per_cell))
    print("--------------------------------------------------")
    print("First cell height  min : {:.4e} m".format(first_cell_h_min))
    print("First cell height  max : {:.4e} m".format(first_cell_h_max))
    print("First cell height mean : {:.4e} m".format(first_cell_h_mean))
    print("--------------------------------------------------")
else:
    first_cell_h_min = float('nan')
    print("WARNING: first cell height could not be computed for any surface cell.")

# ------------------------------------------------------------
# Bar-graph "contour": first cell height of every surface cell,
# ordered around the body by theta and colour-mapped by value.
# ------------------------------------------------------------
try:
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    order = np.argsort(theta_cells[valid])
    th    = theta_cells[valid][order]
    hvals = first_cell_h_per_cell[valid][order]

    norm   = Normalize(vmin=hvals.min(), vmax=hvals.max())
    colors = cm.turbo(norm(hvals))

    # Bar width from the median angular spacing so bars sit side by side
    if th.size > 1:
        dth   = np.diff(th)
        dth   = dth[dth > 0]
        width = (np.median(dth) * 0.95) if dth.size else 1.0
    else:
        width = 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(th, hvals, width=width, color=colors,
           edgecolor='none', align='center')

    sm = cm.ScalarMappable(norm=norm, cmap='turbo')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("First cell height (m)")

    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("First cell height (m)")
    ax.set_title("First Cell Height per Surface Cell")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    out = os.path.join(cycle_folder, "first_cell_height_bar.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(out))
except Exception as e:
    print("WARNING: first cell height bar plot failed ({})".format(e))

# ============================================================
# 7c. Quad layer spacing along the symmetry line (mcfdsol_bc2.vtk)
#     The quad boundary-layer cells are concentric shells about the body,
#     so every node on a given layer shares the same radius. We therefore
#     collapse same-radius nodes into one layer and take the spacing between
#     consecutive shells. Using r (not |x|) and grouping by shell avoids the
#     tangential smearing / sawtooth that a |y|-band + np.diff produces.
# ============================================================
def get_quad_spacing(vtk_file, layer_tol=1e-9):
    """
    Wall-normal spacing of the concentric quad layers.

    layer_tol only needs to be larger than floating-point / intra-shell noise
    (nodes on one layer share a radius) and smaller than the first-cell height.
    1e-9 m is safe for micron-scale first cells.
    """
    qmesh = pv.read(vtk_file)

    # Prefer quad cells if the file is a mixed mesh; else use all points.
    pts = qmesh.points
    try:
        ctypes = qmesh.celltypes
        if np.any(ctypes == 9):                       # VTK_QUAD present
            pts = qmesh.extract_cells(ctypes == 9).points
    except Exception:
        pass

    if pts.shape[0] < 2:
        return np.array([]), np.array([])

    r     = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)      # true radial distance
    dist  = np.sort(r - r.min())

    # Group concentric shells: a new layer starts where the radial gap to the
    # previous node exceeds layer_tol; each layer = mean radius of its nodes.
    brk     = np.where(np.diff(dist) > layer_tol)[0] + 1
    layer_r = np.array([g.mean() for g in np.split(dist, brk)])

    if layer_r.size < 2:
        return np.array([]), np.array([])
    return layer_r[:-1], np.diff(layer_r)

n_quad_layers = None
try:
    bc2_file = os.path.join(os.path.dirname(vtk_current), "mcfdsol_bc2.vtk")
    if not os.path.exists(bc2_file):
        print("INFO: symmetry file not found ({}) — skipping quad spacing."
              .format(bc2_file))
    else:
        qdist, qspacing = get_quad_spacing(bc2_file)
        if qspacing.size == 0:
            print("WARNING: no quad layers extracted from {}".format(bc2_file))
        else:
            n_quad_layers = int(qspacing.size + 1)
            print("Quad layers found     : {}".format(n_quad_layers))
            print("First quad layer h    : {:.4e} m".format(qspacing[0]))

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(qdist, qspacing, 'o-', linewidth=1.5,
                    label="{} ({:.3e} m)".format(cycle_label, qspacing[0]))
            ax.set_xlabel("Distance from wall (m)")
            ax.set_ylabel("Quad layer spacing (m)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(title="Case")
            out = os.path.join(cycle_folder, "quad_spacing.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved: {}".format(out))
except Exception as e:
    print("WARNING: quad spacing plot failed ({})".format(e))

# ============================================================
# 8. Mesh visualisation with statistics table
# ============================================================
try:
    # --------------------------------------------------------
    # Mesh statistics
    # --------------------------------------------------------
    cyl_r_stat      = all_r_vol.min()
    n_nodes_total   = volume_mesh.n_points
    n_cells_total   = volume_mesh.n_cells
    n_nodes_surface = mesh.n_points
    n_cells_surface = mesh.n_cells

    vtk_type_map = {
        5: "Triangle", 9: "Quad",
        10: "Tetrahedron", 12: "Hexahedron",
        13: "Wedge", 14: "Pyramid"
    }
    unique_types, type_counts = np.unique(volume_mesh.celltypes, return_counts=True)

    # First cell height is now computed per surface cell in section 7b.
    # Here we just reuse the summary (minimum) value for the statistics table.

    # --------------------------------------------------------
    # Build polygon lists in ONE pass using vectorised numpy
    # --------------------------------------------------------
    raw_cells  = volume_mesh.cells          # flat int array
    cell_types = volume_mesh.celltypes      # one entry per cell

    # Split into quads and triangles using numpy where possible.
    # pyvista guarantees the flat layout [n, i0, i1, ..., n, i0, i1, ...]
    # We walk it once in Python but build the arrays with numpy slicing.
    quad_polys = []
    tri_polys  = []
    offset = 0
    n_cells = volume_mesh.n_cells

    # Pre-cache x, y as C-contiguous for fast fancy indexing
    x_c = np.ascontiguousarray(x)
    y_c = np.ascontiguousarray(y)

    while offset < len(raw_cells):
        npts = int(raw_cells[offset])
        conn = raw_cells[offset + 1: offset + 1 + npts]
        poly = np.stack((x_c[conn], y_c[conn]), axis=1)
        if npts == 4:
            quad_polys.append(poly)
        elif npts == 3:
            tri_polys.append(poly)
        offset += npts + 1

    def make_poly_collection(polys, lw=0.35):
        return PolyCollection(
            polys,
            facecolors='none',
            edgecolors='black',
            linewidths=lw,
            clip_on=True,           # honour the axes clip box
        )

    def draw_mesh_on(ax, lw=0.35):
        """Add mesh polygons to ax and clip them to the axes patch."""
        collections = []
        if quad_polys:
            c = make_poly_collection(quad_polys, lw)
            ax.add_collection(c)
            collections.append(c)
        if tri_polys:
            c = make_poly_collection(tri_polys, lw)
            ax.add_collection(c)
            collections.append(c)
        # Clip to axes bounding box AFTER limits are set by the caller.
        # We return the collections so the caller can re-clip if needed.
        return collections

    def clip_collections_to_ax(ax, collections):
        """Must be called AFTER ax xlim/ylim are finalised."""
        for c in collections:
            c.set_clip_path(ax.patch)

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------
    fig = plt.figure(figsize=(22, 12))
    gs  = fig.add_gridspec(1, 2,width_ratios=[2.1, 0.6], wspace=0.2)
    ax_mesh  = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    main_colls = draw_mesh_on(ax_mesh, lw=0.2)

    # --------------------------------------------------------
    # Zoomed insets  (reuse the same polygon lists)
    # --------------------------------------------------------
    inset_specs = [
        dict(pos=[0.57, 0.15, 0.13, 0.25],
             xlim=(-0.046, -0.0449), ylim=(-0.0003, 0.003),   lw=0.2),
        dict(pos=[0.57, 0.70, 0.15, 0.20],
             xlim=(-0.0317, -0.0315), ylim=(0.032, 0.0325),   lw=0.2),
        dict(pos=[0.20, 0.70, 0.15, 0.20],
             xlim=(-0.0421, -0.042),  ylim=(0.04499, 0.0455), lw=0.2),
    ]

    for spec in inset_specs:
        axins = fig.add_axes(spec['pos'])
        ins_colls = draw_mesh_on(axins, lw=spec['lw'])
        axins.set_xlim(*spec['xlim'])
        axins.set_ylim(*spec['ylim'])
        # Clip AFTER limits are set so the patch is correctly sized
        clip_collections_to_ax(axins, ins_colls)
        axins.set_xticks([]);  axins.set_yticks([])
        mark_inset(ax_mesh, axins, loc1=2, loc2=4,
                   fc="none", ec="red", lw=0.9)

    # --------------------------------------------------------
    # Boundary scatter
    # --------------------------------------------------------
    ax_mesh.scatter(coords[:, 0], coords[:, 1], s=0.02, color='red', zorder=3)
    ax_mesh.set_aspect('equal')
    ax_mesh.set_xlim(x.min(), x.max())
    ax_mesh.set_ylim(y.min(), y.max())
    # Clip main axes AFTER limits and aspect are finalised
    clip_collections_to_ax(ax_mesh, main_colls)
    ax_mesh.set_title("Mesh", fontsize=14, fontweight='bold')
    ax_mesh.axis('off')

    # --------------------------------------------------------
    # Statistics table
    # --------------------------------------------------------
    table_data = [
        ["GLOBAL MESH", "", ""],
        ["Total nodes",    "{:,}".format(n_nodes_total),   ""],
        ["Total cells",    "{:,}".format(n_cells_total),   ""],
        ["Surface nodes",  "{:,}".format(n_nodes_surface), ""],
        ["Surface cells",  "{:,}".format(n_cells_surface), ""],
        ["", "", ""],
        ["ELEMENT TYPES", "", ""],
    ]

    for t_id, count in zip(unique_types, type_counts):
        t_name = vtk_type_map.get(int(t_id), "Type {}".format(t_id))
        table_data.append([t_name, "{:,}".format(count), ""])
        
    n_quad_layers = next(count for t_id, count in zip(unique_types, type_counts)
                         if int(t_id) == 9) // n_nodes_surface

    table_data.append(["Quad BL layers", "{:d}".format(n_quad_layers), ""])

    table_data += [
        ["", "", ""],
        ["MESH SPACING", "", ""],
        ["First cell height (min)", "{:.4e} m".format(first_cell_h_min), ""],
        ["", "", ""],
        ["PEAK HEATING", "", ""],
        ["Max Qdot",         "{:.3e} W/m²".format(max_qdot),          ""],
        ["Theta @ Max Qdot", "{:.2f} deg".format(theta_max_qdot),     ""],
        ["Max Pressure",     "{:.3e} Pa".format(max_pressure),         ""],
        ["Theta @ Max P",    "{:.2f} deg".format(theta_max_pressure),  ""],
        ["", "", ""],
        ["Y-PLUS", "", ""],
        ["Max Y+",      "{:.4f}".format(np.max(Yplus)),  ""],
        ["Y+ at 0 deg", "{:.4f}".format(yplus_at_0),    ""],
        ["Y+ at 90 deg","{:.4f}".format(yplus_at_90),   ""],
    ]

    ax_table.axis('off')
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Parameter", "Value", ""],
        cellLoc='left', loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Header row
    for c in range(3):
        tbl[(0, c)].set_facecolor('#2c3e50')
        tbl[(0, c)].set_text_props(color='white', fontweight='bold')

    section_colors = {
        "GLOBAL MESH":   '#1a5276',
        "ELEMENT TYPES": '#8e44ad',
        "MESH SPACING":  '#145a32',
        "PEAK HEATING":  '#b03a2e',
        "Y-PLUS":        '#4a235a',
    }

    for row_idx, row in enumerate(table_data, start=1):
        label = row[0].strip()
        if label in section_colors:
            for c in range(3):
                tbl[(row_idx, c)].set_facecolor(section_colors[label])
                tbl[(row_idx, c)].set_text_props(color='white', fontweight='bold')
        elif label == "":
            for c in range(3):
                tbl[(row_idx, c)].set_facecolor('#ffffff')
                tbl[(row_idx, c)].set_height(0.01)
        else:
            shade = '#eaf4fb' if row_idx % 2 == 0 else '#fdfefe'
            for c in range(3):
                tbl[(row_idx, c)].set_facecolor(shade)

    for row_idx in range(len(table_data) + 1):
        tbl[(row_idx, 2)].set_visible(False)

    tbl.auto_set_column_width([0, 1])
    ax_table.set_title("Mesh Statistics", fontsize=11, fontweight='bold', pad=8)

    # --------------------------------------------------------
    # Save outputs — native vector PDF, no Inkscape needed
    # --------------------------------------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib as mpl

    with mpl.rc_context({
        "pdf.fonttype":           42,    # embed TrueType (not bitmap Type 3)
        "ps.fonttype":            42,
        "path.simplify":          False, # keep every polygon vertex
        "path.simplify_threshold": 0.0,
        "figure.dpi":             300,   # raster fallback resolution
    }):
        mesh_plot_pdf = os.path.join(cycle_folder, "mesh_vector.pdf")
        with PdfPages(mesh_plot_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
            d = pdf.infodict()
            d["Title"]   = "Mesh - {}".format(cycle_label)
            d["Subject"] = "CFD mesh visualisation"
        print("Saved:", mesh_plot_pdf)

        mesh_plot_png = os.path.join(cycle_folder, "mesh_vector.png")
        fig.savefig(mesh_plot_png, dpi=300, bbox_inches="tight")
        print("Saved:", mesh_plot_png)

    plt.close(fig)

except Exception as e:
    print("WARNING: Vector mesh plot failed ({})".format(e))

print("Analysis complete for cycle: {}".format(cycle_label))
