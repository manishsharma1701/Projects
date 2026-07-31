import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. LOAD MESH ─────────────────────────────────────────────────────────────
mesh = pv.read('mcfdsol.vtk')  # change to your filename

print("Available arrays:", mesh.array_names)
print("Number of points:", mesh.n_points)
print("Number of cells:", mesh.n_cells)

# ─── 2. GET PRESSURE ──────────────────────────────────────────────────────────
# Your array is called 'P'
pressure_name = 'P'

# Convert cell data to point data if needed
if pressure_name in mesh.cell_data.keys():
    mesh = mesh.cell_data_to_point_data()
    print("Converted cell data to point data")

p = mesh.point_data[pressure_name]
print(f"Pressure range: {p.min():.2f} to {p.max():.2f} Pa")

# Also get Qdot for correlation analysis later
qdot = mesh.point_data.get('Qdot', None)

# ─── 3. BUILD POINT CONNECTIVITY ─────────────────────────────────────────────
# FIXED: Handle mixed cell types (triangles + quads) correctly
print("Building neighbour connectivity...")

n_points = mesh.n_points
n_cells  = mesh.n_cells

# Build point-to-point neighbour map directly from cells
# Works for ANY mixed mesh - triangles, quads, mixed
point_neighbors = [set() for _ in range(n_points)]

for cell_idx in range(n_cells):
    cell = mesh.get_cell(cell_idx)
    cell_point_ids = list(cell.point_ids)
    # Every point in this cell is a neighbour of every other point in the cell
    for i, pt_i in enumerate(cell_point_ids):
        for pt_j in cell_point_ids:
            if pt_i != pt_j:
                point_neighbors[pt_i].add(pt_j)

# Report mesh statistics
n_neighbors = [len(s) for s in point_neighbors]
print(f"Min neighbours per point:     {min(n_neighbors)}")
print(f"Max neighbours per point:     {max(n_neighbors)}")
print(f"Average neighbours per point: {np.mean(n_neighbors):.1f}")

# ─── 4. COMPUTE p_d — NORMAL MODE (one level) ────────────────────────────────
print("\nComputing p_d (Normal mode - 1 level)...")

p_d_normal = np.zeros(n_points)
p_max_arr  = np.zeros(n_points)
p_min_arr  = np.zeros(n_points)

for i in range(n_points):
    neighbors = list(point_neighbors[i])
    all_pts   = [i] + neighbors
    p_local   = p[all_pts]

    pmax = p_local.max()
    pmin = p_local.min()
    denom = pmax + pmin

    p_max_arr[i] = pmax
    p_min_arr[i] = pmin

    if denom > 1e-10:
        p_d_normal[i] = abs(pmax - pmin) / denom
    else:
        p_d_normal[i] = 0.0

print(f"p_d (Normal) range: "
      f"{p_d_normal.min():.6f} to {p_d_normal.max():.4f}")

# ─── 5. COMPUTE p_d — AGGRESSIVE MODE (two levels) ───────────────────────────
print("Computing p_d (Aggressive mode - 2 levels)...")

p_d_aggressive = np.zeros(n_points)

for i in range(n_points):
    level1 = point_neighbors[i]
    level2 = set()
    for j in level1:
        level2.update(point_neighbors[j])

    all_pts = list({i} | level1 | level2)
    p_local = p[all_pts]

    pmax  = p_local.max()
    pmin  = p_local.min()
    denom = pmax + pmin

    if denom > 1e-10:
        p_d_aggressive[i] = abs(pmax - pmin) / denom
    else:
        p_d_aggressive[i] = 0.0

print(f"p_d (Aggressive) range: "
      f"{p_d_aggressive.min():.6f} to {p_d_aggressive.max():.4f}")

# ─── 6. ADD ARRAYS TO MESH AND SAVE ──────────────────────────────────────────
mesh.point_data['p_d_normal']      = p_d_normal
mesh.point_data['p_d_aggressive']  = p_d_aggressive
mesh.point_data['p_max_neighbors'] = p_max_arr
mesh.point_data['p_min_neighbors'] = p_min_arr

mesh.save('solution_with_pd.vtk')
print("\nSaved: solution_with_pd.vtk")
print("Load this in ParaView and color by p_d_normal or p_d_aggressive")

# ─── 7. SWITCH ACTIVATION ANALYSIS ───────────────────────────────────────────
thresholds = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

print("\n=== Switch Activation Analysis ===")
print(f"{'Threshold':>10} | {'Normal (%)':>12} | {'Aggressive (%)':>15}")
print("-" * 45)
for t in thresholds:
    pct_n = np.sum(p_d_normal     > t) / n_points * 100
    pct_a = np.sum(p_d_aggressive > t) / n_points * 100
    marker = " ← your current" if abs(t - 0.0001) < 1e-6 else ""
    print(f"{t:>10.4f} | {pct_n:>11.1f}% | {pct_a:>14.1f}%{marker}")

# ─── 8. REGIONAL ANALYSIS ─────────────────────────────────────────────────────
coords = mesh.points
x = coords[:, 0]
y = coords[:, 1]
R = 0.045  # cylinder radius in metres
r = np.sqrt(x**2 + y**2)

# Stagnation angle: flow in +x direction, stagnation at (-R, 0)
theta = np.degrees(np.arctan2(y, -x))

# Region masks
wall_mask     = (r >= R * 0.95) & (r <= R * 1.10)   # near wall / BL
shock_mask    = (r >= R * 1.15) & (r <= R * 1.40)   # shock layer
free_mask     = r > R * 3.0                          # freestream
expansion_mask = wall_mask & (np.abs(theta) > 30)   # shoulder expansion

print("\n=== p_d Statistics by Region ===")
regions = [
    ('Wall / BL (all)',        wall_mask),
    ('Shoulder expansion BL',  expansion_mask),
    ('Shock layer',            shock_mask),
    ('Freestream',             free_mask),
]

for name, mask in regions:
    if mask.sum() == 0:
        print(f"\n{name}: no points found — adjust r/theta thresholds")
        continue
    pd = p_d_normal[mask]
    print(f"\n{name} ({mask.sum()} points):")
    print(f"  p_d mean            = {pd.mean():.5f}")
    print(f"  p_d max             = {pd.max():.5f}")
    print(f"  p_d 95th percentile = {np.percentile(pd, 95):.5f}")
    print(f"  Active at 0.0001:   "
          f"{(pd > 0.0001).mean()*100:.1f}% of points")
    print(f"  Active at 0.05:     "
          f"{(pd > 0.05).mean()*100:.1f}% of points")

# ─── 9. FIND OPTIMAL THRESHOLD ────────────────────────────────────────────────
print("\n=== Optimal Threshold Recommendation ===")

if wall_mask.sum() > 0 and shock_mask.sum() > 0:
    pd_bl_95     = np.percentile(p_d_normal[wall_mask], 95)
    pd_bl_max    = p_d_normal[wall_mask].max()
    pd_shock_min = p_d_normal[shock_mask].min()
    pd_shock_5   = np.percentile(p_d_normal[shock_mask], 5)

    print(f"BL p_d 95th percentile : {pd_bl_95:.5f}")
    print(f"BL p_d maximum         : {pd_bl_max:.5f}")
    print(f"Shock p_d 5th percentile: {pd_shock_5:.5f}")
    print(f"Shock p_d minimum       : {pd_shock_min:.5f}")

    if pd_bl_max < pd_shock_min:
        optimal = (pd_bl_max + pd_shock_min) / 2.0
        print(f"\n✓ Clean separation exists between BL and shock.")
        print(f"  Optimal threshold = {optimal:.5f}")
    else:
        # Suggest threshold that keeps <5% of BL active
        for t in np.linspace(0.0001, 0.5, 10000):
            if np.mean(p_d_normal[wall_mask] > t) < 0.05:
                print(f"\n✗ BL and shock p_d overlap.")
                print(f"  Suggested threshold (keeps <5% BL active) = {t:.5f}")
                print(f"  Consider switching to Normal detection mode")
                break

# ─── 10. PLOTS ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pressure Switch Analysis — Run 619', fontsize=13)

# ── Plot 1: p_d histogram by region ──
ax = axes[0, 0]
colors = {'Wall/BL': 'blue',
          'Expansion BL': 'purple',
          'Shock layer': 'red',
          'Freestream': 'green'}
for (name, mask), color in zip(
        [('Wall/BL', wall_mask),
         ('Expansion BL', expansion_mask),
         ('Shock layer', shock_mask),
         ('Freestream', free_mask)], colors.values()):
    if mask.sum() > 0:
        ax.hist(p_d_normal[mask], bins=60, alpha=0.5,
                label=name, color=color, density=True)
ax.axvline(0.0001, color='black', ls='--', lw=1.5,
           label='Threshold=0.0001 (current)')
ax.axvline(0.05, color='orange', ls='--', lw=1.5,
           label='Threshold=0.05')
ax.set_xlabel('$p_d$')
ax.set_ylabel('Probability density')
ax.set_title('$p_d$ Distribution by Region')
ax.legend(fontsize=7)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3)

# ── Plot 2: % points active vs threshold ──
ax = axes[0, 1]
t_range = np.logspace(-5, 0, 200)
for name, mask, ls in [('All cells', np.ones(n_points, bool), '-'),
                        ('Wall/BL only', wall_mask, '--'),
                        ('Shock only', shock_mask, ':')]:
    if mask.sum() > 0:
        pct = [np.sum(p_d_normal[mask] > t) / mask.sum() * 100
               for t in t_range]
        ax.semilogx(t_range, pct, ls, label=name, lw=2)
ax.axvline(0.0001, color='black', ls=':', lw=1.5,
           label='Current threshold')
ax.axvline(0.05, color='orange', ls=':', lw=1.5,
           label='Recommended')
ax.set_xlabel('Pressure switch threshold')
ax.set_ylabel('% Points with switch active')
ax.set_title('Switch Activation vs Threshold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# ── Plot 3: p_d along wall surface vs theta ──
ax = axes[1, 0]
wall_idx = np.where(wall_mask)[0]
if len(wall_idx) > 0:
    theta_w = theta[wall_idx]
    pd_w_n  = p_d_normal[wall_idx]
    pd_w_a  = p_d_aggressive[wall_idx]
    sort_i  = np.argsort(theta_w)

    ax.plot(theta_w[sort_i], pd_w_n[sort_i], 'b-',
            lw=2, label='Normal mode')
    ax.plot(theta_w[sort_i], pd_w_a[sort_i], 'r-',
            lw=2, label='Aggressive mode')
    ax.axhline(0.0001, color='black', ls='--', lw=1.5,
               label='Threshold 0.0001')
    ax.axhline(0.05, color='orange', ls='--', lw=1.5,
               label='Threshold 0.05')
    ax.fill_between(theta_w[sort_i], 0, pd_w_n[sort_i],
                    where=(pd_w_n[sort_i] > 0.0001),
                    alpha=0.25, color='red',
                    label='Active region (0.0001)')
    ax.set_xlabel(r'$\Theta$ [degrees from stagnation]')
    ax.set_ylabel('$p_d$ at wall/BL cells')
    ax.set_title('Switch Activation Along Wall Surface')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.3)

# ── Plot 4: Qdot vs p_d at wall (if Qdot available) ──
ax = axes[1, 1]
if qdot is not None and wall_mask.sum() > 0:
    q_wall = qdot[wall_idx]
    pd_wall = p_d_normal[wall_idx]
    sc = ax.scatter(pd_wall, q_wall / 1e6, c=theta_w,
                    cmap='RdBu_r', s=10, alpha=0.7,
                    vmin=-90, vmax=90)
    plt.colorbar(sc, ax=ax, label='Theta [deg]')
    ax.axvline(0.0001, color='black', ls='--', lw=1.5,
               label='Threshold 0.0001')
    ax.axvline(0.05, color='orange', ls='--', lw=1.5,
               label='Threshold 0.05')
    ax.set_xlabel('$p_d$ at wall point')
    ax.set_ylabel('$\\dot{q}$ [MW/m²]')
    ax.set_title('$\\dot{q}$ vs $p_d$ at Wall — '
                 'Correlation Diagnostic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    print("\n=== Qdot Correlation with p_d ===")
    corr = np.corrcoef(pd_wall, q_wall)[0, 1]
    print(f"Pearson correlation (p_d vs Qdot at wall): {corr:.4f}")
    if corr > 0.3:
        print("⚠ Positive correlation: switch may be")
        print("  artificially inflating Qdot at wall")
    elif corr < -0.3:
        print("⚠ Negative correlation: switch may be")
        print("  suppressing Qdot in high-p_d regions")
    else:
        print("✓ Low correlation: switch not strongly")
        print("  affecting local Qdot")
else:
    ax.text(0.5, 0.5, 'Qdot not available\nor no wall points found',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Qdot vs p_d (unavailable)')

plt.tight_layout()
plt.savefig('pd_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: pd_analysis.png")