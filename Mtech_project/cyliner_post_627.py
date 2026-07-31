import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
mesh = pv.read("case_627/manish_adapt/mcfdsol_bc1.vtk")
mesh2 = pv.read("case_627/manish_struct/mcfdsol_bc1.vtk")
experimental_Qdot = pd.read_csv("case_627/Exp_Qdot_627.csv")
experimental_P = pd.read_csv("case_627/Exp_P_627.csv")

# 2. Process Mesh 1 (Adapt)
coords = mesh.points
Qdot = mesh.point_data["Qdot"]
P = mesh.point_data["P"]
y_plus = mesh.point_data["Y_plus"]
theta_deg = np.degrees(np.arctan2(coords[:, 1], -coords[:, 0]))
idx = np.argsort(theta_deg)

# 3. Process Mesh 2 (Struct)
coords2 = mesh2.points
Qdot2 = mesh2.point_data["Qdot"]
P2 = mesh2.point_data["P"]
y_plus2 = mesh2.point_data["Y_plus"]
theta2_deg = np.degrees(np.arctan2(coords2[:, 1], -coords2[:, 0]))
idx2 = np.argsort(theta2_deg)

# --- PLOT 1: HEAT FLUX ---
plt.figure(figsize=(8, 5))
plt.plot(theta_deg[idx], Qdot[idx], '-k', label="manish_adapt")
plt.plot(theta2_deg[idx2], Qdot2[idx2], '--r', label="manish_struct")

# Experimental Qdot with 5% Error
exp_theta = experimental_Qdot.iloc[:, 0]
exp_val = experimental_Qdot.iloc[:, 2] * 4.83 * 10**6
q_error = exp_val * 0.05  # 5 percent error calculation

plt.errorbar(exp_theta, exp_val, yerr=q_error, fmt='^', 
             ecolor='blue', capsize=3, elinewidth=1, markeredgecolor='blue',
             label="experimental (±5%)")

plt.xlabel("Theta (deg)")
plt.ylabel("Qdot ($W/m^2$)")
plt.title("Heat Flux vs Theta (Case 627)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- PLOT 2: PRESSURE ---
plt.figure(figsize=(8, 5))
plt.plot(theta_deg[idx], P[idx], '-k', label="manish_adapt")
plt.plot(theta2_deg[idx2], P2[idx2], '--r', label="manish_struct")
plt.plot(experimental_P.iloc[:, 0], experimental_P.iloc[:, 2] * 68 * 10**3, "s", label="experimental")

plt.xlabel("Theta (deg)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure vs Theta (Case 627)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- PLOT 3: Y-PLUS ---
plt.figure(figsize=(8, 5))
# Added [idx] to fix the line order
plt.plot(theta_deg[idx], y_plus[idx], '-g', label="Y_plus (adapt)") 
plt.plot(theta2_deg[idx2], y_plus2[idx2], '--', label="Y_plus (struct)") 
plt.xlabel("Theta (deg)")
plt.ylabel("$y^{+}$")
plt.title("$y^{+}$ vs Theta (Case 627)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()