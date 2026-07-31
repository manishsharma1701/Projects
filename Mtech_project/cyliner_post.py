import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
mesh = pv.read("mcfdsol_bc1.vtk")
mesh2 = pv.read("manish_struct/mcfdsol_bc1.vtk")
mesh3 = pv.read("shraman_adapt/mcfdsol_bc1.vtk")
mesh4 = pv.read("manish_adapt/mcfdsol_bc1.vtk")
experimental_Qdot = pd.read_csv("Exp_Qdot.csv")
experimental_P = pd.read_csv("Experimental_Pressure_ratio.csv")

# 2. Extract and Process Data (Mesh 1 to 4)
def process_mesh(m):
    coords = m.points
    return (m.point_data["Qdot"], 
            m.point_data["P"], 
            np.degrees(np.arctan2(coords[:, 1], -coords[:, 0])))

Qdot1, P1, theta1 = process_mesh(mesh)
Qdot2, P2, theta2 = process_mesh(mesh2)
Qdot3, P3, theta3 = process_mesh(mesh3)
Qdot4, P4, theta4 = process_mesh(mesh4)

# Sorting indices for clean lines
idx1, idx2, idx3, idx4 = np.argsort(theta1), np.argsort(theta2), np.argsort(theta3), np.argsort(theta4)

# --- PLOT 1: HEAT FLUX (Case 619) ---
plt.figure(figsize=(10, 6))
plt.plot(theta1[idx1], Qdot1[idx1], '-k', label="shraman(struct)")
plt.plot(theta3[idx3], Qdot3[idx3], '-r', label="shraman(adapt)")
plt.plot(theta2[idx2], Qdot2[idx2], '-.b', label="manish_struct")
plt.plot(theta4[idx4], Qdot4[idx4], '--g', label="manish_adapt")

# NEW: Experimental Qdot with 5% Error
exp_q_x = experimental_Qdot.iloc[:, 0]
exp_q_y = experimental_Qdot.iloc[:, 2] * 7.60 * 10**6
q_error = exp_q_y * 0.05  # 5% uncertainty

plt.errorbar(exp_q_x, exp_q_y, yerr=q_error, fmt='^', 
             ecolor='blue', capsize=3, elinewidth=1, markeredgecolor='blue',
             label="experimental (±5%)")

plt.xlabel("Theta (deg)")
plt.ylabel("Qdot ($W/m^2$)")
plt.title("Heat Flux vs Theta (Case 619)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- PLOT 2: PRESSURE (Case 619) ---
plt.figure(figsize=(10, 6))
plt.plot(theta1[idx1], P1[idx1], '-k', label="shraman(struct)")
plt.plot(theta3[idx3], P3[idx3], '-r', label="shraman(adapt)")
plt.plot(theta2[idx2], P2[idx2], '-.b', label="manish_struct")
plt.plot(theta4[idx4], P4[idx4], '--g', label="manish_adapt")

# Experimental Pressure
plt.plot(experimental_P.iloc[:, 0], experimental_P.iloc[:, 2] * 51.9 * 10**3, "^", label="experimental")

plt.xlabel("Theta (deg)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure vs Theta (Case 619)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()