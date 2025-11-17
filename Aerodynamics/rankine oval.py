# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:58:52 2025

@author: Leovo
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parameters
Q = 100    # Source/sink strength
U = 100    # Uniform flow velocity
a = 0.1 # Half the distance between source and sink

# Range for Y
Y_values = np.linspace(-2*a,2* a, 1000)

# Function for psi = 0 stream function
def psi(X, Y, Q, U, a):
    return U * Y + (Q / (2 * np.pi)) * np.arctan2(-2 * a * Y, X**2 + Y**2 - a**2)

# Solve X for psi=0 (both positive and negative X sides)
X_values_pos = []
X_values_neg = []

for Y in Y_values:
    # Solve for positive branch
    X_pos = fsolve(psi, x0=5, args=(Y, Q, U, a))[0]
    X_values_pos.append(X_pos)
    

    # Solve for negative branch
    X_neg = fsolve(psi, x0=-5, args=(Y, Q, U, a))[0]
    X_values_neg.append(X_neg)
    

# Convert to arrays
X_values_pos = np.array(X_values_pos)
X_values_neg = np.array(X_values_neg)

# Plotting the streamline
plt.figure(figsize=(8, 6))
plt.plot(X_values_pos, Y_values, label=r'$\psi=0$ (Positive X branch)', color='blue')
plt.plot(X_values_neg, Y_values, label=r'$\psi=0$ (Negative X branch)', color='red')
plt.title(f'Streamline $\\psi=0$ for Source-Sink Flow (Oval Shape, a={a})')
plt.xlabel('X')
plt.ylabel('Y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.legend()
plt.show()

# Velocity field function
def velocity(X, Y, Q, U, a):
    """Compute velocity components (u, v) in source-sink flow"""
    r2 = X**2 + Y**2 - a**2
    denom = (r2**2 + 4 * a**2 * Y**2)
    
    if np.any(denom == 0):  # Avoid division by zero
        denom += 1e-10  

    ux = U + (Q / (2 * np.pi)) * ((-2 * a * X**2 + 2 * a * Y**2 + 2 * a**3) / denom)
    vy = (Q / np.pi) * ((a * X * Y) / denom)
    
    return ux, vy

# Compute velocity and Cp for both positive and negative branches
ux_pos, vy_pos = velocity(X_values_pos, Y_values, Q, U, a)
vel_mag_pos = np.sqrt(ux_pos**2 + vy_pos**2)
Cp_pos = 1 - (ux_pos**2 + vy_pos**2) / U**2  # Pressure coefficient for positive X

ux_neg, vy_neg = velocity(X_values_neg, Y_values, Q, U, a)
vel_mag_neg = np.sqrt(ux_neg**2 + vy_neg**2)
Cp_neg = 1 - (ux_neg**2 + vy_neg**2) / U**2  # Pressure coefficient for negative X

# Plot Cp vs X along the streamline for both branches
plt.figure(figsize=(10, 6))
plt.plot(X_values_pos, Cp_pos, label='Cp (Positive X branch)', color='blue')
plt.plot(X_values_neg, Cp_neg, label='Cp (Negative X branch)', color='red')
plt.title(f'Pressure Coefficient Distribution (Cp) at a={a}')
plt.xlabel('X')
plt.ylabel('Cp')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(X_values_pos, vel_mag_pos, label='Velocity Magnitude (Positive X)', color='blue')
plt.plot(X_values_neg, vel_mag_neg, label='Velocity Magnitude (Negative X)', color='red')

plt.title(f'Velocity Magnitude Along $\psi=0$ Streamline (Rankine Oval) at a={a}')
plt.xlabel('X')
plt.ylabel('Velocity Magnitude')
plt.grid(True)
plt.legend()
plt.show()
