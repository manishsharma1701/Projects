import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plot

# Parameters
Q = 50    # Source/sink strength
U = 50    # Uniform flow velocity
a = 1.0    # Half the distance between source and sink

# Range for Y
Y_values = np.linspace(-2*a, 2 * a, 50)

# Function for psi = 0 stream function
def psi(X, Y, Q, U, a):
    return U * Y + (Q / (2 * np.pi)) * np.arctan2(-2 * a * Y, X**2 + Y**2 - a**2)

# Solve X for psi=0 (both positive and negative X sides)
X_values_pos = []
X_values_neg = []

for Y in Y_values:
    # Solve for positive branch
    X_pos = fsolve(psi, x0=a, args=(Y, Q, U, a))[0]
    X_values_pos.append(X_pos)

    # Solve for negative branch
    X_neg = fsolve(psi, x0=-a, args=(Y, Q, U, a))[0]
    X_values_neg.append(X_neg)

# Convert to arrays
X_values_pos = np.array(X_values_pos)
X_values_neg = np.array(X_values_neg)

# Plotting
plot.figure(figsize=(8, 6))
plot.plot(X_values_pos, Y_values, label=r'$\psi=0$ (Positive X branch)', color='blue')
plot.plot(X_values_neg, Y_values, label=r'$\psi=0$ (Negative X branch)', color='red')
plot.title('Streamline $\psi=0$ for Source-Sink Flow (Oval Shape)')
plot.xlabel('X')
plot.ylabel('Y')
plot.axhline(0, color='black', linewidth=0.5, linestyle='--')
plot.axvline(0, color='black', linewidth=0.5, linestyle='--')
plot.grid(True)
plot.legend()
plot.axis('equal')
plot.show()

# Define the stream function
def stream_function(X, Y, Q, U, a):
    ux=U + (Q/(2 * np.pi))*(((-2 * a * (x_0**2))+(2* a * (Y**2))+(2 * (a**3)))/(((x_0**2) + (Y**2) - (a**2))**2) + (4 * (a**2) * (Y**2)))
    vx=(Q/np.pi)*((a*x_0*Y)/(((x_0**2) + (Y**2) - (a**2))**2 + 4 * (a**2) * (Y**2)))
    cp=1-(((ux**2) + (vx**2))/(U**2))
    return cp,x_0

# Compute the stream function on the grid
[Cp,x_0] = stream_function(X_values_pos, Y, Q, U, a)




# Plot Cp vs x along the centerline
plot.figure(figsize=(10, 6))
plot.plot(X_values_pos, Cp, label='Pressure Coefficient (Cp) along y=0')
plot.title('Pressure Coefficient Distribution (Cp) along the Centerline')
plot.xlabel('x')
plot.ylabel('Cp')
plot.grid(True)
plot.show()

theta=np.linspace(0,np.pi,50)
cp=1-4*(np.sin(theta))**2
plot.plot(theta,cp)
plot.show()


