import numpy as np
import matplotlib.pyplot as plt

def visualize_metric_mapping():
    # 1. SETUP: Define a Metric Matrix M
    # We construct M using eigenvalues (h^-2) and a rotation angle (theta)
    # Recall: h is the "target size" in that direction.
    
    h1 = 0.5  # Size in direction 1 (small size = large eigenvalue = compressed)
    h2 = 2.0  # Size in direction 2 (large size = small eigenvalue = stretched)
    theta = np.radians(30) # Rotate the metric by 30 degrees
    
    # Construct Rotation Matrix R
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Construct Eigenvalue Matrix Lambda (values are 1/h^2)
    # M = R * Lambda * R^T
    Lambda = np.diag([1/h1**2, 1/h2**2]) 
    M = R @ Lambda @ R.T
    
    print(f"Metric Tensor M:\n{M}")

    # 2. CREATE THE IDENTITY UNIT BALL (A Circle)
    # Generate 100 points around a circle of radius 1
    t = np.linspace(0, 2*np.pi, 100)
    circle_points = np.array([np.cos(t), np.sin(t)]) # Shape (2, 100)

    # 3. COMPUTE THE MAPPING OPERATOR: M^(-1/2)
    # Decompose M to get eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(M)
    print(eig_vals)
    
    # Calculate M^(-1/2) = R * Lambda^(-1/2) * R^T
    # Note: eigenvalues of M are 1/h^2, so lambda^(-1/2) is just h
    Lambda_neg_half = np.diag(1.0 / np.sqrt(eig_vals))
    M_neg_half = eig_vecs @ Lambda_neg_half @ eig_vecs.T
    
    # 4. APPLY THE MAPPING
    # This transforms the circle into the metric ellipsoid
    # formula: x_mapped = M^(-1/2) * x
    ellipsoid_points = M_neg_half @ circle_points

    # 5. PLOTTING
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Identity Unit Ball (Circle)
    ax.plot(circle_points[0, :], circle_points[1, :], 'b--', label='Identity Unit Ball $\mathcal{B}_{\mathcal{I}}$')
    
    # Plot Metric Unit Ball (Ellipsoid)
    ax.plot(ellipsoid_points[0, :], ellipsoid_points[1, :], 'r-', linewidth=2, label='Metric Unit Ball $\mathcal{B}_{\mathcal{M}}$')
    
    # Plot Eigenvectors (The principal axes)
    origin = [0, 0]
    # Scaled by h (size) for visualization
    # Axis 1
    ax.quiver(*origin, *(eig_vecs[:, 0] * h1), color='g', scale=1, scale_units='xy', angles='xy', label=f'Axis 1 ($h_1={h1}$)')
    # Axis 2
    ax.quiver(*origin, *(eig_vecs[:, 1] * h2), color='purple', scale=1, scale_units='xy', angles='xy', label=f'Axis 2 ($h_2={h2}$)')

    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"Natural Metric Mapping\n$\mathcal{{M}}^{{-1/2}}$ maps Circle $\\to$ Ellipse")
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    plt.show()

if __name__ == "__main__":
    visualize_metric_mapping()