import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

plt.style.use('default')
df=pd.read_csv('n0012.csv')
#X=np.flip(df['X'].values/100)
X=df['X'].values
#Y=np.flip(df['Y'].values/100)
Y=df['Y'].values

N_orig = len(X)
m_orig=len(X)-1
print(f"Original number of points: {N_orig}")

def resample_airfoil_panels(X_orig, Y_orig, m_new):

    # Leading edge = minimum X
    le_index = np.argmin(X_orig)

    # --- Split using your ordering ---
    # Input order: TE → lower → LE → upper → TE
    lower_X = X_orig[:le_index+1]
    lower_Y = Y_orig[:le_index+1]

    upper_X = X_orig[le_index:]
    upper_Y = Y_orig[le_index:]

    # --- Cosine spacing in chord direction (0 → 1 → 0) ---
    N_surf = m_new // 2 + 1
    x_dist = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N_surf)))

    # Interpolate lower surface (currently TE→LE, must flip to LE→TE)
    fl = interp1d(lower_X[::-1], lower_Y[::-1], kind='cubic')
    X_lower_new = x_dist
    Y_lower_new = fl(X_lower_new)

    # Interpolate upper surface (already LE→TE)
    fu = interp1d(upper_X, upper_Y, kind='cubic')
    X_upper_new = x_dist
    Y_upper_new = fu(X_upper_new)

    # --- Reorder final points: TE → LE → TE (VPM compatible) ---
    X_final = np.concatenate([X_lower_new[::-1], X_upper_new[1:]])
    Y_final = np.concatenate([Y_lower_new[::-1], Y_upper_new[1:]])

    return X_final, Y_final

m = 100# Set your desired number of panels

if m!=m_orig:
    X, Y = resample_airfoil_panels(X, Y, m)

print(f"New number of panels (m): {m}")

plt.figure(figsize=(10, 3))
plt.plot(X, Y, color='blue', marker='o',markersize='3')
plt.title(f'NACA0012 with m={m} Panels')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.axis('equal')
#plt.xlim([0.9,1.05]) # to zoom at trailing edge
plt.grid(True)
plt.show()

alpha_values=np.linspace(-5,10,16)
alpha_near_0 = np.linspace(-1,1,10)

alpha_values = np.sort(np.unique(np.concatenate((alpha_values, alpha_near_0))))

Cl_values=[]
x_cp_values=[]
Cm_cp_values=[]
for alpha in alpha_values:
    

    #Calculation of control point(xi.yi)
    x=(X[1:]+X[0:-1]) /2
    y=(Y[1:]+Y[0:-1]) /2
    
    S=np.sqrt((X[1:]-X[0:-1])**2 + (Y[1:]-Y[0:-1])**2) #Panel length
    theta=(np.arctan2((Y[1:]-Y[0:-1]),(X[1:]-X[0:-1]))) # orientation of panel
    RHS=np.sin((theta)-np.radians(alpha)) #RHS of equation
    
    
    
    An=np.zeros([m+1,m+1])
    At=np.zeros([m+1,m+1])
    Cn1=np.zeros([m,m])
    Cn2=np.zeros([m,m])
    Ct1=np.zeros([m,m])
    Ct2=np.zeros([m,m])
    
    for i in range(m):
        for j in range(m):
            if i==j:
                Cn1[i,j]=-1
                Cn2[i,j]=1
                Ct1[i,j]=np.pi/2
                Ct2[i,j]=np.pi/2
            else:
                A = - (x[i] - X[j])*(np.cos(theta[j])) - (y[i] - Y[j])*(np.sin(theta[j]))
                B = (x[i] - X[j])**2 + (y[i] - Y[j])**2
                C = np.sin(theta[i] - theta[j])
                D = np.cos(theta[i] - theta[j])
                E = (x[i] - X[j])*np.sin(theta[j]) - (y[i] - Y[j])*np.cos(theta[j])
                F = np.log(1 + ((S[j])**2 + (2*A*S[j])) / B)
                G = np.arctan2((E*S[j]) , (B + A*S[j]))
                P = ((x[i] - X[j]) * np.sin(theta[i] - 2*theta[j])) + ((y[i] - Y[j]) * np.cos(theta[i] - 2*theta[j]))
                Q = ((x[i] - X[j]) * np.cos(theta[i] - 2*theta[j])) - ((y[i] - Y[j]) * np.sin(theta[i] - 2*theta[j]))
                
                Cn2[i,j] = D + ((0.5*Q*F)/S[j]) - ((A*C + D*E)*(G/S[j]))
                Cn1[i,j] = 0.5*D*F + C*G - Cn2[i,j]
                Ct2[i,j] = C + ((0.5*P*F)/S[j]) + ((A*D - C*E)*(G/S[j]))
                Ct1[i,j] = 0.5*C*F - D*G - Ct2[i,j]
                
          
    for i in range(m):
        An[i,0]=Cn1[i,0]
        An[i,m]=Cn2[0,m-1]
        
        At[i,0]=Ct1[i,0]
        At[i,m]=Ct2[i,m-1]
        for j in range(1,m):
            At[i,j]=Ct1[i,j]+Ct2[i,j-1]
            An[i,j]=Cn1[i,j]+Cn2[i,j-1]
            
    
    #For i=m equation          
    An[m,0]=1
    An[m,m]=1
    for j in range(1,m):
        An[m,j]=0
        
    RHS=np.append(RHS,0)
    
    gamma=np.linalg.solve(An,RHS)
    
    #calculating velocity and Cp
    v=np.zeros_like(x)
    C_p=np.zeros_like(x)
    for i in range(m):
        v[i]=np.cos(theta[i]-np.radians(alpha))
        for j in range(m+1):
            v[i] += At[i,j] * gamma[j]
            
        C_p[i]=1-(v[i]**2)
    
    mid = m//2
    CP_lower = C_p[:mid][::-1]   # lower surface
    CP_upper = C_p[mid:]         # upper surface
    x_mid = x[mid:]
    # Area under the Cp plot is lift
    Cl = np.trapz(CP_lower - CP_upper, x_mid)
    print("Lift Coefficient Cl =", Cl)
    Cl_values.append(Cl)
    
    x_moment_arm = x_mid 
   
    moment_integrand = (CP_lower - CP_upper) * x_moment_arm
    
    # Calculate the moment coefficient about the leading edge (x=0)
    # We use np.trapz over the x_mid array (which spans 0 to 1)
    Cm_LE = -np.trapz(moment_integrand, x_mid)
   
    # Handle the case of alpha=0, where Cl is zero (causes division by zero)
    if alpha==0:
        x_cp=np.nan
        Cm_cp=0
    else:
        x_cp = -Cm_LE / Cl
        Cm_cp = Cm_LE + Cl * x_cp 
        
    x_cp_values.append(x_cp)
    # This value will be 0.0 or a very small floating-point residual
    Cm_cp_values.append(Cm_cp)
    if alpha==0 or alpha==5 or alpha==10:
        
        plt.plot(x, C_p,label=f'$\\alpha={alpha}$')
        

plt.title(f'Pressure Coefficient $C_p$ vs. X ')
plt.xlabel('x')
plt.ylabel('$C_p$')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()

plt.plot(x, C_p,label=f'$\\alpha={alpha}$')
plt.title(f'Pressure Coefficient $C_p$ vs. X for $\\alpha = {alpha}$ degrees')
plt.xlabel('x')
plt.ylabel('$C_p$')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(alpha_values, Cl_values)
plt.title(f'Lift Coefficient $C_l$ vs. alpha')
plt.xlabel('alpha')
plt.ylabel('$C_l$')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(alpha_values, x_cp_values)
plt.title(f'center of pressure $x_(cp)$  vs. alpha')
plt.xlabel('alpha')
plt.ylabel('$x_cp$')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(alpha_values, Cm_cp_values)
plt.ylim([-1e-13,1e-13])
plt.title(f'Torque about center of pressure $x_(cp)$  vs. alpha')
plt.xlabel('alpha')
plt.ylabel('$Cm_cp$')
plt.grid(True)
plt.show()

