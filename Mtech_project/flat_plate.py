import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Raw data string provided
data_str = """variables="N","h^2=1/N","h=sqrt(1/N)","CD" , "Cf"
zone t="CFL3D"
208896.  4.78707e-6  2.18794e-3  0.285985288E-02  0.270562153E-02
 52224.  1.91483e-5  4.37588e-3  0.286130951E-02  0.270673749E-02
 13056.  7.65931e-5  8.75175e-3  0.286620917E-02  0.271115173E-02
  3264.  3.06373e-4  1.75035e-2  0.288437885E-02  0.272834697E-02
   816.  1.22549e-3  3.50070e-2  0.295438152E-02  0.279568508E-02
zone t="FUN3D"
208896.  4.78707e-6  2.18794e-3  0.2852469E-02  0.270540472596778E-02
 52224.  1.91483e-5  4.37588e-3  0.2847933E-02  0.270448474835251E-02
 13056.  7.65931e-5  8.75175e-3  0.2840045E-02  0.270215403362043E-02
  3264.  3.06373e-4  1.75035e-2  0.2822641E-02  0.269497594242929E-02
   816.  1.22549e-3  3.50070e-2  0.2773859E-02  0.266737773774490E-02
   
"""

# Manual Comparison Data
nodes_comp = np.array([208896, 52224, 13056, 3264, 816])
h_comp = np.sqrt(1.0 / nodes_comp)
adapt1_nodes=np.array([9848,12685])
adapt1_nodes=np.sqrt(1.0 / adapt1_nodes )


Cf_cfdpp = np.array([0.00269410088125953979,0.00269410088125953979,0.00269585185647821775,0.00270003987721690111,0.00270346464700834625])
# Cf_cfdpp_adapt= np.array([,0.00222638903444133309])

Cd_cfdpp = np.array([2.8504108e-03,2.8504108e-03,2.8536176e-03,2.8563901e-03,2.8611632e-03])
Cd_hifun = np.array([0.001243,0.001825,0.001724,0.002052])
Cd_cfdpp_adapt1=np.array([1.1436487e-03,2.1696334e-03])


# Correct Parsing Logic
zones = []
current_vars = []
current_zone_name = None
current_data = []

for line in data_str.strip().split('\n'):
    clean_line = line.strip()
    if not clean_line or clean_line.startswith('#'):
        continue
    
    # Handle Variable Headers (order changes for FUN3D vs CFL3D)
    if clean_line.upper().startswith('VARIABLES'):
        # Extract everything inside quotes
        current_vars = re.findall(r'"(.*?)"', clean_line)
        # Rename 'h=sqrt(1/N)' to just 'h' for consistency
        current_vars = ["h" if "h=" in v else v for v in current_vars]
        continue
        
    # Handle Zone Headers (Case-Insensitive)
    if clean_line.upper().startswith('ZONE'):
        if current_zone_name and current_data:
            zones.append((current_zone_name, pd.DataFrame(current_data, columns=current_vars)))
            current_data = []
        # Find zone title
        match = re.search(r'T="(.*?)"', clean_line, re.IGNORECASE)
        current_zone_name = match.group(1) if match else "Unknown"
        continue
    
    # Parse Numerical Data
    try:
        values = list(map(float, clean_line.replace(',', ' ').split()))
        if values:
            current_data.append(values)
    except ValueError:
        continue

# Add the final zone
if current_zone_name and current_data:
    zones.append((current_zone_name, pd.DataFrame(current_data, columns=current_vars)))

# --- Plotting ---
plot_vars = ["Cf", "CD"]

for var in plot_vars:
    plt.figure(figsize=(9, 6))
    
    # Plot all parsed zones
    for name, df in zones:
        if var in df.columns:
            plt.plot(df['h'], df[var], marker='o', label=name)
    
    # Overlay manual comparison data
    # if var == 'Cf':
        # plt.plot(h_comp, Cl_hifun, 'ks--', label="hifun")
        # plt.plot(h_comp, Cf_cfdpp, 'rs--', label="cfd++")
        # plt.plot(adapt1_nodes, Cd_cfdpp_adapt1,'o-' , label="cfd++ adapt1")
    if var == 'CD':
        plt.plot(h_comp[:4], Cd_hifun, 'ks--', label="hifun")
        # plt.plot(h_comp, Cd_cfdpp, 'rs--', label="cfd++")
        # plt.plot(adapt1_nodes, Cd_cfdpp_adapt1,'o-' , label="cfd++ adapt1")

    plt.xlabel('$h = \sqrt{1/N}$', fontsize=12)
    if var=='Cf':
        plt.ylabel(f'${var}$ -skin friction coeffiient at x=0.97', fontsize=12)
    else:
        plt.ylabel(f'${var}$', fontsize=12)
    plt.title(f'Grid Convergence: {var} vs Mesh Resolution', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
