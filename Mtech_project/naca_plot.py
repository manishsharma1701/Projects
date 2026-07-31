import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Raw data string provided
data_str = """VARIABLES = "N" "h=sqrt(1/N)" "CL" "CD" "CMy" "CDp" "CDv"
# Note FUN3D uses 2nd order turb advection
ZONE T="FUN3D, Family I, NOPV AW SA model, 10 deg"
      3729  0.1638E-01   0.1060191356E+01   0.2011797523E-01   0.2027220413E-02  0.1463312163E-01   0.5484853602E-02
     14625  0.8269E-02   0.1099906528E+01   0.1407319759E-01   0.2618838680E-02  0.8004924181E-02   0.6068273404E-02
     57921  0.4155E-02   0.1101848022E+01   0.1271896742E-01   0.3927498277E-02  0.6532834550E-02   0.6186132871E-02 
    230529  0.2083E-02   0.1098526159E+01   0.1239777283E-01   0.5014136113E-02  0.6195564068E-02   0.6202208761E-02
    919809  0.1043E-02   0.1095344620E+01   0.1231283688E-01   0.5791222485E-02  0.6108532180E-02   0.6204304703E-02 
   3674625  0.5217E-03   0.1093017395E+01   0.1228623320E-01   0.6321495892E-02  0.6081222546E-02   0.6205010651E-02
  14689281  0.2609E-03   0.1091643113E+01   0.1227630000E-01   0.6629082666E-02  0.6070805111E-02   0.6205494893E-02
ZONE T="FUN3D, Family II, NOPV AW SA model, 10 deg"
      3729  0.1638E-01   0.1039511987E+01   0.2280337883E-01   0.3701645819E-02  0.1748374426E-01   0.5319634563E-02
     14625  0.8269E-02   0.1083258921E+01   0.1444034784E-01   0.5782716424E-02  0.8409755491E-02   0.6030592346E-02
     57921  0.4155E-02   0.1090578875E+01   0.1275673029E-01   0.6297637000E-02  0.6571303465E-02   0.6185426825E-02
    230529  0.2083E-02   0.1090993871E+01   0.1238518102E-01   0.6644036246E-02  0.6179192087E-02   0.6205988930E-02
    919809  0.1043E-02   0.1090900334E+01   0.1229874922E-01   0.6764055464E-02  0.6091884427E-02   0.6206864791E-02
   3674625  0.5217E-03   0.1090999296E+01   0.1227885732E-01   0.6765182720E-02  0.6072634280E-02   0.6206223038E-02
  14689281  0.2609E-03   0.1091021077E+01   0.1227401922E-01   0.6765769311E-02  0.6068122094E-02   0.6205897124E-02


VARIABLES = "N" "h=sqrt(1/N)" "CL" "CD" "CDp" "CDv" "CMy"
ZONE T="CFL3D no PV, Family I, SA model, 10 deg, 1st order turb advection"
 14680064  .000261 0.10890569614E+01  0.12259262246E-01  0.60561788665E-02  0.62030833791E-02  0.72004537157E-02
  3670016  .000522 0.10896887773E+01  0.12262192960E-01  0.60621621886E-02  0.62000307715E-02  0.70505079735E-02
   917504  .00104  0.10904597696E+01  0.12281045030E-01  0.60848337571E-02  0.61962112724E-02  0.68550890878E-02
   229376  .00209  0.10911194303E+01  0.12371218400E-01  0.61785399316E-02  0.61926784683E-02  0.66172558711E-02
    57344  .00418  0.10906108574E+01  0.12775462049E-01  0.65856266569E-02  0.61898353917E-02  0.63516773495E-02
    14336  .00835  0.10835690990E+01  0.14532239861E-01  0.83712011451E-02  0.61610387161E-02  0.62622782982E-02
zone, t="CFL3D no PV, Family II, SA model, 10 deg, 2nd order turb advection"
 14680064  .000261 0.10908536861E+01  0.12271549926E-01  0.60657647377E-02  0.62057851883E-02  0.68042030939E-02
  3670016  .000522 0.10907132965E+01  0.12276226337E-01  0.60704776725E-02  0.62057486648E-02  0.68314574012E-02
   917504  .00104  0.10901477249E+01  0.12298112613E-01  0.60926717260E-02  0.62054408871E-02  0.69382075543E-02
   229376  .00209  0.10895756871E+01  0.12402564791E-01  0.61999219483E-02  0.62026428424E-02  0.69799799721E-02
    57344  .00418  0.10890346717E+01  0.12874777386E-01  0.66881090634E-02  0.61866683223E-02  0.67155804862E-02
    14336  .00835  0.10850207185E+01  0.15029162146E-01  0.89379899119E-02  0.60911722340E-02  0.57621158162E-02
"""

# Manual Comparison Data
nodes_comp = np.array([3674625, 919809, 230529,57921,14625,3729])
h_comp = np.sqrt(1.0 / nodes_comp)
h_comp_tmp = np.sqrt(1.0/nodes_comp[:3])
adapt1_nodes = np.array([448, 7015, 13661, 26736, 54700, 111915, 223881, 440751, 649518])
adapt1_nodes = np.sqrt(1.0 / adapt1_nodes)

Cl_hifun = np.array([1.115724, 1.122666, 1.101582])
Cl_hifun_f2 = np.array([1.093830, 1.091684, 1.089808])
Cl_pravaha_f2_SA = np.array([1.061215, 0.934083, 0.958808])
Cl_pravaha_f2_SST=np.array([1.075916,1.076135,1.077305])
Cl_cfdpp = np.array([1.0953931e+00, 1.0958107e+00, 1.0957384e+00])
Cl_cfdpp_f2 = np.array([1.0886959e+00, 1.0894359e+00, 1.0889901e+00,1.0868807e+00,1.0761526e+00,1.0051964e+00])
Cl_cfdpp_adapt1 = np.array([1.0842427e+00, 1.0961049e+00, 1.1029378e+00, 1.1077322e+00, 1.1054636e+00,
                             1.1049566e+00, 1.1051286e+00, 1.1036222e+00, 1.1031810e+00])

Cd_hifun = np.array([0.013364, 0.013514, 0.012357])
Cd_hifun_f2 = np.array([0.012569, 0.012426, 0.012539])
Cd_pravaha_f2_SA = np.array([0.011269, 0.012536, 0.057407])
Cd_pravaha_f2_SST =np.array([0.012624,0.012659,0.012936])
Cd_cfdpp = np.array([1.1870721e-02, 1.1892372e-02, 1.1980968e-02])
Cd_cfdpp_f2 = np.array([1.1938230e-02, 1.1921243e-02, 1.2020656e-02,1.2498163e-02,1.4822154e-02,2.8017715e-02])
Cd_cfdpp_adapt1 = np.array([6.2216405e-02, 1.0947724e-02, 1.0654115e-02, 1.0386387e-02, 1.0240204e-02,
                             1.0201328e-02, 1.0343574e-02, 1.0644369e-02, 1.0651896e-02])

Cdp_hifun = np.array([0.006335,0.006237,0.006040])
Cdp_hifun_f2 = np.array([0.006358,0.006253,0.006410])
Cdp_pravaha_f2_SA = np.array([0.006059])
Cdp_pravaha_f2_SST = np.array([0.006428,0.006480,0.006810])
Cdp_cfdpp = np.array([5.9728592e-03, 5.9983251e-03, 6.0914276e-03])
Cdp_cfdpp_f2 = np.array([6.0991136e-03, 6.0711621e-03, 6.1798185e-03,6.6793960e-03,9.1383432e-03,2.3021990e-02])
Cdp_cfdpp_adapt1 = np.array([5.7827009e-02, 6.1909697e-03, 5.6832083e-03, 5.4267257e-03, 5.3352039e-03,
                              5.2985267e-03, 5.3424796e-03, 5.4444741e-03, 5.4517805e-03])

Cdv_hifun = np.array([0.007029,0.007276,0.006316])
Cdv_hifun_f2 = np.array([0.006212,0.006173,0.006129])
Cdv_pravaha_f2_SST = np.array([0.006196,0.006179,0.006126])
Cdv_cfdpp = np.array([5.8978646e-03, 5.8940487e-03, 5.8895478e-03])
Cdv_cfdpp_f2 = np.array([5.8391188e-03, 5.8500857e-03, 5.8408335e-03,5.8187642e-03,5.6838104e-03,4.9957283e-03])
Cdv_cfdpp_adapt1 = np.array([4.3893942e-03, 4.7570974e-03, 4.9710355e-03, 4.9591348e-03, 4.9048904e-03,
                              4.9023707e-03, 5.0014986e-03, 5.1647328e-03, 5.2001265e-03])

# --- Parsing Logic ---
# NOTE: a previous version of this parser built each zone's DataFrame lazily
# (only when the *next* ZONE line was encountered, or at EOF). That meant a
# zone's column labels were whatever current_vars happened to be at that
# *later* point in the file, not what was active when the zone's own data
# lines were actually read. Concretely: FUN3D "Family I" and "Family II"
# share a single VARIABLES header ([..., CMy, CDp, CDv]) with no VARIABLES
# line between them, but Family II's DataFrame wasn't finalized until the
# CFL3D ZONE line was hit -- by which point a *second* VARIABLES header
# ([..., CDp, CDv, CMy]) had already overwritten current_vars. So Family II's
# data was mislabeled with the CFL3D column order, silently shifting CMy/CDp/
# CDv by one column (CMy's data got labeled "CDp", CDp's data got labeled
# "CDv", and CDv's data got labeled "CMy").
#
# Fix: snapshot current_vars as soon as we start reading a zone's data, so a
# later VARIABLES line can never retroactively relabel an earlier zone.
zones = []
current_vars = []
current_zone_name = None
current_data = []
current_zone_vars = None  # vars snapshot taken when this zone's data starts

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
            zones.append((current_zone_name, pd.DataFrame(current_data, columns=current_zone_vars)))
            current_data = []
        # Find zone title
        match = re.search(r'T="(.*?)"', clean_line, re.IGNORECASE)
        current_zone_name = match.group(1) if match else "Unknown"
        current_zone_vars = None  # will snapshot on first data line of this zone
        continue

    # Parse Numerical Data
    try:
        values = list(map(float, clean_line.replace(',', ' ').split()))
        if values:
            if current_zone_vars is None:
                current_zone_vars = current_vars  # snapshot vars active right now
            current_data.append(values)
    except ValueError:
        continue

# Add the final zone
if current_zone_name and current_data:
    zones.append((current_zone_name, pd.DataFrame(current_data, columns=current_zone_vars)))

# --- Plotting ---
plot_vars = ["CL", "CD", "CDp", "CDv"]
output_files = []

for var in plot_vars:
    plt.figure(figsize=(9, 6))

    # Plot all parsed zones
    for name, df in zones:
        if var in df.columns:
            plt.plot(df['h'], df[var], marker='o', label=name)

    # # Overlay manual comparison data
    if var == 'CL':
        plt.plot(h_comp_tmp, Cl_hifun, 's--', label="HiFUN F-1")
        plt.plot(h_comp_tmp, Cl_hifun_f2, 'ks--', label="HiFUN F-2")
        plt.plot(h_comp_tmp, Cl_pravaha_f2_SST, 's--', label="PraVaHa F-2 SST")
        # plt.plot(h_comp, Cl_pravaha_f2_SA, 's--', label="PraVaHa F-2 SA")
        plt.plot(h_comp_tmp, Cl_cfdpp, 'rs--', label="CFD++ F-1")
        plt.plot(h_comp, Cl_cfdpp_f2, 'bs--', label="CFD++ F-2")
    # #     # plt.plot(adapt1_nodes[1:], Cl_cfdpp_adapt1[1:], 'o-', label="CFD++ adapt1")
    elif var == 'CD':
        plt.plot(h_comp_tmp, Cd_hifun, 's--', label="HiFUN F-1")
        plt.plot(h_comp_tmp, Cd_hifun_f2, 'ks--', label="HiFUN F-2")
        plt.plot(h_comp_tmp, Cd_pravaha_f2_SST, 's--', label="PraVaHa F-2 SST")
        # plt.plot(h_comp, Cd_pravaha_f2_SA, 's--', label="PraVaHa F-2 SA")
        plt.plot(h_comp_tmp, Cd_cfdpp, 'rs--', label="CFD++ F-1")
        plt.plot(h_comp, Cd_cfdpp_f2, 'bs--', label="CFD++ F-2")
    #     # plt.plot(adapt1_nodes[1:], Cd_cfdpp_adapt1[1:], 'o-', label="CFD++ adapt1")
    elif var == 'CDp':
        plt.plot(h_comp_tmp, Cdp_cfdpp, 'rs--', label="CFD++ F-1")
        plt.plot(h_comp, Cdp_cfdpp_f2, 'bs--', label="CFD++ F-2")
        plt.plot(h_comp_tmp, Cdp_hifun, 's--', label="HiFUN F-1")
        plt.plot(h_comp_tmp, Cdp_hifun_f2, 'ks--', label="HiFUN F-2")
        plt.plot(h_comp_tmp, Cdp_pravaha_f2_SST, 's--', label="PraVaHa F-2 SST")
    #     # plt.plot(adapt1_nodes[1:], Cdp_cfdpp_adapt1[1:], 'rs--', label="CFD++ adapt1")
    elif var == 'CDv':
        plt.plot(h_comp_tmp, Cdv_hifun, 's--', label="HiFUN F-1")
        plt.plot(h_comp_tmp, Cdv_cfdpp, 'rs--', label="CFD++ F-1")
        plt.plot(h_comp, Cdv_cfdpp_f2, 'bs--', label="CFD++ F-2")
        plt.plot(h_comp_tmp, Cdv_pravaha_f2_SST, 's--', label="PraVaHa F-2 SST")
        plt.plot(h_comp_tmp, Cdv_hifun_f2, 'ks--', label="HiFUN F-2")
    #     # plt.plot(adapt1_nodes[1:], Cdv_cfdpp_adapt1[1:], 'rs--', label="CFD++ adapt1")

    plt.xlabel(r'$h = \sqrt{1/N}$', fontsize=12)
    plt.ylabel(f'${var}$', fontsize=12)
    plt.title(f'Grid Convergence: {var} vs Mesh Resolution', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_path = f"Manish/icfd++/naca0012/grid_convergence_{var}.pdf"
    plt.savefig(
    out_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.02,
    )

output_files.append(out_path)
plt.show()

print(f"Saved {len(output_files)} plots: {output_files}")

print(f"Saved {len(output_files)} plots: {output_files}")