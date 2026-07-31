"Reference = https://turbmodels.larc.nasa.gov/Onerawingnumerics_val/SA/combined_forces_pitchmom_maxmut.dat"
import numpy as np
import matplotlib.pyplot as plt

nodes=np.array([60777345,7625153,960225,121841,15705])
adapted_nodes_l2=np.array([12781,18236,27266,47320,85538,177903,352998,709328,1349414,2622920,5146882])
adapted_nodes_l3=np.array([18180,27550,48662])
adapted_nodes_spalding_l4=np.array([1716935,927935,497415,238307,147054])
nodes=nodes**-(1/3)
adapted_nodes_l2=adapted_nodes_l2**-(1/3)
adapted_nodes_l3=adapted_nodes_l3**-(1/3)
adapted_nodes_spalding_l4=adapted_nodes_spalding_l4**-(1/3)
adapted_nodes_l4_new = np.array([    17568,    33951,    59408,    97903,    159765,    276226,    497326])
adapted_nodes_l4_new=adapted_nodes_l4_new**(-1/3)

L2_grad1_3_nodes = np.array([
    17568,
    22525,
    31424,
    54613,
    109004,
    184810,
])
L2_grad1_3_nodes = L2_grad1_3_nodes**-(1/3)

L2_grad1_3_cl = [
    0.24478573,
    0.24321037,
    0.26226705,
    0.28076028,
    0.28369950,
    0.27305807,
]

L2_grad1_3_cd = [
    0.018055954,
    0.016874545,
    0.016474108,
    0.016646991,
    0.019585161,
    0.016936189,
]

L2_grad1_3_cdp = [
    0.013921125,
    0.016827208,
    0.016334162,
    0.016073528,
    0.018028746,
    0.014047040,
]

L2_grad1_3_cdv = [
    0.0041348281,
    0.000047338174,
    0.00013994557,
    0.00057346285,
    0.0015564158,
    0.0028891499,
]

##############C_L###############
cl_adapted_l2=[0.2334009,0.25127366,0.26298342,0.26251893,0.26398322,0.26958126,0.27333959,0.27379752,0.27516978,0.27600282,2.7597409e-01]
cl_adapted_spalding_L4=np.array([2.7571305e-01,2.7453954e-01,2.7143497e-01,2.6297795e-01,2.6696366e-01])
cl_adapted_l4_new =np.array([    0.24478573,    0.26266747,    0.26498746,    0.26606121,    0.26674023,    0.26681291,    0.26657140])

cl_icfd_79KP=[2.7183951e-01,2.6794012e-01,2.6400460e-01,2.5894726e-01]
cl_icfd_79KP_ph=np.array([2.6639351e-01,2.6347894e-01,2.5214443e-01])
cl_icfd_79KP_prism=np.array([2.6774075e-01,2.6520567e-01,2.6120514e-01])

cl_hifun_79KP_new=np.array([0.273841,0.271224,0.265630,2.5894726e-01])
cl_hifun_ph_79KP_new=np.array([0.271192,0.268244,0.267090,0.257984])
cl_hifun_prism_79KP_new=np.array([0.272342,0.270787,0.269900,0.268874])

cl_pravaha_ph=np.array([0.260843,0.253865,0.227890])
cl_pravaha_prism=np.array([0.253891,0.235471])

USM3d_cl=np.array([2.706238E-01,2.690708E-01,2.673551E-01,2.641175E-01,2.504382E-01])
USM3d_cl_ph=np.array([2.693477E-01,2.673999E-01,2.669640E-01,2.612740E-01])
FUN3d_cl=[0.271195,0.270826,0.263215,0.244181]
FUN3d_cl_prism=[0.269545512300000,0.267878736000000,0.265852518000000,0.258628725400000]

#--------------C_D-----------------------

cd_icfd_79KP=[1.7645625e-02,1.7983095e-02,2.0089557e-02,2.9880796e-02]
cd_icfd_79KP_ph=np.array([1.6967686e-02,1.7823977e-02,2.1880754e-02])
cd_icfd_79KP_prism=np.array([1.7054226e-02,1.7797782e-02,2.0608887e-02])

cd_adapted_l2=[0.017167969,0.01849061,0.017153813,0.016249948,0.016101703,0.016134312,0.015776774,0.01558731,0.015411625,0.015360805,1.5304745e-02]
cd_adapted_spalding_L4=np.array([1.5285547e-02,1.5258862e-02,1.5187756e-02,1.5221372e-02,1.7395963e-02])
cd_adapted_l4_new = np.array([    0.018055954,    0.016019255,    0.015633333,    0.015404614,    0.015398519,    0.015331516,    0.015256971])

cd_hifun_79KP_new=np.array([0.017792,0.016194,0.017700,2.9880796e-02])
cd_hifun_ph_79KP_new=np.array([0.016994,0.017100,0.017903,0.021229])
cd_hifun_prism_79KP_new=np.array([0.016961,0.016726,0.016832,0.018704])

cd_pravaha_ph=np.array([0.016830,0.016810,0.017653])
cd_pravaha_prism=np.array([0.016294,0.015585])

USM3d_cd=np.array([1.704970E-02,1.706643E-02,1.738240E-02,1.904645E-02,2.638412E-02])
USM3d_cd_ph=np.array([1.698167E-02,1.703294E-02,1.774529E-02,2.126384E-02])
FUN3d_cd=[0.016979,0.017088,0.018330, 0.025362 ]
FUN3d_cd_prism=[0.0169473249, 0.01692282997, 0.01722662628, 0.02007509074,
            0.03220872702, 0.05303840179, 0.07794936472 ]

#--------------------C_Dp------------------------------------

cdp_icfd=[1.2157927e-02,1.2644714e-02,1.4860176e-02,2.4335787e-02]
cdp_icfd_79KP_ph=np.array([1.1800879e-02,1.2503149e-02,1.6369861e-02])
cdp_icfd_79KP_prism=np.array([1.1856176e-02,1.2495626e-02,1.5197102e-02])

cdp_adapted_l2=[1.7152191e-02,1.4443401e-02,1.3239020e-02,1.2191172e-02,1.1955230e-02,1.2045187e-02,1.1998501e-02,1.1917425e-02,1.1847601e-02,1.1839459e-02,1.1805187e-02]
cdp_adapted_spalding_L4=np.array([1.1818567e-02,1.1802939e-02,1.1677398e-02,1.1509803e-02,1.3113557e-02])
cdp_adapted_l4_new = np.array([    0.013921125,    0.012247762,    0.011944639,    0.011788619,    0.011812354,    0.011759880,    0.011713314])

cdp_hifun_79KP_new=np.array([0.012521,0.012550,0.014499])
cdp_hifun_ph_79KP_new=np.array([0.012107,0.013136,0.017137])
cdp_hifun_prism_79KP_new=np.array([0.011938,0.012559,0.014979])

cdp_pravaha_ph=np.array([0.011773,0.012270,0.014850])
cdp_pravaha_prism=np.array([0.011923,0.012998])

USM3d_cdp=[1.174045E-02,1.178316E-02,1.209829E-02,1.368016E-02,2.113643E-02]
USM3d_cdp_ph=[0.011680187, 0.011701769, 0.012198916, 0.015081715]
FUN3d_cdp=[0.011747,0.011849,0.013270,0.022730]
FUN3d_cdp_prism=[ 0.01166792697, 0.01163270616, 0.01181248629, 0.01390166551,
            0.02399877651, 0.04012950988, 0.06194007610]

#--------------------C_Dv------------------------------------

cdv_icfd=[5.4876977e-03,5.3383809e-03,5.2293812e-03,5.5450086e-03]
cdv_icfd_79KP_ph=np.array([5.1668084e-03,5.3208284e-03,5.5108931e-03])
cdv_icfd_79KP_prism=np.array([5.1980493e-03,5.3021553e-03,5.4117844e-03])

cdv_adapted_l2=[1.5779062e-05,4.0472086e-03,3.9147923e-03,4.0587770e-03,4.1464732e-03,4.0891248e-03,3.7782729e-03,3.5809875e-03,3.5359221e-03,3.5213468e-03,3.4995584e-03]
cdv_adapted_spalding_L4=np.array([3.4669796e-03,3.4559238e-03,3.5103586e-03,3.7115703e-03,4.2824066e-03])
cdv_adapted_l4_new = np.array([    0.0041348281,    0.0037714932,    0.0036886938,    0.0036159954,    0.0035861641,    0.0035716355,    0.0035436567])

cdv_hifun_79KP_new=np.array([0.005272,0.003644,0.003201])
cdv_hifun_ph_79KP_new=np.array([0.004993,0.004766,0.004091])
cdv_hifun_prism_79KP_new=np.array([0.004787,0.004273,0.003725])

cdv_pravaha_ph=np.array([0.005057,0.004540,0.002803])
cdv_pravaha_prism=np.array([0.004371,0.002587])

USM3d_cdv=[5.309252E-03,5.283275E-03,5.284108E-03,5.366293E-03 ,5.247687E-03]
USM3d_cdv_ph=[0.005301483, 0.005331171, 0.005546374, 0.006182125]
FUN3d_cdv=[0.0052317,0.0052393,0.0050567,0.0026322]
FUN3d_cdv_prism=[0.005279397925, 0.005290123807, 0.005414139993, 0.006173425231,
            0.008209950509, 0.01290889191, 0.01600928862]


plt.figure(figsize=[10,5])
# # plt.plot(nodes[1:4],cl_icfd[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cl_icfd_79KP[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cl_icfd_79KP_ph[:3],"--d",label='ICFD++, prism-hex')
# plt.plot(nodes[1:4],cl_icfd_79KP_prism[:3],"--d",label='ICFD++, prism')

# plt.plot(adapted_nodes_l2,cl_adapted_l2,"-^",label='adapted_l2_ICFD++')
# plt.plot(L2_grad1_3_nodes,L2_grad1_3_cl,"-s",label='adapted_l2_gra1.3_ICFD++')
# plt.plot(adapted_nodes_spalding_l4,cl_adapted_spalding_L4,"-v",label='adapted_spalding_l4_ICFD++')
# plt.plot(adapted_nodes_l4_new,cl_adapted_l4_new,"-o",label='adapted_l4_ICFD++ ')

# plt.plot(nodes[1:4],cl_hifun[:3],"-d",label='HiFun_tet')
# plt.plot(nodes[1:4],cl_hifun_ph[:3],"-d",label='HiFun_ph')
# plt.plot(nodes[1:4],cl_hifun_prism[:3],"-d",label='HiFun_prism')


plt.plot(nodes[1:4],cl_hifun_79KP_new[:3],"-.d",label='HiFUN,tet,14.6e6')

plt.plot(nodes[:4],cl_hifun_ph_79KP_new,"-.d",label='HiFUN,prism-hex,14.6e6')

plt.plot(nodes[:4],cl_hifun_prism_79KP_new,"-.d",label='HiFUN,prism,14.6e6')
# plt.plot(nodes[2], 0.271239,'x',label='cd_hifun_su2_reader')
# plt.plot(nodes[1:4],cl_pravaha_ph,"-.d",label='PraVaHa,prism-hex,14.6e6')
# plt.plot(nodes[2:4],cl_pravaha_prism,"-.d",label='PraVaHa,prism,14.6e6')

plt.plot(nodes[:4],USM3d_cl_ph[:4],"-d",label='USM3d,prism-hex')
plt.plot(nodes[:4],FUN3d_cl_prism[:4],"-d",label='FUN3d,prism')
plt.plot(nodes[1:4],USM3d_cl[:3],"-d",label='USM3d,tet')
plt.plot(nodes[1:4],FUN3d_cl[:3],"-d",label='FUN3d,tet')
plt.title("$C_L$ vs h")
plt.ylabel("$C_L$")
plt.xlabel("$h=(1/N)^{(1/3)}$")
plt.grid("True")

plt.legend(loc='lower left')
out_path ="Manish/icfd++/onera/hifun_cl.pdf"
plt.savefig(
out_path,
format="pdf",
bbox_inches="tight",
pad_inches=0.02,
)
plt.show()

#--------------C_D-----------------------

plt.figure(figsize=[10,5])

# plt.plot(nodes[1:4],cd_icfd_79KP[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cd_icfd_79KP_ph[:3],"--d",label='ICFD++, prism-hex')
# plt.plot(nodes[1:4],cd_icfd_79KP_prism[:3],"--d",label='ICFD++, prism')

# plt.plot(adapted_nodes_l2,cd_adapted_l2,"-^",label='adapted_l2_ICFD++')
# plt.plot(L2_grad1_3_nodes,L2_grad1_3_cd,"-s",label='adapted_l2_gra1.3_ICFD++')
# plt.plot(adapted_nodes_spalding_l4,cd_adapted_spalding_L4,"-v",label='adapted_spalding_l4_ICFD++')
# plt.plot(adapted_nodes_l4_new,cd_adapted_l4_new,"-o",label='adapted_l4_ICFD++ ')


plt.plot(nodes[1:4],cd_hifun_79KP_new[:3],"-.d",label='HiFUN,tet,14.6e6')
plt.plot(nodes[:4],cd_hifun_ph_79KP_new,"-.d",label='HiFUN,prism-hex,14.6e6')
plt.plot(nodes[:4],cd_hifun_prism_79KP_new,"-.d",label='HiFUN,prism,14.6e6')
# # plt.plot(nodes[2], 0.016189,'x',label='cd_hifun_su2_reader')
# # plt.plot(nodes[1:4],cd_pravaha_ph,"-.d",label='PraVaHa,prism-hex,14.6e6')
# plt.plot(nodes[2:4],cd_pravaha_prism,"-.d",label='PraVaHa,prism,14.6e6')

plt.plot(nodes[:4],USM3d_cd_ph[:4],"-d",label='USM3d,prism-hex')
plt.plot(nodes[:4],FUN3d_cd_prism[:4],"-d",label='FUN3d,prism')
plt.plot(nodes[1:4],USM3d_cd[:3],"-d",label='USM3d,tet')
plt.plot(nodes[1:4],FUN3d_cd[:3],"-d",label='FUN3d,tet')
plt.title("$C_D$ vs h")
plt.ylabel("$C_D$")
plt.xlabel("$h=(1/N)^{(1/3)}$")
plt.grid("True")
plt.legend(loc='upper left')
out_path ="Manish/icfd++/onera/hifun_cd.pdf"
plt.savefig(
out_path,
format="pdf",
bbox_inches="tight",
pad_inches=0.02,
)
plt.show()

#--------------------C_Dp------------------------------------



plt.figure(figsize=[10,5])
# plt.plot(adapted_nodes_l2,cdp_adapted_l2,"-d",label='adapted_l2_ICFD++')
# plt.plot(L2_grad1_3_nodes,L2_grad1_3_cdp,"-s",label='adapted_l2_gra1.3_ICFD++')
# # plt.plot(adapted_nodes_l4,cdp_adapted_l4,"-d",label='adapted_l4_ICFD++')
# plt.plot(adapted_nodes_spalding_l4,cdp_adapted_spalding_L4,"-v",label='adapted_spalding_l4_ICFD++')
# plt.plot(adapted_nodes_l4_new,cdp_adapted_l4_new,"-o",label='adapted_l4_ICFD++ ')

# plt.plot(nodes[1:4],cdp_icfd[:3],"-d",label='ICFD++')
# plt.plot(nodes[1:4],cdp_icfd[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cdp_icfd_79KP_ph[:3],"--d",label='ICFD++, prism-hex')
# plt.plot(nodes[1:4],cdp_icfd_79KP_prism[:3],"--d",label='ICFD++, prism')
# plt.plot(nodes[1:4],cdp_pravaha_ph,"-.d",label='PraVaHa,prism-hex,14.6e6')
# plt.plot(nodes[2:4],cdp_pravaha_prism,"-.d",label='PraVaHa,prism,14.6e6')
# plt.plot(nodes[1:4],cdp_hifun[:3],"-d",label='HiFun_tet')
# plt.plot(nodes[1:4],cdp_hifun_79KP[:3],"-.d",label='HiFun,tet,14.6e6')
plt.plot(nodes[1:4],cdp_hifun_79KP_new[:3],"-.d",label='HiFUN,tet,14.6e6')
# plt.plot(nodes[1:4],cdp_hifun_ph[:3],"-d",label='HiFun_ph')
plt.plot(nodes[1:4],cdp_hifun_ph_79KP_new[:3],"-.d",label='HiFUN,prism-hex,14.6e6')
# plt.plot(nodes[1:4],cdp_hifun_prism[:3],"-d",label='HiFun_prism')
plt.plot(nodes[1:4],cdp_hifun_prism_79KP_new[:3],"-.d",label='HiFUN,prism,14.6e6')

plt.plot(nodes[:4],USM3d_cdp_ph[:4],"-d",label='USM3d,prism-hex')
plt.plot(nodes[:4],FUN3d_cdp_prism[:4],"-d",label='FUN3d,prism')
plt.plot(nodes[1:4],USM3d_cdp[:3],"-d",label='USM3d,tet')
plt.plot(nodes[1:4],FUN3d_cdp[:3],"-d",label='FUN3d,tet')
plt.title("$C_Dp$ vs h")
plt.ylabel("$C_Dp$")
plt.xlabel("$h=(1/N)^{(1/3)}$")
plt.grid("True")
plt.legend(loc='upper left')
out_path ="Manish/icfd++/onera/hifun_cdp.pdf"
plt.savefig(
out_path,
format="pdf",
bbox_inches="tight",
pad_inches=0.02,
)
plt.show()


#--------------------C_Dv------------------------------------

plt.figure(figsize=[10,5])
# plt.plot(adapted_nodes_l2,cdv_adapted_l2,"-d",label='adapted_l2_ICFD++')
# plt.plot(L2_grad1_3_nodes,L2_grad1_3_cdv,"-s",label='adapted_l2_gra1.3_ICFD++')
# plt.plot(adapted_nodes_l4,cdv_adapted_l4,"-d",label='adapted_l4_ICFD++')
# plt.plot(adapted_nodes_spalding_l4,cdv_adapted_spalding_L4,"-v",label='adapted_spalding_l4_ICFD++')
# plt.plot(adapted_nodes_l4_new,cdv_adapted_l4_new,"-o",label='adapted_l4_ICFD++ ')

# plt.plot(nodes[1:4],cdv_icfd[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cdv_icfd_79KP[:3],"--d",label='ICFD++')
# plt.plot(nodes[1:4],cdv_icfd_79KP_ph[:3],"--d",label='ICFD++, prism-hex')
# plt.plot(nodes[1:4],cdv_icfd_79KP_prism[:3],"--d",label='ICFD++, prism')
# plt.plot(nodes[1:4],cdv_pravaha_ph,"-.d",label='PraVaHa,prism-hex,14.6e6')
# plt.plot(nodes[2:4],cdv_pravaha_prism,"-.d",label='PraVaHa,prism,14.6e6')
# plt.plot(nodes[1:4],cdv_hifun[:3],"-d",label='HiFun_tet')
# plt.plot(nodes[1:4],cdv_hifun_79KP[:3],"-.d",label='HiFun,tet,14.6e6')
plt.plot(nodes[1:4],cdv_hifun_79KP_new[:3],"-.d",label='HiFUN,tet,14.6e6')
# # plt.plot(nodes[1:4],cdv_hifun_ph[:3],"-d",label='HiFun_ph')
plt.plot(nodes[1:4],cdv_hifun_ph_79KP_new[:3],"-.d",label='HiFUN,prism-hex,14.6e6')
# # plt.plot(nodes[1:4],cdv_hifun_prism[:3],"-d",label='HiFun_prism')
plt.plot(nodes[1:4],cdv_hifun_prism_79KP_new[:3],"-.d",label='HiFUN,prism,14.6e6')

plt.plot(nodes[:4],USM3d_cdv_ph[:4],"-d",label='USM3d,prism-hex')
plt.plot(nodes[:4],FUN3d_cdv_prism[:4],"-d",label='FUN3d,prism')
plt.plot(nodes[1:4],USM3d_cdv[:3],"-d",label='USM3d,tet')
plt.plot(nodes[1:4],FUN3d_cdv[:3],"-d",label='FUN3d,tet')
plt.title("$C_Dv$ vs h")
plt.ylabel("$C_Dv$")
plt.xlabel("$h=(1/N)^{(1/3)}$")
plt.grid("True")
plt.legend(loc='lower left')
out_path ="Manish/icfd++/onera/hifun_cdv.pdf"
plt.savefig(
out_path,
format="pdf",
bbox_inches="tight",
pad_inches=0.02,
)
plt.show()






cl_icfd=[2.7137262e-01,2.6849501e-01,2.6479785e-01,2.5894726e-01]
cl_hifun_ph=np.array([0.269592,0.268305,0.258740,0.258740])
cl_hifun_ph_79KP=np.array([0.269195,0.267955,0.258509])
cl_hifun_prism=np.array([0.271679,0.271497,0.269789,0.269789])
cl_hifun_prism_79KP=np.array([0.271208,0.270944,0.269572])
cl_hifun_79KP=np.array([0.279846,0.274205,0.266186,2.5894726e-01])
cl_hifun=np.array([0.274934,0.272258,0.266825,2.5894726e-01])


cd_icfd=[1.7343770e-02,1.7778598e-02,2.0169456e-02,2.9880796e-02]
cd_hifun_79KP=np.array([0.016753,0.016246,0.017538,2.9880796e-02])
cd_hifun=np.array([0.015755,0.015879 ,0.017389,2.9880796e-02])
cd_hifun_ph=np.array([0.016873,0.017716,0.021023,0.021023])
cd_hifun_ph_79KP=np.array([0.017037,0.017884,0.021231])
cd_hifun_prism=np.array([0.016484,0.016592,0.018486,0.018486])
cd_hifun_prism_79KP=np.array([0.016699,0.016802,0.018687])

cdp_hifun=[0.011963,0.012484 ,0.014448,0.014448]
cdp_hifun_79KP=np.array([0.012252,0.012647,0.014461])
cdp_hifun_ph=np.array([0.012070,0.013107,0.017091,0.017091])
cdp_hifun_ph_79KP=np.array([0.012099,0.013136,0.017135])
cdp_hifun_prism=np.array([0.011899,0.012502,0.014950,0.014950])
cdp_hifun_prism_79KP=np.array([0.011936,0.012531,0.014980])


cdv_hifun=[0.003792,0.003395,0.002941,0.002941]
cdv_hifun_79KP=np.array([0.004303,0.003598,0.003076])
cdv_hifun_ph=np.array([0.004804,0.004609,0.003931,0.003931])
cdv_hifun_ph_79KP=np.array([0.004939,0.004748,0.004096])
cdv_hifun_prism=np.array([0.004585,0.004090,0.003535,0.003535])
cdv_hifun_prism_79KP=np.array([0.004762,0.004271,0.003708])