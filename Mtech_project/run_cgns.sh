#!/bin/bash

# ==========================================
# CGNS + HIFUN OpenMPI Environment Loader
# ==========================================

HIFUN_MPI="/home/yash/Manish/hifun-6.1.1_install/hifun-6.1.1/Third_Party_64/openmpi"
CGNS_UTILS="$HOME/Manish/CGNS-4.5.1/build/src/cgnstools/utilities"

# Clean old MPI influence
unset LD_LIBRARY_PATH
unset PATH

# Minimal safe PATH
export PATH=/usr/bin:/bin

# Load correct MPI
export PATH=$HIFUN_MPI/bin:$PATH
export LD_LIBRARY_PATH=$HIFUN_MPI/lib

# ==========================================
# Check input
# ==========================================

if [ $# -lt 1 ]; then
    echo "Usage: ./run_cgns.sh <cgns_command> [arguments]"
    echo "Example:"
    echo "  ./run_cgns.sh cgns_to_vtk mesh.cgns"
    echo "  ./run_cgns.sh print_cgns_details mesh.cgns"
    exit 1
fi

COMMAND=$1
shift

# ==========================================
# Run CGNS command
# ==========================================

$CGNS_UTILS/$COMMAND "$@"

