#!/bin/bash
# ============================================================================
# automate.sh  -  Mesh-adaptation + CFD campaign driver
# ----------------------------------------------------------------------------
# Runs a Hessian-based mesh-adaptation loop across a ramp of complexities.
#
#   - 5 complexity levels, starting at 10000 and DOUBLING each level
#         -> 10000, 20000, 40000, 80000, 160000
#   - 5 adaptation runs PER complexity level (25 runs total)
#
# You run the "00" case manually and hand this script the resulting input
# mesh (+ its CFD solution). From there it produces:
#
#         01  01_2  01_3  01_4  01_5      (complexity = 10000)
#         02  02_2  02_3  02_4  02_5      (complexity = 20000)
#         03  03_2  03_3  03_4  03_5      (complexity = 40000)
#         04  04_2  04_3  04_4  04_5      (complexity = 80000)
#         05  05_2  05_3  05_4  05_5      (complexity = 160000)
#
# Each folder = one full run: refine metric+adapt -> Deepak yplus -> Julia
# hybrid adaptation -> CFD++ solve -> VTK -> heat post.
#
# The runs CHAIN: every run adapts from the previous folder's mesh+solution.
# (The very first run adapts from the 00 case you provide.)
#
# CFD JOBS ARE NOT AUTO-STOPPED. The script just waits and lets the solver
# run; YOU decide when it has converged and 'qdel' it manually. The only
# time the script kills a job itself is when it gets stuck in the 'E' state.
#
# Usage:
#   ./automate.sh <initial_input_mesh.meshb> [initial_solution]
#
# Launch this from the "00" folder (the folder that holds your 00-case CFD
# solution and the input mesh), exactly like the original workflow.
# ============================================================================

source ~/manish_vtk/bin/activate

set -u   # catch unset variables early

# ============================================================================
# USER CONFIG
# ============================================================================

START_COMPLEXITY=5000      # first complexity level
COMPLEXITY_FACTOR=2         # multiply by this at each new level
N_COMPLEXITIES=5            # number of complexity levels
RUNS_PER_COMPLEXITY=4       # adaptation runs at each level

MAX_E_CHECKS=2              # force-kill a job after this many 'E' polls
REFINE_CHECK_INTERVAL=10    # seconds between refine-job polls
CFD_CHECK_INTERVAL=10      # seconds between CFD-job polls

REMOTE_HOST="yoda1"         # host we qsub from
TARGET_DIR="../../2D-Hybrid-Mesh-Adaptation"   # Julia hybrid-adaptation dir

# ============================================================================
# INPUT CHECK
# ============================================================================

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <initial_input_mesh.meshb> [initial_solution]"
    echo "  Run this from the 00 folder that contains the 00-case solution."
    exit 1
fi

INITIAL_INPUT_MESH=$1
INITIAL_SOLUTION=${2:-}     # optional; only checked for existence

START_DIR=$(pwd)

echo "==============================================================="
echo " Adaptation campaign"
echo "---------------------------------------------------------------"
echo " Launch (00) dir     : $START_DIR"
echo " Initial input mesh  : $INITIAL_INPUT_MESH"
echo " Initial solution    : ${INITIAL_SOLUTION:-<relying on files already in this dir>}"
echo " Complexity levels   : $N_COMPLEXITIES (start $START_COMPLEXITY, x$COMPLEXITY_FACTOR each)"
echo " Runs per complexity : $RUNS_PER_COMPLEXITY"
echo "==============================================================="

[ -f "$INITIAL_INPUT_MESH" ] || { echo "Missing initial input mesh: $INITIAL_INPUT_MESH"; exit 1; }
if [ -n "$INITIAL_SOLUTION" ]; then
    [ -f "$INITIAL_SOLUTION" ] || { echo "Missing initial solution: $INITIAL_SOLUTION"; exit 1; }
fi

# ============================================================================
# HELPER: folder name for (complexity_index, run_index)
#   run 1        -> "01", "02", ...
#   run 2,3,4,5  -> "01_2", "01_3", ...
# ============================================================================

folder_name() {
    local ci=$1 ri=$2
    if [ "$ri" -eq 1 ]; then
        printf "%02d" "$ci"
    else
        printf "%02d_%d" "$ci" "$ri"
    fi
}

# ============================================================================
# HELPER: submit a PBS job over ssh and wait for it.
#   $1 = job kind ("refine" or "cfd")   -> only used for logging
#   $2 = absolute working directory the job is submitted from
#   $3 = qsub command line (the whole thing, e.g. 'qsub -N .. -v .. file.pbs')
#   $4 = seconds between polls
#
# Waits until the job leaves the queue (finishes OR you manually qdel it).
# The ONLY job the script kills on its own is one stuck in the 'E' state.
# ============================================================================

submit_and_wait() {
    local KIND=$1
    local WORKDIR=$2
    local QSUB_CMD=$3
    local INTERVAL=$4

    local SSH_SCRIPT="$WORKDIR/${KIND}_ssh_$$.sh"

    cat > "$SSH_SCRIPT" << ENDSSH
#!/bin/bash
cd $WORKDIR

JOB_ID=\$($QSUB_CMD)
JOB_ID=\$(echo \$JOB_ID | tr -d '\r')
echo "Submitted ${KIND} job: \$JOB_ID"

echo "Waiting for ${KIND} job to finish (manual qdel expected; only E-state is auto-killed)..."
E_COUNTER=0

while qstat | grep -q "\$JOB_ID"; do

    JOB_STATE=\$(qstat -f "\$JOB_ID" 2>/dev/null | grep "job_state" | awk '{print \$3}')
    echo "${KIND} job state: \$JOB_STATE"

    if [ "\$JOB_STATE" = "E" ]; then
        E_COUNTER=\$((E_COUNTER + 1))
        echo "${KIND} job in E state for \$E_COUNTER check(s)..."
        if [ "\$E_COUNTER" -ge $MAX_E_CHECKS ]; then
            echo "${KIND} job stuck in E state. Force deleting..."
            qdel -W force "\$JOB_ID"
            break
        fi
    else
        E_COUNTER=0
    fi

    sleep $INTERVAL
done

echo "${KIND} job left the queue."
exit
ENDSSH

    chmod +x "$SSH_SCRIPT"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash $SSH_SCRIPT"
    rm -f "$SSH_SCRIPT"
}

# ============================================================================
# HELPER: run ONE adaptation + CFD cycle.
#   $1 = input mesh filename (must exist in the CURRENT directory, whose
#        CFD solution we adapt from)
#   $2 = output (adapted) mesh filename
#   $3 = complexity
#   $4 = destination folder name (created as ../<name>)
#
# Precondition : cwd = folder holding the previous solution + input mesh.
# Postcondition: cwd = ../<dest folder>, holding this run's CFD solution and
#                the Deepak .meshb ready to feed the next run. The name of
#                that .meshb is written to the global RESULT_MESHB.
# ============================================================================

RESULT_MESHB=""

run_one() {
    local INPUT_MESH=$1
    local OUTPUT_MESH=$2
    local COMPLEXITY=$3
    local NEW_FILE=$4

    local CURRENT_DIR
    CURRENT_DIR=$(pwd)

    local SU2_FILE="${OUTPUT_MESH%.meshb}.su2"
    local SU2_FILE_DEEPAK="${OUTPUT_MESH%.meshb}_deepak.su2"
    local SU2_FILE_DEEPAK_MESHB="${SU2_FILE_DEEPAK%.su2}.meshb"

    echo ""
    echo "---------------------------------------------------------------"
    echo " RUN  ->  folder $NEW_FILE"
    echo "   input mesh  : $INPUT_MESH   (in $CURRENT_DIR)"
    echo "   output mesh : $OUTPUT_MESH"
    echo "   complexity  : $COMPLEXITY"
    echo "---------------------------------------------------------------"

    [ -f "$INPUT_MESH" ] || { echo "Missing input mesh in $CURRENT_DIR: $INPUT_MESH"; exit 1; }

    # ------------------------------------------------------------------
    # 1. REFINE: build metric from current solution + adapt + translate
    #    (writesol_volnsurf.py -> mach.sol / yplus.sol is run inside the job)
    # ------------------------------------------------------------------
    echo ""
    echo "Submitting refinement PBS job..."
    submit_and_wait "refine" "$CURRENT_DIR" \
        "qsub -N refine_job -v INPUT_MESH=$INPUT_MESH,OUTPUT_MESH=$OUTPUT_MESH,COMPLEXITY=$COMPLEXITY,NEW_FILE=$NEW_FILE jobscript_refine.pbs" \
        "$REFINE_CHECK_INTERVAL"

    # ------------------------------------------------------------------
    # 2. DEEPAK step: Config.txt for the yplus / hybrid-layer tool
    # ------------------------------------------------------------------
    echo ""
    echo "Preparing Config.txt..."
    cat <<EOF > Config.txt
IMPORT FILENAME   : $SU2_FILE
EXPORT FILENAME   : $SU2_FILE_DEEPAK
YPLUS FILENAME    : yplus.sol
WALL BC NO        : 4
MAX COEFFICIENT   : 1.5
RELAXATION FACTOR : 0.9
EOF

    [ -f "$SU2_FILE" ]  || { echo "Missing $SU2_FILE (refine did not produce it)"; exit 1; }
    [ -f "yplus.sol" ]  || { echo "Missing yplus.sol"; exit 1; }
    [ -d "$TARGET_DIR" ] || { echo "Missing $TARGET_DIR"; exit 1; }

    # ------------------------------------------------------------------
    # 3. JULIA hybrid mesh adaptation
    # ------------------------------------------------------------------
    mv Config.txt "$TARGET_DIR/"
    mv "$SU2_FILE" "$TARGET_DIR/"
    cp yplus.sol "$TARGET_DIR/"

    cd "$TARGET_DIR"
    echo ""
    echo "Running Julia adaptation..."
    julia main.jl > julia.log 2>&1
    cd "$CURRENT_DIR"

    # ------------------------------------------------------------------
    # 4. Assemble the new output folder (sibling of current dir)
    # ------------------------------------------------------------------
    mkdir -p "../$NEW_FILE"
    mv "$TARGET_DIR/$SU2_FILE_DEEPAK" "../$NEW_FILE/"
    mv "$TARGET_DIR/$SU2_FILE"        "../$NEW_FILE/"
    mv "$TARGET_DIR/julia.log"        "../$NEW_FILE/"
    
    cp analyze_heat.py cfd_post.sh \
       residue.py \
       Exp* \
       automate.sh \
       *.inp \
       writesol_volnsurf.py \
       jobscript_* \
       fix_su2.py \
       "../$NEW_FILE/"

    # ------------------------------------------------------------------
    # 5. Move into the new folder and convert Deepak SU2 -> MESHB
    #    (this .meshb becomes the input mesh for the NEXT run)
    # ------------------------------------------------------------------
    cd "../$NEW_FILE/"
    local CFD_DIR
    CFD_DIR=$(pwd)

    echo ""
    echo "Converting SU2 -> MESHB..."
    ref translate "$SU2_FILE_DEEPAK" "$SU2_FILE_DEEPAK_MESHB"

    mkdir -p OU_files ER_files

    # ------------------------------------------------------------------
    # 6. CFD: fix volumes -> convert -> submit -> wait (manual stop)
    # ------------------------------------------------------------------
    echo ""
    echo "Fixing negative volumes..."
    python3 fix_su2.py "$SU2_FILE_DEEPAK"

    echo ""
    echo "Converting SU2 -> CFD++ (.msh)..."
    convert25 "$SU2_FILE_DEEPAK"
    echo "Generated: ${SU2_FILE_DEEPAK%.su2}.msh"

    echo ""
    echo "Submitting CFD PBS job..."
    submit_and_wait "cfd" "$CFD_DIR" \
        "qsub jobscript_cfd.pbs" \
        "$CFD_CHECK_INTERVAL"

    # ------------------------------------------------------------------
    # 7. Export VTK
    # ------------------------------------------------------------------
    if [ -f "pltosout.bin" ]; then
        echo ""
        echo "Generating VTK..."
        genplif pltosout.bin vtk
        source ~/manish_vtk/bin/activate
        python3 residue.py
    else
        echo "WARNING: pltosout.bin not found - skipping VTK export"
    fi

    # ------------------------------------------------------------------
    # 8. Heat analysis
    # ------------------------------------------------------------------
    if [ -f "mcfdsol_bc1.vtk" ]; then
        echo ""
        echo "Running heat analysis..."
        ./cfd_post.sh
    else
        echo "WARNING: mcfdsol_bc1.vtk not found - skipping heat analysis"
    fi

    # cwd is now the new folder; hand back the mesh for the next run.
    RESULT_MESHB="$SU2_FILE_DEEPAK_MESHB"
    echo ""
    echo "Run for folder $NEW_FILE complete. Next input mesh: $RESULT_MESHB"
}

# ============================================================================
# MAIN CAMPAIGN LOOP
# ============================================================================

# We must be sitting in the folder that holds the previous solution before
# each run_one call. We start in the 00 folder.
cd "$START_DIR"

CURRENT_INPUT_MESH="$INITIAL_INPUT_MESH"
COMPLEXITY=$START_COMPLEXITY

for (( ci=1; ci<=N_COMPLEXITIES; ci++ )); do

    echo ""
    echo "###############################################################"
    echo "# COMPLEXITY LEVEL $ci / $N_COMPLEXITIES   (complexity = $COMPLEXITY)"
    echo "###############################################################"

    for (( ri=1; ri<=RUNS_PER_COMPLEXITY; ri++ )); do

        FOLDER=$(folder_name "$ci" "$ri")
        OUTPUT_MESH="adapt_${FOLDER}.meshb"

        echo ""
        echo "==============================================================="
        echo "  Complexity $COMPLEXITY  |  run $ri/$RUNS_PER_COMPLEXITY  |  folder $FOLDER"
        echo "==============================================================="

        run_one "$CURRENT_INPUT_MESH" "$OUTPUT_MESH" "$COMPLEXITY" "$FOLDER"

        # After run_one, cwd = ../<FOLDER> and RESULT_MESHB holds this run's
        # Deepak .meshb. It becomes the input for the next run (chained).
        CURRENT_INPUT_MESH="$RESULT_MESHB"
    done

    # ramp complexity for the next level
    COMPLEXITY=$(( COMPLEXITY * COMPLEXITY_FACTOR ))
done

echo ""
echo "==============================================================="
echo " ALL COMPLEXITY LEVELS AND RUNS COMPLETED"
echo "==============================================================="
