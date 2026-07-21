#!/bin/bash
source ~/manish_vtk/bin/activate
# ============================================================
# CONFIG: complexity sweep + repeat structure
# ============================================================
#
# Folder / file naming:
#   Complexity level 1 -> 01, 01_2, 01_3, 01_4, 01_5   (5 repeats)
#   Complexity level 2 -> 02, 02_2, 02_3, 02_4, 02_5
#   ...
#
# Each complexity level uses the SAME complexity value for all 5 repeats.
# Going from one level to the next MULTIPLIES the complexity by 2.
# Within a level, each repeat re-adapts the mesh using the PREVIOUS
# repeat's adapted mesh as its input (chained refinement), so the mesh
# keeps refining even though the target complexity is held fixed.
#
# A CFD job is only considered "done" when residuals in mcfd.rhsav have
# PLATEAUED (stagnated) for STAGNATION_WINDOW consecutive samples, using
# the same normalization/plotting technique as residue.py. The job is
# then qdel'd (or left to finish naturally if it ends first).

# ============================================================
# CHECK INPUT
# ============================================================

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_mesh>"
    echo ""
    echo "All other parameters (starting complexity, multiplier, number of"
    echo "complexity levels, repeats per level, residual plateau settings)"
    echo "are configured in the CONFIG block at the top of this script."
    exit 1
fi

INPUT_MESH_ORIG=$1

CURRENT_DIR=$(pwd)

# ------------------------------------------------------------
# SWEEP PARAMETERS
# ------------------------------------------------------------
START_COMPLEXITY=10000          # complexity used for level 01
COMPLEXITY_MULTIPLIER=2        # complexity x2 at each new level
NUM_LEVELS=5                   # number of complexity levels (01..05)
REPEATS_PER_LEVEL=5            # repeats per level (01, 01_2 .. 01_5)

# ------------------------------------------------------------
# RESIDUAL PLATEAU / CONVERGENCE PARAMETERS
# ------------------------------------------------------------
CHECK_INTERVAL=60             # seconds between residual checks
STAGNATION_WINDOW=5           # number of trailing samples to inspect
STAGNATION_TOL=0.05            # max allowed spread (log10 units) in window
                                # over STAGNATION_WINDOW samples to call it
                                # "plateaued" -> converged
MAX_WAIT_CYCLES=0              # 0 = no hard cap; otherwise force-stop after
                                # this many CHECK_INTERVAL cycles even if not
                                # converged (safety net for runaway jobs)

# ============================================================
# DERIVED (per-iteration) FILE NAMES -- fixed names used INSIDE
# each run folder; only the run folder name changes.
# ============================================================

METRIC_FILE="output-metric.solb"
ADAPTED_MESH_NAME="adapt_tmp.meshb"     # local name of refined mesh
SU2_NAME="adapt_tmp.su2"

echo "----------------------------------"
echo "Starting mesh    : $INPUT_MESH_ORIG"
echo "Start complexity  : $START_COMPLEXITY"
echo "Multiplier        : x$COMPLEXITY_MULTIPLIER"
echo "Levels            : $NUM_LEVELS"
echo "Repeats/level     : $REPEATS_PER_LEVEL"
echo "Working dir       : $CURRENT_DIR"
echo "----------------------------------"

# This will be updated to point at the previous repeat's adapted mesh
# as we chain through repeats/levels.
CURRENT_INPUT_MESH="$INPUT_MESH_ORIG"
COMPLEXITY=$START_COMPLEXITY

# ============================================================
# OUTER LOOP: complexity levels
# ============================================================

for (( level=1; level<=NUM_LEVELS; level++ )); do

    LEVEL_TAG=$(printf "%02d" "$level")

    echo ""
    echo "=================================================="
    echo "COMPLEXITY LEVEL $LEVEL_TAG  (complexity = $COMPLEXITY)"
    echo "=================================================="

    # ========================================================
    # INNER LOOP: repeats at this complexity (chained refinement)
    # ========================================================

    for (( rep=1; rep<=REPEATS_PER_LEVEL; rep++ )); do

        if [ "$rep" -eq 1 ]; then
            RUN_TAG="$LEVEL_TAG"
        else
            RUN_TAG="${LEVEL_TAG}_${rep}"
        fi

        echo ""
        echo "--------------------------------------------------"
        echo "RUN $RUN_TAG  (level $LEVEL_TAG, repeat $rep/$REPEATS_PER_LEVEL, complexity $COMPLEXITY)"
        echo "Input mesh for this run: $CURRENT_INPUT_MESH"
        echo "--------------------------------------------------"

        OUTPUT_MESH="$ADAPTED_MESH_NAME"
        SU2_FILE="$SU2_NAME"

        # ====================================================
        # SUBMIT REFINE PBS JOB
        # ====================================================

        echo ""
        echo "Submitting refinement PBS job..."

        REFINE_SSH_SCRIPT="$CURRENT_DIR/refine_ssh_$$.sh"

        cat > "$REFINE_SSH_SCRIPT" << ENDREFINE
#!/bin/bash

cd $CURRENT_DIR

JOB_ID=\$(qsub -N refine_job \
    -v INPUT_MESH=$CURRENT_INPUT_MESH,OUTPUT_MESH=$OUTPUT_MESH,COMPLEXITY=$COMPLEXITY,NEW_FILE=$RUN_TAG \
    jobscript_refine.pbs)

echo "Submitted Job ID: \$JOB_ID"

echo ""
echo "Waiting for refine job to finish..."

E_COUNTER=0

while qstat -f "\$JOB_ID" >/dev/null 2>&1; do

    JOB_STATE=\$(qstat -f "\$JOB_ID" 2>/dev/null | grep "job_state" | awk '{print \$3}')

    echo "Current job state: \$JOB_STATE"

    if [ "\$JOB_STATE" = "E" ]; then

        E_COUNTER=\$((E_COUNTER + 1))

        echo "Job in E state for \$E_COUNTER checks..."

        if [ "\$E_COUNTER" -ge 2 ]; then

            echo "Job stuck in E state. Force deleting..."

            qdel -W force "\$JOB_ID"

            break
        fi

    else
        E_COUNTER=0
    fi

    sleep 10

done

echo "Refine job completed!"

exit
ENDREFINE

        chmod +x "$REFINE_SSH_SCRIPT"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no yoda1 "bash $REFINE_SSH_SCRIPT"
        rm -f "$REFINE_SSH_SCRIPT"

        cd "$CURRENT_DIR"

        # ====================================================
        # RUN FOLDER SETUP
        # ====================================================

        mkdir -p "../$RUN_TAG"

        mv "$SU2_FILE" "../$RUN_TAG/"
        mv "$OUTPUT_MESH" "../$RUN_TAG/"

        cp automate.sh \
           cfd_post.sh analyze_heat.py *.csv residue.py \
           *.inp \
           writesol_volnsurf.py \
           jobscript_* "../$RUN_TAG/"

        cd "../$RUN_TAG/"

        echo ""
        echo "----------------------------------"
        echo "Refinement for run $RUN_TAG completed"
        echo "----------------------------------"

        # ====================================================
        # FIX SU2
        # ====================================================

        CURRENT_SU2="$SU2_FILE"

        echo ""
        echo "Fixing negative volumes..."

        

        # ====================================================
        # CONVERT TO CFD++
        # ====================================================

        echo ""
        echo "Converting SU2 -> CFD++"

        convert25 "$CURRENT_SU2"

        echo "Generated"

        # ====================================================
        # SUBMIT CFD PBS JOB
        # ====================================================

        CFD_DIR=$(pwd)

        echo ""
        echo "Submitting CFD PBS job..."

        SSH_SCRIPT="$CFD_DIR/cfd_ssh_$$.sh"
        cat > "$SSH_SCRIPT" << ENDSSH
#!/bin/bash

cd $CFD_DIR

JOB_ID=\$(qsub jobscript_cfd.pbs)
JOB_ID=\$(echo \$JOB_ID | tr -d '\r')

echo "Submitted CFD Job ID: \$JOB_ID"
echo "\$JOB_ID" > .cfd_job_id

exit
ENDSSH

        chmod +x "$SSH_SCRIPT"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no yoda1 "bash $SSH_SCRIPT"
        rm -f "$SSH_SCRIPT"

        # Fetch the job ID we just wrote on the remote side back to here.
        # (We are running in the same shared filesystem path, so the file
        # written by the ssh session lands directly in $CFD_DIR.)
        if [ -f "$CFD_DIR/.cfd_job_id" ]; then
            CFD_JOB_ID=$(cat "$CFD_DIR/.cfd_job_id")
        else
            echo "WARNING: could not determine CFD job ID, skipping wait."
            CFD_JOB_ID=""
        fi

        echo "CFD Job ID: $CFD_JOB_ID"

        # ====================================================
        # MONITOR RESIDUALS UNTIL PLATEAU (residue.py technique)
        # via SSH so polling happens on the cluster side where
        # mcfd.rhsav actually lives.
        # ====================================================

        if [ -n "$CFD_JOB_ID" ]; then

            echo ""
            echo "Monitoring mcfd.rhsav for residual plateau..."

            MONITOR_SSH_SCRIPT="$CFD_DIR/monitor_ssh_$$.sh"

            cat > "$MONITOR_SSH_SCRIPT" << ENDMON
#!/bin/bash

cd $CFD_DIR

JOB_ID="$CFD_JOB_ID"
E_COUNTER=0
WAIT_CYCLES=0
CONVERGED=0

while qstat | grep -q "\$JOB_ID"; do

    JOB_STATE=\$(qstat -f "\$JOB_ID" 2>/dev/null | grep "job_state" | awk '{print \$3}')
    echo "CFD job state: \$JOB_STATE"

    if [ "\$JOB_STATE" = "E" ]; then
        E_COUNTER=\$((E_COUNTER + 1))
        echo "CFD job in E state for \$E_COUNTER checks..."
        if [ "\$E_COUNTER" -ge 2 ]; then
            echo "CFD job stuck in E state. Force deleting..."
            qdel -W force "\$JOB_ID"
            break
        fi
    else
        E_COUNTER=0
    fi

    if [ -f "mcfd.rhsav" ]; then
        STATUS=\$(python3 check_convergence.py mcfd.rhsav $STAGNATION_WINDOW $STAGNATION_TOL)
        echo "Residual status: \$STATUS"

        if [ "\$STATUS" = "CONVERGED" ]; then
            echo "Residuals have plateaued. Stopping CFD job."
            qdel -W force "\$JOB_ID" 2>/dev/null
            CONVERGED=1
            break
        fi
    else
        echo "mcfd.rhsav not yet present..."
    fi

    WAIT_CYCLES=\$((WAIT_CYCLES + 1))
    if [ "$MAX_WAIT_CYCLES" -gt 0 ] && [ "\$WAIT_CYCLES" -ge "$MAX_WAIT_CYCLES" ]; then
        echo "Reached MAX_WAIT_CYCLES ($MAX_WAIT_CYCLES) without plateau. Force stopping."
        qdel -W force "\$JOB_ID" 2>/dev/null
        break
    fi

    sleep $CHECK_INTERVAL

done

if [ "\$CONVERGED" -eq 1 ]; then
    echo "CONVERGED" > .convergence_status
else
    echo "STOPPED" > .convergence_status
fi

echo "CFD monitoring loop finished."
exit
ENDMON

            chmod +x "$MONITOR_SSH_SCRIPT"
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no yoda1 "bash $MONITOR_SSH_SCRIPT"
            rm -f "$MONITOR_SSH_SCRIPT"

            if [ -f "$CFD_DIR/.convergence_status" ]; then
                CONV_STATUS=$(cat "$CFD_DIR/.convergence_status")
                echo "Convergence status for run $RUN_TAG: $CONV_STATUS"
                rm -f "$CFD_DIR/.convergence_status"
            fi
            rm -f "$CFD_DIR/.cfd_job_id"
        fi

        # ====================================================
        # EXPORT VTK
        # ====================================================

        if [ -f "pltosout.bin" ]; then
            echo ""
            echo "Generating VTK..."
            genplif pltosout.bin vtk
            source ~/manish_vtk/bin/activate
            python3 residue.py
            ./cfd_post.sh
        else
            echo "WARNING: pltosout.bin not found"
        fi

        
        # ====================================================
        # CHAIN: next repeat's input mesh = this repeat's adapted mesh
        # ====================================================

        CURRENT_INPUT_MESH="$(pwd)/$OUTPUT_MESH"
        CURRENT_DIR="$(pwd)"

        echo ""
        echo "Run $RUN_TAG complete. Next input mesh: $CURRENT_INPUT_MESH"

    done  # end repeats loop

    # increase complexity for the next level
    COMPLEXITY=$((COMPLEXITY * COMPLEXITY_MULTIPLIER))

done  # end levels loop

echo ""
echo "===================================================="
echo "All $NUM_LEVELS complexity levels x $REPEATS_PER_LEVEL repeats completed."
echo "===================================================="
