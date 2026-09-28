#!/bin/bash -l
#SBATCH --job-name=tabicl-traj-watch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly TRAJ_ROOT="${TRAJECTORY_ROOT:?TRAJECTORY_ROOT is required}"
readonly EVAL_SCRIPT="${PROJECT_DIR}/scripts/slurm/evaluate_tabicl_v2_decoder_trajectory.sbatch"
previous_eval=""

for step in $(seq 500 500 10000); do
    checkpoints=(
        "${TRAJ_ROOT}/slot-k4-seed2402/step-${step}.ckpt"
        "${TRAJ_ROOT}/slot-k8-seed2402/step-${step}.ckpt"
        "${TRAJ_ROOT}/slot-k16-seed2402/step-${step}.ckpt"
        "${TRAJ_ROOT}/slot-k32-seed2402/step-${step}.ckpt"
        "${TRAJ_ROOT}/mlp-random-seed2402/step-${step}.ckpt"
        "${TRAJ_ROOT}/tabpfnv3-random-seed2402/step-${step}.ckpt"
    )
    echo "Waiting for all six step-${step} checkpoints."
    while :; do
        ready=true
        for checkpoint in "${checkpoints[@]}"; do
            if [[ ! -s "${checkpoint}" ]]; then ready=false; break; fi
        done
        if [[ "${ready}" == true ]]; then break; fi
        sleep 20
    done
    for checkpoint in "${checkpoints[@]}"; do
        previous_size=""
        stable=0
        while (( stable < 2 )); do
            size=$(stat -c %s "${checkpoint}")
            if [[ "${size}" == "${previous_size}" && "${size}" -gt 0 ]]; then
                ((stable += 1))
            else
                stable=0
                previous_size="${size}"
            fi
            sleep 2
        done
    done
    if [[ -n "${previous_eval}" ]]; then
        eval_job=$(sbatch --parsable --dependency="afterany:${previous_eval}" \
            --export="ALL,TRAJECTORY_ROOT=${TRAJ_ROOT},TRAJECTORY_STEP=${step}" "${EVAL_SCRIPT}")
    else
        eval_job=$(sbatch --parsable \
            --export="ALL,TRAJECTORY_ROOT=${TRAJ_ROOT},TRAJECTORY_STEP=${step}" "${EVAL_SCRIPT}")
    fi
    previous_eval="${eval_job%%;*}"
    echo "Submitted step-${step} TabArena evaluation job ${previous_eval}."
done
echo "All trajectory evaluation jobs have been submitted."
