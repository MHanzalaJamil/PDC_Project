
#!/bin/bash

set -euo pipefail

RUNS=5
CSV="benchmark_results.csv"
LOG="benchmark_log.txt"

CONFIGS=(
    "2 2"
    "2 4"
    "4 2"
    "4 4"
    "4 8"
)

echo "Implementation,MPI_Processes,OpenMP_Threads,Workers,Run,Iterations,Execution_Time_Seconds" > "$CSV"
: > "$LOG"

echo "Starting K-Means Benchmarking..."
echo "Runs per configuration: $RUNS"

run_benchmark() {
    local implementation=$1
    local mpi=$2
    local threads=$3
    local workers=$4
    local run=$5

    local output
    local elapsed
    local iterations

    if [ "$implementation" = "Sequential" ]; then
        output=$(./sequential)
    else
        output=$(OMP_NUM_THREADS="$threads" \
                 OMP_DYNAMIC=FALSE \
                 mpirun --allow-run-as-root -np "$mpi" ./hybrid)
    fi

    {
        echo "========================================"
        echo "$implementation | MPI=$mpi | OMP=$threads | Run=$run"
        echo "========================================"
        echo "$output"
        echo
    } >> "$LOG"

    # Extract execution time and iterations
    elapsed=$(echo "$output" | awk '/Execution Time/ {print $4}')
    iterations=$(echo "$output" | awk '/Iterations/ {print $3}')

    if [[ -z "$elapsed" || -z "$iterations" ]]; then
        echo "Failed to extract results. Check $LOG"
        exit 1
    fi

    # Save CSV row
    echo "$implementation,$mpi,$threads,$workers,$run,$iterations,$elapsed" >> "$CSV"

    echo "$implementation | MPI=$mpi | OMP=$threads | Run=$run | Time=$elapsed s | Iterations=$iterations"
}

echo
echo "========== SEQUENTIAL BENCHMARK =========="

for ((run=1; run<=RUNS; run++)); do
    run_benchmark "Sequential" 1 1 1 "$run"
done

echo
echo "========== HYBRID BENCHMARKS =========="

for config in "${CONFIGS[@]}"; do
    read -r mpi threads <<< "$config"

    workers=$((mpi * threads))

    echo
    echo "Testing MPI=$mpi, OpenMP=$threads, Workers=$workers"

    for ((run=1; run<=RUNS; run++)); do
        run_benchmark "Hybrid" "$mpi" "$threads" "$workers" "$run"
    done
done

echo
echo "========== BENCHMARKING COMPLETED =========="
echo "CSV Results : $CSV"
echo "Full Log    : $LOG"
