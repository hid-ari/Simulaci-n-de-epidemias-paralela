import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    grid_size = 1000
    days = 100
    
    results = []
    
    # SIMULACIÓN SECUENCIAL
    print("=== Running Sequential ===")
    start = time.time()
    run_cmd(f"py sequential/sir_sequential.py --size {grid_size} --days {days}")
    seq_time = time.time() - start
    results.append({'cores': 1, 'type': 'Sequential', 'time': seq_time})
    
    seq_grid = np.load("results/sequential_final_grid.npy")
    
    # ESCALADO PARALELO
    for cores in [1, 2, 4, 8]:
        print(f"=== Running Parallel with {cores} cores ===")
        start = time.time()
        run_cmd(f"py parallel/sir_parallel.py --size {grid_size} --days {days} --cores {cores}")
        par_time = time.time() - start
        
        # VALIDACIÓN
        par_grid = np.load("results/parallel_final_grid.npy")
        is_identical = np.array_equal(seq_grid, par_grid)
        print(f"Identical to sequential: {is_identical}")
        
        results.append({'cores': cores, 'type': f'Parallel {cores}', 'time': par_time, 'identical': is_identical})

    # EXPORTACIÓN
    df = pd.DataFrame(results)
    df.to_csv("results/scaling_results.csv", index=False)
    
    # GRÁFICAS DE DESEMPEÑO
    
    plt.figure(figsize=(10, 5))
    
    # GRÁFICA DE TIEMPO
    plt.subplot(1, 2, 1)
    cores_list = [1, 2, 4, 8]
    p_times = [res['time'] for res in results[1:]]
    plt.plot(cores_list, p_times, marker='o', label='Parallel')
    plt.axhline(y=seq_time, color='r', linestyle='--', label='Sequential')
    plt.xlabel('Cores')
    plt.ylabel('Time (s)')
    plt.title('Execution Time vs Cores')
    plt.legend()
    plt.grid(True)
    
    # GRÁFICA DE SPEEDUP
    plt.subplot(1, 2, 2)
    speedups = [seq_time / t for t in p_times]
    plt.plot(cores_list, speedups, marker='s', color='g', label='Measured')
    plt.plot(cores_list, cores_list, linestyle=':', color='gray', label='Ideal')
    plt.xlabel('Cores')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Cores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/scaling_plot.png")
    print("Scaling plot saved to results/scaling_plot.png")

if __name__ == "__main__":
    main()
