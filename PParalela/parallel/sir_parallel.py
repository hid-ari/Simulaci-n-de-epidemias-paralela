import numpy as np
import time
import os
import argparse
from multiprocessing import Process, Barrier, Value, Array, shared_memory
import pandas as pd

# ESTADOS: 0=S, 1=I, 2=R

def worker(rank, num_processes, size, days, beta, gamma, seed, barrier, sm_name_A, sm_name_B, stats_array, stats_output):
    # ACCESO A MEMORIA COMPARTIDA
    existing_shm_A = shared_memory.SharedMemory(name=sm_name_A)
    grid_A = np.ndarray((size, size), dtype=np.int8, buffer=existing_shm_A.buf)
    existing_shm_B = shared_memory.SharedMemory(name=sm_name_B)
    grid_B = np.ndarray((size, size), dtype=np.int8, buffer=existing_shm_B.buf)
    
    # PARÁMETROS DE PROCESO
    rows_per_proc = size // num_processes
    start_row = rank * rows_per_proc
    end_row = (rank + 1) * rows_per_proc if rank != num_processes - 1 else size
    
    # CONFIGURACIÓN DETERMINISTA
    grids = [grid_A, grid_B]
    
    # BUCLE DE SIMULACIÓN
    for day in range(days):
        grid_now = grids[day % 2]
        grid_next = grids[(day + 1) % 2]
        
        # REDUCCIÓN DE ESTADÍSTICAS
        local_S = (grid_now[start_row:end_row] == 0).sum()
        local_I = (grid_now[start_row:end_row] == 1).sum()
        local_R = (grid_now[start_row:end_row] == 2).sum()
        
        stats_array[rank * 3 + 0] = int(local_S)
        stats_array[rank * 3 + 1] = int(local_I)
        stats_array[rank * 3 + 2] = int(local_R)
        
        # SINCRONIZACIÓN
        barrier.wait()
        
        # PROCESAMIENTO DE FILAS
        for r in range(start_row, end_row):
            r_prev = (r - 1) % size
            r_next = (r + 1) % size
            
            row_prev = grid_now[r_prev]
            row_curr = grid_now[r]
            row_next_vals = grid_now[r_next]
            
            inf_prev = (row_prev == 1).astype(np.int8)
            inf_curr = (row_curr == 1).astype(np.int8)
            inf_next = (row_next_vals == 1).astype(np.int8)
            
            # VECINDAD
            counts = (
                np.roll(inf_prev, 1) + inf_prev + np.roll(inf_prev, -1) +
                np.roll(inf_curr, 1) +            np.roll(inf_curr, -1) +
                np.roll(inf_next, 1) + inf_next + np.roll(inf_next, -1)
            )
            
            rng = np.random.default_rng([seed, day, r])
            
            # REGLAS DE TRANSICIÓN
            mask_S = row_curr == 0
            prob_inf = 1 - (1 - beta)**counts
            newly_infected = mask_S & (rng.random(size) < prob_inf)
            
            mask_I = row_curr == 1
            newly_removed = mask_I & (rng.random(size) < gamma)
            
            next_row = row_curr.copy()
            next_row[newly_infected] = 1
            next_row[newly_removed] = 2
            grid_next[r] = next_row
            
        # PUNTO DE SINCRONIZACIÓN
        barrier.wait()
        
        # AGREGACIÓN GLOBAL
        if rank == 0:
            total_S = 0
            total_I = 0
            total_R = 0
            for i in range(num_processes):
                total_S += stats_array[i * 3 + 0]
                total_I += stats_array[i * 3 + 1]
                total_R += stats_array[i * 3 + 2]
            
            if not hasattr(worker, 'daily_stats'):
                worker.daily_stats = []
            worker.daily_stats.append({
                'day': day,
                'S': int(total_S),
                'I': int(total_I),
                'R': int(total_R)
            })

    if rank == 0:
        df = pd.DataFrame(worker.daily_stats)
        df.to_csv(stats_output, index=False)
        
    existing_shm_A.close()
    existing_shm_B.close()

def run_parallel(size=1000, days=365, beta=0.3, gamma=0.1, num_processes=4, seed=42, stats_output="results/parallel_stats.csv"):
    # ASIGNACIÓN DE MEMORIA
    shm_A = shared_memory.SharedMemory(create=True, size=size*size)
    shm_B = shared_memory.SharedMemory(create=True, size=size*size)
    
    grid_A = np.ndarray((size, size), dtype=np.int8, buffer=shm_A.buf)
    grid_B = np.ndarray((size, size), dtype=np.int8, buffer=shm_B.buf)
    
    grid_A[:] = 0
    grid_B[:] = 0
    grid_A[size // 2, size // 2] = 1
    
    # CONTROL DE PROCESOS
    stats_array = Array('i', 3 * num_processes)
    barrier = Barrier(num_processes)
    
    processes = []
    start_time = time.time()
    
    for i in range(num_processes):
        p = Process(target=worker, args=(i, num_processes, size, days, beta, gamma, seed, barrier, shm_A.name, shm_B.name, stats_array, stats_output))
        p.start()
        processes.append(p)
        
    # GESTIÓN DE SALIDA
    
    for p in processes:
        p.join()
        
    total_time = time.time() - start_time
    
    # EXTRACCIÓN DE RESULTADOS
    final_grid = grid_A if (days % 2 == 0) else grid_B
    final_grid_result = final_grid.copy()
    
    # LIMPIEZA
    shm_A.close()
    shm_A.unlink()
    shm_B.close()
    shm_B.unlink()
    
    return total_time, final_grid_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_output", type=str, default="results/parallel_stats.csv")
    parser.add_argument("--time_output", type=str, default="results/parallel_time.txt")
    args = parser.parse_args()
    
    print(f"Running parallel simulation {args.size}x{args.size} with {args.cores} cores for {args.days} days (Seed: {args.seed})...")
    
    # EJECUCIÓN
    duration, final_grid = run_parallel(args.size, args.days, args.beta, args.gamma, args.cores, args.seed, args.stats_output)
    print(f"Completed in {duration:.2f} seconds.")
    
    with open(args.time_output, "w") as f:
        f.write(str(duration))
    
    np.save("results/parallel_final_grid.npy", final_grid)
