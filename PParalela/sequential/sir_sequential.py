import numpy as np
import time
import os
import argparse
import pandas as pd

def run_simulation(grid_size=1000, days=365, beta=0.3, gamma=0.1, seed=42):
    np.random.seed(seed)
    
    # CONFIGURACIÓN INICIAL
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid[grid_size // 2, grid_size // 2] = 1
    
    stats = []
    
    # LÓGICA DE VECINDAD
    def count_infected_neighbors(g):
        infected = (g == 1).astype(np.int8)
        counts = np.zeros_like(infected, dtype=np.int8)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                counts += np.roll(np.roll(infected, di, axis=0), dj, axis=1)
        return counts

    start_time = time.time()
    
    # SIMULACIÓN DIARIA
    for day in range(days):
        current_infected = (grid == 1).sum()
        current_susceptible = (grid == 0).sum()
        current_removed = (grid == 2).sum()
        
        stats.append({
            'day': day,
            'S': int(current_susceptible),
            'I': int(current_infected),
            'R': int(current_removed)
        })
        
        if current_infected == 0 and day > 0:
            print(f"Epidemic ended on day {day}")
            break
            
        next_grid = grid.copy()
        
        # PROCESAMIENTO DE FILAS
        for r in range(grid_size):
            r_prev = (r - 1) % grid_size
            r_next = (r + 1) % grid_size
            
            row_prev = grid[r_prev]
            row_curr = grid[r]
            row_next_vals = grid[r_next]
            
            inf_prev = (row_prev == 1).astype(np.int8)
            inf_curr = (row_curr == 1).astype(np.int8)
            inf_next = (row_next_vals == 1).astype(np.int8)
            
            counts = (
                np.roll(inf_prev, 1) + inf_prev + np.roll(inf_prev, -1) +
                np.roll(inf_curr, 1) +            np.roll(inf_curr, -1) +
                np.roll(inf_next, 1) + inf_next + np.roll(inf_next, -1)
            )
            
            rng = np.random.default_rng([seed, day, r])
            
            # REGLAS DE TRANSICIÓN
            mask_S = row_curr == 0
            prob_inf = 1 - (1 - beta)**counts
            newly_infected = mask_S & (rng.random(grid_size) < prob_inf)
            
            mask_I = row_curr == 1
            newly_removed = mask_I & (rng.random(grid_size) < gamma)
            
            next_row = row_curr.copy()
            next_row[newly_infected] = 1
            next_row[newly_removed] = 2
            next_grid[r] = next_row
            
        grid = next_grid
        
    end_time = time.time()
    total_time = end_time - start_time
    
    return stats, grid, total_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="results/sequential_stats.csv")
    args = parser.parse_args()
    
    print(f"Running sequential simulation {args.size}x{args.size} for {args.days} days...")
    stats, final_grid, duration = run_simulation(args.size, args.days, args.beta, args.gamma)
    print(f"Completed in {duration:.2f} seconds.")
    
    df = pd.DataFrame(stats)
    df.to_csv(args.output, index=False)
    
    # EXPORTACIÓN
    np.save("results/sequential_final_grid.npy", final_grid)
    
    with open("results/sequential_time.txt", "w") as f:
        f.write(str(duration))
