import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import os
from sequential.sir_sequential import run_simulation as run_seq
# CONFIGURACIÓN VISUAL

def create_animation(size=400, days=250):
    print(f"Generando 'Carrera de Simulación' para grilla de {size}x{size}...")
    
    cmap = ListedColormap(['#e6f3ff', '#ff4d4d', '#737373'])
    
    # Basado en los resultados de scaling (10.5s vs 2.4s)
    # El paralelo es ~4x veces más rápido.
    speedup = 4.0 
    
    frames = []
    
    # Parámetros
    beta = 0.3
    gamma = 0.05
    seed = 42
    
    # Grillas independientes
    grid_seq = np.zeros((size, size), dtype=np.int8)
    grid_seq[size//2, size//2] = 1
    grid_par = grid_seq.copy()
    
    day_seq = 0
    day_par = 0
    
    def update_grid(grid, day_start, num_steps, size, seed, beta, gamma):
        current_grid = grid.copy()
        for d_offset in range(int(num_steps)):
            if day_start + d_offset >= days: break
            if (current_grid == 1).sum() == 0 and day_start + d_offset > 0: break
            
            next_grid = current_grid.copy()
            for r in range(size):
                r_prev, r_next = (r - 1) % size, (r + 1) % size
                row_prev, row_curr, row_next = current_grid[r_prev], current_grid[r], current_grid[r_next]
                inf_prev, inf_curr, inf_next = (row_prev==1).astype(np.int8), (row_curr==1).astype(np.int8), (row_next==1).astype(np.int8)
                counts = (np.roll(inf_prev, 1) + inf_prev + np.roll(inf_prev, -1) +
                          np.roll(inf_curr, 1) +            np.roll(inf_curr, -1) +
                          np.roll(inf_next, 1) + inf_next + np.roll(inf_next, -1))
                rng = np.random.default_rng([int(seed), int(day_start + d_offset), int(r)])
                newly_infected = (row_curr == 0) & (rng.random(size) < (1 - (1 - beta)**counts))
                newly_removed = (row_curr == 1) & (rng.random(size) < gamma)
                next_row = row_curr.copy()
                next_row[newly_infected], next_row[newly_removed] = 1, 2
                next_grid[r] = next_row
            current_grid = next_grid
        return current_grid

    # BUCLE DE GENERACIÓN DE FRAMES
    for f in range(120):
        # Paso base: 2 días por frame
        step_base = 2
        
        if day_seq < days:
            grid_seq = update_grid(grid_seq, day_seq, step_base, size, seed, beta, gamma)
            day_seq += step_base
            
        if day_par < days:
            grid_par = update_grid(grid_par, day_par, step_base * speedup, size, seed, beta, gamma)
            day_par += step_base * speedup
            
        # RENDERIZADO
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        ax1.imshow(grid_seq, cmap=cmap, vmin=0, vmax=2)
        ax1.set_title(f"SECUENCIAL (1 Núcleo)\nDía Actual: {min(day_seq, days)}\nProgreso: {min(day_seq/days*100, 100):.1f}%")
        ax1.axis('off')
        
        ax2.imshow(grid_par, cmap=cmap, vmin=0, vmax=2)
        ax2.set_title(f"PARALELO (8 Núcleos)\nDía Actual: {int(min(day_par, days))}\nPROGRESO: {min(day_par/days*100, 100):.1f}%")
        ax2.axis('off')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image_full = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(image_full)
        plt.close()
        
        if day_seq >= days and day_par >= days:
            break

    imageio.mimsave('animations/sir_comparison_speed.gif', frames, fps=12)
    print("Animación de 'Carrera' guardada en animations/sir_comparison_speed.gif")

if __name__ == "__main__":
    create_animation()
