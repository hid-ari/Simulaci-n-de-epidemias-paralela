import subprocess
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def run_monte_carlo(iterations=10, size=1000, days=200):
    print(f"Iniciando estudio de Monte Carlo con {iterations} iteraciones...")
    
    all_runs = []
    
    if not os.path.exists("results/mc"):
        os.makedirs("results/mc")
        
    for i in range(iterations):
        seed = 100 + i
        stats_file = f"results/mc/run_{i}.csv"
        print(f"Ejecutando iteración {i+1}/{iterations} (Semilla: {seed})...")
        
        # Ejecutar simulación paralela usando 8 núcleos por defecto
        cmd = f"py parallel/sir_parallel.py --size {size} --days {days} --cores 8 --seed {seed} --stats_output {stats_file}"
        subprocess.run(cmd, shell=True)
        
        # Leer resultados
        df = pd.read_csv(stats_file)
        all_runs.append(df)
        
    # Agregar y calcular estadísticas (Media y Desviación Estándar)
    # Concatenar todos los DataFrames
    combined = pd.concat(all_runs)
    
    # Agrupar por día y calcular estadísticas
    stats = combined.groupby('day').agg(['mean', 'std']).reset_index()
    
    # Aplanar las columnas multinivel
    stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]
    
    stats.to_csv("results/monte_carlo_aggregated.csv", index=False)
    print("Resultados de Monte Carlo guardados en results/monte_carlo_aggregated.csv")
    
    return stats

def plot_monte_carlo(stats):
    plt.figure(figsize=(12, 8))
    
    days = stats['day']
    
    colors = {'S': 'blue', 'I': 'red', 'R': 'green'}
    
    for state in ['S', 'I', 'R']:
        mean = stats[f'{state}_mean']
        std = stats[f'{state}_std']
        
        plt.plot(days, mean, label=f'{state} (Media)', color=colors[state], linewidth=2)
        plt.fill_between(days, mean - std, mean + std, color=colors[state], alpha=0.2, label=f'{state} (±1 std dev)')
        
    plt.title('Simulación SIR: Estudio de Monte Carlo (10 Iteraciones)')
    plt.xlabel('Día')
    plt.ylabel('Población')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/monte_carlo_plot.png')
    print("Gráfica de Monte Carlo guardada en results/monte_carlo_plot.png")

if __name__ == "__main__":
    stats = run_monte_carlo(iterations=10, size=1000, days=200)
    plot_monte_carlo(stats)
