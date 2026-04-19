# Simulación SIR en Grilla 2-D (Paralelización con Python)

Este proyecto implementa un modelo epidemiológico SIR (Suscrito-Infectado-Recuperado) en una grilla de $1000 \times 1000$ personnas. Se comparan una implementación secuencial y una paralela utilizando `multiprocessing` y memoria compartida.

Reporte
https://github.com/hid-ari/Simulaci-n-de-epidemias-paralela/blob/main/PParalela/report.md

## Estructura del Proyecto

- `sequential/`: Código de la implementación secuencial.
- `parallel/`: Código de la implementación paralela (Domain Decomposition).
- `results/`: CSVs de estadísticas, tiempos y gráficas de scaling.
- `animations/`: GIFs comparativos del brote.
- `run_experiments.py`: Script para ejecutar las pruebas de desempeño (1, 2, 4, 8 núcleos).
- `visualize_results.py`: Script para generar la animación side-by-side.

## Requisitos

- Python 3.x
- NumPy, Pandas, Matplotlib, ImageIO

## Cómo ejecutar

1. **Simulación Secuencial**:
   ```bash
   python sequential/sir_sequential.py --size 1000 --days 365
   ```

2. **Simulación Paralela**:
   ```bash
   python parallel/sir_parallel.py --cores 4 --size 1000 --days 365
   ```

3. **Experimentos de Scaling**:
   ```bash
   python run_experiments.py
   ```

4. **Generar Animación**:
   ```bash
   python visualize_results.py
   ```

## Resultados Destacados

- Se logró **coincidencia bit a bit** entre la versión secuencial y paralela mediante el uso de generadores aleatorios sembrados por fila/día.
- El escalado fuerte muestra una reducción significativa de tiempo hasta 8 núcleos.
