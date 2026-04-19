# Informe: Simulación SIR en Grilla 2-D con Paralelismo en Python

## 1. Modelo Matemático

El modelo SIR (Susceptible-Infectado-Recuperado) se implementa sobre una grilla bidimensional de $1000 \times 1000$ autómatas celulares. Cada celda representa una persona en uno de tres estados:

1.  **S (Suscrito - 0)**: Persona sana que puede contraer la enfermedad.
2.  **I (Infectado - 1)**: Persona enferma que puede contagiar a sus vecinos.
3.  **R (Removido - 2)**: Persona recuperada o fallecida, que ya no participa en el ciclo epidemiológico.

### Dinámica de Transición
En cada paso de tiempo (día), las transiciones se rigen por:

*   **S → I**: Una celda susceptible se infecta con una probabilidad dependiente de sus vecinos infectados. Usamos la vecindad de **Moore (8 vecinos)**.
    $$P(S \to I) = 1 - (1 - \beta)^k$$
    Donde $\beta$ es la tasa de transmisión por contacto y $k$ es el número de vecinos infectados.
*   **I → R**: Una celda infectada se recupera con una probabilidad fija $\gamma$.
    $$P(I \to R) = \gamma$$

$R_0$ teórico para este modelo en grilla es aproximadamente $R_0 \approx 8 \cdot (\beta / \gamma)$ si ignoramos la saturación espacial inicial.

## 2. Implementación de Paralelismo

### Descomposición de Dominio
La grilla se divide en **franjas horizontales** (strips). Cada proceso es responsable de actualizar una franja.

### Celdas Fantasma (Ghost Cells)
Debido a que la actualización de una celda depende de sus 8 vecinos, las celdas en los bordes de una franja necesitan información de las filas adyacentes pertenecientes a otros procesos. 
En esta implementación, utilizamos **Memoria Compartida (`multiprocessing.shared_memory`)**, lo que permite que todos los procesos visualicen la grilla completa (evitando copias pesadas) mientras se sincronizan mediante **Barreras** para asegurar consistencia entre el día $t$ y $t+1$.

### Reducción de Estadísticas
Al final de cada día, cada proceso calcula sus totales locales de (S, I, R). Estos se agregan en un arreglo compartido para obtener el conteo global (Reducción Paralela).

## 3. Análisis de Resultados y Scaling

### Validación de Resultados
Se implementó un esquema de generación de números aleatorios basado en la tupla `(semilla, día, fila)`. Esto garantiza que, independientemente del número de núcleos utilizados, las decisiones estocásticas sean idénticas para cada celda. Se verificó que las grillas finales de las versiones de 1, 2, 4 y 8 núcleos son **binariamente idénticas** a la versión secuencial.

### Rendimiento (Strong Scaling)
Experimento ejecutado en grilla 1000x1000 por 100 días:

| Núcleos | Tiempo (s) | Speedup |
| :--- | :--- | :--- |
| 1 (Secuencial) | 10.57 s | 1.00x |
| 1 (Paralelo) | 10.80 s | 0.98x |
| 2 | 5.94 s | 1.78x |
| 4 | 3.46 s | 3.05x |
| 8 | 2.38 s | 4.43x |

*(Nota: Los valores muestran una ganancia significativa, alcanzando un speedup de 4.43x con 8 núcleos. La eficiencia disminuye ligeramente debido al overhead de sincronización (Barrier) y la gestión de procesos en Windows).*

## 4. Conclusiones

La paralelización basada en memoria compartida en Python es altamente efectiva para simulaciones de gran escala como el modelo SIR. La clave del éxito en este proyecto fue la gestión eficiente de la memoria y la sincronización mediante barreras, logrando bit-identity que facilita el debugging y la validación científica.
