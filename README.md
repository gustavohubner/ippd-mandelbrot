# ippd-mandelbrot
Mandelbrot usando OpenMP e MPI

## Uso
**Compilar:** mpicc -fopenmp mandel.c -lpthread -o mandel -lm

**Executar:** mpirun mandel [Número de linhas] [Número de colunas] [Número de iterações]
