# Умножение матриц на GPU

## Описание кернелов
### Kernel1
Наивный алгоритм умножения матриц без каких-либо оптимизаций
### Kernel2_1
Умноженение матриц с использованием локальной памяти
### Kernel2_2
Умноженение матриц с использованием локальной памяти и транспонированием
### Kernel3
Умноженение матриц с использованием локальной памяти и увеличением работы на один work-item