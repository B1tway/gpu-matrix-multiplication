
__kernel void matrix_copy(__global float* a, __global float* b, unsigned int rows, unsigned int cols)
{
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    if(gy * rows + gx < rows * cols) b[gy * rows + gx] = a[gy * rows + gx];
}