
__kernel void matrix_transpose(__global float* a,
    __global float* b,
    unsigned int rows,
    unsigned int cols)
{
    
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    if (gx < rows && gy < cols)
        b[gx * rows + gy] = a[gy * rows + gx];

}