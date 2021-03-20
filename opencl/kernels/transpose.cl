
#define TILE_SIZE 32

__kernel void matrix_transpose(__global float* a,
    __global float* b,
    unsigned int rows,
    unsigned int cols)
{
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    if (gx < rows && gy < cols) {
        tile[ly][lx] = a[gy * rows + gx];
        barrier(CLK_LOCAL_MEM_FENCE);
        b[gy * rows + gx] = tile[lx][ly];
    }
   
}