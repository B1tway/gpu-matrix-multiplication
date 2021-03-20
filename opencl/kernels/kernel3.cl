#define TILE_SIZE 32
#define WPT 16
#define RTS 2
__kernel void local_per_thread_mul(
    __global float* X,
    __global float* Y,
    __global float* S,
    const int K) {
    const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    const int globalRow = TILE_SIZE * get_group_id(0) + row;
    const int globalCol = TILE_SIZE * get_group_id(1) + col; 
    __local float localX[TILE_SIZE][TILE_SIZE];
    __local float localY[TILE_SIZE][TILE_SIZE];
    float res[WPT];
    for (int w = 0; w < WPT; w++) {
        res[w] = 0.0f;
    }
    for (int t = 0; t < K / TILE_SIZE; t++) {
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TILE_SIZE * t + row;
            const int tiledCol = TILE_SIZE * t + col;
            localX[col + w * RTS][row] = X[(tiledCol + w * RTS) * K + globalRow];
            localY[col + w * RTS][row] = Y[(globalCol + w * RTS) * K + tiledRow];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int w = 0; w < WPT; w++) {
                res[w] += localX[k][row] * localY[col + w * RTS][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++) {
        S[(globalCol + w * RTS) * K + globalRow] = res[w];
    }
}
