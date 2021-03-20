#define TILE_SIZE 32
#define WPT 16
#define RTS 2
__kernel void wide_matrix_mul(const int M, const int N, const int K,
    const __global floatX* A,
    const __global floatX* B,
    __global floatX* C) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = (TILE_SIZE / 4) * get_group_id(0) + lx; /
    const int gy = TILE_SIZE * get_group_id(1) + ly; 
    __local floatX tileA[TILE_SIZE][TILE_SIZE / 4];
    __local floatX tileB[TILE_SIZE][TILE_SIZE / 4];
    const int numTiles = K / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TILE_SIZE / WIDTH) * tile + lx;
        const int tiledCol = TILE_SIZE * tile + ly;
        tileA[ly][lx] = A[tiledCol * (M / WIDTH) + gx];
        tileB[ly][lx] = B[gy * (K / WIDTH) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valB;
        for (int k = 0; k < TILE_SIZE / WIDTH; k++) {
            vecB = tileB[ly][k];
            for (int w = 0; w < WIDTH; w++) {
                vecA = tileA[WIDTH * k + w][lx];
#if WIDTH == 1
                valB = vecB;
                acc += vecA * valB;
#elif WIDTH == 2
                switch (w) {
                case 0: valB = vecB.x; break;
                case 1: valB = vecB.y; break;
                }
                acc.x += vecA.x * valB;
                acc.y += vecA.y * valB;
#elif WIDTH == 4
                switch (w) {
                case 0: valB = vecB.x; break;
                case 1: valB = vecB.y; break;
                case 2: valB = vecB.z; break;
                case 3: valB = vecB.w; break;
                }
                acc.x += vecA.x * valB;
                acc.y += vecA.y * valB;
                acc.z += vecA.z * valB;
                acc.w += vecA.w * valB;
#endif
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[gy * (M / WIDTH) + gx] = acc;
}
