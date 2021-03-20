#define TILE_SIZE 32
#define WPT 16
#define RTS 2
#define WIDTH 4
__kernel void wide_matrix_mul(const int M, const int N, const int K,
    const __global float4* A,
    const __global float4* B,
    __global float4* C) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = (TILE_SIZE / 4) * get_group_id(0) + lx; /
    const int gy = TILE_SIZE * get_group_id(1) + ly; 
    __local float4 tileA[TILE_SIZE][TILE_SIZE / WIDTH];
    __local float4 tileB[TILE_SIZE][TILE_SIZE / WIDTH];
    const int numTiles = K / TILE_SIZE;
    float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t = 0; t < numTiles; t++) {
        const int tx = (TILE_SIZE / WIDTH) * tile + lx;
        const int ty = TILE_SIZE * tile + ly;
        tileA[ly][lx] = A[ty * (M / WIDTH) + gx];
        tileB[ly][lx] = B[gy * (K / WIDTH) + tx];
        barrier(CLK_LOCAL_MEM_FENCE);

       
        float4 vecA, vecB;
        float valB;
        for (int k = 0; k < TILE_SIZE / WIDTH; k++) {
            vecB = tileB[ly][k];
            for (int w = 0; w < WIDTH; w++) {
                vecA = tileA[WIDTH * k + w][lx];
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
