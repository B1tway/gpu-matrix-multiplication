#define TILE_SIZE 32
__kernel void local_mul(__global float *X, __global float *Y, __global float *S,const int M, const int N,
                        const int K) {
  size_t lx = get_local_id(0);
  size_t ly = get_local_id(1);
  size_t gx = get_global_id(0);
  size_t gy = get_global_id(1);
  __local tileA[TILE_SIZE][TILE_SIZE];
  __local tileB[TILE_SIZE][TILE_SIZE];

  float res = 0.0f;
  size_t tiles = K /TILE_SIZE;
  for(size_t tile = 0; tile < tiles; tile++) {
    const int tx = TILE_SIZE * tile + lx;
    const int ty = TILE_SIZE * tile + ly;
    tileA[ly][lx] = X[tx * M + gx];
    tileB[ly][lx] = Y[gy * k + tx];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(size_t k = 0; k < TILE_SIZE; k++) {
      res += tileA[k][lx] * tileB[ly][k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  S[gy * M + gx] = res;

}