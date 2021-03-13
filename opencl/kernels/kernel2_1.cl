#define TILE_SIZE 32
__kernel void local_mul(__global float *X, __global float *Y, __global float *S,
                        const int K) {
  int global_row = get_global_id(0);
  int global_col = get_global_id(1);
  int local_row = get_local_id(0);
  int local_col = get_local_id(1);

  __local float localX[TILE_SIZE][TILE_SIZE];
  __local float localY[TILE_SIZE][TILE_SIZE];
  float res = 0;
  for (int kg = 0; kg < K / TILE_SIZE; kg++) {
    int aid = global_col * K + (kg * TILE_SIZE + local_row);
    int bid = (kg * TILE_SIZE + local_col) * K + global_row;
    localX[local_col][local_row] = X[aid];
    localY[local_col][local_row] = Y[bid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < TILE_SIZE; i++) {
      res += localX[local_col][i] * localY[i][local_row];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  S[global_col * K + global_row] = res;
}