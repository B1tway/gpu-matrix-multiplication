__kernel void simple_add(
   __global float* X,
   __global float* Y,
   __global float* S,
    const int N){
    int i = get_global_id(0);
    int j = get_global_id(1);
    S[j * N + i] = X[j * N + i] + Y[j * N + i];

}   