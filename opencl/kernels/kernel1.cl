__kernel void simple_mul(
        __global float* X,
        __global float* Y,
        __global float* S,
        const int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float res = 0;
    for (int k = 0; k < N; k++) {
        res += X[j * N + k] * Y[k * N + i];
    }
    S[j * N + i] = res;

}