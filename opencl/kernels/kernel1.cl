__kernel void simple_mul(
        __global float* X,
        __global float* Y,
        __global float* S,
        const int M, const int N, const int K) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float res = 0;
    for (int k = 0; k < K; k++) {
        res += X[j * M + k] * Y[k * K + i];
    }
    S[j * M + i] = res;

}