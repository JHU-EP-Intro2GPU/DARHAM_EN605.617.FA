
__kernel void add(__global const int *a,
						__global const int *b,
						__global int *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}


__kernel void sub(__global const int *a,
						__global int  *b,
						__global int *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] - b[gid];
}


__kernel void mult(__global const int *a,
						__global const int *b,
						__global int *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] * b[gid];
}
