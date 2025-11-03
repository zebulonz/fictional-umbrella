/*
 Abstract:
 A command-line tool to benchmark Metal compute performance for operations
 relevant to LLM inference, specifically fp16 GEMM and softmax.
 It compares GPU results against CPU reference implementations for validation.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

//================ Embedded Metal Shaders ================
static NSString *kMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// Kernel: y = a*x + y (Single-precision A*X Plus Y)
kernel void saxpy(const device float *x    [[buffer(0)]],
                  device float *y          [[buffer(1)]],
                  constant float &a        [[buffer(2)]],
                  constant uint  &n        [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = fma(a, x[gid], y[gid]);
}

/*
 * Kernel: hgemm_outer_64x64
 *
 * Performs half-precision (fp16) matrix multiplication (C = A * B)
 * using a tiled approach.
 *
 * Tiling Strategy:
 * - Threadgroup Tile (Output): 64x64 (TILE_M, TILE_N)
 * - Thread Tile (Output): 4x4 (BLOCK_M, BLOCK_N)
 * - Threads per Threadgroup: 16x16 = 256
 * - K-Dimension Slice: 16 (TILE_K)
 */
kernel void hgemm_outer_64x64(const device half *A [[buffer(0)]],
                              const device half *B [[buffer(1)]],
                              device half *C       [[buffer(2)]],
                              constant uint &M     [[buffer(3)]],
                              constant uint &N     [[buffer(4)]],
                              constant uint &K     [[buffer(5)]],
                              uint2 tid [[thread_position_in_threadgroup]],
                              uint2 bid [[threadgroup_position_in_grid]]) {

    const uint BLOCK_M = 4, BLOCK_N = 4;    // 4x4 output per thread
    const uint TILE_M = 64, TILE_N = 64;    // 64x64 output tile per threadgroup
    const uint TILE_K = 16;                 // K-slice depth

    // Calculate thread's base indices within the 64x64 tile
    const uint thread_row = tid.y * BLOCK_M;
    const uint thread_col = tid.x * BLOCK_N;
    // Calculate threadgroup's base indices in the global C matrix
    const uint tile_row = bid.y * TILE_M;
    const uint tile_col = bid.x * TILE_N;

    // Tile storage in fast threadgroup memory
    threadgroup half Asub[TILE_M][TILE_K];
    threadgroup half Bsub[TILE_K][TILE_N];

    // Local register accumulators for the 4x4 thread tile
    float acc[BLOCK_M][BLOCK_N];
    for (uint i = 0; i < BLOCK_M; i++)
        for (uint j = 0; j < BLOCK_N; j++)
            acc[i][j] = 0.0f;

    // Iterate over K dimension in TILE_K-sized chunks
    for (uint k0 = 0; k0 < K; k0 += TILE_K) {
        
        // --- Collaborative Load from Global to Threadgroup Memory ---
        // Each of the 256 threads loads 4 elements of A and 4 elements of B
        // (256 * 4 = 1024 elements each, matching tile sizes)
        uint load_m = tid.y * 4 + tid.x / 4;       // 0..63
        uint load_k_a = (tid.x % 4) * 4 + (tid.y % 4); // spread across 16

        if (tile_row + load_m < M && k0 + load_k_a < K) {
            Asub[load_m][load_k_a] = A[(tile_row + load_m) * K + k0 + load_k_a];
        } else {
            Asub[load_m][load_k_a] = half(0); // Pad with zero
        }

        uint load_k_b = tid.y;                  // 0..15
        uint load_n = tid.x * 4 + (tid.y % 4);    // 0..63

        if (k0 + load_k_b < K && tile_col + load_n < N) {
            Bsub[load_k_b][load_n] = B[(k0 + load_k_b) * N + tile_col + load_n];
        } else {
            Bsub[load_k_b][load_n] = half(0); // Pad with zero
        }

        // Wait for all threads to finish loading tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute 4x4 output block via outer product over K-slice ---
        // All threads perform this compute on the same shared data
        #pragma unroll
        for (uint k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (uint i = 0; i < BLOCK_M; i++) {
                float a_val = float(Asub[thread_row + i][k]);
                #pragma unroll
                for (uint j = 0; j < BLOCK_N; j++) {
                    float b_val = float(Bsub[k][thread_col + j]);
                    acc[i][j] = fma(a_val, b_val, acc[i][j]);
                }
            }
        }
        
        // Wait for all threads to finish compute before loading next K-slice
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write 4x4 block from registers to global memory ---
    for (uint i = 0; i < BLOCK_M; i++) {
        for (uint j = 0; j < BLOCK_N; j++) {
            uint out_row = tile_row + thread_row + i;
            uint out_col = tile_col + thread_col + j;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = half(acc[i][j]);
            }
        }
    }
}

/*
 * Kernel: softmax_rowwise
 *
 * Computes softmax row-wise (over dimension N) for M rows.
 * Uses a stable, two-pass parallel reduction within each threadgroup.
 * Input/Output is fp32 for numerical stability.
 * Each threadgroup (of size e.g. 256) processes one row.
 */
kernel void softmax_rowwise(const device float *X [[buffer(0)]],
                            device float *Y       [[buffer(1)]],
                            constant uint &M      [[buffer(2)]],
                            constant uint &N      [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint3 tpg [[threads_per_threadgroup]],
                            uint3 tgpos [[threadgroup_position_in_grid]]) {

    // Shared memory for the parallel reduction
    threadgroup float sdataMax[256];
    threadgroup float sdataSum[256];
    
    // Each threadgroup processes one row
    uint row = tgpos.y;
    if (row >= M) return;

    // --- Pass 1: Find max value in the row ---
    float localMax = -INFINITY;
    for (uint j = tid; j < N; j += tpg.x) { // Grid-stride loop
        localMax = max(localMax, X[row*N + j]);
    }
    sdataMax[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find the single max value
    for (uint stride = tpg.x>>1; stride>0; stride >>= 1) {
        if (tid < stride) sdataMax[tid] = max(sdataMax[tid], sdataMax[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxv = sdataMax[0]; // Row's max value

    // --- Pass 2: Calculate sum of exp(x - maxv) ---
    float localSum = 0.0f;
    for (uint j = tid; j < N; j += tpg.x) {
        localSum += exp(X[row*N + j] - maxv);
    }
    sdataSum[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find the sum
    for (uint stride = tpg.x>>1; stride>0; stride >>= 1) {
        if (tid < stride) sdataSum[tid] += sdataSum[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = sdataSum[0]; // Row's sum (denominator)

    // --- Pass 3: Normalize and write output ---
    for (uint j = tid; j < N; j += tpg.x) {
        Y[row*N + j] = exp(X[row*N + j] - maxv) / denom;
    }
}
)MSL";

//================ Host helpers ================
/**
 @brief CPU utility to convert float32 to float16 (half)
 */
static uint16_t f32_to_f16(float f){
    union { uint32_t u; float f; } v; v.f=f;
    uint32_t x=v.u, sign=(x>>16)&0x8000u;
    int exp=((x>>23)&0xFF)-127+15;
    uint32_t mant=(x>>13)&0x3FFu;
    if (exp<=0) return (uint16_t)sign;
    if (exp>=31) return (uint16_t)(sign|0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp<<10) | mant);
}
/**
 @brief CPU utility to convert float16 (half) to float32
 */
static float f16_to_f32(uint16_t h){
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1F, f=h&0x3FF;
    float out;
    if(e==0){ out=(f? (f/1024.0f)*powf(2,-14):0.0f); }
    else if(e==31){ out = f? NAN : INFINITY; }
    else{ out = (1.0f+f/1024.0f)*powf(2,(int)e-15); }
    return s?-out:out;
}

/**
 @brief Selects the first available Metal GPU.
 @return An object conforming to id<MTLDevice> or nil if none found.
 */
static id<MTLDevice> pickDevice(void){
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    for (id<MTLDevice> d in all) NSLog(@"Found Metal device: %@", d.name);
    return all.count? all.firstObject : nil;
}
/**
 @brief Compiles the embedded Metal shader string into a library.
 @param dev The Metal device.
 @param src The MSL source code string.
 @return A new id<MTLLibrary> object or nil on failure.
 */
static id<MTLLibrary> compileMSL(id<MTLDevice> dev, NSString *src){
    NSError *err=nil; MTLCompileOptions *opts=[MTLCompileOptions new];
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:opts error:&err];
    if(!lib){ NSLog(@"Metal compile error: %@", err.localizedDescription); exit(2); }
    return lib;
}
/**
 @brief Calculates GPU execution time from a command buffer.
 @discussion Falls back to host-side timing if GPU-side timing is unavailable.
 @param cb The completed command buffer.
 @param t0 The host-side start time (NSDate).
 @return The execution time in seconds (double).
 */
static double gpuSeconds(id<MTLCommandBuffer> cb, NSDate *t0){
    double dt = cb.GPUEndTime - cb.GPUStartTime;
    if (dt>0) return dt;
    return -[t0 timeIntervalSinceNow]; // Fallback for host-side time
}

//================ CPU refs for tests ================
/**
 @brief CPU reference implementation for GEMM (fp16 in, fp32 out) for validation.
 */
static void cpu_gemm_fp16(const uint16_t* A, const uint16_t* B, float* C,
                          uint M,uint N,uint K){
    for(uint i=0;i<M;i++){
        for(uint j=0;j<N;j++){
            float acc=0.f;
            for(uint k=0;k<K;k++){
                acc += f16_to_f32(A[i*K+k]) * f16_to_f32(B[k*N+j]);
            }
            C[i*N+j]=acc;
        }
    }
}
/**
 @brief CPU reference for row-wise softmax for validation.
 */
static void cpu_softmax_rowwise(const float* X, float* Y, uint M,uint N){
    for(uint i=0;i<M;i++){
        const float* x=&X[i*N]; float* y=&Y[i*N];
        float m=-INFINITY;
        for(uint j=0;j<N;j++) m=std::max(m,x[j]);
        float s=0.f;
        for(uint j=0;j<N;j++) s += std::exp(x[j]-m);
        for(uint j=0;j<N;j++) y[j]=std::exp(x[j]-m)/s;
    }
}
/**
 @brief Calculates L2 relative error between two float buffers.
 */
static float l2_rel_err(const float* a,const float* b,size_t n){
    double num=0,den=0;
    for(size_t i=0;i<n;i++){ double d=a[i]-b[i]; num+=d*d; den+=b[i]*b[i]; }
    return den>0? std::sqrt(num/den):0.f;
}

//================ Main ================
int main(){
    @autoreleasepool {
        
        // --- 1. Initialization: Device, Queue, Library ---
        id<MTLDevice> dev = pickDevice();
        if(!dev){ NSLog(@"No Metal device"); return 1; }
        NSLog(@"Using device: %@", dev.name);

        // Log device capabilities
        MTLSize m = dev.maxThreadsPerThreadgroup;
        NSLog(@"device maxThreadsPerThreadgroup dims: %lu x %lu x %lu",
              (unsigned long)m.width,(unsigned long)m.height,(unsigned long)m.depth);
        NSLog(@"device maxThreadgroupMemoryLength: %llu",
              (unsigned long long)dev.maxThreadgroupMemoryLength);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLLibrary> lib = compileMSL(dev, kMSL);
        NSError *err=nil;
        
        // --- 2. Create Pipeline State Objects (PSOs) ---
        auto fnSax  = [lib newFunctionWithName:@"saxpy"];
        auto fnGemm = [lib newFunctionWithName:@"hgemm_outer_64x64"];
        auto fnSm   = [lib newFunctionWithName:@"softmax_rowwise"];
        auto psoSax = [dev newComputePipelineStateWithFunction:fnSax  error:&err];
        auto psoGem = [dev newComputePipelineStateWithFunction:fnGemm error:&err];
        auto psoSm  = [dev newComputePipelineStateWithFunction:fnSm   error:&err];
        if(!psoSax||!psoGem||!psoSm){ NSLog(@"PSO build failed"); return 2; }
        
        NSLog(@"PSO maxTotalThreadsPerThreadgroup: saxpy=%lu gemm=%lu softmax=%lu",
              (unsigned long)psoSax.maxTotalThreadsPerThreadgroup,
              (unsigned long)psoGem.maxTotalThreadsPerThreadgroup,
              (unsigned long)psoSm.maxTotalThreadsPerThreadgroup);

        //---------- 1) SAXPY smoke test ----------
        const uint N = 1u<<22;
        // Allocate shared memory (accessible by both CPU and GPU)
        id<MTLBuffer> x = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> y = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
        float *xp=(float*)x.contents, *yp=(float*)y.contents;
        for(uint i=0;i<N;i++){ xp[i]=1.f; yp[i]=2.f; }
        float a=3.f;

        {
            // Encode, commit, and time SAXPY kernel
            id<MTLCommandBuffer> cb=[q commandBuffer];
            NSDate *t0=[NSDate date];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:psoSax];
            [e setBuffer:x offset:0 atIndex:0];
            [e setBuffer:y offset:0 atIndex:1];
            [e setBytes:&a length:sizeof(float) atIndex:2];
            [e setBytes:&N length:sizeof(uint) atIndex:3];
            NSUInteger tpg=psoSax.maxTotalThreadsPerThreadgroup;
            MTLSize tg=MTLSizeMake(std::min<NSUInteger>(256,tpg),1,1);
            MTLSize grid=MTLSizeMake(((N+tg.width-1)/tg.width)*tg.width,1,1);
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            double sec=gpuSeconds(cb,t0);
            
            // Log SAXPY results (y = 3*1 + 2 = 5)
            NSLog(@"SAXPY y[123]=%f  time=%.6f s  BW≈%.1f GB/s",
                  ((float*)y.contents)[123], sec, (3.0*N*sizeof(float))/sec/1e9);
        }

        //---------- 2) GEMM + SOFTMAX tests ----------
        struct Test { uint M,N,K; };
        std::vector<Test> sizes={{256,256,256},{512,256,512}};

        for (auto t : sizes){
            uint M=t.M, Nn=t.N, K=t.K;

            // Allocate buffers for A, B, C (fp16)
            id<MTLBuffer> A=[dev newBufferWithLength:M*K*sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> B=[dev newBufferWithLength:K*Nn*sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> C=[dev newBufferWithLength:M*Nn*sizeof(uint16_t) options:MTLResourceStorageModeShared];
            
            // Initialize test data on CPU
            uint16_t *Ap=(uint16_t*)A.contents, *Bp=(uint16_t*)B.contents;
            for(uint i=0;i<M;i++) for(uint k=0;k<K;k++) Ap[i*K+k]=f32_to_f16(((i+k)%7)*0.01f);
            for(uint k=0;k<K;k++) for(uint j=0;j<Nn;j++) Bp[k*Nn+j]=f32_to_f16(((k+j)%5)*0.02f);

            // Configure GEMM kernel launch parameters
            const int TILE_M = 64, TILE_N = 64; // Must match kernel
            MTLSize tg = MTLSizeMake(16, 16, 1); // 256 threads
            MTLSize grid = MTLSizeMake((Nn + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

            // Encode, commit, and time GEMM kernel
            id<MTLCommandBuffer> cb=[q commandBuffer];
            NSDate *t0=[NSDate date];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:psoGem];
            [e setBuffer:A offset:0 atIndex:0];
            [e setBuffer:B offset:0 atIndex:1];
            [e setBuffer:C offset:0 atIndex:2];
            [e setBytes:&M length:sizeof(uint) atIndex:3];
            [e setBytes:&Nn length:sizeof(uint) atIndex:4];
            [e setBytes:&K length:sizeof(uint) atIndex:5];
            [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            double sec=gpuSeconds(cb,t0);

            // Log GEMM performance
            double flops = 2.0*(double)M*Nn*K / sec / 1e9;
            uint16_t h = ((uint16_t*)C.contents)[0];
            NSLog(@"GEMM %ux%ux%u  C[0]=%.6f  time=%.6f s  %.2f GFLOP/s",
                  M,Nn,K, f16_to_f32(h), sec, flops);

            // --- Softmax (fp32) ---
            
            // Convert fp16 GEMM result to fp32 for softmax input
            std::vector<float> C32(M*Nn);
            for(uint i=0;i<M*Nn;i++) C32[i]=f16_to_f32(((uint16_t*)C.contents)[i]);

            id<MTLBuffer> Xin=[dev newBufferWithLength:M*Nn*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> Yout=[dev newBufferWithLength:M*Nn*sizeof(float) options:MTLResourceStorageModeShared];
            memcpy(Xin.contents, C32.data(), M*Nn*sizeof(float));

            // Encode, commit, and time Softmax kernel
            id<MTLCommandBuffer> cb2=[q commandBuffer];
            NSDate *t1=[NSDate date];
            id<MTLComputeCommandEncoder> e2=[cb2 computeCommandEncoder];
            [e2 setComputePipelineState:psoSm];
            [e2 setBuffer:Xin offset:0 atIndex:0];
            [e2 setBuffer:Yout offset:0 atIndex:1];
            [e2 setBytes:&M length:sizeof(uint) atIndex:2];
            [e2 setBytes:&Nn length:sizeof(uint) atIndex:3];
            // Launch one threadgroup per row
            NSUInteger tpg_sm = std::min<NSUInteger>(256, psoSm.maxTotalThreadsPerThreadgroup);
            MTLSize tg2=MTLSizeMake(tpg_sm,1,1);
            MTLSize grid2=MTLSizeMake(1,M,1);
            [e2 dispatchThreadgroups:grid2 threadsPerThreadgroup:tg2];
            [e2 endEncoding];
            [cb2 commit]; [cb2 waitUntilCompleted];
            double sec2=gpuSeconds(cb2,t1);

            // --- Validation ---
            // Run CPU reference implementations
            std::vector<float> Cref(M*Nn), Sref(M*Nn);
            cpu_gemm_fp16((uint16_t*)A.contents,(uint16_t*)B.contents,Cref.data(),M,Nn,K);
            cpu_softmax_rowwise(Cref.data(),Sref.data(),M,Nn);

            // Compare GPU softmax output to CPU reference
            float *Sgpu=(float*)Yout.contents;
            float err = l2_rel_err(Sgpu, Sref.data(), (size_t)M*Nn);
            NSLog(@"Softmax rows M=%u N=%u  time=%.6f s  relL2=%.3e  sample=%.5f",
                  M,Nn,sec2,err,Sgpu[(M>0?0:0)*Nn + (Nn>1?1:0)]);
        }

        return 0;
    }
}

