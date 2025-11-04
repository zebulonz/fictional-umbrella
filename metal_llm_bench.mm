/*
    metal1.mm
    
    A command-line tool to benchmark Metal compute performance for operations
    relevant to LLM inference, specifically fp16 GEMM and softmax.
    It compares GPU results against CPU reference implementations.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

// MARK: - Embedded Metal Shaders

static NSString *metalShaderSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// MARK: SAXPY Kernel
// Performs y = a*x + y (Single-precision A*X Plus Y)
kernel void saxpy(const device float *inputX    [[buffer(0)]],
                  device float *inputOutputY    [[buffer(1)]],
                  constant float &scalarA       [[buffer(2)]],
                  constant uint  &elementCount  [[buffer(3)]],
                  uint globalThreadID [[thread_position_in_grid]]) {
    if (globalThreadID < elementCount) {
        inputOutputY[globalThreadID] = fma(scalarA, inputX[globalThreadID], inputOutputY[globalThreadID]);
    }
}

// MARK: Half-Precision GEMM Kernel
/*
    hgemm_outer_64x64
    
    Performs half-precision (fp16) matrix multiplication (C = A * B)
    using a tiled approach.
    
    Tiling Strategy:
    - Threadgroup Tile (Output): 64x64 (TILE_M, TILE_N)
    - Thread Tile (Output): 4x4 (BLOCK_M, BLOCK_N)
    - Threads per Threadgroup: 16x16 = 256
    - K-Dimension Slice: 16 (TILE_K)
*/
kernel void hgemm_outer_64x64(const device half *matrixA [[buffer(0)]],
                              const device half *matrixB [[buffer(1)]],
                              device half *matrixC       [[buffer(2)]],
                              constant uint &rowsM       [[buffer(3)]],
                              constant uint &colsN       [[buffer(4)]],
                              constant uint &innerDimK   [[buffer(5)]],
                              uint2 threadIDInThreadgroup [[thread_position_in_threadgroup]],
                              uint2 threadgroupID [[threadgroup_position_in_grid]]) {

    const uint outputTileRowsPerThread = 4;
    const uint outputTileColsPerThread = 4;
    const uint threadgroupTileRows = 64;
    const uint threadgroupTileCols = 64;
    const uint kDimensionTileDepth = 16;

    // Calculate thread's base indices within the 64x64 tile
    const uint threadBaseRow = threadIDInThreadgroup.y * outputTileRowsPerThread;
    const uint threadBaseCol = threadIDInThreadgroup.x * outputTileColsPerThread;
    
    // Calculate threadgroup's base indices in the global C matrix
    const uint threadgroupBaseRow = threadgroupID.y * threadgroupTileRows;
    const uint threadgroupBaseCol = threadgroupID.x * threadgroupTileCols;

    // Tile storage in fast threadgroup memory
    threadgroup half matrixATile[threadgroupTileRows][kDimensionTileDepth];
    threadgroup half matrixBTile[kDimensionTileDepth][threadgroupTileCols];

    // Local register accumulators for the 4x4 thread tile
    float accumulators[outputTileRowsPerThread][outputTileColsPerThread];
    for (uint i = 0; i < outputTileRowsPerThread; i++) {
        for (uint j = 0; j < outputTileColsPerThread; j++) {
            accumulators[i][j] = 0.0f;
        }
    }

    // Iterate over K dimension in kDimensionTileDepth-sized chunks
    for (uint kTileStart = 0; kTileStart < innerDimK; kTileStart += kDimensionTileDepth) {
        
        // MARK: Collaborative Load from Global to Threadgroup Memory
        // Each of the 256 threads loads 4 elements of A and 4 elements of B
        // (256 * 4 = 1024 elements each, matching tile sizes)
        uint loadRowIndexA = threadIDInThreadgroup.y * 4 + threadIDInThreadgroup.x / 4;
        uint loadColIndexA = (threadIDInThreadgroup.x % 4) * 4 + (threadIDInThreadgroup.y % 4);

        if (threadgroupBaseRow + loadRowIndexA < rowsM && kTileStart + loadColIndexA < innerDimK) {
            matrixATile[loadRowIndexA][loadColIndexA] = 
                matrixA[(threadgroupBaseRow + loadRowIndexA) * innerDimK + kTileStart + loadColIndexA];
        } else {
            matrixATile[loadRowIndexA][loadColIndexA] = half(0);
        }

        uint loadRowIndexB = threadIDInThreadgroup.y;
        uint loadColIndexB = threadIDInThreadgroup.x * 4 + (threadIDInThreadgroup.y % 4);

        if (kTileStart + loadRowIndexB < innerDimK && threadgroupBaseCol + loadColIndexB < colsN) {
            matrixBTile[loadRowIndexB][loadColIndexB] = 
                matrixB[(kTileStart + loadRowIndexB) * colsN + threadgroupBaseCol + loadColIndexB];
        } else {
            matrixBTile[loadRowIndexB][loadColIndexB] = half(0);
        }

        // Wait for all threads to finish loading tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MARK: Compute 4x4 output block via outer product over K-slice
        #pragma unroll
        for (uint k = 0; k < kDimensionTileDepth; k++) {
            #pragma unroll
            for (uint i = 0; i < outputTileRowsPerThread; i++) {
                float valueA = float(matrixATile[threadBaseRow + i][k]);
                #pragma unroll
                for (uint j = 0; j < outputTileColsPerThread; j++) {
                    float valueB = float(matrixBTile[k][threadBaseCol + j]);
                    accumulators[i][j] = fma(valueA, valueB, accumulators[i][j]);
                }
            }
        }
        
        // Wait for all threads to finish compute before loading next K-slice
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // MARK: Write 4x4 block from registers to global memory
    for (uint i = 0; i < outputTileRowsPerThread; i++) {
        for (uint j = 0; j < outputTileColsPerThread; j++) {
            uint outputRow = threadgroupBaseRow + threadBaseRow + i;
            uint outputCol = threadgroupBaseCol + threadBaseCol + j;
            if (outputRow < rowsM && outputCol < colsN) {
                matrixC[outputRow * colsN + outputCol] = half(accumulators[i][j]);
            }
        }
    }
}

// MARK: Softmax Kernel
/*
    softmax_rowwise
    
    Computes softmax row-wise (over dimension N) for M rows.
    Uses a stable, two-pass parallel reduction within each threadgroup.
    Input/Output is fp32 for numerical stability.
    Each threadgroup (of size e.g. 256) processes one row.
*/
kernel void softmax_rowwise(const device float *inputMatrix [[buffer(0)]],
                            device float *outputMatrix     [[buffer(1)]],
                            constant uint &numberOfRows    [[buffer(2)]],
                            constant uint &numberOfCols    [[buffer(3)]],
                            uint threadIndexInGroup [[thread_index_in_threadgroup]],
                            uint3 threadsPerThreadgroup [[threads_per_threadgroup]],
                            uint3 threadgroupPosition [[threadgroup_position_in_grid]]) {

    // Shared memory for the parallel reduction
    threadgroup float sharedMaxValues[256];
    threadgroup float sharedSumValues[256];
    
    // Each threadgroup processes one row
    uint currentRow = threadgroupPosition.y;
    if (currentRow >= numberOfRows) return;

    // MARK: Pass 1 - Find maximum value in the row
    float localMaximum = -INFINITY;
    for (uint col = threadIndexInGroup; col < numberOfCols; col += threadsPerThreadgroup.x) {
        localMaximum = max(localMaximum, inputMatrix[currentRow * numberOfCols + col]);
    }
    sharedMaxValues[threadIndexInGroup] = localMaximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find the single max value
    for (uint reductionStride = threadsPerThreadgroup.x >> 1; reductionStride > 0; reductionStride >>= 1) {
        if (threadIndexInGroup < reductionStride) {
            sharedMaxValues[threadIndexInGroup] = max(sharedMaxValues[threadIndexInGroup], 
                                                      sharedMaxValues[threadIndexInGroup + reductionStride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMaximum = sharedMaxValues[0];

    // MARK: Pass 2 - Calculate sum of exp(x - maxv)
    float localSum = 0.0f;
    for (uint col = threadIndexInGroup; col < numberOfCols; col += threadsPerThreadgroup.x) {
        localSum += exp(inputMatrix[currentRow * numberOfCols + col] - rowMaximum);
    }
    sharedSumValues[threadIndexInGroup] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find the sum
    for (uint reductionStride = threadsPerThreadgroup.x >> 1; reductionStride > 0; reductionStride >>= 1) {
        if (threadIndexInGroup < reductionStride) {
            sharedSumValues[threadIndexInGroup] += sharedSumValues[threadIndexInGroup + reductionStride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = sharedSumValues[0];

    // MARK: Pass 3 - Normalize and write output
    for (uint col = threadIndexInGroup; col < numberOfCols; col += threadsPerThreadgroup.x) {
        outputMatrix[currentRow * numberOfCols + col] = 
            exp(inputMatrix[currentRow * numberOfCols + col] - rowMaximum) / denominator;
    }
}
)MSL";

// MARK: - Host Helpers

// Converts float32 to float16 (half precision)
static uint16_t convertFloat32ToFloat16(float inputFloat) {
    union { uint32_t u; float f; } converter;
    converter.f = inputFloat;
    
    uint32_t bits = converter.u;
    uint32_t signBit = (bits >> 16) & 0x8000u;
    int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3FFu;
    
    if (exponent <= 0) return (uint16_t)signBit;
    if (exponent >= 31) return (uint16_t)(signBit | 0x7C00);
    
    return (uint16_t)(signBit | ((uint32_t)exponent << 10) | mantissa);
}

// Converts float16 (half precision) to float32
static float convertFloat16ToFloat32(uint16_t inputHalf) {
    uint32_t signBit = (inputHalf >> 15) & 1;
    uint32_t exponent = (inputHalf >> 10) & 0x1F;
    uint32_t fraction = inputHalf & 0x3FF;
    
    float result;
    if (exponent == 0) {
        result = fraction ? (fraction / 1024.0f) * powf(2, -14) : 0.0f;
    } else if (exponent == 31) {
        result = fraction ? NAN : INFINITY;
    } else {
        result = (1.0f + fraction / 1024.0f) * powf(2, (int)exponent - 15);
    }
    
    return signBit ? -result : result;
}

// Selects the first available Metal GPU
static id<MTLDevice> selectMetalDevice(void) {
    NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
    for (id<MTLDevice> device in availableDevices) {
        NSLog(@"Found Metal device: %@", device.name);
    }
    return availableDevices.count ? availableDevices.firstObject : nil;
}

// Compiles the embedded Metal shader string into a library
static id<MTLLibrary> compileMetalShaderLibrary(id<MTLDevice> device, NSString *shaderSource) {
    NSError *compilationError = nil;
    MTLCompileOptions *compileOptions = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource 
                                                  options:compileOptions 
                                                    error:&compilationError];
    if (!library) {
        NSLog(@"Metal compile error: %@", compilationError.localizedDescription);
        exit(2);
    }
    return library;
}

// Calculates GPU execution time from a command buffer
// Falls back to host-side timing if GPU-side timing is unavailable
static double calculateGPUExecutionTimeInSeconds(id<MTLCommandBuffer> commandBuffer, NSDate *hostStartTime) {
    double gpuTimeDelta = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    if (gpuTimeDelta > 0) {
        return gpuTimeDelta;
    }
    return -[hostStartTime timeIntervalSinceNow];
}

// MARK: - CPU Reference Implementations

// CPU reference implementation for GEMM (fp16 in, fp32 out) for validation
static void cpuReferenceGEMMFloat16(const uint16_t *matrixA, 
                                    const uint16_t *matrixB, 
                                    float *matrixC,
                                    uint rowsM, 
                                    uint colsN, 
                                    uint innerDimK) {
    for (uint i = 0; i < rowsM; i++) {
        for (uint j = 0; j < colsN; j++) {
            float accumulator = 0.0f;
            for (uint k = 0; k < innerDimK; k++) {
                accumulator += convertFloat16ToFloat32(matrixA[i * innerDimK + k]) * 
                               convertFloat16ToFloat32(matrixB[k * colsN + j]);
            }
            matrixC[i * colsN + j] = accumulator;
        }
    }
}

// CPU reference for row-wise softmax for validation
static void cpuReferenceSoftmaxRowwise(const float *inputMatrix, 
                                       float *outputMatrix, 
                                       uint numberOfRows, 
                                       uint numberOfCols) {
    for (uint i = 0; i < numberOfRows; i++) {
        const float *inputRow = &inputMatrix[i * numberOfCols];
        float *outputRow = &outputMatrix[i * numberOfCols];
        
        float rowMaximum = -INFINITY;
        for (uint j = 0; j < numberOfCols; j++) {
            rowMaximum = std::max(rowMaximum, inputRow[j]);
        }
        
        float exponentialSum = 0.0f;
        for (uint j = 0; j < numberOfCols; j++) {
            exponentialSum += std::exp(inputRow[j] - rowMaximum);
        }
        
        for (uint j = 0; j < numberOfCols; j++) {
            outputRow[j] = std::exp(inputRow[j] - rowMaximum) / exponentialSum;
        }
    }
}

// Calculates L2 relative error between two float buffers
static float calculateL2RelativeError(const float *bufferA, 
                                     const float *bufferB, 
                                     size_t elementCount) {
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (size_t i = 0; i < elementCount; i++) {
        double difference = bufferA[i] - bufferB[i];
        numerator += difference * difference;
        denominator += bufferB[i] * bufferB[i];
    }
    
    return denominator > 0 ? std::sqrt(numerator / denominator) : 0.0f;
}

// MARK: - Main Entry Point

int main() {
    @autoreleasepool {
        
        // MARK: Device Initialization
        id<MTLDevice> metalDevice = selectMetalDevice();
        if (!metalDevice) {
            NSLog(@"No Metal device");
            return 1;
        }
        NSLog(@"Using device: %@", metalDevice.name);

        // Log device capabilities
        MTLSize maxThreadsPerThreadgroup = metalDevice.maxThreadsPerThreadgroup;
        NSLog(@"device maxThreadsPerThreadgroup dims: %lu x %lu x %lu",
              (unsigned long)maxThreadsPerThreadgroup.width,
              (unsigned long)maxThreadsPerThreadgroup.height,
              (unsigned long)maxThreadsPerThreadgroup.depth);
        NSLog(@"device maxThreadgroupMemoryLength: %llu",
              (unsigned long long)metalDevice.maxThreadgroupMemoryLength);

        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
        id<MTLLibrary> shaderLibrary = compileMetalShaderLibrary(metalDevice, metalShaderSource);
        NSError *pipelineError = nil;
        
        // MARK: Create Pipeline State Objects
        id<MTLFunction> saxpyFunction = [shaderLibrary newFunctionWithName:@"saxpy"];
        id<MTLFunction> gemmFunction = [shaderLibrary newFunctionWithName:@"hgemm_outer_64x64"];
        id<MTLFunction> softmaxFunction = [shaderLibrary newFunctionWithName:@"softmax_rowwise"];
        
        id<MTLComputePipelineState> saxpyPipelineState = 
            [metalDevice newComputePipelineStateWithFunction:saxpyFunction error:&pipelineError];
        id<MTLComputePipelineState> gemmPipelineState = 
            [metalDevice newComputePipelineStateWithFunction:gemmFunction error:&pipelineError];
        id<MTLComputePipelineState> softmaxPipelineState = 
            [metalDevice newComputePipelineStateWithFunction:softmaxFunction error:&pipelineError];
            
        if (!saxpyPipelineState || !gemmPipelineState || !softmaxPipelineState) {
            NSLog(@"PSO build failed");
            return 2;
        }
        
        NSLog(@"PSO maxTotalThreadsPerThreadgroup: saxpy=%lu gemm=%lu softmax=%lu",
              (unsigned long)saxpyPipelineState.maxTotalThreadsPerThreadgroup,
              (unsigned long)gemmPipelineState.maxTotalThreadsPerThreadgroup,
              (unsigned long)softmaxPipelineState.maxTotalThreadsPerThreadgroup);

        // MARK: SAXPY Smoke Test
        const uint vectorElementCount = 1u << 22;
        
        // Allocate shared memory (accessible by both CPU and GPU)
        id<MTLBuffer> vectorX = [metalDevice newBufferWithLength:vectorElementCount * sizeof(float) 
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectorY = [metalDevice newBufferWithLength:vectorElementCount * sizeof(float) 
                                                         options:MTLResourceStorageModeShared];
        float *xPointer = (float *)vectorX.contents;
        float *yPointer = (float *)vectorY.contents;
        
        for (uint i = 0; i < vectorElementCount; i++) {
            xPointer[i] = 1.0f;
            yPointer[i] = 2.0f;
        }
        float scalarA = 3.0f;

        {
            // Encode, commit, and time SAXPY kernel
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            NSDate *hostStartTime = [NSDate date];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            
            [computeEncoder setComputePipelineState:saxpyPipelineState];
            [computeEncoder setBuffer:vectorX offset:0 atIndex:0];
            [computeEncoder setBuffer:vectorY offset:0 atIndex:1];
            [computeEncoder setBytes:&scalarA length:sizeof(float) atIndex:2];
            [computeEncoder setBytes:&vectorElementCount length:sizeof(uint) atIndex:3];
            
            NSUInteger maxThreadsPerGroup = saxpyPipelineState.maxTotalThreadsPerThreadgroup;
            MTLSize threadgroupSize = MTLSizeMake(std::min<NSUInteger>(256, maxThreadsPerGroup), 1, 1);
            MTLSize gridSize = MTLSizeMake(((vectorElementCount + threadgroupSize.width - 1) / 
                                           threadgroupSize.width) * threadgroupSize.width, 1, 1);
            
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            double executionTime = calculateGPUExecutionTimeInSeconds(commandBuffer, hostStartTime);
            
            // Log SAXPY results (y = 3*1 + 2 = 5)
            NSLog(@"SAXPY y[123]=%f  time=%.6f s  BW≈%.1f GB/s",
                  ((float *)vectorY.contents)[123], 
                  executionTime, 
                  (3.0 * vectorElementCount * sizeof(float)) / executionTime / 1e9);
        }

        // MARK: GEMM + Softmax Tests
        struct BenchmarkConfiguration {
            uint rowsM;
            uint colsN;
            uint innerDimK;
        };
        
        std::vector<BenchmarkConfiguration> testConfigurations = {
            {256, 256, 256},
            {512, 256, 512}
        };

        for (auto configuration : testConfigurations) {
            uint rowsM = configuration.rowsM;
            uint colsN = configuration.colsN;
            uint innerDimK = configuration.innerDimK;

            // Allocate buffers for A, B, C (fp16)
            id<MTLBuffer> matrixABuffer = [metalDevice newBufferWithLength:rowsM * innerDimK * sizeof(uint16_t) 
                                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> matrixBBuffer = [metalDevice newBufferWithLength:innerDimK * colsN * sizeof(uint16_t) 
                                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> matrixCBuffer = [metalDevice newBufferWithLength:rowsM * colsN * sizeof(uint16_t) 
                                                                   options:MTLResourceStorageModeShared];
            
            // Initialize test data on CPU
            uint16_t *matrixAPointer = (uint16_t *)matrixABuffer.contents;
            uint16_t *matrixBPointer = (uint16_t *)matrixBBuffer.contents;
            
            for (uint i = 0; i < rowsM; i++) {
                for (uint k = 0; k < innerDimK; k++) {
                    matrixAPointer[i * innerDimK + k] = convertFloat32ToFloat16(((i + k) % 7) * 0.01f);
                }
            }
            
            for (uint k = 0; k < innerDimK; k++) {
                for (uint j = 0; j < colsN; j++) {
                    matrixBPointer[k * colsN + j] = convertFloat32ToFloat16(((k + j) % 5) * 0.02f);
                }
            }

            // Configure GEMM kernel launch parameters
            const int threadgroupTileRows = 64;
            const int threadgroupTileCols = 64;
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1); // 256 threads
            MTLSize gridSize = MTLSizeMake((colsN + threadgroupTileCols - 1) / threadgroupTileCols, 
                                          (rowsM + threadgroupTileRows - 1) / threadgroupTileRows, 
                                          1);

            // Encode, commit, and time GEMM kernel
            id<MTLCommandBuffer> gemmCommandBuffer = [commandQueue commandBuffer];
            NSDate *gemmStartTime = [NSDate date];
            id<MTLComputeCommandEncoder> gemmEncoder = [gemmCommandBuffer computeCommandEncoder];
            
            [gemmEncoder setComputePipelineState:gemmPipelineState];
            [gemmEncoder setBuffer:matrixABuffer offset:0 atIndex:0];
            [gemmEncoder setBuffer:matrixBBuffer offset:0 atIndex:1];
            [gemmEncoder setBuffer:matrixCBuffer offset:0 atIndex:2];
            [gemmEncoder setBytes:&rowsM length:sizeof(uint) atIndex:3];
            [gemmEncoder setBytes:&colsN length:sizeof(uint) atIndex:4];
            [gemmEncoder setBytes:&innerDimK length:sizeof(uint) atIndex:5];
            [gemmEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [gemmEncoder endEncoding];
            [gemmCommandBuffer commit];
            [gemmCommandBuffer waitUntilCompleted];
            
            double gemmExecutionTime = calculateGPUExecutionTimeInSeconds(gemmCommandBuffer, gemmStartTime);

            // Log GEMM performance
            double gemmGigaflops = 2.0 * (double)rowsM * colsN * innerDimK / gemmExecutionTime / 1e9;
            uint16_t firstResultHalf = ((uint16_t *)matrixCBuffer.contents)[0];
            NSLog(@"GEMM %ux%ux%u  C[0]=%.6f  time=%.6f s  %.2f GFLOP/s",
                  rowsM, colsN, innerDimK, 
                  convertFloat16ToFloat32(firstResultHalf), 
                  gemmExecutionTime, 
                  gemmGigaflops);

            // MARK: Softmax (fp32)
            
            // Convert fp16 GEMM result to fp32 for softmax input
            std::vector<float> matrixCFloat32(rowsM * colsN);
            for (uint i = 0; i < rowsM * colsN; i++) {
                matrixCFloat32[i] = convertFloat16ToFloat32(((uint16_t *)matrixCBuffer.contents)[i]);
            }

            id<MTLBuffer> softmaxInputBuffer = [metalDevice newBufferWithLength:rowsM * colsN * sizeof(float) 
                                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> softmaxOutputBuffer = [metalDevice newBufferWithLength:rowsM * colsN * sizeof(float) 
                                                                         options:MTLResourceStorageModeShared];
            memcpy(softmaxInputBuffer.contents, matrixCFloat32.data(), rowsM * colsN * sizeof(float));

            // Encode, commit, and time Softmax kernel
            id<MTLCommandBuffer> softmaxCommandBuffer = [commandQueue commandBuffer];
            NSDate *softmaxStartTime = [NSDate date];
            id<MTLComputeCommandEncoder> softmaxEncoder = [softmaxCommandBuffer computeCommandEncoder];
            
            [softmaxEncoder setComputePipelineState:softmaxPipelineState];
            [softmaxEncoder setBuffer:softmaxInputBuffer offset:0 atIndex:0];
            [softmaxEncoder setBuffer:softmaxOutputBuffer offset:0 atIndex:1];
            [softmaxEncoder setBytes:&rowsM length:sizeof(uint) atIndex:2];
            [softmaxEncoder setBytes:&colsN length:sizeof(uint) atIndex:3];
            
            // Launch one threadgroup per row
            NSUInteger softmaxThreadsPerGroup = std::min<NSUInteger>(256, 
                                                                     softmaxPipelineState.maxTotalThreadsPerThreadgroup);
            MTLSize softmaxThreadgroupSize = MTLSizeMake(softmaxThreadsPerGroup, 1, 1);
            MTLSize softmaxGridSize = MTLSizeMake(1, rowsM, 1);
            
            [softmaxEncoder dispatchThreadgroups:softmaxGridSize 
                           threadsPerThreadgroup:softmaxThreadgroupSize];
            [softmaxEncoder endEncoding];
            [softmaxCommandBuffer commit];
            [softmaxCommandBuffer waitUntilCompleted];
            
            double softmaxExecutionTime = calculateGPUExecutionTimeInSeconds(softmaxCommandBuffer, 
                                                                            softmaxStartTime);

            // MARK: Validation
            // Run CPU reference implementations
            std::vector<float> cpuGEMMResult(rowsM * colsN);
            std::vector<float> cpuSoftmaxResult(rowsM * colsN);
            
            cpuReferenceGEMMFloat16((uint16_t *)matrixABuffer.contents,
                                   (uint16_t *)matrixBBuffer.contents,
                                   cpuGEMMResult.data(),
                                   rowsM, colsN, innerDimK);
            
            cpuReferenceSoftmaxRowwise(cpuGEMMResult.data(),
                                      cpuSoftmaxResult.data(),
                                      rowsM, colsN);

            // Compare GPU softmax output to CPU reference
            float *gpuSoftmaxResult = (float *)softmaxOutputBuffer.contents;
            float relativeError = calculateL2RelativeError(gpuSoftmaxResult, 
                                                          cpuSoftmaxResult.data(), 
                                                          (size_t)rowsM * colsN);
            NSLog(@"Softmax rows M=%u N=%u  time=%.6f s  relL2=%.3e  sample=%.5f",
                  rowsM, colsN, 
                  softmaxExecutionTime, 
                  relativeError,
                  gpuSoftmaxResult[(rowsM > 0 ? 0 : 0) * colsN + (colsN > 1 ? 1 : 0)]);
        }

        return 0;
    }
}
