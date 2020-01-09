/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***************************************************************************/
#include "Common.hlsl"

shared cbuffer PerFrameCB
{
    float4x4 viewProjMat;
    int2 taskDim;
    int2 screenDim;
    float2 randomOffset;
    float normalThreshold;
    float distanceThreshold;
    float planarThreshold;
    float pixelLuminanceThreshold;
    float minPhotonPixelSize;
};

struct DrawArguments
{
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int  baseVertexLocation;
    uint startInstanceLocation;
};

RWStructuredBuffer<Photon> gPhotonBuffer;
//RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
RWStructuredBuffer<RayTask> gRayTask;
StructuredBuffer<PixelInfo> gPixelInfo;
Texture2D gDepthTex;
Texture2D<float4> gRayDensityTex;

int getRayTaskID(uint2 pos)
{
    return pos.y * taskDim.x + pos.x;
}

#define GROUP_DIM_X 32
#define GROUP_DIM_Y 4

groupshared float4 cornerDensity[GROUP_DIM_X * GROUP_DIM_Y];
groupshared int taskCount[GROUP_DIM_X * GROUP_DIM_Y+1];

groupshared int pixelIdx0;
groupshared int sampleIdx0;
groupshared int2 threadTask[GROUP_DIM_X * GROUP_DIM_Y];

groupshared int globalRayOffset;

[numthreads(GROUP_DIM_X, GROUP_DIM_Y, 1)]
void addPhotonTaskFromTexture(
    uint3 groupThreadIdx : SV_GroupThreadID,
    uint3 threadIdx : SV_DispatchThreadID,
    uint3 groupIdx:SV_GroupID
)
{
    uint rayIdx = getRayTaskID(threadIdx.xy);
    int idx0 = gPixelInfo[rayIdx].photonIdx;
    int threadOffset = groupThreadIdx.y * GROUP_DIM_X + groupThreadIdx.x;

    int2 pixel00 = threadIdx.xy;
    float v00 = gRayDensityTex.Load(int3(pixel00 + int2(0, 0), 0)).r;
    float v10 = gRayDensityTex.Load(int3(pixel00 + int2(1, 0), 0)).r;
    float v01 = gRayDensityTex.Load(int3(pixel00 + int2(0, 1), 0)).r;
    float v11 = gRayDensityTex.Load(int3(pixel00 + int2(1, 1), 0)).r;
    float sampleCountF =  0.25 * (v00 + v10 + v01 + v11);
    int sampleCount = int(sampleCountF);
    sampleCount = clamp(sampleCount, 1, 1024 * 8);
    cornerDensity[threadOffset] = float4(v00, v10, v01, v11);
    taskCount[threadOffset] = sampleCount;
    GroupMemoryBarrierWithGroupSync();

    // calculate prefix sum
    const int blockThreads = GROUP_DIM_X * GROUP_DIM_Y;
    if (threadOffset == 0)
    {
        int sum = 0;
        for (int i = 0; i < blockThreads; i++)
        {
            int value = taskCount[i];
            taskCount[i] = sum;
            sum += value;
        }
        taskCount[blockThreads] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // allocate global space
    int rayTaskCount = taskCount[blockThreads];
    if (threadOffset == 0)
    {
        InterlockedAdd(gRayArgument[0].rayTaskCount, rayTaskCount, globalRayOffset);
        pixelIdx0 = 0;
        sampleIdx0 = 0;
    }
    AllMemoryBarrierWithGroupSync();

    int iterations = (rayTaskCount + blockThreads - 1) / blockThreads;
    int2 groupIdx2 = groupIdx.xy * int2(GROUP_DIM_X, GROUP_DIM_Y);
    for (int i = 0; i < iterations; i++)
    {
        // allocate new task
        if (threadOffset == 0)
        {
            for (int t = 0; t < blockThreads; t++)
            {
                while (pixelIdx0 < blockThreads && sampleIdx0 >= taskCount[pixelIdx0 + 1] - taskCount[pixelIdx0] )
                {
                    sampleIdx0 = 0;
                    pixelIdx0++;
                }
                threadTask[t] = int2(pixelIdx0, sampleIdx0);
                sampleIdx0++;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // calculate and store ray task
        int accumulateTaskCount = i * blockThreads + threadOffset;
        if (accumulateTaskCount < rayTaskCount)
        {
            int2 pixelAndSampleIdx = threadTask[threadOffset];
            int pixelIdx = pixelAndSampleIdx.x;
            int sampleIdx = pixelAndSampleIdx.y;

            int nSamples = taskCount[pixelIdx + 1] - taskCount[pixelIdx];
            float sampleWeight = 1.0 / sqrt(float(nSamples));
            float pixelSize = 1 * sampleWeight;
            float g = 1.32471795724474602596;
            float a1 = 1.0 / g;
            float a2 = 1.0 / (g * g);
            float x = frac(randomOffset.x + a1 * (sampleIdx + 1));
            float y = frac(randomOffset.y + a2 * (sampleIdx + 1));

            float4 corner = cornerDensity[pixelIdx];
            float2 uv = float2(x, y);// bilinearSample(corner.x, corner.y, corner.z, corner.w, float2(x, y));
            int2 pixelPos = groupIdx2 + int2(pixelIdx % GROUP_DIM_X, pixelIdx / GROUP_DIM_X);
            RayTask newTask;
            newTask.screenCoord = pixelPos + uv + 0.5;
            newTask.pixelSize = pixelSize;// *sqrt(nSamples / (bilinearIntepolation(corner.x, corner.y, corner.z, corner.w, uv)));
            gRayTask[globalRayOffset + accumulateTaskCount] = newTask;
        }
        //AllMemoryBarrierWithGroupSync();
        GroupMemoryBarrierWithGroupSync();
    }
}
