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

[numthreads(16, 16, 1)]
void addPhotonTaskFromTexture(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{
    uint rayIdx = getRayTaskID(threadIdx.xy);
    //gPixelInfo[rayIdx].screenArea = 0;
    //gPixelInfo[rayIdx].screenAreaSq = 0;
    //gPixelInfo[rayIdx].count = 0;

    int idx0 = gPixelInfo[rayIdx].photonIdx;

    int2 pixel00 = threadIdx.xy;
    float v00 = gRayDensityTex.Load(int3(pixel00 + int2(0, 0), 0)).r;
    float v10 = gRayDensityTex.Load(int3(pixel00 + int2(1, 0), 0)).r;
    float v01 = gRayDensityTex.Load(int3(pixel00 + int2(0, 1), 0)).r;
    float v11 = gRayDensityTex.Load(int3(pixel00 + int2(1, 1), 0)).r;

    float sampleCountF =  0.25 * (v00 + v10 + v01 + v11); //max(v00, max(v10, max(v01, v11)));//
    int sampleCount = int(sampleCountF);
    sampleCount = clamp(sampleCount, 1, 1024 * 8);

    float sampleWeight = 1.0 / sqrt(float(sampleCount));
    float2 pixelSize = float2(1, 1) * sampleWeight;

    int taskIdx = 0;
    InterlockedAdd(gRayArgument[0].rayTaskCount, sampleCount, taskIdx);
    float g = 1.32471795724474602596;
    float a1 = 1.0 / g;
    float a2 = 1.0 / (g * g);
    for (uint i = 0; i < sampleCount; i++)
    {
        float x = frac(randomOffset.x + a1 * (i + 1));
        float y = frac(randomOffset.y + a2 * (i + 1));

        float2 uv = bilinearSample(v00, v10, v01, v11, float2(x, y));
        RayTask newTask;
        newTask.screenCoord = pixel00 + uv +0.5;
        newTask.pixelSize = pixelSize *sqrt(sampleCount / (bilinearIntepolation(v00, v10, v01, v11, uv)));
        gRayTask[taskIdx + i] = newTask;
    }
}
