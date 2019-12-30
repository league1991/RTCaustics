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
RWStructuredBuffer<PixelInfo> gPixelInfo;
Texture2D gDepthTex;
Texture2D gRayDensityTex;

float getNeighbourArea(uint2 pixelCoord0, uint2 offset, out int isInFrustum)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + offset));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

    RayTask task1 = gRayTask[pixelIdx1];
    isInFrustum = task1.inFrustum;
    return task1.pixelArea;
}

float checkPixelNeighbour(uint2 pixelCoord0, RayTask task0, Photon photon0, uint2 offset)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + offset));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

    RayTask task1 = gRayTask[pixelIdx1];
    bool isContinue = task1.photonIdx != -1;
    float3 dColor = photon0.color;
    Photon photon1;
    if (isContinue)
    {
        photon1 = gPhotonBuffer[task1.photonIdx];
        isContinue &= dot(photon0.normalW, photon1.normalW) < normalThreshold;
        isContinue &= length(photon0.posW - photon1.posW) < distanceThreshold;
        isContinue &= dot(photon0.normalW, photon0.posW - photon1.posW) < planarThreshold;
    }

    float3 posW0 = photon0.posW;
    float area0 = length(cross(photon0.dPdx,photon0.dPdy));
    float3 color0 = photon0.color;// / area0;
    float3 posW1;
    float3 color1;
    if (true || isContinue)
    {
        posW1 = photon1.posW;
        float area1 = length(cross(photon1.dPdx, photon1.dPdy));
        color1 = photon1.color;// / area1;
    }
    else
    {
        posW1 = photon0.posW + photon0.dPdx* offset.x + photon0.dPdy * offset.y;
        color1 = 0;
    }
    float4 posS0 = mul(float4(posW0, 1), viewProjMat);
    float4 posS1 = mul(float4(posW1, 1), viewProjMat);
    posS0 /= posS0.w;
    posS1 /= posS1.w;
    float2 dPosS = (posS1.xy - posS0.xy) * screenDim.xy * 0.5;
    float distS = max(length(dPosS),1);
    float dLuminance = abs(dot(color1 - color0, float3(0.2126, 0.7152, 0.0722)));// / distS;
    float luminanceSubd = 0;// dLuminancePerPixel / pixelLuminanceThreshold;
    float maxPixelSubd = distS / minPhotonPixelSize;
    //return (dLuminance > pixelLuminanceThreshold) ? min(4, maxPixelSubd) : 0;
    return min(20, maxPixelSubd);
    //float subd = min(luminanceSubd, maxPixelSubd);
    //return int(subd + 0.5);
}

int getRayTaskID(uint2 pos)
{
    return pos.y * taskDim.x + pos.x;
}

[numthreads(16, 16, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{
    uint rayIdx = getRayTaskID(threadIdx.xy);
    RayTask task0 = gRayTask[rayIdx];
    int idx0 = task0.photonIdx;
    if (idx0 == -1)
    {
        return;
    }

    Photon photon0 = gPhotonBuffer[idx0];
    float4 posS0 = mul(float4(photon0.posW, 1), viewProjMat);
    posS0 /= posS0.w;
    if (any(abs(posS0.xy) > 1) || posS0.z < 0 || posS0.z > 1)
    {
        return;
    }
    int2 screenPos = (posS0.xy * float2(1, -1) + 1) * 0.5 * screenDim;
    float depth = gDepthTex.Load(int3(screenPos.xy, 0)).x;
    if (posS0.w > depth + 0.1)
    {
        return;
    }

    float4 posSDx = mul(float4(photon0.posW + photon0.dPdx, 1), viewProjMat);
    float4 posSDy = mul(float4(photon0.posW + photon0.dPdy, 1), viewProjMat);
    posSDx /= posSDx.w;
    posSDy /= posSDy.w;
    float2 dsx = (posSDx.xy - posS0.xy) * screenDim;
    float2 dsy = (posSDy.xy - posS0.xy) * screenDim;
    float screenArea = abs(dsx.x * dsy.y - dsx.y * dsy.x);

    //int sampleCount = (subdW + subdE) * (subdN + subdS);
    int sampleCount = min(screenArea / (minPhotonPixelSize * minPhotonPixelSize),1024*8);
    if (sampleCount <= 1)
    {
        return;
    }
    float sampleWeight = 1.0 / sqrt(float(sampleCount));
    float2 pixelSize = float2(1, 1) * sampleWeight;
    gPhotonBuffer[idx0].dPdx *= sampleWeight;
    gPhotonBuffer[idx0].dPdy *= sampleWeight;
    gPhotonBuffer[idx0].color /= sampleCount;
    float2 screenCoord0 = task0.screenCoord;

    int taskIdx = 0;
    InterlockedAdd(gRayArgument[0].rayTaskCount, sampleCount, taskIdx);
    float g = 1.32471795724474602596;
    float a1 = 1.0 / g;
    float a2 = 1.0 / (g * g);
    for (uint i = 0; i < sampleCount; i++)
    {
        float x = frac(0.5 + a1 * (i + 1));
        float y = frac(0.5 + a2 * (i + 1));
        x = x * 2 - 1;
        y = y * 2 - 1;
        RayTask newTask;
        newTask.screenCoord = screenCoord0 + float2(x, y)*0.5;
        newTask.pixelSize = pixelSize;
        newTask.photonIdx = -1;
        gRayTask[taskIdx + i] = newTask;
    }
}

[numthreads(16, 16, 1)]
void addPhotonTaskFromTexture(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{
    uint rayIdx = getRayTaskID(threadIdx.xy);
    gPixelInfo[rayIdx].screenArea = 0;
    gPixelInfo[rayIdx].count = 0;

    RayTask task0 = gRayTask[rayIdx];
    int idx0 = task0.photonIdx;

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

    if (idx0 != -1)
    {
        //Photon photon0 = gPhotonBuffer[idx0];
        gPhotonBuffer[idx0].dPdx *= sampleWeight;
        gPhotonBuffer[idx0].dPdy *= sampleWeight;
        gPhotonBuffer[idx0].color /= sampleCount;
    }

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
        newTask.photonIdx = -1;
        gRayTask[taskIdx + i] = newTask;
    }
}
