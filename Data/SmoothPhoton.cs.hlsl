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
Texture2D gDepthTex;

//float getNeighbourArea(uint2 pixelCoord0, uint2 offset, out int isInFrustum)
//{
//    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + offset));
//    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;
//
//    RayTask task1 = gRayTask[pixelIdx1];
//    isInFrustum = task1.inFrustum;
//    return task1.pixelArea;
//}

bool checkPixelNeighbour(uint2 pixelCoord0, RayTask task0, Photon photon0, uint2 offset)
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
    return isContinue;
}

[numthreads(16, 16, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{

    //uint length, stride;
    //gPhotonBuffer.GetDimensions(length, stride);
    uint rayIdx = threadIdx.y * taskDim.x + threadIdx.x;
    RayTask task0 = gRayTask[rayIdx];
    int idx0 = task0.photonIdx;
    if (idx0 == -1)
    {
        return;
    }

    Photon photon0 = gPhotonBuffer[idx0];
    //float4 posS0 = mul(float4(photon0.posW, 1), viewProjMat);
    //posS0 /= posS0.w;
    //if (any(abs(posS0.xy) > 1) || posS0.z < 0 || posS0.z > 1)
    //{
    //    return;
    //}
    //int2 screenPos = (posS0.xy * float2(1,-1) + 1) * 0.5 * screenDim;
    //float depth = gDepthTex.Load(int3(screenPos.xy, 0)).x;
    //if (posS0.w > depth + 0.1)
    //{
    //    return;
    //}

    //float subdW = checkPixelNeighbour(threadIdx.xy, task0, photon0, uint2(-1,0));
    //float subdE = checkPixelNeighbour(threadIdx.xy, task0, photon0, uint2(1, 0));
    //float subdN = checkPixelNeighbour(threadIdx.xy, task0, photon0, uint2(0, -1));
    //float subdS = checkPixelNeighbour(threadIdx.xy, task0, photon0, uint2(0, 1));

    //int subdX = max(subdW, subdE);
    //int subdY = max(subdN, subdS);
    //if (subdX == 0 || subdY == 0)
    //{
    //    return;
    //}
    //int sampleX = subdX * 2;
    //int sampleY = subdY * 2;
    //int sampleCount = sampleX * sampleY;
    //gPhotonBuffer[idx0].dPdx /= (subdX + 1);
    //gPhotonBuffer[idx0].dPdy /= (subdY + 1);
    //gPhotonBuffer[idx0].color /= sampleCount;
    //float2 pixelSize = float2(1, 1) / float2(sampleX, sampleY);
    //gRayTask[rayIdx].pixelSize = pixelSize;
    //float2 screenCoord0 = task0.screenCoord - pixelSize * float2(subdX - 1, subdY - 1);

    //int taskIdx = 0;
    //InterlockedAdd(gRayArgument[0].rayTaskCount, sampleCount -1, taskIdx);

    //for (uint i = 0, offset = 0; i < sampleCount; i++)
    //{
    //    uint y = i / sampleX;
    //    uint x = i - sampleX * y;
    //    if (x == subdX-1 && y == subdY-1)
    //    {
    //        continue;
    //    }
    //    RayTask newTask;
    //    newTask.screenCoord = screenCoord0 + pixelSize * float2(x, y);
    //    newTask.pixelSize = pixelSize;
    //    newTask.photonIdx = -1;
    //    gRayTask[taskIdx + offset] = newTask;
    //    offset++;
    //}

    uint2 dir[8] = {
        uint2(-1,-1),
        uint2(-1,0),
        uint2(-1,1),
        uint2(0,-1),
        //uint2(0,0),
        uint2(0,1),
        uint2(1,-1),
        uint2(1,0),
        uint2(1,1),
    };
    float3 color = photon0.color;// / 9;
    int continueCount = 0;
    for (int i = 0; i < 8; i++)
    {
        //bool isContinue = checkPixelNeighbour(threadIdx.xy, task0, photon0, dir[i]);
        uint2 pixelCoord0 = threadIdx.xy;
        uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + dir[i]));
        uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

        RayTask task1 = gRayTask[pixelIdx1];
        bool isContinue = task1.photonIdx != -1;
        Photon photon1;
        if (isContinue)
        {
            photon1 = gPhotonBuffer[task1.photonIdx];
            isContinue &= dot(photon0.normalW, photon1.normalW) > normalThreshold;
            isContinue &= length(photon0.posW - photon1.posW) < distanceThreshold;
            isContinue &= dot(photon0.normalW, photon0.posW - photon1.posW) < planarThreshold;
        }
        if (isContinue)
        {
            //color += photon1.color / 9;
            continueCount++;
        }
    }
    //if (continueCount < 5)
    {
        float w = saturate((continueCount - 0.0) / (8.0 - 0.0));
        color *= lerp(0.0,1,w);
    }
    gPhotonBuffer[idx0].color = color;
}
