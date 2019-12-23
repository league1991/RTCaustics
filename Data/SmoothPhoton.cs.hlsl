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
    float trimDirectionThreshold;
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
        isContinue &= isPhotonAdjecent(photon0, photon1, normalThreshold, distanceThreshold, planarThreshold);
    }
    return isContinue;
}

bool getPhoton(uint2 pixelCoord, inout Photon photon)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;
    RayTask task1 = gRayTask[pixelIdx1];
    if (task1.photonIdx == -1)
    {
        return false;
    }
    photon = gPhotonBuffer[task1.photonIdx];
    return true;
}

void getTrimLength(float3 P1, float3 D1, float3 P2, float3 D2, inout float l)
{
    D1 = normalize(D1);
    D2 = normalize(D2);
    float D1D2 = dot(D1, D2);
    if (abs(D1D2) > trimDirectionThreshold)
    {
        return;
    }
    //float3 P12 = P1 - P2;
    //float P12D1 = dot(P12, D1);
    //float P12D2 = dot(P12, D2);
    //float det = -1 + D1D2 * D1D2;
    //float t1 = P12D1 - P12D2 * D1D2;
    //l = min(l, abs(t1 / det));

    float3 P21 = P2 - P1;
    float proj = length(P21);// dot(P21, D1);
    l = min(l, abs(proj));
}

[numthreads(16, 16, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{

    //uint length, stride;
    //gPhotonBuffer.GetDimensions(length, stride);
    uint2 pixelCoord0 = threadIdx.xy;
    uint rayIdx = threadIdx.y * taskDim.x + threadIdx.x;
    RayTask task0 = gRayTask[rayIdx];
    int idx0 = task0.photonIdx;
    if (idx0 == -1)
    {
        return;
    }

    Photon photon0 = gPhotonBuffer[idx0];

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
    uint continueFlag = 0;
    float3 color = photon0.color;// / 9;
    int continueCount = 0;
    for (int i = 0; i < 8; i++)
    {
        //bool isContinue = checkPixelNeighbour(threadIdx.xy, task0, photon0, dir[i]);
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
            continueFlag |= (1 << i);
        }
    }
    //if (continueCount < 5)
    {
        float w = saturate((continueCount - 0.0) / (8.0 - 0.0));
        color *= lerp(0.0,1,w);
    }
    //gPhotonBuffer[idx0].color = color;

    //Photon photon1;
    //float trimmedLength = length(photon0.dPdx);
    //if (getPhoton(pixelCoord0 + uint2(1, 0), photon1) && (continueFlag & (1 << 6)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdx, photon1.posW, photon1.dPdx, trimmedLength);
    //}
    //if (getPhoton(pixelCoord0 + uint2(-1, 0), photon1) && (continueFlag & (1 << 1)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdx, photon1.posW, photon1.dPdx, trimmedLength);
    //}
    //photon0.dPdx = photon0.dPdx * trimmedLength / length(photon0.dPdx);

    //trimmedLength = length(photon0.dPdy);
    //if (getPhoton(pixelCoord0 + uint2(0, 1), photon1) && (continueFlag & (1 << 4)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdy, photon1.posW, photon1.dPdy, trimmedLength);
    //}
    //if (getPhoton(pixelCoord0 + uint2(0, -1), photon1) && (continueFlag & (1 << 3)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdy, photon1.posW, photon1.dPdy, trimmedLength);
    //}
    //photon0.dPdy = photon0.dPdy * trimmedLength / length(photon0.dPdy);

    //uint fourNeighbourFlag = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6);
    //if ((continueFlag & fourNeighbourFlag) == 0)
    //{
    //    photon0.color = 0;
    //}

    float width = length(photon0.dPdx);
    float height = length(photon0.dPdy);
    float area0 = width * height;
    float area = length(cross(photon0.dPdx, photon0.dPdy));
    float ratio = area / area0;
    if (ratio < trimDirectionThreshold)
    {
        photon0.color = 0;
    }

    gPhotonBuffer[idx0] = photon0;
}
