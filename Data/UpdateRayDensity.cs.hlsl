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
    int2 coarseDim;
    float minPhotonPixelSize;
    float smoothWeight;
    int maxTaskPerPixel;
    float updateSpeed;
};

RWStructuredBuffer<PixelInfo> gPixelInfo;
RWTexture2D<float> gRayDensityTex;

float getAvgArea(uint2 pos)
{
    int idx = coarseDim.x * pos.y + pos.x;
    float avgArea = (gPixelInfo[idx].screenArea / float(gPixelInfo[idx].count + 0.01));
    return avgArea;
}

[numthreads(16, 16, 1)]
void updateRayDensityTex(uint3 threadIdx : SV_DispatchThreadID)
{
    int idx = coarseDim.x * threadIdx.y + threadIdx.x;

    float a[5];
    a[0] = getAvgArea(threadIdx.xy);
    a[1] = getAvgArea(threadIdx.xy + uint2(1, 0));
    a[2] = getAvgArea(threadIdx.xy + uint2(-1, 0));
    a[3] = getAvgArea(threadIdx.xy + uint2(0, 1));
    a[4] = getAvgArea(threadIdx.xy + uint2(0, -1));

    float area = a[0];
    //for (int i = 1; i < 5; i++)
    //{
    //    //area = max(area, a[i]);
    //    area += a[i]* smoothWeight;
    //}
    //area /= (1 + smoothWeight * 4);

    float targetArea = minPhotonPixelSize * minPhotonPixelSize;

    float oldDensity = gRayDensityTex[threadIdx.xy].r;
    float weight = smoothWeight * updateSpeed;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(1, 0)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(0, 1)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(-1, 0)].r * weight;
    oldDensity += gRayDensityTex[threadIdx.xy + uint2(0, -1)].r * weight;
    oldDensity /= (1 + weight * 4);

    float newDensity = oldDensity * area / targetArea;// oldDensity + (area - targetArea) * 0.1;
    newDensity = newDensity * updateSpeed + oldDensity * (1- updateSpeed);
    gRayDensityTex[threadIdx.xy] = clamp(newDensity, 0.1, maxTaskPerPixel);

    //gPixelInfo[idx].screenArea = 0;
    //gPixelInfo[idx].count = 0;
}
