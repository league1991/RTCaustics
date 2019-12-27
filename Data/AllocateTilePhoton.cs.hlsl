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

__import ShaderCommon;

#include "Common.hlsl"

shared cbuffer PerFrameCB
{
    float4x4 gViewProjMat;
    int2 screenDim;
    int2 tileDim;
    float gSplatSize;
};


RWStructuredBuffer<DrawArguments> gDrawArgument;
StructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<IDBlock> gTileInfo;
RWByteAddressBuffer gIDBuffer;
RWByteAddressBuffer gIDCounter;

int getTileOffset(int x, int y)
{
    return tileDim.x * y + x;
}

void GetPhotonScreenRange(Photon photon, out int2 minTileID, out int2 maxTileID)
{
    // get screen position
    float3 corner0 = photon.dPdx + photon.dPdy;
    float3 corner1 = photon.dPdx - photon.dPdy;
    float4 px0 = mul(float4(photon.posW + corner0 * gSplatSize, 1), gViewProjMat);
    float4 px1 = mul(float4(photon.posW - corner0 * gSplatSize, 1), gViewProjMat);
    float4 py0 = mul(float4(photon.posW + corner1 * gSplatSize, 1), gViewProjMat);
    float4 py1 = mul(float4(photon.posW - corner1 * gSplatSize, 1), gViewProjMat);
    px0.xyz /= px0.w;
    px1.xyz /= px1.w;
    py0.xyz /= py0.w;
    py1.xyz /= py1.w;
    px0.y *= -1;
    px1.y *= -1;
    py0.y *= -1;
    py1.y *= -1;

    // get range
    float2 minCoord = 10000;
    float2 maxCoord = -10000;
    minCoord = min(minCoord, px0.xy);
    minCoord = min(minCoord, px1.xy);
    minCoord = min(minCoord, py0.xy);
    minCoord = min(minCoord, py1.xy);
    maxCoord = max(maxCoord, px0.xy);
    maxCoord = max(maxCoord, px1.xy);
    maxCoord = max(maxCoord, py0.xy);
    maxCoord = max(maxCoord, py1.xy);

    if (any(minCoord > 1) || any(maxCoord < -1))
    {
        minTileID = 1;
        maxTileID = -1;
        return;
    }
    minCoord = (minCoord + 1) * 0.5;
    maxCoord = (maxCoord + 1) * 0.5;
    minCoord = clamp(minCoord, 0, 1);
    maxCoord = clamp(maxCoord, 0, 1);

    minTileID = minCoord * screenDim / GATHER_TILE_SIZE;
    maxTileID = maxCoord * screenDim / GATHER_TILE_SIZE;
}

[numthreads(64, 1, 1)]
void CountTilePhoton(uint3 threadIdx : SV_DispatchThreadID)
{
    int photonID = threadIdx.x;
    if (photonID >= gDrawArgument[0].instanceCount)
    {
        return;
    }
    if (photonID == 0)
    {
        gIDCounter.Store(0, 0);
    }

    Photon photon = gPhotonBuffer[photonID];
    int2 minTileID, maxTileID;
    GetPhotonScreenRange(photon, minTileID, maxTileID);
    for (int x = minTileID.x; x <= maxTileID.x; x++)
    {
        for (int y = minTileID.y; y <= maxTileID.y; y++)
        {
            int offset = getTileOffset(x, y);
            int count;
            InterlockedAdd(gTileInfo[offset].count, 1, count);
        }
    }
}

[numthreads(16, 16, 1)]
void AllocateMemory(uint3 threadIdx : SV_DispatchThreadID)
{
    int2 tileID = threadIdx.xy;
    if (any(tileID >= tileDim))
    {
        return;
    }

    int offset = getTileOffset(tileID.x, tileID.y);
    uint count = gTileInfo[offset].count;
    uint address = 0;
    gIDCounter.InterlockedAdd(0, count, address);
    gTileInfo[offset].address = address;
    gTileInfo[offset].count = 0;
}

[numthreads(64, 1, 1)]
void StoreTilePhoton(uint3 threadIdx : SV_DispatchThreadID)
{
    int photonID = threadIdx.x;
    if (photonID >= gDrawArgument[0].instanceCount)
    {
        return;
    }

    Photon photon = gPhotonBuffer[photonID];
    int2 minTileID, maxTileID;
    GetPhotonScreenRange(photon, minTileID, maxTileID);
    for (int x = minTileID.x; x <= maxTileID.x; x++)
    {
        for (int y = minTileID.y; y <= maxTileID.y; y++)
        {
            int offset = getTileOffset(x, y);
            int address = gTileInfo[offset].address;
            int idx = 0;
            InterlockedAdd(gTileInfo[offset].count, 1, idx);
            gIDBuffer.Store((address + idx)*4, photonID);
        }
    }
}
