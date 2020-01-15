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
    float4x4 gInvViewProjMat;
    int2 screenDim;
    int2 tileDim;
    float gSplatSize;
    float gDepthRadius;
    int gShowTileCount;
    int gTileCountScale;
    float gKernelPower;
    int causticsMapResRatio;
};

struct DrawArguments
{
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int  baseVertexLocation;
    uint startInstanceLocation;
};


StructuredBuffer<Photon> gPhotonBuffer;
StructuredBuffer<IDBlock> gTileInfo;
ByteAddressBuffer gIDBuffer;

Texture2D gDepthTex;
Texture2D gNormalTex;
RWTexture2D gPhotonTex;

int getTileOffset(int x, int y)
{
    return tileDim.x * y + x;
}

float2 getLocalCoordinate(float3 a, float3 b, float3 P)
{
    //float xa = length(a);
    //float xb = dot(a, b) / xa;
    //float xc = dot(a, P) / xa;
    //float3 y = b - a * (xb / xa);
    //float yb = dot(y, b);
    //float yc = dot(y, P);
    //float cb = yc / yb;
    //float ca = (xc - xb * cb) / xa;
    //return float2(ca, cb);

    float ca = dot(a, P) / dot(a, a);
    float cb = dot(b, P) / dot(b, b);
    return float2(ca, cb);
}

float getLightFactor(float3 pos, float3 photonPos, float3 dPdx, float3 dPdy, float3 normal)
{
    float3 dPos = pos - photonPos;
    //float dist = length(dPos) / (gSplatSize);
    //return saturate(1 - dist);
    //dPdx *= gSplatSize;
    //dPdy *= gSplatSize;
    //float3 normal = normalize(cross(dPdx, dPdy));
    float z = dot(dPos, normal) / (gDepthRadius * gSplatSize);
    if (abs(z) > 1)
    {
        return 0;
    }
    dPos -= dot(dPos, normal) * normal;
    float2 localCoord = getLocalCoordinate(dPdx, dPdy, dPos);
    float r2 = dot(localCoord, localCoord);
    float dist2 = r2 + z * z;
    return pow(smoothKernel(sqrt(dist2)), gKernelPower);
}

#define PHOTON_CACHE_SIZE 64
groupshared Photon photonList[PHOTON_CACHE_SIZE];
groupshared float3 normalList[PHOTON_CACHE_SIZE];
groupshared int photonCount;
groupshared int beginAddress;

[numthreads(GATHER_TILE_SIZE, GATHER_TILE_SIZE, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID, uint3 threadIdx : SV_DispatchThreadID)
{
    uint2 tileID = groupID.xy;
    uint2 pixelTileIndex = groupThreadID.xy;
    uint2 pixelLocation = tileID * GATHER_TILE_SIZE + pixelTileIndex;

    if (any(pixelLocation >= screenDim))
    {
        //return;
    }
    float depth = gDepthTex.Load(int3(pixelLocation* causticsMapResRatio,0)).r;
    //float3 normal = gNormalTex.Load(int3(pixelLocation* causticsMapResRatio,0)).rgb;
    float2 uv = pixelLocation / float2(screenDim);
    float4 ndc = float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
    float4 worldPnt = mul(ndc,gInvViewProjMat);
    worldPnt /= worldPnt.w;

    if (all(groupThreadID == uint3(0,0,0)))
    {
        int tileOffset = getTileOffset(tileID.x, tileID.y);
        beginAddress = gTileInfo[tileOffset].address;
        photonCount = gTileInfo[tileOffset].count;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gShowTileCount)
    {
        float intensity = float(photonCount) / gTileCountScale;
        float4 color;
        if (intensity <= 1)
        {
            color = float4(intensity.xxx, 1);
        }
        else
        {
            color = float4(1, 0, 0, 1);
        }
        if (all(pixelLocation < screenDim))
            gPhotonTex[pixelLocation] = color;
        return;
    }

    float3 totalLight = 0;
    int threadGroupOffset = pixelTileIndex.y * GATHER_TILE_SIZE + pixelTileIndex.x;
    for (int photonIdx = 0; photonIdx < photonCount; photonIdx+= PHOTON_CACHE_SIZE)
    {
        int idOffset = photonIdx + threadGroupOffset;
        if (threadGroupOffset < PHOTON_CACHE_SIZE && idOffset < photonCount)
        {
            int id = gIDBuffer.Load((beginAddress + idOffset) * 4);
            //Photon p = gPhotonBuffer[id];
            //PhotonData pd;
            //pd.posW = p.posW;
            //pd.color = p.color;
            //pd.dPdx = p.dPdx * gSplatSize;
            //pd.dPdy = p.dPdy * gSplatSize;
            //pd.normalW = normalize(cross(p.dPdx, p.dPdy));
            photonList[threadGroupOffset] = gPhotonBuffer[id];
            //photonList[threadGroupOffset].dPdx *= gSplatSize;
            //photonList[threadGroupOffset].dPdy *= gSplatSize;
            normalList[threadGroupOffset] =
                normalize(cross(photonList[threadGroupOffset].dPdx, photonList[threadGroupOffset].dPdy));
        }
        GroupMemoryBarrierWithGroupSync();

        for (int i = 0; i < min(PHOTON_CACHE_SIZE, photonCount - photonIdx); i++)
        {
            Photon p = photonList[i];
            float lightFactor = getLightFactor(worldPnt.xyz, p.posW, p.dPdx, p.dPdy, normalList[i]);
            totalLight += lightFactor * p.color;
        }
    }

    if (all(pixelLocation < screenDim))
        gPhotonTex[pixelLocation] = float4(totalLight, 1);
}
