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
__import Raytracing;
__import ShaderCommon;
__import Shading;
import Helpers;

RWStructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
RWStructuredBuffer<RayTask> gRayTask;
RWStructuredBuffer<PixelInfo> gPixelInfo;
Texture2D gUniformNoise;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    uint2 coarseDim;
    float2 randomOffset;
    float emitSize;
    float roughThreshold;
    float jitter;
    int launchRayTask;
    int rayTaskOffset;
    int maxDepth;
    float iorOverride;
    uint colorPhotonID;
    int photonIDScale;
    float traceColorThreshold;
    float cullColorThreshold;
    uint  gAreaType;
    float gIntensity;
    float gSplatSize;
};

struct PrimaryRayData
{
    float3 color;
    float hitT;
    float3 nextDir;
    uint isContinue;
    float3 dPdx, dPdy, dDdx, dDdy;  // ray differentials
};

struct ShadowRayData
{
    bool hit;
};

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.color = float3(0, 0, 0);
    hitData.isContinue = 0;
    hitData.hitT = 0;
}

void getVerticesAndNormals(
    uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs,
    out float3 P0, out float3 P1, out float3 P2,
    out float3 N0, out float3 N1, out float3 N2,
    out float3 N)
{
    uint3 indices = getIndices(triangleIndex);
    int address = (indices[0] * 3) * 4;
    P0 = asfloat(gPositions.Load3(address)).xyz;
    P0 = mul(float4(P0, 1.f), gWorldMat[0]).xyz;
    N0 = asfloat(gNormals.Load3(address)).xyz;
    N0 = normalize(mul(float4(N0, 0.f), gWorldMat[0]).xyz);

    address = (indices[1] * 3) * 4;
    P1 = asfloat(gPositions.Load3(address)).xyz;
    P1 = mul(float4(P1, 1.f), gWorldMat[0]).xyz;
    N1 = asfloat(gNormals.Load3(address)).xyz;
    N1 = normalize(mul(float4(N1, 0.f), gWorldMat[0]).xyz);

    address = (indices[2] * 3) * 4;
    P2 = asfloat(gPositions.Load3(address)).xyz;
    P2 = mul(float4(P2, 1.f), gWorldMat[0]).xyz;
    N2 = asfloat(gNormals.Load3(address)).xyz;
    N2 = normalize(mul(float4(N2, 0.f), gWorldMat[0]).xyz);

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    N = (N0 * barycentrics.x + N1 * barycentrics.y + N2 * barycentrics.z);
}

void updateTransferRayDifferential(
    float3 N,
    inout float3 dPdx,
    inout float3 dDdx)
{
    float3 D = WorldRayDirection();
    float t = RayTCurrent();
    float dtdx = -1 * dot(dPdx + t * dDdx, N) / dot(D, N);
    dPdx = dPdx + t * dDdx + dtdx * D;
}

void calculateDNdx(
    float3 P0, float3 P1, float3 P2,
    float3 N0, float3 N1, float3 N2,
    float3 N,
    float3 dPdx, out float3 dNdx)
{
    float3 dP1 = P1 - P0;
    float3 dP2 = P2 - P0;

    float P11 = dot(dP1, dP1);
    float P12 = dot(dP1, dP2);
    float P22 = dot(dP2, dP2);

    float Q1 = dot(dP1, dPdx);
    float Q2 = dot(dP2, dPdx);

    float delta = P11 * P22 - P12 * P12;
    float dudx = (Q1 * P22 - Q2 * P12) / delta;
    float dvdx = (Q2 * P11 - Q1 * P12) / delta;

    float n2 = dot(N, N);
    float n1 = sqrt(n2);
    dNdx = dudx * (N1 - N0) + dvdx * (N2 - N0);
    dNdx = (n2 * dNdx - dot(N, dNdx) * N) / (n2 * n1);
}

void updateReflectRayDifferential(
    float3 P0, float3 P1, float3 P2,
    float3 N0, float3 N1, float3 N2, float3 N,
    float3 dPdx,
    inout float3 dDdx)
{
    float3 dNdx=0;
    calculateDNdx(P0, P1, P2, N0, N1, N2, N, dPdx, dNdx);

    N = normalize(N);
    float3 D = WorldRayDirection();
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    dDdx = dDdx - 2 * (dot(D, N) * dNdx + dDNdx * N);
}

float getPhotonScreenArea(float3 posW, float3 dPdx, float3 dPdy, out bool inFrustum)
{
    dPdx = dPdx * gSplatSize;
    dPdy = dPdy * gSplatSize;
    float4 s0 = mul(float4(posW, 1), gCamera.viewProjMat);
    float4 s00 = mul(float4(posW + dPdx + dPdy, 1), gCamera.viewProjMat);
    float4 s01 = mul(float4(posW + dPdx - dPdy, 1), gCamera.viewProjMat);
    float4 s10 = mul(float4(posW - dPdx + dPdy, 1), gCamera.viewProjMat);
    float4 s11 = mul(float4(posW - dPdx - dPdy, 1), gCamera.viewProjMat);
    inFrustum = isInFrustum(s00) || isInFrustum(s01) || isInFrustum(s10) || isInFrustum(s11);
    s0 /= s0.w;
    s00 /= s00.w;
    s01 /= s01.w;
    float2 dx = (s00.xy - s0.xy) * viewportDims;
    float2 dy = (s01.xy - s0.xy) * viewportDims;
    float area = abs(dx.x * dy.y - dy.x * dx.y) / (gSplatSize * gSplatSize);
    //float area = 0.5 * (dot(dx, dx) + dot(dy, dy)) / (gSplatSize * gSplatSize);
    return area;
}


void updateRefractRayDifferential(
    float3 P0, float3 P1, float3 P2,
    float3 N0, float3 N1, float3 N2,
    float3 D, float3 R, float3 N, float eta,
    float3 dPdx,
    inout float3 dDdx)
{
    float3 dNdx = 0;
    calculateDNdx(P0, P1, P2, N0, N1, N2, N, dPdx, dNdx);

    N = normalize(N);
    float DN = dot(D, N);
    float RN = dot(R, N);
    float mu = eta * DN - RN;
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    float dmudx = (eta - eta * eta * DN / RN) * dDNdx;
    dDdx = eta * dDdx - (mu * dNdx + dmudx * N);
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Get the hit-point data
    float3 rayOrigW = WorldRayOrigin();
    float3 rayDirW = WorldRayDirection(); 
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();
    float3 posW = rayOrigW + hitT * rayDirW;

    // prepare the shading data
    VertexOut v = getVertexAttributes(triangleIndex, attribs);
    ShadingData sd = prepareShadingData(v, gMaterial, rayOrigW, 0);
    hitData.hitT = hitT;

    float3 P0, P1, P2, N0, N1, N2, N;
    getVerticesAndNormals(PrimitiveIndex(), attribs, P0, P1, P2, N0, N1, N2, N);
    updateTransferRayDifferential(N, hitData.dPdx, hitData.dDdx);
    updateTransferRayDifferential(N, hitData.dPdy, hitData.dDdy);

    hitData.isContinue = 0;
    bool isSpecular = (sd.linearRoughness > roughThreshold || sd.opacity < 1);
    if (isSpecular)
    {
        bool isReflect = (sd.opacity == 1);
        float3 R;
        float eta = iorOverride > 0 ? 1.0 / iorOverride : 1.0 / sd.IoR;
        if (!isReflect)
        {
            if (dot(v.normalW, rayDirW) > 0)
            {
                eta = 1.0 / eta;
                N0 *= -1;
                N1 *= -1;
                N2 *= -1;
                N *= -1;
                v.normalW *= -1;
            }
            isReflect = isTotalInternalReflection(rayDirW, v.normalW, eta);
        }

        if (isReflect)
        {
            updateReflectRayDifferential(P0, P1, P2, N0, N1, N2, N, hitData.dPdx, hitData.dDdx);
            updateReflectRayDifferential(P0, P1, P2, N0, N1, N2, N, hitData.dPdy, hitData.dDdy);
            R = reflect(rayDirW, v.normalW);
        }
        else
        {
            getRefractVector(rayDirW, v.normalW, R, eta);
            updateRefractRayDifferential(P0, P1, P2, N0, N1, N2, rayDirW, R, N, eta, hitData.dPdx, hitData.dDdx);
            updateRefractRayDifferential(P0, P1, P2, N0, N1, N2, rayDirW, R, N, eta, hitData.dPdy, hitData.dDdy);
        }

        float3 baseColor = lerp(1, sd.diffuse, sd.opacity);
        hitData.color = baseColor * hitData.color;// *float4(sd.specular, 1);
        float area = (dot(hitData.dPdx, hitData.dPdx) + dot(hitData.dPdy, hitData.dPdy)) * 0.5;
        //hitData.color.rgb = hitData.color.rgb / area;
        hitData.nextDir = R;

        if (dot(hitData.color/ area, float3(0.299, 0.587, 0.114)) > traceColorThreshold)
        {
            hitData.isContinue = 1;
        }
    }
    else
    {
        hitData.color.rgb = dot(-rayDirW, sd.N)* sd.diffuse* hitData.color.rgb;
        hitData.nextDir = sd.N;
    }
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    uint3 launchDimension = DispatchRaysDimensions();
    RayDesc ray;
    float3 lightOrigin = gLights[0].dirW * -100;// gLights[0].posW;
    float3 lightDirZ = gLights[0].dirW;
    float3 lightDirX = normalize(float3(-lightDirZ.z, 0, lightDirZ.x));
    float3 lightDirY = normalize(cross(lightDirZ, lightDirX));
    float2 lightUV;
    float2 pixelSize = float2(1,1);
    uint taskIdx = launchIndex.y * launchDimension.x + launchIndex.x;
    uint2 pixelCoord;
    if (launchRayTask)
    {
        taskIdx += rayTaskOffset;
        if (taskIdx >= gRayArgument[0].rayTaskCount)
        {
            return;
        }
        RayTask task = gRayTask[taskIdx];
        pixelCoord = task.screenCoord;
        lightUV = (task.screenCoord) / float2(coarseDim);
        pixelSize = task.pixelSize;
    }
    else
    {
        if (any(launchIndex.xy >= coarseDim))
        {
            return;
        }
        pixelCoord = launchIndex.xy;
        lightUV = float2(launchIndex.xy) / float2(coarseDim.xy);
        uint nw, nh, nl;
        gUniformNoise.GetDimensions(0, nw, nh, nl);
        float2 noise = gUniformNoise.Load(uint3(launchIndex.xy % uint2(nw, nh), 0)).rg;
        lightUV += (noise * jitter + randomOffset*0) * pixelSize;
    }
    lightUV = lightUV * 2 - 1;
    pixelSize *= emitSize / float2(coarseDim.xy);

    ray.Origin = lightOrigin + (lightDirX * lightUV.x + lightDirY * lightUV.y) * emitSize;
    ray.Direction = lightDirZ;
    ray.TMin = 0.0;
    ray.TMax = 1e10;

    PrimaryRayData hitData;
    float3 color0 = 1;
    if (colorPhotonID)
    {
        color0.xyz = frac(launchIndex.xyz / float(photonIDScale)) * 0.8 + 0.2;
    }
    hitData.color = color0* pixelSize.x* pixelSize.y * 512 * 512 * 0.5 * gIntensity;
    hitData.dDdx = 0;// lightDirX* pixelSize.x * 2.0;
    hitData.dDdy = 0;// lightDirY* pixelSize.y * 2.0;
    hitData.dPdx = lightDirX * pixelSize.x * 2.0;
    hitData.dPdy = -lightDirY * pixelSize.y * 2.0;
    hitData.nextDir = ray.Direction;
    hitData.isContinue = 1;

    gRayTask[taskIdx].photonIdx = -1;
    if (!launchRayTask)
    {
        gRayTask[taskIdx].screenCoord = launchIndex.xy;
        gRayTask[taskIdx].pixelSize = float2(1, 1);
        gRayTask[taskIdx].pixelArea = 0;
        gRayTask[taskIdx].inFrustum = 0;
    }

    int depth;
    for (depth = 0; depth < maxDepth && hitData.isContinue; depth++)
    {
        TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, hitData);

        ray.Origin = ray.Origin + ray.Direction * hitData.hitT;
        ray.Direction = hitData.nextDir;
        ray.TMin = 0.001;
        ray.TMax = 1e10;
    }

    if (any(hitData.color > 0) && depth > 1)
    {
        float area;
        if (gAreaType == 0)
        {
            area = (dot(hitData.dPdx, hitData.dPdx) + dot(hitData.dPdy, hitData.dPdy)) * 0.5;
        }
        else if (gAreaType == 1)
        {
            area = (length(hitData.dPdx) + length(hitData.dPdy));
            area *= area;
        }
        else if (gAreaType == 2)
        {
            area = max(dot(hitData.dPdx, hitData.dPdx), dot(hitData.dPdy, hitData.dPdy));
        }
        else
        {
            float3 areaVector = cross(hitData.dPdx, hitData.dPdy);
            area = length(areaVector);
        }

        float3 color = hitData.color.rgb / area;
        bool isInFrustum;
        float3 posW = ray.Origin;
        float pixelArea = getPhotonScreenArea(posW, hitData.dPdx, hitData.dPdy, isInFrustum);
        if (dot(color, float3(0.299, 0.587, 0.114)) > cullColorThreshold && isInFrustum)
        {
            uint instanceIdx = 0;
            InterlockedAdd(gDrawArgument[0].instanceCount, 1, instanceIdx);

            Photon photon;
            photon.posW = posW;
            photon.normalW = hitData.nextDir;
            photon.color = color;
            photon.dPdx = hitData.dPdx;
            photon.dPdy = hitData.dPdy;
            gPhotonBuffer[instanceIdx] = photon;

            if (!launchRayTask)
            {
                uint3 dim3 = DispatchRaysDimensions();
                uint3 idx3 = DispatchRaysIndex();
                uint idx = idx3.y * dim3.x + idx3.x;
                gRayTask[idx].photonIdx = instanceIdx;
                gRayTask[idx].pixelArea = pixelArea;
                gRayTask[idx].inFrustum = isInFrustum ? 1 : 0;

            }
            uint oldV;
            uint pixelLoc = pixelCoord.y * coarseDim.x + pixelCoord.x;
            InterlockedAdd(gPixelInfo[pixelLoc].screenArea, uint(pixelArea.x), oldV);
            InterlockedAdd(gPixelInfo[pixelLoc].screenAreaSq, uint(pixelArea.x * pixelArea.x), oldV);
            InterlockedAdd(gPixelInfo[pixelLoc].count, 1, oldV);
        }
    }
}
