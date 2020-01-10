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
    uint updatePhoton;

    float gMaxScreenRadius;
};

struct PrimaryRayData
{
    float3 color;
    float hitT;
    float3 nextDir;
    uint isContinue;
#ifdef RAY_DIFFERENTIAL
    float3 dPdx, dPdy, dDdx, dDdy;  // ray differentials
#elif defined(RAY_CONE)
    float radius;
    float dRadius;
#elif defined(RAY_NONE)
#endif
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

VertexOut getVertexAttributes(
    uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs,
    out float3 dP1, out float3 dP2, out float3 dN1, out float3 dN2)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint3 indices = getIndices(triangleIndex);
    VertexOut v;
    v.texC = 0;
    v.normalW = 0;
    v.bitangentW = 0;
#ifdef USE_INTERPOLATED_POSITION
    v.posW = 0;
#else
    v.posW = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
#endif
    v.colorV = 0;
    v.prevPosH = 0;
    v.lightmapC = 0;

    float3 P[3], N[3];
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        int address = (indices[i] * 3) * 4;
        P[i] = asfloat(gPositions.Load3(address));
        N[i] = asfloat(gNormals.Load3(address));

        v.texC += asfloat(gTexCrds.Load2(address)) * barycentrics[i];
        v.normalW += N[i] * barycentrics[i];
        v.bitangentW += asfloat(gBitangents.Load3(address)) * barycentrics[i];
        v.lightmapC += asfloat(gLightMapUVs.Load2(address)) * barycentrics[i];

#ifdef USE_INTERPOLATED_POSITION
        v.posW += P[i] * barycentrics[i];
#endif
    }

#ifdef USE_INTERPOLATED_POSITION
    v.posW = mul(float4(v.posW, 1.f), gWorldMat[0]).xyz;
#endif

#ifndef _MS_DISABLE_INSTANCE_TRANSFORM
    // Transform normal/bitangent to world space
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMat[0]).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMat[0]).xyz;
#endif

    dP1 = mul(P[1] - P[0], (float3x3)gWorldMat[0]).xyz;
    dP2 = mul(P[2] - P[0], (float3x3)gWorldMat[0]).xyz;
    dN1 = mul(N[1] - N[0], (float3x3)gWorldInvTransposeMat[0]).xyz;
    dN2 = mul(N[2] - N[0], (float3x3)gWorldInvTransposeMat[0]).xyz;

    // Handle invalid bitangents gracefully (avoid NaN from normalization).
    v.bitangentW = dot(v.bitangentW, v.bitangentW) > 0.f ? normalize(v.bitangentW) : float3(0, 0, 0);
    return v;
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
    float3 dDdx)
{
    float3 D = WorldRayDirection();
    float t = RayTCurrent();
    float dtdx = -1 * dot(dPdx + t * dDdx, N) / dot(D, N);
    dPdx = dPdx + t * dDdx + dtdx * D;
}

void calculateDNdx(
    float3 dP1, float3 dP2,
    float3 dN1, float3 dN2,
    float3 N,
    float3 dPdx, out float3 dNdx)
{
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
    dNdx = dudx * dN1 + dvdx * dN2;
    dNdx = (n2 * dNdx - dot(N, dNdx) * N) / (n2 * n1);
}

void updateReflectRayDifferential(
    float3 N,
    float3 dPdx,
    float3 dNdx,
    inout float3 dDdx)
{
    N = normalize(N);
    float3 D = WorldRayDirection();
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    dDdx = dDdx - 2 * (dot(D, N) * dNdx + dDNdx * N);
}

void getPhotonDifferential(PrimaryRayData hitData, RayDesc ray, out float3 dPdx, out float3 dPdy)
{
#ifdef RAY_DIFFERENTIAL
    dPdx = hitData.dPdx;
    dPdy = hitData.dPdy;
#elif defined(RAY_CONE)
    float radius = hitData.radius;
    float3 normal = hitData.nextDir;
    float cosVal = dot(normal, ray.Direction);
    dPdx = normalize(ray.Direction - normal * cosVal) * radius;
    dPdy = cross(normal, dPdx);
    dPdx /= cosVal;
#elif defined(RAY_NONE)
    float radius = 0.2;
    float3 normal = hitData.nextDir;
    float cosVal = dot(normal, ray.Direction);
    dPdx = normalize(ray.Direction - normal * cosVal) * radius;
    dPdy = cross(normal, dPdx);
    dPdx /= cosVal;
#endif
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

    //float zRadius = max(1,length(dPdx) + length(dPdy)) * 0.5 / length(posW - gCamera.posW) * gCamera.nearZ * 0.5 * (viewportDims.x + viewportDims.y);
    //float area = zRadius * zRadius;
    //float area = 0.5 * (dot(dx, dx) + dot(dy, dy)) / (gSplatSize * gSplatSize);
    return area;
}


void updateRefractRayDifferential(
    float3 D, float3 R, float3 N, float eta,
    float3 dPdx,
    float3 dNdx,
    inout float3 dDdx)
{
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

    // prepare the shading data
    float3 dP1, dP2, dN1, dN2;
    VertexOut v = getVertexAttributes(triangleIndex, attribs, dP1, dP2, dN1, dN2);
    float3 N = v.normalW;
    v.normalW = normalize(v.normalW);
#ifdef RAY_DIFFERENTIAL
    updateTransferRayDifferential(v.normalW, hitData.dPdx, hitData.dDdx);
    updateTransferRayDifferential(v.normalW, hitData.dPdy, hitData.dDdy);
#elif defined(RAY_CONE)
    hitData.radius = abs(hitData.radius + hitT * hitData.dRadius);
#endif
    ShadingData sd = prepareShadingData(v, gMaterial, rayOrigW, 0);

    hitData.isContinue = 0;
    hitData.hitT = hitT;
    bool isSpecular = (sd.linearRoughness > roughThreshold || sd.opacity < 1);
    if (isSpecular)
    {
        float3 N_ = v.normalW;
        bool isReflect = (sd.opacity == 1);
        float3 R;
        float eta = iorOverride > 0 ? 1.0 / iorOverride : 1.0 / sd.IoR;
        if (!isReflect)
        {
            if (dot(N_, rayDirW) > 0)
            {
                eta = 1.0 / eta;
                N *= -1;
                N_ *= -1;
#ifdef RAY_DIFFERENTIAL
                dN1 *= -1;
                dN2 *= -1;
#endif
            }
            isReflect = isTotalInternalReflection(rayDirW, N_, eta);
        }

#ifdef RAY_DIFFERENTIAL
        float3 dNdx = 0;
        float3 dNdy = 0;
        calculateDNdx(dP1, dP2, dN1, dN2, N, hitData.dPdx, dNdx);
        calculateDNdx(dP1, dP2, dN1, dN2, N, hitData.dPdy, dNdy);
        if (isReflect)
        {
            updateReflectRayDifferential(N, hitData.dPdx, dNdx, hitData.dDdx);
            updateReflectRayDifferential(N, hitData.dPdy, dNdy, hitData.dDdy);
            R = reflect(rayDirW, N_);
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
            updateRefractRayDifferential(rayDirW, R, N, eta, hitData.dPdx, dNdx, hitData.dDdx);
            updateRefractRayDifferential(rayDirW, R, N, eta, hitData.dPdy, dNdy, hitData.dDdy);
        }
        float area = (dot(hitData.dPdx, hitData.dPdx) + dot(hitData.dPdy, hitData.dPdy)) * 0.5;
#elif defined(RAY_CONE)
        float cosVal = abs(dot(N_, rayDirW));
        float area = hitData.radius * hitData.radius / cosVal;
        float triArea = length(cross(dP1, dP2)) * 0.5;
        float triNormalArea = max(dot(dN1, dN1),dot(dN2, dN2));
        float isConcave = (dot(dP1, dN1) + dot(dP2, dN2) >= 0) ? 1.0 : -1.0;
        float dH = area / triArea * triNormalArea * isConcave * 2;
        float dO;
        if (isReflect)
        {
            R = reflect(rayDirW, N_);
            dO = dH * 4 * cosVal;
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
            float etaI = eta;
            float etaO = 1;
            float3 ht = -1 * (-etaI * rayDirW + etaO * R);
            float dHdO = dot(ht, ht) / (etaO * etaO * abs(dot(ht, R)));
            dO = dH * dHdO;
        }
        hitData.dRadius += sqrt(dO);
#elif defined(RAY_NONE)
        if (isReflect)
        {
            R = reflect(rayDirW, N_);
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
        }
        float area = 0.0001;
#endif
        hitData.nextDir = R;
        float3 baseColor = lerp(1, sd.diffuse, sd.opacity);
        hitData.color = baseColor * hitData.color;// *float4(sd.specular, 1);

        if (dot(hitData.color/ area, float3(0.299, 0.587, 0.114)) > traceColorThreshold)
        {
            hitData.isContinue = 1;
        }
        else
        {
            hitData.color = 0;
        }
    }
    else
    {
        hitData.color.rgb = dot(-rayDirW, sd.N)* sd.diffuse* hitData.color.rgb;
        hitData.nextDir = sd.N;
    }
}

float getArea(float3 dPdx, float3 dPdy)
{
    float area;
    if (gAreaType == 0)
    {
        area = (dot(dPdx, dPdx) + dot(dPdy, dPdy)) * 0.5;
    }
    else if (gAreaType == 1)
    {
        area = (length(dPdx) + length(dPdy));
        area *= area;
    }
    else if (gAreaType == 2)
    {
        area = max(dot(dPdx, dPdx), dot(dPdy, dPdy));
    }
    else
    {
        float3 areaVector = cross(dPdx, dPdy);
        area = length(areaVector);
    }
    return area;
}


void initFromLight(float2 lightUV, float2 pixelSize, out RayDesc ray, out PrimaryRayData hitData)
{
    lightUV = lightUV * 2 - 1;
    pixelSize *= emitSize / float2(coarseDim.xy);

    float3 lightOrigin = gLights[0].dirW * -100;// gLights[0].posW;
    float3 lightDirZ = gLights[0].dirW;
    float3 lightDirX = normalize(float3(-lightDirZ.z, 0, lightDirZ.x));
    float3 lightDirY = normalize(cross(lightDirZ, lightDirX));

    ray.Origin = lightOrigin + (lightDirX * lightUV.x + lightDirY * lightUV.y) * emitSize;
    ray.Direction = lightDirZ;
    ray.TMin = 0.001;
    ray.TMax = 1e10;

    float3 color0 = 1;
    if (colorPhotonID)
    {
        uint3 launchIndex = DispatchRaysIndex();
        color0.xyz = frac(launchIndex.xyz / float(photonIDScale)) * 0.8 + 0.2;
    }
    hitData.color = color0 * pixelSize.x * pixelSize.y * 512 * 512 * 0.5 * gIntensity;
    hitData.nextDir = ray.Direction;
    hitData.isContinue = 1;
#ifdef RAY_DIFFERENTIAL
    hitData.dDdx = 0;// lightDirX* pixelSize.x * 2.0;
    hitData.dDdy = 0;// lightDirY* pixelSize.y * 2.0;
    hitData.dPdx = lightDirX * pixelSize.x * 2.0;
    hitData.dPdy = -lightDirY * pixelSize.y * 2.0;
#elif defined(RAY_CONE)
    hitData.radius = 0.5 * (pixelSize.x + pixelSize.y);
    hitData.dRadius = 0.0;
#endif
}

void StorePhoton(RayDesc ray, PrimaryRayData hitData, uint2 pixelCoord)
{
    bool isInFrustum;
    float3 dPdx, dPdy;
    float3 posW = ray.Origin;
    getPhotonDifferential(hitData, ray, dPdx, dPdy);
    float area = getArea(dPdx, dPdy);
    float3 color = hitData.color.rgb / area;
    float pixelArea = getPhotonScreenArea(posW, dPdx, dPdy, isInFrustum);
    if (dot(color, float3(0.299, 0.587, 0.114)) > cullColorThreshold&& isInFrustum && updatePhoton)
    {
        uint pixelLoc = pixelCoord.y * coarseDim.x + pixelCoord.x;

        if (pixelArea < gMaxScreenRadius* gMaxScreenRadius)
        {
            uint instanceIdx = 0;
            InterlockedAdd(gDrawArgument[0].instanceCount, 1, instanceIdx);

            Photon photon;
            photon.posW = posW;
            //photon.normalW = hitData.nextDir;
            if (dot(cross(dPdx,dPdy), hitData.nextDir) < 0)
            {
                dPdy *= -1;
            }
            photon.color = color;
            photon.dPdx = dPdx;
            photon.dPdy = dPdy;
            gPhotonBuffer[instanceIdx] = photon;
            if (!launchRayTask)
            {
                gPixelInfo[pixelLoc].photonIdx = instanceIdx;
            }
        }

        uint oldV;
        pixelArea = clamp(pixelArea, 1, 100 * 100);
        InterlockedAdd(gPixelInfo[pixelLoc].screenArea, uint(pixelArea.x), oldV);
        InterlockedAdd(gPixelInfo[pixelLoc].screenAreaSq, uint(pixelArea.x * pixelArea.x), oldV);
        InterlockedAdd(gPixelInfo[pixelLoc].count, 1, oldV);
    }
}

bool getTask(out float2 lightUV, out uint2 pixelCoord, out float2 pixelSize)
{
    uint3 launchIndex = DispatchRaysIndex();
    uint3 launchDimension = DispatchRaysDimensions();
    uint taskIdx = launchIndex.y * launchDimension.x + launchIndex.x;
    if (launchRayTask)
    {
        if (taskIdx >= gRayArgument[0].rayTaskCount)
        {
            return false;
        }
        RayTask task = gRayTask[taskIdx];
        pixelCoord = task.screenCoord;
        lightUV = (task.screenCoord + randomOffset * task.pixelSize) / float2(coarseDim);
        pixelSize = task.pixelSize;
    }
    else
    {
        if (any(launchIndex.xy >= coarseDim))
        {
            return false;
        }
        pixelSize = float2(1, 1);
        pixelCoord = launchIndex.xy;
        lightUV = float2(launchIndex.xy) / float2(coarseDim.xy);
        uint nw, nh, nl;
        gUniformNoise.GetDimensions(0, nw, nh, nl);
        //float2 noise = gUniformNoise.Load(uint3(launchIndex.xy % uint2(nw, nh), 0)).rg;
        lightUV += randomOffset / float2(coarseDim.xy);
    }

    if (updatePhoton)
    {
        gPixelInfo[taskIdx].photonIdx = -1;
    }
    if (!launchRayTask)
    {
        gRayTask[taskIdx].screenCoord = launchIndex.xy;
        gRayTask[taskIdx].pixelSize = 1;
    }
    return true;
}

[shader("raygeneration")]
void rayGen()
{
    // fetch task
    float2 lightUV;
    uint2 pixelCoord;
    float2 pixelSize;
    if (!getTask(lightUV, pixelCoord, pixelSize))
        return;

    // Init ray and hit data
    RayDesc ray;
    PrimaryRayData hitData;
    initFromLight(lightUV, pixelSize, ray, hitData);

    // Photon trace
    int depth;
    for (depth = 0; depth < maxDepth && hitData.isContinue; depth++)
    {
        ray.Direction = hitData.nextDir;
        TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, hitData);
        ray.Origin = ray.Origin + ray.Direction * hitData.hitT;
    }

    // write result
    if (any(hitData.color > 0) && depth > 1)
    {
        StorePhoton(ray, hitData, pixelCoord);
    }
}
