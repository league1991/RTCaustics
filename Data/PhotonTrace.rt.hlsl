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
RWTexture2D<float4> gOutput;
__import Raytracing;
__import ShaderCommon;
__import Shading;
import Helpers;

RWStructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
RWStructuredBuffer<RayTask> gRayTask;
Texture2D gUniformNoise;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    uint2 coarseDim;
    float emitSize;
    float roughThreshold;
    float jitter;
    int launchRayTask;
    int rayTaskOffset;
    //float tanHalfFovY;
    //uint sampleIndex;
    //bool useDOF;
};

struct PrimaryRayData
{
    float4 color;
    float3 dPdx, dPdy, dDdx, dDdy;  // ray differentials
    uint depth;
    float hitT;
};

struct ShadowRayData
{
    bool hit;
};

//[shader("miss")]
//void shadowMiss(inout ShadowRayData hitData)
//{
//    hitData.hit = false;
//}

//[shader("anyhit")]
//void shadowAnyHit(inout ShadowRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
//{
//    hitData.hit = true;
//}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.color = float4(1, 0, 0, 1);
    hitData.hitT = -1;
}

//bool checkLightHit(uint lightIndex, float3 origin)
//{
//    float3 direction = gLights[lightIndex].posW - origin;
//    RayDesc ray;
//    float epsilon = 0.01;
//    ray.Origin = origin;
//    ray.Direction = normalize(direction);
//    ray.TMin = epsilon;
//    ray.TMax = max(0.01, length(direction)- epsilon);
//
//    ShadowRayData rayData;
//    rayData.hit = true;
//    TraceRay(gRtScene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, hitProgramCount, 1, ray, rayData);
//    return rayData.hit;
//}

//float3 getReflectionColor(float3 worldOrigin, VertexOut v, float3 worldRayDir, uint hitDepth)
//{
//    float3 reflectColor = float3(0, 0, 0);
//    if (hitDepth == 0)
//    {
//        PrimaryRayData secondaryRay;
//        secondaryRay.depth = 1;
//        RayDesc ray;
//        ray.Origin = worldOrigin;
//        ray.Direction = reflect(worldRayDir, v.normalW);
//        ray.TMin = 0.001;
//        ray.TMax = 100000;
//        TraceRay(gRtScene, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, hitProgramCount, 0, ray, secondaryRay);
//        reflectColor = secondaryRay.hitT == -1 ? 0 : secondaryRay.color.rgb;
//        float falloff = max(1, (secondaryRay.hitT * secondaryRay.hitT));
//        reflectColor *= 20 / falloff;
//    }
//    return reflectColor;
//}

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
    inout float3 dPdx,
    inout float3 dDdx)
{
    float3 dNdx=0;
    calculateDNdx(P0, P1, P2, N0, N1, N2, N, dPdx, dNdx);

    N = normalize(N);
    float3 D = WorldRayDirection();
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    dDdx = dDdx - 2 * (dot(D, N) * dNdx + dDNdx * N);
}

//void updateRayPartialDifferential(
//    float3 P0, float3 P1, float3 P2,
//    float3 N0, float3 N1, float3 N2, float3 N,
//    inout float3 dPdx, inout float3 dDdx)
//{
//    updateTransferRayDifferential(dPdx, dDdx, N);
//    updateReflectRayDifferential(P0, P1, P2, N0, N1, N2, N, dPdx, dDdx);
//}
//
//void updateRayDifferential(inout PrimaryRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
//{
//    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
//    float3 P0, P1, P2, N0, N1, N2;
//    getVerticesAndNormals(PrimitiveIndex(), P0, P1, P2, N0, N1, N2);
//    float3 N = N0 * barycentrics.x + N1 * barycentrics.y + N2 * barycentrics.z;
//
//    updateRayPartialDifferential(P0, P1, P2, N0, N1, N2, N, hitData.dPdx, hitData.dDdx);
//    updateRayPartialDifferential(P0, P1, P2, N0, N1, N2, N, hitData.dPdy, hitData.dDdy);
//}

float getPhotonScreenArea(Photon p, out bool isInfrustum)
{
    float4 s0 = mul(float4(p.posW, 1), gCamera.viewProjMat);
    float4 sx = mul(float4(p.posW + p.dPdx, 1), gCamera.viewProjMat);
    float4 sy = mul(float4(p.posW + p.dPdy, 1), gCamera.viewProjMat);
    s0 /= s0.w;
    sx /= sx.w;
    sy /= sy.w;
    isInfrustum = all(abs(s0.xy) < 1) && s0.z > 0 && s0.z < 1;
    float2 dx = (sx.xy - s0.xy) * viewportDims;
    float2 dy = (sy.xy - s0.xy) * viewportDims;
    float area = abs(dx.x * dy.y - dy.x * dx.y);
    //float area = 0.5 * (dx.x * dx.y + dy.x * dy.y);
    return area;
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

    PrimaryRayData hitData2;
    hitData2.depth = hitData.depth + 1;
    hitData2.dPdx = hitData.dPdx;
    hitData2.dDdx = hitData.dDdx;
    hitData2.dPdy = hitData.dPdy;
    hitData2.dDdy = hitData.dDdy;
    float3 P0, P1, P2, N0, N1, N2, N;
    getVerticesAndNormals(PrimitiveIndex(), attribs, P0, P1, P2, N0, N1, N2, N);
    updateTransferRayDifferential(N, hitData2.dPdx, hitData2.dDdx);
    updateTransferRayDifferential(N, hitData2.dPdy, hitData2.dDdy);

    // Shoot a reflection ray
    float3 reflectColor = float3(0, 0, 0);
    //reflectColor = getReflectionColor(posW, v, rayDirW, hitData.depth);
    if (sd.linearRoughness > roughThreshold)
    {
        if (hitData.depth < 1)
        {
            hitData2.color = hitData.color* float4(sd.specular, 1);
            updateReflectRayDifferential(P0, P1, P2, N0, N1, N2, N, hitData2.dPdx, hitData2.dDdx);
            updateReflectRayDifferential(P0, P1, P2, N0, N1, N2, N, hitData2.dPdy, hitData2.dDdy);

            RayDesc ray;
            ray.Origin = posW;
            ray.Direction = reflect(rayDirW, v.normalW);
            ray.TMin = 0.01;
            ray.TMax = 100000;
            TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, hitData2);
            //reflectColor = hitData2.hitT == -1 ? 0 : hitData2.color.rgb;
            //float falloff = max(1, (hitData2.hitT * hitData2.hitT));
            //reflectColor *= 20 / falloff;
        }
    }
    else if(hitData.depth > 0)
    {
        //LightData ld;
        //ld.type = 1;
        //ld.dirW = rayDirW;
        //ld.intensity = hitData.color.rgb;
        //ShadingResult sr = evalMaterial(sd, ld, 1);

        Photon photon;
        photon.posW = posW;
        photon.normalW = sd.N;
        photon.color = dot(-rayDirW, sd.N)* sd.diffuse * hitData.color.rgb;//sr.color.rgb;
        photon.dPdx = hitData2.dPdx;
        photon.dPdy = hitData2.dPdy;

        uint instanceIdx = 0;
        InterlockedAdd(gDrawArgument[0].instanceCount, 1, instanceIdx);
        gPhotonBuffer[instanceIdx] = photon;
        uint3 dim3 = DispatchRaysDimensions();
        uint3 idx3 = DispatchRaysIndex();
        uint idx = idx3.y * dim3.x + idx3.x;
        bool isInFrustum;
        gRayTask[idx].photonIdx = instanceIdx;
        gRayTask[idx].pixelArea = getPhotonScreenArea(photon, isInFrustum);
        gRayTask[idx].inFrustum = isInFrustum ? 1 : 0;
    }

    if (hitData.depth == 0)
    {
        float4 cameraPnt = mul(float4(posW, 1), gCamera.viewProjMat);
        cameraPnt.xyz /= cameraPnt.w;
        float2 screenPosF = saturate((cameraPnt.xy * float2(1, -1) + 1.0) * 0.5);
        uint2 screenDim;
        gOutput.GetDimensions(screenDim.x, screenDim.y);

        int2 screenPosI = screenPosF * screenDim;
        screenPosI = DispatchRaysIndex().xy;
        gOutput[screenPosI.xy] = float4(abs(sd.N), 1);
    }
    //float3 color = 0;

    //[unroll]
    //for (int i = 0; i < gLightsCount; i++)
    //{
    //    if (checkLightHit(i, posW) == false)
    //    {
    //        color += evalMaterial(sd, gLights[i], 1).color.xyz;
    //    }
    //}

    //hitData.color = float4(sd.N, 1.0);
    hitData.hitT = hitT;
    //hitData.color.rgb = color;
    // A very non-PBR inaccurate way to do reflections
    //float roughness = min(0.5, max(1e-8, sd.roughness));
    //hitData.color.rgb += sd.specular * reflectColor * (roughness * roughness);
    //hitData.color.rgb += sd.emissive;
    //hitData.color.a = 1;
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
    if (launchRayTask)
    {
        taskIdx += rayTaskOffset;
        if (taskIdx >= gRayArgument[0].rayTaskCount)
        {
            return;
        }
        RayTask task = gRayTask[taskIdx];
        lightUV = (task.screenCoord) / float2(coarseDim);// float2(launchDimension.xy);
        pixelSize = task.pixelSize;
        //lightUV = float2(launchIndex.xy+0.5) / float2(launchDimension.xy);
        //pixelSize = 0.1;
    }
    else
    {
        lightUV = float2(launchIndex.xy) / float2(coarseDim.xy);
    }
    lightUV = lightUV * 2 - 1;
    //float dispatchFactor = (coarseDim.x / 512.0) * (coarseDim.y / 512.0);
    pixelSize *= emitSize / float2(coarseDim.xy);

    uint nw, nh, nl;
    gUniformNoise.GetDimensions(0, nw, nh, nl);
    float2 noise = gUniformNoise.Load(uint3(launchIndex.xy % uint2(nw, nh), 0)).rg;
    lightUV += noise * pixelSize * jitter;

    ray.Origin = lightOrigin + (lightDirX * lightUV.x + lightDirY * lightUV.y) * emitSize;
    ray.Direction = lightDirZ;// lightDirZ;
    ray.TMin = 0.0;
    ray.TMax = 1e10;

    PrimaryRayData hitData;
    hitData.depth = 0;
    hitData.color = float4(1, 1, 1, 1) *pixelSize.x* pixelSize.y * 512 * 512;
    //hitData.dPdx = 0;
    //hitData.dPdy = 0;
    hitData.dDdx = 0;// lightDirX* pixelSize.x * 2.0;
    hitData.dDdy = 0;// lightDirY* pixelSize.y * 2.0;
    hitData.dPdx = lightDirX * pixelSize.x * 2.0;
    hitData.dPdy = -lightDirY * pixelSize.y * 2.0;

    gRayTask[taskIdx].photonIdx = -1;
    if (!launchRayTask)
    {
        gRayTask[taskIdx].screenCoord = launchIndex.xy;
        gRayTask[taskIdx].pixelSize = float2(1, 1);
        gRayTask[taskIdx].pixelArea = 0;
        gRayTask[taskIdx].inFrustum = 0;
    }

    TraceRay( gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, hitData );
    //gOutput[launchIndex.xy] = hitData.color;
    //gOutput[int2(0,0)] = hitData.color;

    //Photon photon;
    //photon.posW = float3(0, 0, 0);
    //photon.normalW = float3(0, 0, 0);
    //photon.color = float3(0, 0, 0);
    //gPhotonBuffer[0] = photon;
}
