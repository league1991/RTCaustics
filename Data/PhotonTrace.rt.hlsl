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
RWTexture2D<float4> gOutput;
__import Raytracing;
__import ShaderCommon;
import Helpers;

struct Photon
{
    float3 posW;
    float3 normalW;
    float3 color;
};
RWStructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<DrawArguments> gDrawArgument;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    float emitSize;
    float roughThreshold;
    //float tanHalfFovY;
    //uint sampleIndex;
    //bool useDOF;
};

struct PrimaryRayData
{
    float4 color;
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

    // Shoot a reflection ray
    float3 reflectColor = float3(0, 0, 0);
    //reflectColor = getReflectionColor(posW, v, rayDirW, hitData.depth);
    if (sd.linearRoughness > roughThreshold && hitData.depth < 1)
    {
        PrimaryRayData secondaryRay;
        secondaryRay.depth = hitData.depth + 1;
        RayDesc ray;
        ray.Origin = posW;
        ray.Direction = reflect(rayDirW, v.normalW);
        ray.TMin = 0.01;
        ray.TMax = 100000;
        TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, secondaryRay);
        //reflectColor = secondaryRay.hitT == -1 ? 0 : secondaryRay.color.rgb;
        //float falloff = max(1, (secondaryRay.hitT * secondaryRay.hitT));
        //reflectColor *= 20 / falloff;
    }
    else if(hitData.depth > 0)
    {
        int2 rayId  = DispatchRaysIndex().xy;
        int2 rayDim = DispatchRaysDimensions().xy;
        Photon photon;
        photon.posW = posW;
        photon.normalW = sd.N;
        photon.color = sd.diffuse;// float3(1, 1, 1);
        //gPhotonBuffer.Append(photon);
        //int idx = rayId.y * rayDim.x + rayId.x;

        uint instanceIdx = 0;
        InterlockedAdd(gDrawArgument[0].instanceCount, 1, instanceIdx);
        gPhotonBuffer[instanceIdx] = photon;

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

    hitData.color = float4(sd.N, 1.0);
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
    //uint randSeed = rand_init(launchIndex.x + launchIndex.y * viewportDims.x, sampleIndex, 16);
    //if (!useDOF)
    //{
    //    ray = generateRay(gCamera, launchIndex.xy, viewportDims);
    //}
    //else
    //{
    //    ray = generateDOFRay(gCamera, launchIndex.xy, viewportDims, randSeed);
    //}
    float3 lightOrigin = gLights[0].dirW * -100;// gLights[0].posW;
    float3 lightDirZ = gLights[0].dirW;
    float3 lightDirX = normalize(float3(-lightDirZ.z, 0, lightDirZ.x));
    float3 lightDirY = normalize(cross(lightDirZ, lightDirX));
    float2 lightUV = float2(launchIndex.xy) / float2(launchDimension.xy);
    lightUV = lightUV * 2 - 1;

    ray.Origin = lightOrigin + (lightDirX * lightUV.x + lightDirY * lightUV.y)* emitSize;
    ray.Direction = lightDirZ;// lightDirZ;
    ray.TMin = 0.0;
    ray.TMax = 1e10;

    PrimaryRayData hitData;
    hitData.depth = 0;
    hitData.color = float4(0, 0, 1, 1);
    TraceRay( gRtScene, 0, 0xFF, 0, hitProgramCount, 0, ray, hitData );
    //gOutput[launchIndex.xy] = hitData.color;
    //gOutput[int2(0,0)] = hitData.color;

    Photon photon;
    photon.posW = float3(0, 0, 0);
    photon.normalW = float3(0, 0, 0);
    photon.color = float3(0, 0, 0);
    //gPhotonBuffer[0] = photon;
}
