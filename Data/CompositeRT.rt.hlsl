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
Texture2D<float4> gCausticsTex;
Texture2D gDepthTex;
Texture2D gNormalTex;
SamplerState gLinearSampler;
SamplerState gPointSampler;
import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float4x4 invProj;
    float2 viewportDims;
    float tanHalfFovY;
    uint sampleIndex;
    bool useDOF;
    float roughThreshold;
    int maxDepth;
    float iorOverride;
    int causticsResRatio;
    float gPosKernel;
    float gZKernel;
    float gNormalKernel;
};

struct HitData
{
    float3 color;
    float hitT;
    float3 nextDir;
    float3 throughput;
};

struct ShadowRayData
{
    bool hit;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

[shader("miss")]
void primaryMiss(inout HitData hitData)
{
    hitData.color = 0;// float4(0.38f, 0.52f, 0.10f, 1);
    hitData.hitT = -1;
    hitData.throughput = 0;
}

bool checkLightHit(uint lightIndex, float3 origin)
{
    const LightData light = gScene.getLight(lightIndex);
    float3 direction = light.posW - origin;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = normalize(direction);
    ray.TMin = 0.001;
    ray.TMax = max(0.01, length(direction));

    ShadowRayData rayData;
    rayData.hit = true;
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, ray, rayData);
    return rayData.hit;
}

void getSample(SamplerState samplerState, float2 uv, out float3 causticsClr, out float3 normal, out float depth)
{
    causticsClr = gCausticsTex.SampleLevel(samplerState, uv, 0).rgb;
    normal = gNormalTex.SampleLevel(samplerState, uv, 0).rgb;
    depth = gDepthTex.SampleLevel(samplerState, uv, 0).r;

    depth = toViewSpace(invProj, depth);
    //float4 screenPnt = float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
    //float4 worldPnt = mul(screenPnt, invProj);
    //pos = worldPnt.xyz / worldPnt.w;
}

float getWeight(float2 offset, float3 normal0, float3 normal1, float z0, float z1)
{
    float normalDiff = gNormalKernel * (1 - saturate(dot(normal0, normal1)));
    //float3 dPos = pos0 - pos1;
    float zDiff = gZKernel * abs(z0 - z1);// (length(dPos));
    float uvDiff = gPosKernel * length(offset);
    //float x = zDiff * normalDiff * uvDiff * gPosKernel;// *zDiff * 1;
    return exp(-1 * (zDiff * zDiff + normalDiff * normalDiff + uvDiff * uvDiff));
}

[shader("closesthit")]
void primaryClosestHit(inout HitData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    float3 rayOrigW = WorldRayOrigin();
    float3 rayDirW = WorldRayDirection();
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    // prepare the shading data
    float3 posW = rayOrigW + hitT * rayDirW;
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, triangleIndex, attribs);
    //VertexOut v = getVertexAttributes(triangleIndex, attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW, lod);
    //ShadingData sd = prepareShadingData(v, gMaterial, rayOrigW, 0);

    float3 color = 0;
    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);
    if(bsdfProperties.roughness > roughThreshold || /*sd.opacity < 1*/bsdfProperties.isTransmissive)
    {
        bool isReflect = (sd.opacity == 1);
        float3 R;
        float eta = iorOverride > 0 ? 1.0 / iorOverride : 1.0 / sd.IoR;
        float3 N = v.normalW;
        if (!isReflect)
        {
            if (dot(N, rayDirW) > 0)
            {
                eta = 1.0 / eta;
                N *= -1;
            }
            isReflect = isTotalInternalReflection(rayDirW, N, eta);
        }

        if (isReflect)
        {
            R = reflect(rayDirW, N);
        }
        else
        {
            getRefractVector(rayDirW, N, R, eta);
        }

        float3 baseColor = bsdfProperties.diffuseReflectionAlbedo;// lerp(1, /*sd.diffuse*/bsdfProperties.diffuseReflectionAlbedo, sd.opacity);
        color = 0;
        hitData.nextDir = R;
        hitData.throughput = baseColor;
    }
    else
    {
        float4 posS = mul(float4(posW,1), gScene.camera.data.viewProjMat);
        posS /= posS.w;
        if (all(posS.xy < 1))
        {
            posS.y *= -1;
            float2 texCoord = (posS.xy + 1) * 0.5;
            float3 clr0, nor0;
            float z0;
            getSample(gLinearSampler, texCoord, clr0, nor0, z0);

            float totalWeight = 1;
            float3 causticsClr = clr0;
            if (gPosKernel > 0 || gZKernel > 0 || gNormalKernel > 0)
            {
                float2 dir[] = {
                    float2(-1,-1),    float2(-1,0),    float2(-1,1),
                    float2(0,-1),                      float2(0,1),
                    float2(1,-1),     float2(1,0),     float2(1,1),
                };
                [unroll]
                for (int i = 0; i < 8; i++)
                {
                    float2 sampleUV = texCoord + dir[i] / (viewportDims / causticsResRatio);
                    float3 clr, nor;
                    float z;
                    getSample(gLinearSampler, sampleUV, clr, nor, z);
                    float w = getWeight(dir[i],nor0, nor, z0, z);
                    causticsClr += clr * w;
                    totalWeight += w;
                    //color += w;
                }
            }
            causticsClr /= totalWeight;

            color += /*sd.diffuse*/bsdfProperties.diffuseReflectionAlbedo * causticsClr;// gCausticsTex.SampleLevel(gLinearSampler, texCoord, 0).rgb;
        }

        uint3 launchIndex = DispatchRaysIndex();
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);
        [unroll]
        for (int i = 0; i < gScene.getLightCount(); i++)
        {
            AnalyticLightSample ls;
            if (evalLightApproximate(sd.posW, gScene.getLight(i), ls))
            {
                if (checkLightHit(i, posW) == false)
                {
                    color += bsdf.eval(sd, ls.dir, sg) * ls.Li;// .color.xyz;
                }
            }
        }

        hitData.throughput = 0;
    }

    hitData.color.rgb = color;
    hitData.hitT = hitT;
    hitData.color.rgb += bsdfProperties.emission;// sd.emissive;
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    //uint randSeed = rand_init(launchIndex.x + launchIndex.y * viewportDims.x, sampleIndex, 16);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);
    RayDesc ray;
    if (!useDOF)
    {
        //ray = generateRay(gCamera, launchIndex.xy, viewportDims);
        ray = gScene.camera.computeRayPinhole(launchIndex.xy, viewportDims).toRayDesc();
    }
    else
    {
        //ray = generateDOFRay(gCamera, launchIndex.xy, viewportDims, randSeed);
        float2 u = sampleNext2D(sg);
        ray = gScene.camera.computeRayThinlens(launchIndex.xy, viewportDims, u).toRayDesc();
    }

    float3 totalColor = 0;
    float3 totalThroughput = 1;
    for (int i = 0; i < maxDepth && any(totalThroughput > 0); i++)
    {
        HitData hitData;
        TraceRay(gScene.rtAccel, 0, 0xFF, 0, rayTypeCount, 0, ray, hitData);

        totalColor += totalThroughput * hitData.color.rgb;
        totalThroughput *= hitData.throughput;

        ray.Origin = ray.Origin + ray.Direction * hitData.hitT;
        ray.Direction = hitData.nextDir;
        ray.TMin = 0.001;
        ray.TMax = 100000;
    }

    gOutput[launchIndex.xy] = float4(totalColor,1);
}
