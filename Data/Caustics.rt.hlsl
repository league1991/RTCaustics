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
import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;
//import Rendering.Lights.Helpers;
import Rendering.Lights.LightHelpers;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    float tanHalfFovY;
    uint sampleIndex;
    bool useDOF;
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
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.color = float4(0.38f, 0.52f, 0.10f, 1);
    hitData.hitT = -1;
}

bool checkLightHit(uint lightIndex, float3 origin)
{
    const LightData light = gScene.getLight(lightIndex);
    float3 direction = light.posW - origin;
    RayDesc ray;
    float epsilon = 0.01;
    ray.Origin = origin;
    ray.Direction = normalize(direction);
    ray.TMin = epsilon;
    ray.TMax = max(0.01, length(direction)- epsilon);

    ShadowRayData rayData;
    rayData.hit = true;
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, ray, rayData);
    return rayData.hit;
}

//float3 getReflectionColor(float3 worldOrigin, VertexOut v, float3 worldRayDir, uint hitDepth)
//{
//    float3 reflectColor = float3(0, 0, 0);
//    if (hitDepth == 0)
//    {
//        PrimaryRayData secondaryRay;
//        secondaryRay.depth.r = 1;
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
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, triangleIndex, attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW, lod);

    // Shoot a reflection ray
    float3 reflectColor = 0;// getReflectionColor(posW, v, rayDirW, hitData.depth.r);
    float3 color = 0;

    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);
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
                //color += evalMaterial(sd, gLights[i], 1).color.xyz;
                color += bsdf.eval(sd, ls.dir, sg) * ls.Li;
            }
        }
    }

    hitData.color.rgb = color;
    hitData.hitT = hitT;
    // A very non-PBR inaccurate way to do reflections
    let bsdfProp = bsdf.getProperties(sd);
    float oriRoughness = bsdfProp.roughness;
    float roughness = min(0.5f, max(1e-8f, oriRoughness));
    hitData.color.rgb += bsdfProp.specularReflectionAlbedo * reflectColor * (roughness * roughness);
    hitData.color.rgb += bsdfProp.emission;
    //hitData.color.rgb = sd.N;
    hitData.color.a = 1;
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
    PrimaryRayData hitData;
    hitData.depth = 0;
    TraceRay( gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, hitData );
    gOutput[launchIndex.xy] = hitData.color;
}
