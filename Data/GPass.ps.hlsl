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
//__import ShaderCommon;
//__import Shading;
//__import DefaultVS;
import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.MathHelpers;
import Rendering.Lights.LightHelpers;
//import GBufferHelpers;
import Rendering.Materials.TexLODHelpers;

struct GPassPsOut
{
    float4 normal: SV_TARGET0;
    float4 diffuse: SV_TARGET1;
    float4 specular: SV_TARGET2;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

GPassPsOut gpassPS(VSOut vOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    let lod = ImplicitLodTextureSampler();
    float3 viewDir = normalize(gScene.camera.getPosition() - vOut.posW);
    ShadingData sd = prepareShadingData(vOut, triangleIndex, viewDir, lod);
    //ShadingData sd = prepareShadingData(vOut, gMaterial, gCamera.posW);

    // Create BSDF instance and query its properties.
    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);

    if (bsdfProperties.isTransmissive < 1)
    {
        discard;
    }
    GPassPsOut output;
    output.normal = float4(normalize(vOut.normalW.xyz), 1);
    output.diffuse = float4(bsdfProperties.diffuseReflectionAlbedo, bsdfProperties.roughness);
    output.specular = float4(bsdfProperties.specularReflectionAlbedo, /*sd.opacity*/0.5);
    return output;
}
