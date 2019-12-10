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
__import Shading;
__import DefaultVS;

#include "Common.hlsl"

StructuredBuffer<Photon> gPhotonBuffer;

Texture2D gDepthTex;
Texture2D gNormalTex;
Texture2D gDiffuseTex;
Texture2D gSpecularTex;

Texture2D gGaussianTex;

SamplerState gLinearSampler;

cbuffer PerFrameCB : register(b0)
{
    float4x4 gWvpMat;
    float4x4 gWorldMat;
    float3 gEyePosW;
    float gLightIntensity;
    float gSurfaceRoughness;
    float gSplatSize;
    float gIntensity;
    uint  gPhotonMode;
    float gKernelPower;
    uint  gShowPhoton;
    uint  gAreaType;
};
#define AnisotropicPhoton 0
#define IsotropicPhoton 1
#define VisualizePhoton 2

struct PhotonVSOut
{
    //INTERPOLATION_MODE float3 normalW    : NORMAL;
    //INTERPOLATION_MODE float3 bitangentW : BITANGENT;
    //INTERPOLATION_MODE float2 texC       : TEXCRD;
    //INTERPOLATION_MODE float3 posW       : POSW;
    //INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;
    //INTERPOLATION_MODE float2 lightmapC  : LIGHTMAPUV;
    float4 posH : SV_POSITION;
    float2 texcoord: TEXCOORD0;
    float4 color     : COLOR;
};

PhotonVSOut photonScatterVS(VertexIn vIn)
{
    PhotonVSOut vOut;
    //float4x4 worldMat = getWorldMat(vIn);
    //float4 posW = mul(vIn.pos, worldMat);
    ////vOut.posW = posW.xyz;
    //vOut.posH = mul(posW, gCamera.viewProjMat);

    Photon p = gPhotonBuffer[vIn.instanceID];
    float3 normal = normalize(p.normalW);

    float3 tangent, bitangent;
    if (gPhotonMode == AnisotropicPhoton)
    {
        tangent = p.dPdx;
        bitangent = p.dPdy;
        //float tangentLength = length(tangent);
        //float bitangentLength = length(bitangent);
        //float maxLength = 1.0;
        //float minLength = 0.1;
        //tangent *= clamp(tangentLength, minLength, maxLength) / tangentLength;
        //bitangent *= clamp(bitangentLength, minLength, maxLength) / bitangentLength;
    }
    else if (gPhotonMode == IsotropicPhoton)
    {
        tangent = normalize(float3(normal.y, -normal.x, 0));
        bitangent = cross(tangent, normal);
        float radius = max(length(p.dPdx), length(p.dPdy));
        tangent *= radius;
        bitangent *= radius;
    }
    else
    {
        tangent = normalize(float3(normal.y, -normal.x, 0));
        bitangent = cross(tangent, normal);
    }

    tangent *= gSplatSize;
    bitangent *= gSplatSize;

    float3 areaVector = cross(tangent, bitangent);
    float area;
    if (gAreaType == 0)
    {
        area = 0.5 * (dot(tangent, tangent) + dot(bitangent, bitangent));
    }
    else if (gAreaType == 1)
    {
        area = length(tangent) + length(bitangent);
    }
    else
    {
        area = length(areaVector);
    }

    float3 localPoint = tangent * vIn.pos.x + bitangent * vIn.pos.z + normal * vIn.pos.y;
    vOut.texcoord = (vIn.pos.xz + 1) * 0.5;
    vIn.pos.xyz = localPoint + p.posW;
    vOut.posH = mul(vIn.pos, gWvpMat);
    vOut.color = float4(p.color / area * gIntensity, 1);
    return vOut;
}

float4 photonScatterPS(PhotonVSOut vOut) : SV_TARGET
{
    //ShadingData sd = prepareShadingData(vOut, gMaterial, gCamera.posW);
    //float4 color = 0;
    //color.a = 1;

    //[unroll]
    //for (uint i = 0; i < 3; i++)
    //{
    //    color += evalMaterial(sd, gLights[i], 1).color;
    //}
    //color.rgb += sd.emissive;
    float depth = gDepthTex.Load(int3(vOut.posH.xy, 0)).x;
    if (abs(depth- vOut.posH.z) > 0.00001)
    {
        discard;
    }

    float alpha;
    if (gShowPhoton)
    {
        alpha = 1;
    }
    else
    {
        alpha = gGaussianTex.Sample(gLinearSampler, vOut.texcoord).r;
        alpha = pow(alpha, gKernelPower);
    }
    return float4(vOut.color.rgb * alpha, 1);
}
