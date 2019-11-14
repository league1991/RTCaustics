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

SamplerState gPointSampler : register(s1);

Texture2D gDepthTex;
Texture2D gNormalTex;
Texture2D gDiffuseTex;
Texture2D gSpecularTex;
Texture2D gPhotonTex;

// Debug modes
#define ShowDepth       1
#define ShowNormal      2
#define ShowDiffuse     3
#define ShowSpecular    4
#define ShowPhoton      5
#define ShowWorld       6
#define ShowRoughness   7

cbuffer PerImageCB
{
    // Lighting params
    LightData gLightData[16];
    float4x4 gInvWvpMat;
    float3 gCameraPos;
    uint gNumLights;
    uint gDebugMode;
};

float4 main(float2 texC  : TEXCOORD) : SV_TARGET
{
    float depth = gDepthTex.Sample(gPointSampler, texC).r;
    float4 screenPnt = float4(texC * float2(2,-2) + float2(-1,1), depth, 1);
    float4 worldPnt = mul(screenPnt, gInvWvpMat);
    worldPnt /= worldPnt.w;
    float4 normalVal = gNormalTex.Sample(gPointSampler, texC);
    float4 diffuseVal = gDiffuseTex.Sample(gPointSampler, texC);
    float4 specularVal = gSpecularTex.Sample(gPointSampler, texC);

    float4 color = 0;
    if (gDebugMode == ShowDepth)
        color = depth;
    else if (gDebugMode == ShowNormal)
        color = normalVal;
    else if (gDebugMode == ShowDiffuse)
        color = gDiffuseTex.Sample(gPointSampler, texC);
    else if (gDebugMode == ShowSpecular)
        color = gSpecularTex.Sample(gPointSampler, texC);
    else if (gDebugMode == ShowPhoton)
        color = gPhotonTex.Sample(gPointSampler, texC);
    else if (gDebugMode == ShowWorld)
        color = frac(worldPnt * 0.01 + 0.01);
    else if (gDebugMode == ShowRoughness)
        color = gDiffuseTex.Sample(gPointSampler, texC).a;
    else
    {
        ShadingData sd = initShadingData();
        sd.posW = worldPnt.xyz;
        sd.V = normalize(gCameraPos - sd.posW);
        sd.N = normalVal.xyz;
        sd.NdotV = abs(dot(sd.V, sd.N));
        sd.linearRoughness = diffuseVal.a;
        sd.specular = specularVal.xyz;
        sd.diffuse = diffuseVal.rgb;
        sd.opacity = 0;

        for (uint l = 0; l < gNumLights; l++)
        {
            ShadingResult sr = evalMaterial(sd, gLightData[l], 1);
            color.rgb += sr.color.rgb;
        }
    }

    return color;
}
