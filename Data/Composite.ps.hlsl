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

SamplerState gColorSampler : register(s1);
Texture2D gColorTex;

//cbuffer PerFrameCB : register(b0)
//{
//    float4x4 gWvpMat;
//    float4x4 gWorldMat;
//    float3 gEyePosW;
//    float gLightIntensity;
//    float gSurfaceRoughness;
//};

//struct GPassVsOut
//{
//    float4 posH : SV_POSITION;
//    float4 normal: NORMAL;
//};
//
//struct GPassPsOut
//{
//    float4 normal: SV_TARGET;
//};

//GPassVsOut gpassVS(VertexIn input)
//{
//    GPassVsOut output;
//    output.posH = mul(input.pos, gWvpMat);
//    output.normal = input.normal;
//    return output;
//}
//
//GPassPsOut gpassPS(GPassVsOut input)
//{
//    GPassPsOut output;
//    output.normal = input.normal;
//    return output;
//}

float4 main(float2 texC  : TEXCOORD) : SV_TARGET
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
    //return color;
    float4 color = gColorTex.Sample(gColorSampler, texC);
    return color;
}
