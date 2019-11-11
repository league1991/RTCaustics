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

struct Photon
{
    float3 posW;
    float3 normalW;
    float3 color;
};
AppendStructuredBuffer<Photon> gPhotonBuffer;

struct PhotonVSOut
{
    //INTERPOLATION_MODE float3 normalW    : NORMAL;
    //INTERPOLATION_MODE float3 bitangentW : BITANGENT;
    //INTERPOLATION_MODE float2 texC       : TEXCRD;
    //INTERPOLATION_MODE float3 posW       : POSW;
    //INTERPOLATION_MODE float3 colorV     : COLOR;
    //INTERPOLATION_MODE float4 prevPosH   : PREVPOSH;
    //INTERPOLATION_MODE float2 lightmapC  : LIGHTMAPUV;
    float4 posH : SV_POSITION;
};

PhotonVSOut photonScatterVS(VertexIn vIn)
{
    PhotonVSOut vOut;
    float4x4 worldMat = getWorldMat(vIn);
    float4 posW = mul(vIn.pos, worldMat);
    //vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gCamera.viewProjMat);

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

    //return color;
    return float4(1,0,0,1);
}
