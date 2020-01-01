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
#include "common.hlsl"
struct DrawArguments
{
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int  baseVertexLocation;
    uint startInstanceLocation;
};

shared cbuffer PerFrameCB
{
    uint2 coarseDim;
    uint initRayCount;
};

//StructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
//RWStructuredBuffer<PixelInfo> gPixelInfo;

[numthreads(1, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    //uint length, stride;
    //gPhotonBuffer.GetDimensions(length, stride);
    if (all(threadIdx == uint3(0,0,0)))
    {
        gDrawArgument[0].indexCountPerInstance = 6;
        gDrawArgument[0].instanceCount = 0;
        gDrawArgument[0].startIndexLocation = 0;
        gDrawArgument[0].baseVertexLocation = 0;
        gDrawArgument[0].startInstanceLocation = 0;

        //gRayArgument[0].rayTaskCount = initRayCount;
    }

    //uint value = gRayDensityTex[threadIdx.xy];
    //float avgArea = float(value >> 16) / float(value & 0xff);
    //uint lastCount = 1;
    //int idx = coarseDim.x * threadIdx.y + threadIdx.x;
    //gPixelInfo[idx].screenArea = 0;// (uint(avgArea * lastCount) << 16) | lastCount;
    //gPixelInfo[idx].count = 0;
}
