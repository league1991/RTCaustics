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
#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;

class Caustics : public IRenderer
{
public:
    Caustics();
    void onLoad(RenderContext* pRenderContext) override;
    void onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(uint32_t width, uint32_t height) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onGuiRender(Gui* pGui) override;

private:
    // Photon trace
    int mTraceType = 0;
    int mDispatchSize=64;
    int mMaxTraceDepth = 10;
    float mEmitSize = 30.0;
    float mIntensity = 0.4f;
    float mRoughThreshold = 0.1f;
    uint32_t mAreaType = 0;
    float mJitter = 0.f;
    float mIOROveride = 1.5f;
    int mColorPhoton = 0;
    int mPhotonIDScale = 50;
    float mTraceColorThreshold = 0.002f;
    float mCullColorThreshold = 0.2f;
    bool mUpdatePixelInfo = true;
    bool mUpdatePhoton = true;

    // Adaptive photon refine
    float mNormalThreshold = 0.2f;
    float mDistanceThreshold = 10.0f;
    float mPlanarThreshold = 2.0f;
    float mMinPhotonPixelSize = 8.0f;
    float mSmoothWeight = 0.04f;
    int mMaxTaskCountPerPixel = 8192;
    float mUpdateSpeed = 0.2f;
    float mVarianceGain = 0.0f;
    float mDerivativeGain = 0.0f;

    // smooth photon
    bool mMedianFilter = false;
    int mMinNeighbourCount = 2;
    bool mRemoveIsolatedPhoton = false;
    float mPixelLuminanceThreshold = 0.5f;
    float trimDirectionThreshold = 0.5f;

    // Photon Scatter
    int   mScatterOrGather = 0;
    float mSplatSize = 4.0f;
    float mKernelPower = 1.f;
    int mPhotonDisplayMode = 0;
    uint32_t mPhotonMode = 0;
    float mScatterNormalThreshold = 0.2f;
    float mScatterDistanceThreshold = 10.0f;
    float mScatterPlanarThreshold = 2.0f;
    float mMaxAnisotropy = 20.0f;
    float mMaxPhotonPixelRadius = 90.0f;

    // Photon Gather
    int mTileCountScale = 10;
    int mTileSize = 16;
    int2 mTileDim = int2(1, 1);
    bool mShowTileCount = false;
    float mDepthRadius = 0.1f;

    // Composite
    bool mRayTrace = true;
    uint32_t mDebugMode = 9;
    float mMaxPixelArea = 100;
    int mRayTexScaleFactor = 4;

    // Others
    bool mUseDOF = false;
    uint32_t mSampleIndex = 0xdeadbeef;
    float2 mLightAngle{0.4f,2.f};
    float3 mLightDirection;
    float2 mLightAngleSpeed{0,0};

    Model::SharedPtr mpQuad;
    RtScene::SharedPtr mpScene;
    Camera::SharedPtr mpCamera;
    FirstPersonCameraController mCamController;

    // forward shading pass
    RasterScenePass::SharedPtr mpRasterPass;

    // Clear draw argument
    ComputeProgram::SharedPtr mpDrawArgumentProgram;
    ComputeVars::SharedPtr mpDrawArgumentVars;
    ComputeState::SharedPtr mpDrawArgumentState;
    StructuredBuffer::SharedPtr  mpDrawArgumentBuffer;

    // g-pass
    RasterScenePass::SharedPtr mpGPass;
    Texture::SharedPtr mpNormalTex;
    Texture::SharedPtr mpDiffuseTex;
    Texture::SharedPtr mpSpecularTex;
    Texture::SharedPtr mpDepthTex;
    Texture::SharedPtr mpPhotonMapTex;
    Fbo::SharedPtr mpGPassFbo;

    // photon trace
    RtProgram::SharedPtr mpPhotonTraceProgram;
    RtProgramVars::SharedPtr mpPhotonTraceVars;
    RtState::SharedPtr mpPhotonTraceState;
    Texture::SharedPtr mpUniformNoise;

    // update ray density result
    ComputeProgram::SharedPtr mpUpdateRayDensityProgram;
    ComputeVars::SharedPtr mpUpdateRayDensityVars;
    ComputeState::SharedPtr mpUpdateRayDensityState;

    // analyse trace result
    ComputeProgram::SharedPtr mpAnalyseProgram;
    ComputeVars::SharedPtr mpAnalyseVars;
    ComputeState::SharedPtr mpAnalyseState;
    StructuredBuffer::SharedPtr  mpRayArgumentBuffer;

    // smooth photon
    ComputeProgram::SharedPtr mpSmoothProgram;
    ComputeVars::SharedPtr mpSmoothVars;
    ComputeState::SharedPtr mpSmoothState;

    // photon scatter
    GraphicsProgram::SharedPtr mpPhotonScatterProgram;
    GraphicsVars::SharedPtr mpPhotonScatterVars;
    GraphicsState::SharedPtr mpPhotonScatterBlendState;
    GraphicsState::SharedPtr mpPhotonScatterNoBlendState;
    Fbo::SharedPtr mpCausticsFbo;
    Texture::SharedPtr mpGaussianKernel;
    Sampler::SharedPtr mpLinearSampler;

    // photon gather
    ComputeProgram::SharedPtr   mpAllocateTileProgram[3];
    ComputeVars::SharedPtr      mpAllocateTileVars[3];
    ComputeState::SharedPtr     mpAllocateTileState[3];
    ComputeProgram::SharedPtr   mpPhotonGatherProgram;
    ComputeVars::SharedPtr      mpPhotonGatherVars;
    ComputeState::SharedPtr     mpPhotonGatherState;
    StructuredBuffer::SharedPtr mpTileIDInfoBuffer;
    Buffer::SharedPtr           mpIDBuffer;
    Buffer::SharedPtr           mpIDCounterBuffer;

    // raytrace
    RtProgram::SharedPtr mpRaytraceProgram;
    RtProgramVars::SharedPtr mpRtVars;
    RtState::SharedPtr mpRtState;
    RtSceneRenderer::SharedPtr mpRtRenderer;
    Texture::SharedPtr mpRtOut;

    // composite pass
    Sampler::SharedPtr mpPointSampler;
    FullScreenPass::SharedPtr mpCompositePass;

    // RT composite pass
    RtProgram::SharedPtr mpCompositeRTProgram;
    RtProgramVars::SharedPtr mpCompositeRTVars;
    RtState::SharedPtr mpCompositeRTState;

    // Caustics map
    StructuredBuffer::SharedPtr  mpPhotonBuffer;
    StructuredBuffer::SharedPtr  mpPhotonBuffer2;
    StructuredBuffer::SharedPtr  mpRayTaskBuffer;
    StructuredBuffer::SharedPtr  mpPixelInfoBuffer;
    Texture::SharedPtr mpRayDensityTex;

    void setPerFrameVars(const Fbo* pTargetFbo);
    void renderRT(RenderContext* pContext, Fbo::SharedPtr pTargetFbo);
    void loadScene(const std::string& filename, const Fbo* pTargetFbo);
    void loadShader();
    void setCommonVars(GraphicsVars* pVars, const Fbo* pTargetFbo);
    void setPhotonTracingCommonVariable();
};
