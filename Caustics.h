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

#define MAX_CAUSTICS_MAP_SIZE 2048
#define MAX_PHOTON_COUNT 2048*2048

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
    enum TraceType
    {
        TRACE_FIXED = 0,
        TRACE_ADAPTIVE = 1,
        TRACE_NONE = 2,
        TRACE_ADAPTIVE_RAY_MIP_MAP = 3,
    };
    TraceType mTraceType = TRACE_FIXED;
    int mDispatchSize = 512;
    int mMaxTraceDepth = 10;
    float mEmitSize = 30.0;
    float mIntensity = 2.0f;
    float mRoughThreshold = 0.1f;
    enum AreaType
    {
        AREA_AVG_SQUARE = 0,
        AREA_AVG_LENGTH = 1,
        AREA_MAX_SQUARE = 2,
        AREA_EXACT = 3
    } mAreaType = AREA_AVG_SQUARE;
    float mIOROveride = 1.5f;
    int mColorPhoton = 0;
    int mPhotonIDScale = 50;
    float mTraceColorThreshold = 0.0005f;
    float mCullColorThreshold = 1.0f;
    bool mUpdatePhoton = true;
    float mMaxPhotonPixelRadius = 900.0f;
    float mSmallPhotonCompressScale = 1.0f;
    float mFastPhotonPixelRadius = 19.0f;
    float mFastPhotonDrawCount = 0.f;
    bool mFastPhotonPath = false;
    bool mShrinkPayload = true;

    // Adaptive photon refine
    float mNormalThreshold = 0.2f;
    float mDistanceThreshold = 10.0f;
    float mPlanarThreshold = 2.0f;
    float mMinPhotonPixelSize = 8.0f;
    float mSmoothWeight = 0.15f;
    float mMaxTaskCountPerPixel = 8192;
    float mUpdateSpeed = 0.2f;
    float mVarianceGain = 0.0f;
    float mDerivativeGain = 0.0f;
    enum SamplePlacement
    {
        SAMPLE_PLACEMENT_RANDOM = 0,
        SAMPLE_PLACEMENT_GRID = 1
    } mSamplePlacement = SAMPLE_PLACEMENT_GRID;

    // smooth photon
    bool mMedianFilter = false;
    int mMinNeighbourCount = 2;
    bool mRemoveIsolatedPhoton = false;
    float mPixelLuminanceThreshold = 0.5f;
    float trimDirectionThreshold = 0.5f;

    // Photon Scatter
    enum ScatterGeometry
    {
        SCATTER_GEOMETRY_QUAD = 0,
        SCATTER_GEOMETRY_SPHERE = 1,
    };
    ScatterGeometry mScatterGeometry= SCATTER_GEOMETRY_QUAD;
    int   mCausticsMapResRatio = 1;
    enum DensityEstimation
    {
        DENSITY_ESTIMATION_SCATTER = 0,
        DENSITY_ESTIMATION_GATHER = 1,
        DENSITY_ESTIMATION_NONE = 2
    } mScatterOrGather = DENSITY_ESTIMATION_SCATTER;
    float mSplatSize = 0.4f;
    float mKernelPower = 0.01f;
    enum PhotonDisplayMode
    {
        PHOTON_DISPLAY_MODE_KERNEL = 0,
        PHOTON_DISPLAY_MODE_SOLID = 1,
        PHOTON_DISPLAY_MODE_SHADED = 2,
    } mPhotonDisplayMode = PHOTON_DISPLAY_MODE_KERNEL;
    enum PhotonMode
    {
        PHOTON_MODE_ANISOTROPIC = 0,
        PHOTON_MODE_ISOTROPIC = 1,
        PHOTON_MODE_PHOTON_MESH = 2,
        PHOTON_MODE_SCREEN_DOT = 3,
        PHOTON_MODE_SCREEN_DOT_WITH_COLOR = 4,
    } mPhotonMode = PHOTON_MODE_ANISOTROPIC;
    float mScatterNormalThreshold = 0.2f;
    float mScatterDistanceThreshold = 10.0f;
    float mScatterPlanarThreshold = 2.0f;
    float mMaxAnisotropy = 20.0f;
    float mZTolerance = 0.2f;

    // Photon Gather
    int mTileCountScale = 10;
    uint2 mTileSize = uint2(32,32);
    bool mShowTileCount = false;
    float mDepthRadius = 0.1f;
    float mMinGatherColor = 0.001f;

    // Temporal Filter
    bool mTemporalFilter = false;
    float mFilterWeight = 0.8f;
    float mJitter = 0.6f;
    float mTemporalNormalKernel = 0.7f;
    float mTemporalDepthKernel = 3.0f;

    // Spacial Filter
    bool mSpacialFilter = false;
    int mSpacialPasses = 1;
    float mSpacialNormalKernel = 0.7f;
    float mSpacialDepthKernel = 3.0f;
    float mSpacialColorKernel = 0.5f;
    float mSpacialScreenKernel = 1.0f;

    // Composite
    bool mRayTrace = true;
    enum Display
    {
        ShowRasterization = 0,
        ShowDepth = 1,
        ShowNormal = 2,
        ShowDiffuse = 3,
        ShowSpecular = 4,
        ShowPhoton = 5,
        ShowWorld = 6,
        ShowRoughness = 7,
        ShowRayTex = 8,
        ShowRayTracing = 9,
        ShowAvgScreenArea = 10,
        ShowAvgScreenAreaVariance = 11,
        ShowCount = 12,
        ShowTotalPhoton = 13,
        ShowRayCountMipmap = 14,
        ShowPhotonDensity = 15,
        ShowSmallPhoton = 16,
        ShowSmallPhotonCount = 17,
    };
    Display mDebugMode = ShowRayTracing;
    float mMaxPixelArea = 100;
    float mMaxPhotonCount = 1000000;
    int mRayCountMipIdx = 5;
    int mRayTexScaleFactor = 4;
    float mUVKernel = 0.7f;
    float mZKernel = 4.5f;
    float mNormalKernel = 4.0f;
    bool mFilterCausticsMap = false;

    // Others
    int mFrameCounter = 0;
    bool mUseDOF = false;
    uint32_t mSampleIndex = 0xdeadbeef;
    float2 mLightAngle{3.01f,2.f};
    float3 mLightDirection;
    float2 mLightAngleSpeed{0,0};

    Model::SharedPtr mpQuad;
    Model::SharedPtr mpSphere;
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
    struct GBuffer
    {
        Texture::SharedPtr mpNormalTex;
        Texture::SharedPtr mpDiffuseTex;
        Texture::SharedPtr mpSpecularTex;
        Texture::SharedPtr mpDepthTex;
        Fbo::SharedPtr mpGPassFbo;
    };
    GBuffer mGBuffer[2];
    Texture::SharedPtr mpSmallPhotonTex;

    // photon trace
    struct PhotonTraceShader
    {
        RtProgram::SharedPtr mpPhotonTraceProgram;
        RtProgramVars::SharedPtr mpPhotonTraceVars;
        RtState::SharedPtr mpPhotonTraceState;
    };
    enum PhotonTraceMacro
    {
        RAY_DIFFERENTIAL = 0,
        RAY_CONE = 1,
        RAY_NONE = 2
    };
    PhotonTraceMacro mPhotonTraceMacro = RAY_DIFFERENTIAL;
    std::unordered_map<uint32_t, PhotonTraceShader> mPhotonTraceShaderList;
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

    // generate ray count
    ComputeProgram::SharedPtr mpGenerateRayCountProgram;
    ComputeVars::SharedPtr mpGenerateRayCountVars;
    ComputeState::SharedPtr mpGenerateRayCountState;
    StructuredBuffer::SharedPtr mpRayCountQuadTree;

    // generate ray count mipmap
    ComputeProgram::SharedPtr mpGenerateRayCountMipProgram;
    ComputeVars::SharedPtr mpGenerateRayCountMipVars;
    ComputeState::SharedPtr mpGenerateRayCountMipState;

    // smooth photon
    ComputeProgram::SharedPtr mpSmoothProgram;
    ComputeVars::SharedPtr mpSmoothVars;
    ComputeState::SharedPtr mpSmoothState;

    // photon scatter
    GraphicsProgram::SharedPtr mpPhotonScatterProgram;
    GraphicsVars::SharedPtr mpPhotonScatterVars;
    GraphicsState::SharedPtr mpPhotonScatterBlendState;
    GraphicsState::SharedPtr mpPhotonScatterNoBlendState;
    Fbo::SharedPtr mpCausticsFbo[2];
    Texture::SharedPtr mpGaussianKernel;
    Sampler::SharedPtr mpLinearSampler;

    // photon gather
#define GATHER_PROCESSING_SHADER_COUNT 4
    ComputeProgram::SharedPtr   mpAllocateTileProgram[GATHER_PROCESSING_SHADER_COUNT];
    ComputeVars::SharedPtr      mpAllocateTileVars[GATHER_PROCESSING_SHADER_COUNT];
    ComputeState::SharedPtr     mpAllocateTileState[GATHER_PROCESSING_SHADER_COUNT];
    ComputeProgram::SharedPtr   mpPhotonGatherProgram;
    ComputeVars::SharedPtr      mpPhotonGatherVars;
    ComputeState::SharedPtr     mpPhotonGatherState;
    StructuredBuffer::SharedPtr mpTileIDInfoBuffer;
    Buffer::SharedPtr           mpIDBuffer;
    Buffer::SharedPtr           mpIDCounterBuffer;

    // temporal filter
    ComputeProgram::SharedPtr mpFilterProgram;
    ComputeVars::SharedPtr mpFilterVars;
    ComputeState::SharedPtr mpFilterState;

    // spacial filter
    ComputeProgram::SharedPtr mpSpacialFilterProgram;
    ComputeVars::SharedPtr mpSpacialFilterVars;
    ComputeState::SharedPtr mpSpacialFilterState;

    // raytrace
    RtProgram::SharedPtr mpRaytraceProgram;
    RtProgramVars::SharedPtr mpRtVars;
    RtState::SharedPtr mpRtState;
    RtSceneRenderer::SharedPtr mpRtRenderer;
    Texture::SharedPtr mpRtOut;

    // composite pass
    Sampler::SharedPtr mpPointSampler;
    FullScreenPass::SharedPtr mpCompositePass;
    Texture::SharedPtr mpPhotonCountTex;

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
    void setPhotonTracingCommonVariable(PhotonTraceShader& shader);
    PhotonTraceShader getPhotonTraceShader();
    void loadSceneSetting(std::string path);
    void saveSceneSetting(std::string path);
    void createCausticsMap();
    void createGBuffer(int width, int height, GBuffer& gbuffer);
    int2 getTileDim() const;
    float resolutionFactor();
    uint photonMacroToFlags()
    {
        uint flags = 0;
        flags |= (1 << mPhotonTraceMacro); // 3 bits
        flags |= ((1 << mTraceType) << 3); // 4 bits
        if (mFastPhotonPath)
            flags |= (1 << 7); // 1 bits
        if (mShrinkPayload)
        {
            flags |= (1 << 8); // 1 bits
        }
        return flags;
    }
};
