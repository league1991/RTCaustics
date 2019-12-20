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
#include "Caustics.h"
#include "Scene/Model/Model.h"

static const glm::vec4 kClearColor(0.f, 0.f, 0.f, 1);
static const std::string kDefaultScene = "Caustics/ring.fscene";

std::string to_string(const vec3& v)
{
    std::string s;
    s += "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
    return s;
}

void Caustics::onGuiRender(Gui* pGui)
{
    pGui->addCheckBox("Ray Trace", mRayTrace);

    {
        Gui::DropdownList debugModeList;
        debugModeList.push_back({ 0, "Rasterize" });
        debugModeList.push_back({ 1, "Depth" });
        debugModeList.push_back({ 2, "Normal" });
        debugModeList.push_back({ 3, "Diffuse" });
        debugModeList.push_back({ 4, "Specular" });
        debugModeList.push_back({ 5, "Photon" });
        debugModeList.push_back({ 6, "World" });
        debugModeList.push_back({ 7, "Roughness" });
        debugModeList.push_back({ 8, "Ray" });
        debugModeList.push_back({ 9, "Raytrace" });
        pGui->addDropdown("Debug mode", debugModeList, (uint32_t&)mDebugMode);
    }

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileExtensionFilters, filename))
        {
            loadScene(filename, gpFramework->getTargetFbo().get());
            loadShader();
        }
    }
    if (pGui->addButton("Update Shader"))
    {
        loadShader();
    }

    if (pGui->beginGroup("Photon Trace", true))
    {
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 64, "64" });
            debugModeList.push_back({ 128, "128" });
            debugModeList.push_back({ 256, "256" });
            debugModeList.push_back({ 512, "512" });
            debugModeList.push_back({ 1024, "1024" });
            debugModeList.push_back({ 2048, "2048" });
            pGui->addDropdown("Dispatch Size", debugModeList, (uint32_t&)mDispatchSize);
        }
        pGui->addFloatVar("Emit size", mEmitSize, 0, 1000, 1);
        pGui->addFloatVar("Rough Threshold", mRoughThreshold, 0, 1, 0.01f);
        pGui->addFloatVar("Jitter", mJitter, 0, 1, 0.01f);
        pGui->addIntVar("Max Trace Depth", mMaxTraceDepth, 0, 30);
        pGui->addFloatVar("IOR Override", mIOROveride, 0, 3, 0.01f);
        pGui->addCheckBox("ID As Color", mColorPhoton);
        pGui->addIntVar("Photon ID Scale", mPhotonIDScale);
        pGui->addFloatVar("Min Trace Luminance", mTraceColorThreshold, 0, 10000,1);
        pGui->addFloatVar("Min Cull Luminance", mCullColorThreshold, 0, 10000, 0.005f);
        pGui->endGroup();
    }
    if (pGui->beginGroup("Photon Splat", true))
    {
        pGui->addFloatVar("Splat size", mSplatSize, 0, 10, 0.01f);
        pGui->addFloatVar("Intensity", mIntensity, 0, 10, 0.0002f);
        pGui->addFloatVar("Kernel Power", mKernelPower, 0.01f, 10, 0.01f);

        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Kernel" });
            debugModeList.push_back({ 1, "Solid" });
            debugModeList.push_back({ 2, "Shaded" });
            pGui->addDropdown("Photon Display Mode", debugModeList, (uint32_t&)mPhotonDisplayMode);
        }

        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Anisotropic" });
            debugModeList.push_back({ 1, "Isotropic" });
            debugModeList.push_back({ 2, "Photon Mesh" });
            pGui->addDropdown("Photon mode", debugModeList, (uint32_t&)mPhotonMode);
        }

        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Avg. Square" });
            debugModeList.push_back({ 1, "Avg. Length" });
            debugModeList.push_back({ 2, "Exact Area" });
            pGui->addDropdown("Area Type", debugModeList, (uint32_t&)mAreaType);
        }
        pGui->endGroup();
    }
    if(pGui->beginGroup("Refine Photon", false))
    {
        pGui->addCheckBox("Enable Refine", mRefinePhoton);
        pGui->addFloatVar("Luminance Threshold", mPixelLuminanceThreshold, 0.01f, 10.0, 0.01f);
        pGui->addFloatVar("Photon Size Threshold", mMinPhotonPixelSize, 1.f, 1000.0f, 0.1f);
        pGui->endGroup();
    }
    if (pGui->beginGroup("Smooth Photon", false))
    {
        pGui->addCheckBox("Enable Smooth", mSmoothPhoton);
        pGui->addFloatVar("Normal Threshold", mNormalThreshold, 0.01f, 1.0, 0.01f);
        pGui->addFloatVar("Distance Threshold", mDistanceThreshold, 0.1f, 10.0f, 0.1f);
        pGui->addFloatVar("Planar Threshold", mPlanarThreshold, 0.01f, 10.0, 0.1f);
        pGui->addFloatVar("Trim Direction Threshold", trimDirectionThreshold, 0, 1);
        pGui->endGroup();
    }
    mLightDirection = vec3(
        cos(mLightAngle.x) * sin(mLightAngle.y),
        cos(mLightAngle.y),
        sin(mLightAngle.x) * sin(mLightAngle.y));
    if (pGui->beginGroup("Light", true))
    {
        pGui->addFloat2Var("Light Angle", mLightAngle, -FLT_MAX, FLT_MAX, 0.01f);
        if (mpScene)
        {
            auto light0 = dynamic_cast<DirectionalLight*>(mpScene->getLight(0).get());
            light0->setWorldDirection(mLightDirection);
        }
        pGui->addFloat2Var("Light Angle Speed", mLightAngleSpeed, -FLT_MAX, FLT_MAX, 0.001f);
        mLightAngle += mLightAngleSpeed*0.01f;
        pGui->endGroup();
    }

    if (pGui->beginGroup("Camera"))
    {
        mpCamera->renderUI(pGui);
        pGui->endGroup();
    }
}

void Caustics::loadScene(const std::string& filename, const Fbo* pTargetFbo)
{
    mpScene = RtScene::loadFromFile(filename, RtBuildFlags::None, Model::LoadFlags::None);
    if (!mpScene) return;

    mpQuad = Model::createFromFile("Caustics/quad.obj");

    Model::SharedPtr pModel = mpScene->getModel(0);
    float radius = pModel->getRadius();

    mpCamera = mpScene->getActiveCamera();
    assert(mpCamera);

    mCamController.attachCamera(mpCamera);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    Sampler::SharedPtr pSampler = Sampler::create(samplerDesc);
    pModel->bindSamplerToMaterials(pSampler);

    // Update the controllers
    mCamController.setCameraSpeed(radius * 0.2f);
    auto sceneBBox = mpScene->getBoundingBox();
    float sceneRadius = sceneBBox.getSize().length() * 0.5f;
    //mCamController.setModelParams(mpScene->getCenter(), sceneRadius, sceneRadius);
    float nearZ = std::max(0.1f, pModel->getRadius() / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    mpGaussianKernel = Texture::createFromFile("Caustics/gaussian.png", true, false);
    mpUniformNoise = Texture::createFromFile("Caustics/uniform.png", true, false);
}

void Caustics::loadShader()
{
    // raytrace
    RtProgram::Desc rtProgDesc;
    rtProgDesc.addShaderLibrary("Caustics.rt.hlsl");
    rtProgDesc.setRayGen("rayGen");
    rtProgDesc.addHitGroup(0, "primaryClosestHit", "");
    rtProgDesc.addMiss(0, "primaryMiss");
    rtProgDesc.addHitGroup(1, "", "shadowAnyHit");
    rtProgDesc.addMiss(1, "shadowMiss");
    mpRaytraceProgram = RtProgram::create(rtProgDesc);
    mpRtState = RtState::create();
    mpRtState->setProgram(mpRaytraceProgram);
    mpRtState->setMaxTraceRecursionDepth(3);
    mpRtVars = RtProgramVars::create(mpRaytraceProgram, mpScene);

    // clear draw argument program
    mpDrawArgumentProgram = ComputeProgram::createFromFile("ResetDrawArgument.cs.hlsl", "main");
    mpDrawArgumentState = ComputeState::create();
    mpDrawArgumentState->setProgram(mpDrawArgumentProgram);
    mpDrawArgumentVars = ComputeVars::create(mpDrawArgumentProgram.get());

    // photon trace
    {
        RtProgram::Desc desc;
        desc.addShaderLibrary("PhotonTrace.rt.hlsl");
        desc.setRayGen("rayGen");
        desc.addHitGroup(0, "primaryClosestHit", "");
        desc.addMiss(0, "primaryMiss");
        //photonTraceProgDesc.addHitGroup(1, "", "shadowAnyHit");
        //photonTraceProgDesc.addMiss(1, "shadowMiss");
        mpPhotonTraceProgram = RtProgram::create(desc, 72U);
        mpPhotonTraceState = RtState::create();
        mpPhotonTraceState->setProgram(mpPhotonTraceProgram);
        mpPhotonTraceVars = RtProgramVars::create(mpPhotonTraceProgram, mpScene);
        //mpPhotonTraceRenderer = RtSceneRenderer::create(mpScene);
    }

    // composite rt
    {
        RtProgram::Desc desc;
        desc.addShaderLibrary("CompositeRT.rt.hlsl");
        desc.setRayGen("rayGen");
        desc.addHitGroup(0, "primaryClosestHit", "");
        desc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
        desc.addMiss(0, "primaryMiss");
        desc.addMiss(1, "shadowMiss");
        mpCompositeRTProgram = RtProgram::create(desc);
        mpCompositeRTState = RtState::create();
        mpCompositeRTState->setProgram(mpCompositeRTProgram);
        mpCompositeRTVars = RtProgramVars::create(mpCompositeRTProgram, mpScene);
    }

    // analyse trace result
    mpAnalyseProgram = ComputeProgram::createFromFile("AnalyseTraceResult.cs.hlsl", "main");
    mpAnalyseState = ComputeState::create();
    mpAnalyseState->setProgram(mpAnalyseProgram);
    mpAnalyseVars = ComputeVars::create(mpAnalyseProgram.get());

    // smooth photon
    mpSmoothProgram = ComputeProgram::createFromFile("SmoothPhoton.cs.hlsl", "main");
    mpSmoothState = ComputeState::create();
    mpSmoothState->setProgram(mpSmoothProgram);
    mpSmoothVars = ComputeVars::create(mpSmoothProgram.get());

    // photon scatter
    {
        BlendState::Desc blendDesc;
        blendDesc.setRtBlend(0, true);
        blendDesc.setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One);
        BlendState::SharedPtr scatterBlendState = BlendState::create(blendDesc);
        mpPhotonScatterProgram = GraphicsProgram::createFromFile("PhotonScatter.ps.hlsl", "photonScatterVS", "photonScatterPS");
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthEnabled(false);
        auto depthStencilState = DepthStencilState::create(dsDesc);
        RasterizerState::Desc rasterDesc;
        rasterDesc.setCullMode(RasterizerState::CullMode::None);
        auto rasterState = RasterizerState::create(rasterDesc);
        mpPhotonScatterBlendState = GraphicsState::create();
        mpPhotonScatterBlendState->setProgram(mpPhotonScatterProgram);
        mpPhotonScatterBlendState->setBlendState(scatterBlendState);
        mpPhotonScatterBlendState->setDepthStencilState(depthStencilState);
        mpPhotonScatterBlendState->setRasterizerState(rasterState);
        mpPhotonScatterNoBlendState = GraphicsState::create();
        mpPhotonScatterNoBlendState->setProgram(mpPhotonScatterProgram);
        mpPhotonScatterBlendState->setDepthStencilState(depthStencilState);
        mpPhotonScatterNoBlendState->setRasterizerState(rasterState);
        mpPhotonScatterVars = GraphicsVars::create(mpPhotonScatterProgram->getReflector());
        //mpPhotonScatterPass = RasterScenePass::create(mpScene, "PhotonScatter.ps.hlsl", "photonScatterVS", "photonScatterPS");
    }

    mpRtRenderer = RtSceneRenderer::create(mpScene);

    mpRasterPass = RasterScenePass::create(mpScene, "Caustics.ps.hlsl", "", "main");

    mpGPass = RasterScenePass::create(mpScene, "GPass.ps.hlsl", "", "gpassPS");

    mpCompositePass = FullScreenPass::create("Composite.ps.hlsl");

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpLinearSampler = Sampler::create(samplerDesc);
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(samplerDesc);
}

Caustics::Caustics() {}

void Caustics::onLoad(RenderContext* pRenderContext)
{
    if (gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        logErrorAndExit("Device does not support raytracing!");
    }

    loadScene(kDefaultScene, gpFramework->getTargetFbo().get());
    loadShader();
}

void Caustics::setCommonVars(GraphicsVars* pVars, const Fbo* pTargetFbo)
{
    ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
    //pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
    //pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    //pCB["emitSize"] = mEmitSize;
    //float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    //pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
    //pCB["sampleIndex"] = mSampleIndex;
    //pCB["useDOF"] = false;// mUseDOF;
}

void Caustics::setPerFrameVars(const Fbo* pTargetFbo)
{
    PROFILE("setPerFrameVars");
    {
        GraphicsVars* pVars = mpRtVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
        pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
        pCB["sampleIndex"] = mSampleIndex;
        pCB["useDOF"] = mUseDOF;
    }
    //setCommonVars(mpRtVars->getGlobalVars().get(), pTargetFbo);

    mSampleIndex++;
}

void Caustics::renderRT(RenderContext* pContext, Fbo::SharedPtr pTargetFbo)
{
    PROFILE("renderRT");
    setPerFrameVars(pTargetFbo.get());

    // gpass
    {
        pContext->clearFbo(mpGPassFbo.get(), vec4(0, 0, 0, 1), 1.0, 0);
        mpGPass->renderScene(pContext, mpGPassFbo);
    }

    // set draw argument
    {
        ConstantBuffer::SharedPtr pPerFrameCB = mpDrawArgumentVars["PerFrameCB"];
        pPerFrameCB["initRayCount"] = uint(mDispatchSize * mDispatchSize);
        mpDrawArgumentVars->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
        mpDrawArgumentVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        //mpDrawArgumentVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        pContext->dispatch(mpDrawArgumentState.get(), mpDrawArgumentVars.get(), uvec3(1, 1, 1));
    }

    // coarse photon tracing
    {
        pContext->clearTexture(mpRtOut.get());
        GraphicsVars* pVars = mpPhotonTraceVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        pCB["emitSize"] = mEmitSize;
        pCB["roughThreshold"] = mRoughThreshold;
        pCB["jitter"] = mJitter;
        pCB["launchRayTask"] = 0;
        pCB["rayTaskOffset"] = mDispatchSize * mDispatchSize;
        pCB["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
        pCB["maxDepth"] = mMaxTraceDepth;
        pCB["iorOverride"] = mIOROveride;
        pCB["colorPhotonID"] = (uint32_t)mColorPhoton;
        pCB["photonIDScale"] = mPhotonIDScale;
        pCB["traceColorThreshold"] = mTraceColorThreshold * (512 * 512) / (mDispatchSize * mDispatchSize);
        pCB["cullColorThreshold"] = mCullColorThreshold / 255;
        auto rayGenVars = mpPhotonTraceVars->getRayGenVars();
        rayGenVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        rayGenVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        rayGenVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        rayGenVars->setTexture("gOutput", mpRtOut);
        rayGenVars->setTexture("gUniformNoise", mpUniformNoise);
        auto hitVars = mpPhotonTraceVars->getHitVars(0);
        for (auto& hitVar : hitVars)
        {
            hitVar->setTexture("gOutput", mpRtOut);
            hitVar->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
            hitVar->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
            hitVar->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        }
        mpPhotonTraceState->setMaxTraceRecursionDepth(mMaxTraceDepth + 1);
        mpRtRenderer->renderScene(pContext, mpPhotonTraceVars, mpPhotonTraceState, uvec3(mDispatchSize, mDispatchSize, 1), mpCamera.get());
    }

    // analysis output
    if(mRefinePhoton)
    {
        ConstantBuffer::SharedPtr pPerFrameCB = mpAnalyseVars["PerFrameCB"];
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        pPerFrameCB["viewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
        pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
        pPerFrameCB["screenDim"] = int2(mpDepthTex->getWidth(), mpDepthTex->getHeight());
        pPerFrameCB["normalThreshold"] = mNormalThreshold;
        pPerFrameCB["distanceThreshold"] = mDistanceThreshold;
        pPerFrameCB["planarThreshold"] = mPlanarThreshold;
        pPerFrameCB["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
        pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize;
        mpAnalyseVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        mpAnalyseVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        mpAnalyseVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        mpAnalyseVars->setTexture("gDepthTex", mpGPassFbo->getDepthStencilTexture());
        static int groupSize = 16;
        pContext->dispatch(mpAnalyseState.get(), mpAnalyseVars.get(), uvec3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
    }

    // fine photon tracing
    if (mRefinePhoton)
    {
        GraphicsVars* pVars = mpPhotonTraceVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        pCB["emitSize"] = mEmitSize;
        pCB["roughThreshold"] = mRoughThreshold;
        pCB["jitter"] = mJitter;
        pCB["launchRayTask"] = 1;
        pCB["rayTaskOffset"] = mDispatchSize * mDispatchSize;
        auto rayGenVars = mpPhotonTraceVars->getRayGenVars();
        rayGenVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        rayGenVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        rayGenVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        rayGenVars->setTexture("gOutput", mpRtOut);
        rayGenVars->setTexture("gUniformNoise", mpUniformNoise);
        auto hitVars = mpPhotonTraceVars->getHitVars(0);
        for (auto& hitVar : hitVars)
        {
            hitVar->setTexture("gOutput", mpRtOut);
            hitVar->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
            hitVar->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
            hitVar->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        }
        mpRtRenderer->renderScene(pContext, mpPhotonTraceVars, mpPhotonTraceState, uvec3(4096, 4096, 1), mpCamera.get());
    }

    // smooth photon
    if (mSmoothPhoton)
    {
        ConstantBuffer::SharedPtr pPerFrameCB = mpSmoothVars["PerFrameCB"];
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        pPerFrameCB["viewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
        pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
        pPerFrameCB["screenDim"] = int2(mpDepthTex->getWidth(), mpDepthTex->getHeight());
        pPerFrameCB["normalThreshold"] = mNormalThreshold;
        pPerFrameCB["distanceThreshold"] = mDistanceThreshold;
        pPerFrameCB["planarThreshold"] = mPlanarThreshold;
        pPerFrameCB["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
        pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize;
        pPerFrameCB["trimDirectionThreshold"] = trimDirectionThreshold;
        mpSmoothVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        mpSmoothVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        mpSmoothVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        mpSmoothVars->setTexture("gDepthTex", mpGPassFbo->getDepthStencilTexture());
        static int groupSize = 16;
        pContext->dispatch(mpSmoothState.get(), mpSmoothVars.get(), uvec3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
    }

    // photon scattering
    {
        pContext->clearFbo(mpCausticsFbo.get(), vec4(0, 0, 0, 0), 1.0, 0);
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        ConstantBuffer::SharedPtr pPerFrameCB = mpPhotonScatterVars["PerFrameCB"];
        pPerFrameCB["gWorldMat"] = glm::mat4();
        pPerFrameCB["gWvpMat"] = wvp;
        pPerFrameCB["gEyePosW"] = mpCamera->getPosition();
        pPerFrameCB["gSplatSize"] = mSplatSize;
        pPerFrameCB["gIntensity"] = mIntensity;
        pPerFrameCB["gPhotonMode"] = mPhotonMode;
        pPerFrameCB["gAreaType"] = mAreaType;
        pPerFrameCB["gKernelPower"] = mKernelPower;
        pPerFrameCB["gShowPhoton"] = uint32_t(mPhotonDisplayMode);
        pPerFrameCB["gLightDir"] = mLightDirection;
        pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
        mpPhotonScatterVars["gLinearSampler"] = mpLinearSampler;
        mpPhotonScatterVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        mpPhotonScatterVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        mpPhotonScatterVars->setTexture("gDepthTex", mpGPassFbo->getDepthStencilTexture());
        mpPhotonScatterVars->setTexture("gNormalTex", mpGPassFbo->getColorTexture(0));
        mpPhotonScatterVars->setTexture("gDiffuseTex", mpGPassFbo->getColorTexture(1));
        mpPhotonScatterVars->setTexture("gSpecularTex", mpGPassFbo->getColorTexture(2));
        mpPhotonScatterVars->setTexture("gGaussianTex", mpGaussianKernel);
        int instanceCount = mDispatchSize * mDispatchSize;
        //pContext->drawIndexedInstanced(mpPhotonScatterState.get(), mpPhotonScatterVars.get(), mpQuad->getMesh(0)->getIndexCount(), instanceCount, 0, 0, 0);
        if (mPhotonDisplayMode == 2)
        {
            mpPhotonScatterNoBlendState->setVao(mpQuad->getMesh(0)->getVao());
            mpPhotonScatterNoBlendState->setFbo(mpCausticsFbo);
            pContext->drawIndexedIndirect(mpPhotonScatterNoBlendState.get(), mpPhotonScatterVars.get(), mpDrawArgumentBuffer.get(), 0);
        }
        else
        {
            mpPhotonScatterBlendState->setVao(mpQuad->getMesh(0)->getVao());
            mpPhotonScatterBlendState->setFbo(mpCausticsFbo);
            pContext->drawIndexedIndirect(mpPhotonScatterBlendState.get(), mpPhotonScatterVars.get(), mpDrawArgumentBuffer.get(), 0);
        }
    }

    // Render output
    if (mDebugMode == 9)
    {
        pContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
        GraphicsVars* pVars = mpCompositeRTVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
        pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
        pCB["sampleIndex"] = mSampleIndex++;
        pCB["useDOF"] = mUseDOF;
        pCB["roughThreshold"] = mRoughThreshold;
        pCB["maxDepth"] = mMaxTraceDepth;
        pCB["iorOverride"] = mIOROveride;
        auto hitVars = mpCompositeRTVars->getHitVars(0);
        for (auto& hitVar : hitVars)
        {
            hitVar->setTexture("gCausticsTex", mpCausticsFbo->getColorTexture(0));
            hitVar["gLinearSampler"] = mpLinearSampler;
        }
        auto rayGenVars = mpCompositeRTVars->getRayGenVars();
        rayGenVars->setTexture("gOutput", mpRtOut);
        mpCompositeRTState->setMaxTraceRecursionDepth(mMaxTraceDepth + 1);
        mpRtRenderer->renderScene(pContext, mpCompositeRTVars, mpCompositeRTState, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1), mpCamera.get());
        pContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
    }
    else
    {
        mpCompositePass["gDepthTex"] = mpGPassFbo->getDepthStencilTexture();
        mpCompositePass["gNormalTex"] = mpGPassFbo->getColorTexture(0);
        mpCompositePass["gDiffuseTex"] = mpGPassFbo->getColorTexture(1);
        mpCompositePass["gSpecularTex"] = mpGPassFbo->getColorTexture(2);
        mpCompositePass["gPhotonTex"] = mpCausticsFbo->getColorTexture(0);
        mpCompositePass["gRayTex"] = mpRtOut;
        mpCompositePass["gPointSampler"] = mpPointSampler;
        ConstantBuffer::SharedPtr pCompCB = mpCompositePass["PerImageCB"];
        pCompCB["gNumLights"] = mpScene->getLightCount();
        pCompCB["gDebugMode"] = (uint32_t)mDebugMode;
        pCompCB["gInvWvpMat"] = mpCamera->getInvViewProjMatrix();
        pCompCB["gCameraPos"] = mpCamera->getPosition();
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            mpScene->getLight(i)->setIntoProgramVars(mpCompositePass->getVars().get(), pCompCB.get(), "gLightData[" + std::to_string(i) + "]");
        }
        mpCompositePass->execute(pContext, pTargetFbo);
    }
}

void Caustics::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene)
    {
        mCamController.update();
        if (mRayTrace)
            renderRT(pRenderContext, pTargetFbo);
        else
            mpRasterPass->renderScene(pRenderContext, pTargetFbo);
    }

    TextRenderer::render(pRenderContext, gpFramework->getFrameRate().getMsg(), pTargetFbo, { 20, 20 });
}

bool Caustics::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mCamController.onKeyEvent(keyEvent))
    {
        return true;
    }
    if (keyEvent.key == KeyboardEvent::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mRayTrace = !mRayTrace;
        return true;
    }
    return false;
}

bool Caustics::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCamController.onMouseEvent(mouseEvent);
}

void Caustics::onResizeSwapChain(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }

#define CAUSTICS_MAP_SIZE 2048
#define TASK_SIZE 4096*4096
    mpRayTaskBuffer = StructuredBuffer::create(mpAnalyseProgram.get(), std::string("gRayTask"), TASK_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer = StructuredBuffer::create(mpPhotonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), TASK_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpDrawArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gDrawArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg);
    mpRayArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gRayArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg);
    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpDepthTex = Texture::create2D(width, height, ResourceFormat::D24UnormS8, 1, 1, nullptr, Resource::BindFlags::DepthStencil | Resource::BindFlags::ShaderResource);
    mpCausticsFbo = Fbo::create2D(width, height, ResourceFormat::RGBA16Float, ResourceFormat::D24UnormS8);

    mpNormalTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    mpDiffuseTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    mpSpecularTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    mpGPassFbo = Fbo::create({ mpNormalTex , mpDiffuseTex ,mpSpecularTex }, mpDepthTex);//Fbo::create2D(width, height, ResourceFormat::RGBA16Float, ResourceFormat::D24UnormS8);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    Caustics::UniquePtr pRenderer = std::make_unique<Caustics>();
    SampleConfig config;
    config.windowDesc.title = "Caustics";
    config.windowDesc.resizableWindow = true;

    Sample::run(config, pRenderer);
}
