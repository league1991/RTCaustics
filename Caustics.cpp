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

const FileDialogFilterVec settingFilter = { {"ini", "Scene Setting File"} };

void Caustics::onGuiRender(Gui* pGui)
{
    pGui->addCheckBox("Ray Trace", mRayTrace);

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileExtensionFilters, filename))
        {
            loadScene(filename, gpFramework->getTargetFbo().get());
            loadShader();
        }
    }

    if (pGui->addButton("Load Scene Settings"))
    {
        std::string filename;
        if (openFileDialog(settingFilter, filename))
        {
            loadSceneSetting(filename);
        }
    }
    if (pGui->addButton("Save Scene Settings"))
    {
        std::string filename;
        if (saveFileDialog(settingFilter, filename))
        {
            saveSceneSetting(filename);
        }
    }
    if (pGui->addButton("Update Shader"))
    {
        loadShader();
    }

    if (pGui->beginGroup("Display", true))
    {
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
            debugModeList.push_back({ 8, "Ray Info" });
            debugModeList.push_back({ 9, "Raytrace" });
            debugModeList.push_back({ 10, "Avg. Screen Area" });
            debugModeList.push_back({ 11, "Screen Area Std. Variance" });
            debugModeList.push_back({ 12, "Photon Count" });
            debugModeList.push_back({ 13, "Photon Total Count" });
            pGui->addDropdown("Composite mode", debugModeList, (uint32_t&)mDebugMode);
        }
        pGui->addFloatVar("Max Pixel Value", mMaxPixelArea, 0, 1000000000, 5.f);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 1, "x1" });
            debugModeList.push_back({ 2, "x2" });
            debugModeList.push_back({ 4, "x4" });
            debugModeList.push_back({ 8, "x8" });
            debugModeList.push_back({ 16, "x16" });
            pGui->addDropdown("Ray Tex Scale", debugModeList, (uint32_t&)mRayTexScaleFactor);
        }
        pGui->endGroup();
    }

    if (pGui->beginGroup("Photon Trace", true))
    {
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Fixed Resolution" });
            debugModeList.push_back({ 1, "Adaptive Resolution" });
            debugModeList.push_back({ 2, "None" });
            pGui->addDropdown("Trace Type", debugModeList, (uint32_t&)mTraceType);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Ray Differential" });
            debugModeList.push_back({ 1, "Ray Cone" });
            debugModeList.push_back({ 2, "None" });
            pGui->addDropdown("Ray Type", debugModeList, (uint32_t&)mPhotonTraceMacro);
        }
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
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Avg Square" });
            debugModeList.push_back({ 1, "Avg Length" });
            debugModeList.push_back({ 2, "Max Square" });
            debugModeList.push_back({ 3, "Exact Area" });
            pGui->addDropdown("Area Type", debugModeList, (uint32_t&)mAreaType);
        }
        pGui->addFloatVar("Intensity", mIntensity, 0, 10, 0.1f);
        pGui->addFloatVar("Emit size", mEmitSize, 0, 1000, 1);
        pGui->addFloatVar("Rough Threshold", mRoughThreshold, 0, 1, 0.01f);
        pGui->addIntVar("Max Trace Depth", mMaxTraceDepth, 0, 30);
        pGui->addFloatVar("IOR Override", mIOROveride, 0, 3, 0.01f);
        pGui->addCheckBox("ID As Color", mColorPhoton);
        pGui->addIntVar("Photon ID Scale", mPhotonIDScale);
        pGui->addFloatVar("Min Trace Luminance", mTraceColorThreshold, 0, 10,0.005f);
        pGui->addFloatVar("Min Cull Luminance", mCullColorThreshold, 0, 10000, 0.001f);
        pGui->addFloatVar("Max Pixel Radius", mMaxPhotonPixelRadius, 0, 5000, 1.f);
        pGui->addCheckBox("Update Photon", mUpdatePhoton);
        pGui->endGroup();
    }

    if (pGui->beginGroup("Adaptive Resolution", true))
    {
        pGui->addFloatVar("Luminance Threshold", mPixelLuminanceThreshold, 0.01f, 10.0, 0.01f);
        pGui->addFloatVar("Photon Size Threshold", mMinPhotonPixelSize, 1.f, 1000.0f, 0.1f);
        pGui->addFloatVar("Smooth Weight", mSmoothWeight, 0, 10.0f, 0.001f);
        pGui->addFloatVar("Proportional Gain", mUpdateSpeed, 0, 1, 0.01f);
        pGui->addFloatVar("Variance Gain", mVarianceGain, 0, 10, 0.0001f);
        pGui->addFloatVar("Derivative Gain", mDerivativeGain, -10, 10, 0.1f);
        pGui->addIntVar("Max Task Per Pixel", mMaxTaskCountPerPixel, 1);
        pGui->endGroup();
    }

    if (pGui->beginGroup("Smooth Photon", false))
    {
        pGui->addCheckBox("Remove Isolated Photon", mRemoveIsolatedPhoton);
        pGui->addCheckBox("Enable Median Filter", mMedianFilter);
        pGui->addFloatVar("Normal Threshold", mNormalThreshold, 0.01f, 1.0, 0.01f);
        pGui->addFloatVar("Distance Threshold", mDistanceThreshold, 0.1f, 100.0f, 0.1f);
        pGui->addFloatVar("Planar Threshold", mPlanarThreshold, 0.01f, 10.0, 0.1f);
        pGui->addFloatVar("Trim Direction Threshold", trimDirectionThreshold, 0, 1);
        pGui->addIntVar("Min Neighbour Count", mMinNeighbourCount, 0, 8);
        pGui->endGroup();
    }

    if (pGui->beginGroup("Photon Splatting", true))
    {
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Scatter" });
            debugModeList.push_back({ 1, "Gather" });
            debugModeList.push_back({ 2, "None" });
            pGui->addDropdown("Density Estimation", debugModeList, (uint32_t&)mScatterOrGather);
        }
        pGui->addFloatVar("Splat size", mSplatSize, 0, 100, 0.01f);
        pGui->addFloatVar("Kernel Power", mKernelPower, 0.01f, 10.f, 0.01f);

        if(pGui->beginGroup("Scatter Parameters", true))
        {
            pGui->addFloatVar("Z Tolerance", mZTolerance, 0.001f, 1, 0.001f);
            pGui->addFloatVar("Scatter Normal Threshold", mScatterNormalThreshold, 0.01f, 1.0, 0.01f);
            pGui->addFloatVar("Scatter Distance Threshold", mScatterDistanceThreshold, 0.1f, 10.0f, 0.1f);
            pGui->addFloatVar("Scatter Planar Threshold", mScatterPlanarThreshold, 0.01f, 10.0, 0.1f);
            pGui->addFloatVar("Max Anisotropy", mMaxAnisotropy, 1, 100, 0.1f);
            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({ 0, "Quad" });
                debugModeList.push_back({ 1, "Sphere" });
                pGui->addDropdown("Photon Geometry", debugModeList, (uint32_t&)mScatterGeometry);
            }
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
                debugModeList.push_back({ 3, "Screen Dot" });
                pGui->addDropdown("Photon mode", debugModeList, (uint32_t&)mPhotonMode);
            }
            pGui->endGroup();
        }

        if(pGui->beginGroup("Gather Parameters", false))
        {
            pGui->addFloatVar("Gather Depth Radius", mDepthRadius, 0, 10, 0.01f);
            pGui->addCheckBox("Gather Show Tile Count", mShowTileCount);
            pGui->addIntVar("Gather Tile Count Scale", mTileCountScale, 0, 1000);
            pGui->endGroup();
        }

        pGui->endGroup();
    }

    if (pGui->beginGroup("Temporal Filter", true))
    {
        pGui->addCheckBox("Enable Filter", mTemporalFilter);
        pGui->addFloatVar("Filter Weight", mFilterWeight, 0.0f, 1.0f, 0.001f);
        pGui->addFloatVar("Jitter", mJitter, 0, 2, 0.01f);
        pGui->addFloatVar("Normal Filter Strength", mTemporalNormalKernel, 0.0001f, 10, 0.01f);
        pGui->addFloatVar("Depth Filter Strength", mTemporalDepthKernel, 0.0001f, 10, 0.01f);
        pGui->endGroup();
    }

    if (pGui->beginGroup("Composite", true))
    {
        {
            int oldResRatio = mCausticsMapResRatio;
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 1, "x 1" });
            debugModeList.push_back({ 2, "x 1/2" });
            debugModeList.push_back({ 4, "x 1/4" });
            debugModeList.push_back({ 8, "x 1/8" });
            pGui->addDropdown("Caustics Resolution", debugModeList, (uint32_t&)mCausticsMapResRatio);
            if (oldResRatio != mCausticsMapResRatio)
            {
                createCausticsMap();
            }
        }
        pGui->addCheckBox("Filter Caustics Map", mFilterCausticsMap);
        pGui->addFloatVar("UV Kernel", mUVKernel, 0.0f, 1000.f, 0.1f);
        pGui->addFloatVar("Depth Kernel", mZKernel, 0.0f, 1000.f, 0.1f);
        pGui->addFloatVar("Normal Kernel", mNormalKernel, 0.0f, 1000.f, 0.1f);
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
    mpSphere = Model::createFromFile("Caustics/sphere.obj");

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

Caustics::PhotonTraceShader Caustics::getPhotonTraceShader()
{
    auto pIter = mPhotonTraceShaderList.find(mPhotonTraceMacro);
    if (pIter == mPhotonTraceShaderList.end())
    {
        RtProgram::Desc desc;
        desc.addShaderLibrary("PhotonTrace.rt.hlsl");
        desc.setRayGen("rayGen");
        desc.addHitGroup(0, "primaryClosestHit", "");
        desc.addMiss(0, "primaryMiss");
        switch (mPhotonTraceMacro)
        {
        case Caustics::RAY_DIFFERENTIAL:
            desc.addDefine("RAY_DIFFERENTIAL", "1");
            break;
        case Caustics::RAY_CONE:
            desc.addDefine("RAY_CONE", "1");
            break;
        case Caustics::RAY_NONE:
            desc.addDefine("RAY_NONE", "1");
            break;
        default:
            break;
        }
        //photonTraceProgDesc.addHitGroup(1, "", "shadowAnyHit");
        //photonTraceProgDesc.addMiss(1, "shadowMiss");
        auto pPhotonTraceProgram = RtProgram::create(desc, 80U);
        auto pPhotonTraceState = RtState::create();
        pPhotonTraceState->setProgram(pPhotonTraceProgram);
        auto pPhotonTraceVars = RtProgramVars::create(pPhotonTraceProgram, mpScene);
        //mpPhotonTraceRenderer = RtSceneRenderer::create(mpScene);
        mPhotonTraceShaderList[mPhotonTraceMacro] = { pPhotonTraceProgram , pPhotonTraceVars ,pPhotonTraceState };
    }
    return mPhotonTraceShaderList[mPhotonTraceMacro];
}

void Caustics::loadSceneSetting(std::string path)
{
    std::ifstream file(path, std::ios::in);
    if (!file)
    {
        return;
    }

    file >> mLightAngle.x >> mLightAngle.y;

    float3 camOri, camTarget;
    file >> camOri.x >> camOri.y >> camOri.z;
    file >> camTarget.x >> camTarget.y >> camTarget.z;
    mpCamera->setPosition(camOri);
    mpCamera->setTarget(camTarget);
}

void Caustics::saveSceneSetting(std::string path)
{
    if (path.find(".ini") == std::string::npos)
    {
        path += ".ini";
    }
    std::ofstream file(path, std::ios::out);
    if (!file)
    {
        return;
    }

    file << mLightAngle.x << " " << mLightAngle.y << std::endl;
    float3 camOri = mpCamera->getPosition();
    float3 camTarget = mpCamera->getTarget();
    file << camOri.x << " " << camOri.y << " " << camOri.z << std::endl;
    file << camTarget.x << " " << camTarget.y << " " << camTarget.z << std::endl;
}

void Caustics::createCausticsMap()
{
    uint32_t width = mpRtOut->getWidth();
    uint32_t height = mpRtOut->getHeight();
    uint2 dim(width / mCausticsMapResRatio, height / mCausticsMapResRatio);

    auto pPhotonMapTex = Texture::create2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    auto depthTex = Texture::create2D(dim.x, dim.y, ResourceFormat::D24UnormS8, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
    mpCausticsFbo[0] = Fbo::create({ pPhotonMapTex }, depthTex);

    pPhotonMapTex = Texture::create2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    mpCausticsFbo[1] = Fbo::create({ pPhotonMapTex }, depthTex);
}

int2 Caustics::getTileDim() const
{
    int2 tileDim;
    tileDim.x = (mpRtOut->getWidth() / mCausticsMapResRatio + mTileSize - 1) / mTileSize;
    tileDim.y = (mpRtOut->getHeight() / mCausticsMapResRatio + mTileSize - 1) / mTileSize;
    return tileDim;
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
    mPhotonTraceShaderList.clear();
    getPhotonTraceShader();

    // composite rt
    {
        RtProgram::Desc desc;
        desc.addShaderLibrary("CompositeRT.rt.hlsl");
        desc.setRayGen("rayGen");
        desc.addHitGroup(0, "primaryClosestHit", "");
        desc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
        desc.addMiss(0, "primaryMiss");
        desc.addMiss(1, "shadowMiss");
        mpCompositeRTProgram = RtProgram::create(desc,48);
        mpCompositeRTState = RtState::create();
        mpCompositeRTState->setProgram(mpCompositeRTProgram);
        mpCompositeRTVars = RtProgramVars::create(mpCompositeRTProgram, mpScene);
    }

    // update ray density texture
    mpUpdateRayDensityProgram = ComputeProgram::createFromFile("UpdateRayDensity.cs.hlsl", "updateRayDensityTex");
    mpUpdateRayDensityState = ComputeState::create();
    mpUpdateRayDensityState->setProgram(mpUpdateRayDensityProgram);
    mpUpdateRayDensityVars = ComputeVars::create(mpUpdateRayDensityProgram.get());

    // analyse trace result
    mpAnalyseProgram = ComputeProgram::createFromFile("AnalyseTraceResult.cs.hlsl", "addPhotonTaskFromTexture");
    mpAnalyseState = ComputeState::create();
    mpAnalyseState->setProgram(mpAnalyseProgram);
    mpAnalyseVars = ComputeVars::create(mpAnalyseProgram.get());

    // smooth photon
    mpSmoothProgram = ComputeProgram::createFromFile("SmoothPhoton.cs.hlsl", "main");
    mpSmoothState = ComputeState::create();
    mpSmoothState->setProgram(mpSmoothProgram);
    mpSmoothVars = ComputeVars::create(mpSmoothProgram.get());

    // allocate tile
    const char* shaderEntries[] = { "CountTilePhoton","AllocateMemory","StoreTilePhoton" };
    for (int i = 0; i < 3; i++)
    {
        mpAllocateTileProgram[i] = ComputeProgram::createFromFile("AllocateTilePhoton.cs.hlsl", shaderEntries[i]);
        mpAllocateTileState[i] = ComputeState::create();
        mpAllocateTileState[i]->setProgram(mpAllocateTileProgram[i]);
        mpAllocateTileVars[i] = ComputeVars::create(mpAllocateTileProgram[i].get());
    }

    // photon gather
    mpPhotonGatherProgram = ComputeProgram::createFromFile("PhotonGather.cs.hlsl", "main");
    mpPhotonGatherState = ComputeState::create();
    mpPhotonGatherState->setProgram(mpPhotonGatherProgram);
    mpPhotonGatherVars = ComputeVars::create(mpPhotonGatherProgram.get());

    // photon scatter
    {
        BlendState::Desc blendDesc;
        blendDesc.setRtBlend(0, true);
        blendDesc.setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One);
        BlendState::SharedPtr scatterBlendState = BlendState::create(blendDesc);
        mpPhotonScatterProgram = GraphicsProgram::createFromFile("PhotonScatter.ps.hlsl", "photonScatterVS", "photonScatterPS");
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthEnabled(false);
        dsDesc.setDepthWriteMask(false);
        auto depthStencilState = DepthStencilState::create(dsDesc);
        RasterizerState::Desc rasterDesc;
        rasterDesc.setCullMode(RasterizerState::CullMode::None);
        static int32_t depthBias = -8;
        static float slopeBias = -16;
        rasterDesc.setDepthBias(depthBias, slopeBias);
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
    }

    // smooth photon
    mpFilterProgram = ComputeProgram::createFromFile("TemporalFilter.cs.hlsl", "main");
    mpFilterState = ComputeState::create();
    mpFilterState->setProgram(mpFilterProgram);
    mpFilterVars = ComputeVars::create(mpFilterProgram.get());

    mpRtRenderer = RtSceneRenderer::create(mpScene);

    mpRasterPass = RasterScenePass::create(mpScene, "Caustics.ps.hlsl", "", "main");

    mpGPass = RasterScenePass::create(mpScene, "GPass.ps.hlsl", "", "gpassPS");

    mpCompositePass = FullScreenPass::create("Composite.ps.hlsl");

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    samplerDesc.setAddressingMode(Sampler::AddressMode::Border, Sampler::AddressMode::Border, Sampler::AddressMode::Border);
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

float2 getRandomPoint(int i)
{
    const double g = 1.32471795724474602596;
    const double a1 = 1.0 / g;
    const double a2 = 1.0 / (g * g);
    double x = 0.5 + a1 * (i + 1);
    double y = 0.5 + a2 * (i + 1);
    float xF = float(x - floor(x));
    float yF = float(y - floor(y));
    return float2(xF, yF);
}

void Caustics::setPhotonTracingCommonVariable(Caustics::PhotonTraceShader& shader)
{
    GraphicsVars* pVars = shader.mpPhotonTraceVars->getGlobalVars().get();
    ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
    float2 randomOffset = (getRandomPoint(mFrameCounter) * 2.0f - 1.0f) * mJitter;
    pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
    pCB["viewportDims"] = vec2(mpRtOut->getWidth(), mpRtOut->getHeight());
    pCB["emitSize"] = mEmitSize;
    pCB["roughThreshold"] = mRoughThreshold;
    pCB["randomOffset"] = mTemporalFilter ? randomOffset : float2(0,0);
    pCB["launchRayTask"] = 0;
    pCB["rayTaskOffset"] = mDispatchSize * mDispatchSize;
    pCB["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
    pCB["maxDepth"] = mMaxTraceDepth;
    pCB["iorOverride"] = mIOROveride;
    pCB["colorPhotonID"] = (uint32_t)mColorPhoton;
    pCB["photonIDScale"] = mPhotonIDScale;
    pCB["traceColorThreshold"] = mTraceColorThreshold * (512 * 512) / (mDispatchSize * mDispatchSize);
    pCB["cullColorThreshold"] = mCullColorThreshold / 255;
    pCB["gAreaType"] = mAreaType;
    pCB["gIntensity"] = mIntensity / 1000;
    pCB["gSplatSize"] = mSplatSize;
    pCB["updatePhoton"] = (uint32_t)mUpdatePhoton;
    pCB["gMaxScreenRadius"] = mMaxPhotonPixelRadius;
    auto rayGenVars = shader.mpPhotonTraceVars->getRayGenVars();
    rayGenVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
    rayGenVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
    rayGenVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
    rayGenVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
    rayGenVars->setTexture("gUniformNoise", mpUniformNoise);
    rayGenVars->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
    auto hitVars = shader.mpPhotonTraceVars->getHitVars(0);
    for (auto& hitVar : hitVars)
    {
        hitVar->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
        hitVar->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        hitVar->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
        hitVar->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
    }
    shader.mpPhotonTraceState->setMaxTraceRecursionDepth(1);
}

void Caustics::renderRT(RenderContext* pContext, Fbo::SharedPtr pTargetFbo)
{
    PROFILE("renderRT");
    //setPerFrameVars(pTargetFbo.get());

    // reset data
    uint32_t statisticsOffset = uint32_t(mFrameCounter % mpPhotonCountTex->getWidth());
    int thisIdx = mFrameCounter % 2;
    int lastIdx = 1- thisIdx;
    GBuffer* gBuffer = mGBuffer + thisIdx;
    GBuffer* gBufferLast = mGBuffer + lastIdx;
    Fbo::SharedPtr causticsFbo = mpCausticsFbo[thisIdx];
    Fbo::SharedPtr causticsFboLast = mpCausticsFbo[lastIdx];

    if (mUpdatePhoton)
    {
        ConstantBuffer::SharedPtr pPerFrameCB = mpDrawArgumentVars["PerFrameCB"];
        pPerFrameCB["initRayCount"] = uint(mDispatchSize * mDispatchSize);
        pPerFrameCB["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
        pPerFrameCB["textureOffset"] = statisticsOffset;
        pPerFrameCB["scatterGeoIdxCount"] = mScatterGeometry == SCATTER_GEOMETRY_QUAD ? 6U : 12U;
        mpDrawArgumentVars->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
        mpDrawArgumentVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        //mpDrawArgumentVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        mpDrawArgumentVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
        mpDrawArgumentVars->setTexture("gPhotonCountTexture", mpPhotonCountTex);
        pContext->dispatch(mpDrawArgumentState.get(), mpDrawArgumentVars.get(), uvec3(mDispatchSize/16, mDispatchSize / 16, 1));
    }

    // gpass
    {
        pContext->clearFbo(gBuffer->mpGPassFbo.get(), vec4(0, 0, 0, 1), 1.0, 0);
        mpGPass->renderScene(pContext, gBuffer->mpGPassFbo);
    }

    // photon tracing
    if (mTraceType != 2)
    {
        auto photonTraceShader = getPhotonTraceShader();
        setPhotonTracingCommonVariable(photonTraceShader);
        GraphicsVars* pVars = photonTraceShader.mpPhotonTraceVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["launchRayTask"] = int(mTraceType == 1);
        uvec3 resolution = mTraceType == 0 ? uvec3(mDispatchSize, mDispatchSize, 1) : uvec3(4096, 4096, 1);
        mpRtRenderer->renderScene(pContext, photonTraceShader.mpPhotonTraceVars, photonTraceShader.mpPhotonTraceState, resolution, mpCamera.get());
    }

    // analysis output
    bool refinePhoton = mTraceType == 1;
    if(refinePhoton && mUpdatePhoton)
    {
        {
            ConstantBuffer::SharedPtr pPerFrameCB = mpUpdateRayDensityVars["PerFrameCB"];
            pPerFrameCB["coarseDim"] = int2(mDispatchSize, mDispatchSize);
            pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize;
            pPerFrameCB["smoothWeight"] = mSmoothWeight;
            pPerFrameCB["maxTaskPerPixel"] = mMaxTaskCountPerPixel;
            pPerFrameCB["updateSpeed"] = mUpdateSpeed;
            pPerFrameCB["varianceGain"] = mVarianceGain;
            pPerFrameCB["derivativeGain"] = mDerivativeGain;
            mpUpdateRayDensityVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
            mpUpdateRayDensityVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
            mpUpdateRayDensityVars->setTexture("gRayDensityTex", mpRayDensityTex);
            static int groupSize = 16;
            pContext->dispatch(mpUpdateRayDensityState.get(), mpUpdateRayDensityVars.get(), uvec3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
        }

        {
            ConstantBuffer::SharedPtr pPerFrameCB = mpAnalyseVars["PerFrameCB"];
            glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
            pPerFrameCB["viewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
            pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
            pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
            pPerFrameCB["normalThreshold"] = mNormalThreshold;
            pPerFrameCB["distanceThreshold"] = mDistanceThreshold;
            pPerFrameCB["planarThreshold"] = mPlanarThreshold;
            pPerFrameCB["pixelLuminanceThreshold"] = mPixelLuminanceThreshold; 
            pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize;
            static float2 offset(0.5,0.5);
            static float speed = 0.0f;
            pPerFrameCB["randomOffset"] = offset;// mMinPhotonPixelSize;
            offset += speed;
            mpAnalyseVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
            mpAnalyseVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
            mpAnalyseVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
            mpAnalyseVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
            mpAnalyseVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
            mpAnalyseVars->setTexture("gRayDensityTex", mpRayDensityTex);
            int2 groupSize(32,16);
            pContext->dispatch(mpAnalyseState.get(), mpAnalyseVars.get(), uvec3(mDispatchSize / groupSize.x, mDispatchSize / groupSize.y, 1));
        }
    }

    // smooth photon
    StructuredBuffer::SharedPtr photonBuffer = mpPhotonBuffer;
    if (mRemoveIsolatedPhoton || mMedianFilter)
    {
        ConstantBuffer::SharedPtr pPerFrameCB = mpSmoothVars["PerFrameCB"];
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        pPerFrameCB["viewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
        pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
        pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        pPerFrameCB["normalThreshold"] = mNormalThreshold;
        pPerFrameCB["distanceThreshold"] = mDistanceThreshold;
        pPerFrameCB["planarThreshold"] = mPlanarThreshold;
        pPerFrameCB["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
        pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize;
        pPerFrameCB["trimDirectionThreshold"] = trimDirectionThreshold;
        pPerFrameCB["enableMedianFilter"] = uint32_t(mMedianFilter);
        pPerFrameCB["removeIsolatedPhoton"] = uint32_t(mRemoveIsolatedPhoton);
        pPerFrameCB["minNeighbourCount"] = mMinNeighbourCount;
        mpSmoothVars->setStructuredBuffer("gSrcPhotonBuffer", mpPhotonBuffer);
        mpSmoothVars->setStructuredBuffer("gDstPhotonBuffer", mpPhotonBuffer2);
        mpSmoothVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        mpSmoothVars->setStructuredBuffer("gRayTask", mpPixelInfoBuffer);
        mpSmoothVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        static int groupSize = 16;
        pContext->dispatch(mpSmoothState.get(), mpSmoothVars.get(), uvec3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
        photonBuffer = mpPhotonBuffer2;
    }

    // photon scattering
    if(mScatterOrGather == 0)
    {
        //pContext->clearFbo(mpCausticsFbo.get(), vec4(0, 0, 0, 0), 1.0, 0);
        pContext->clearRtv(causticsFbo->getColorTexture(0)->getRTV().get(), vec4(0, 0, 0, 0));
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        glm::mat4 invP = glm::inverse(mpCamera->getProjMatrix());
        ConstantBuffer::SharedPtr pPerFrameCB = mpPhotonScatterVars["PerFrameCB"];
        pPerFrameCB["gWorldMat"] = glm::mat4();
        pPerFrameCB["gWvpMat"] = wvp;
        pPerFrameCB["gInvProjMat"] = invP;
        pPerFrameCB["gEyePosW"] = mpCamera->getPosition();
        pPerFrameCB["gSplatSize"] = mSplatSize;
        pPerFrameCB["gPhotonMode"] = mPhotonMode;
        pPerFrameCB["gKernelPower"] = mKernelPower;
        pPerFrameCB["gShowPhoton"] = uint32_t(mPhotonDisplayMode);
        pPerFrameCB["gLightDir"] = mLightDirection;
        pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
        pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        pPerFrameCB["normalThreshold"] = mScatterNormalThreshold;
        pPerFrameCB["distanceThreshold"] = mScatterDistanceThreshold;
        pPerFrameCB["planarThreshold"] = mScatterPlanarThreshold; 
        pPerFrameCB["gMaxAnisotropy"] = mMaxAnisotropy;
        pPerFrameCB["gCameraPos"] = mpCamera->getPosition();
        pPerFrameCB["gZTolerance"] = mZTolerance;
        pPerFrameCB["gResRatio"] = mCausticsMapResRatio;
        //pPerFrameCB["gMaxScreenRadius"] = mMaxPhotonPixelRadius;
        mpPhotonScatterVars["gLinearSampler"] = mpLinearSampler;
        mpPhotonScatterVars->setStructuredBuffer("gPhotonBuffer", photonBuffer);
        mpPhotonScatterVars->setStructuredBuffer("gRayTask", mpPixelInfoBuffer);
        mpPhotonScatterVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        mpPhotonScatterVars->setTexture("gNormalTex", gBuffer->mpGPassFbo->getColorTexture(0));
        mpPhotonScatterVars->setTexture("gDiffuseTex", gBuffer->mpGPassFbo->getColorTexture(1));
        mpPhotonScatterVars->setTexture("gSpecularTex", gBuffer->mpGPassFbo->getColorTexture(2));
        mpPhotonScatterVars->setTexture("gGaussianTex", mpGaussianKernel);
        int instanceCount = mDispatchSize * mDispatchSize;
        GraphicsState::SharedPtr scatterState;
        if (mPhotonDisplayMode == 2)
        {
            scatterState = mpPhotonScatterNoBlendState;
        }
        else
        {
            scatterState = mpPhotonScatterBlendState;
        }
        if (mScatterGeometry == SCATTER_GEOMETRY_QUAD)
            scatterState->setVao(mpQuad->getMesh(0)->getVao());
        else
            scatterState->setVao(mpSphere->getMesh(0)->getVao());
        scatterState->setFbo(causticsFbo);
        if (mPhotonMode == 2)
        {
            pContext->drawIndexedInstanced(scatterState.get(), mpPhotonScatterVars.get(), 6, mDispatchSize* mDispatchSize, 0, 0, 0);
        }
        else
        {
            pContext->drawIndexedIndirect(scatterState.get(), mpPhotonScatterVars.get(), mpDrawArgumentBuffer.get(), 0);
        }
    }
    else if (mScatterOrGather == 1)
    {
        int2 tileDim = getTileDim();
        int sqrtCount = int(sqrt(mMaxTaskCountPerPixel));
        //int photonCount = mDispatchSize * mDispatchSize * mMaxTaskCountPerPixel;
        int dimX = mDispatchSize * sqrtCount;
        int dimY = mDispatchSize * sqrtCount;
        int blockSize = 32;
        uvec3 dispatchDim[] = {
            uvec3((dimX + blockSize-1) / blockSize,   (dimY + blockSize-1) / blockSize, 1),
            uvec3((tileDim.x + 15) / 16,(tileDim.y + 15) / 16,1),
            uvec3((dimX + blockSize-1) / blockSize,   (dimY + blockSize-1) / blockSize, 1)
        };
        // build tile data
        int2 screenSize(mpRtOut->getWidth() / mCausticsMapResRatio, mpRtOut->getHeight() / mCausticsMapResRatio);
        for (int i = 0; i < 3; i++)
        {
            auto vars = mpAllocateTileVars[i];
            auto states = mpAllocateTileState[i];
            ConstantBuffer::SharedPtr pPerFrameCB = vars["PerFrameCB"];
            glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
            pPerFrameCB["gViewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
            pPerFrameCB["screenDim"] = screenSize;
            pPerFrameCB["tileDim"] = tileDim;
            pPerFrameCB["gSplatSize"] = mSplatSize;
            pPerFrameCB["blockCount"] = int2(dispatchDim[i].x, dispatchDim[i].y);
            vars->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
            vars->setStructuredBuffer("gPhotonBuffer", photonBuffer);
            vars->setStructuredBuffer("gTileInfo", mpTileIDInfoBuffer);
            vars->setRawBuffer("gIDBuffer", mpIDBuffer);
            vars->setRawBuffer("gIDCounter", mpIDCounterBuffer);
            pContext->dispatch(states.get(), vars.get(), dispatchDim[i]);
        }
        // gathering
        ConstantBuffer::SharedPtr pPerFrameCB = mpPhotonGatherVars["PerFrameCB"];
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        pPerFrameCB["gInvViewProjMat"] = mpCamera->getInvViewProjMatrix();
        pPerFrameCB["screenDim"] = screenSize;
        pPerFrameCB["tileDim"] = tileDim;
        pPerFrameCB["gSplatSize"] = mSplatSize;
        pPerFrameCB["gDepthRadius"] = mDepthRadius;
        pPerFrameCB["gShowTileCount"] = int(mShowTileCount);
        pPerFrameCB["gTileCountScale"] = int(mTileCountScale);
        pPerFrameCB["gKernelPower"] = mKernelPower;
        pPerFrameCB["causticsMapResRatio"] = mCausticsMapResRatio;
        mpPhotonGatherVars->setStructuredBuffer("gPhotonBuffer", photonBuffer);
        mpPhotonGatherVars->setStructuredBuffer("gTileInfo", mpTileIDInfoBuffer);
        mpPhotonGatherVars->setRawBuffer("gIDBuffer", mpIDBuffer);
        mpPhotonGatherVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
        mpPhotonGatherVars->setTexture("gNormalTex", gBuffer->mpGPassFbo->getColorTexture(0));
        mpPhotonGatherVars->setTexture("gPhotonTex", causticsFbo->getColorTexture(0));
        static int groupSize = 16;
        uvec3 dispatchSize(
            (screenSize.x + groupSize - 1) / groupSize,
            (screenSize.y + groupSize - 1) / groupSize, 1);
        pContext->dispatch(mpPhotonGatherState.get(), mpPhotonGatherVars.get(), dispatchSize);
    }

    // Temporal filter
    if (mTemporalFilter)
    {
        static float4x4 lastViewProj;
        static float4x4 lastProj;
        float4x4 thisViewProj = mpCamera->getViewProjMatrix();
        float4x4 thisProj = mpCamera->getProjMatrix();
        float4x4 reproj = lastViewProj *glm::inverse(thisViewProj);
        int2 causticsDim(causticsFbo->getWidth(), causticsFbo->getHeight());
        ConstantBuffer::SharedPtr pPerFrameCB = mpFilterVars["PerFrameCB"];
        pPerFrameCB["causticsDim"] = causticsDim;
        pPerFrameCB["gBufferDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        pPerFrameCB["blendWeight"] = mFilterWeight;
        pPerFrameCB["reprojMatrix"] = reproj;
        pPerFrameCB["invProjMatThis"] = glm::inverse(thisProj);
        pPerFrameCB["invProjMatLast"] = glm::inverse(lastProj);
        pPerFrameCB["normalKernel"] = mTemporalNormalKernel;
        pPerFrameCB["depthKernel"] = mTemporalDepthKernel;
        mpFilterVars->setTexture("causticsTexThis", causticsFbo->getColorTexture(0));
        mpFilterVars->setTexture("causticsTexLast", causticsFboLast->getColorTexture(0));
        mpFilterVars->setTexture("depthTexThis", gBuffer->mpDepthTex);
        mpFilterVars->setTexture("depthTexLast", gBufferLast->mpDepthTex);
        mpFilterVars->setTexture("normalTexThis", gBuffer->mpNormalTex);
        mpFilterVars->setTexture("normalTexLast", gBufferLast->mpNormalTex);
        static int groupSize = 16;
        uvec3 dim((causticsDim.x + groupSize - 1) / groupSize, (causticsDim.y + groupSize - 1) / groupSize, 1);
        pContext->dispatch(mpFilterState.get(), mpFilterVars.get(), dim);
        lastViewProj = thisViewProj;
        lastProj = thisProj;
    }


    // Render output
    if(mDebugMode == ShowRayTracing ||
        mDebugMode == ShowAvgScreenArea ||
        mDebugMode == ShowAvgScreenAreaVariance ||
        mDebugMode == ShowCount ||
        mDebugMode == ShowTotalPhoton ||
        mDebugMode == ShowRayTex)
    {
        pContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
        GraphicsVars* pVars = mpCompositeRTVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["invProj"] = glm::inverse(mpCamera->getProjMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
        pCB["tanHalfFovY"] = tanf(fovY * 0.5f);
        pCB["sampleIndex"] = mSampleIndex++;
        pCB["useDOF"] = mUseDOF;
        pCB["roughThreshold"] = mRoughThreshold;
        pCB["maxDepth"] = mMaxTraceDepth;
        pCB["iorOverride"] = mIOROveride;
        pCB["causticsResRatio"] = mCausticsMapResRatio;
        pCB["gPosKernel"] = mFilterCausticsMap ? mUVKernel : 0.0f;
        pCB["gZKernel"] = mFilterCausticsMap ? mZKernel : 0.0f;
        pCB["gNormalKernel"] = mFilterCausticsMap ? mNormalKernel : 0.0f;
        auto hitVars = mpCompositeRTVars->getHitVars(0);
        for (auto& hitVar : hitVars)
        {
            hitVar->setTexture("gCausticsTex", causticsFbo->getColorTexture(0));
            hitVar->setTexture("gNormalTex", gBuffer->mpGPassFbo->getColorTexture(0));
            hitVar->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
            hitVar["gLinearSampler"] = mpLinearSampler;
            hitVar["gPointSampler"] = mpPointSampler;
        }
        auto rayGenVars = mpCompositeRTVars->getRayGenVars();
        rayGenVars->setTexture("gOutput", mpRtOut);
        mpCompositeRTState->setMaxTraceRecursionDepth(2);
        mpRtRenderer->renderScene(pContext, mpCompositeRTVars, mpCompositeRTState, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1), mpCamera.get());
    }

    {
        mpCompositePass["gDepthTex"] = gBuffer->mpGPassFbo->getDepthStencilTexture();
        mpCompositePass["gNormalTex"] = gBuffer->mpGPassFbo->getColorTexture(0);
        mpCompositePass["gDiffuseTex"] = gBuffer->mpGPassFbo->getColorTexture(1);
        mpCompositePass["gSpecularTex"] = gBuffer->mpGPassFbo->getColorTexture(2);
        mpCompositePass["gPhotonTex"] = causticsFbo->getColorTexture(0);
        mpCompositePass["gRaytracingTex"] = mpRtOut;
        mpCompositePass["gRayTex"] = mpRayDensityTex;
        mpCompositePass["gStatisticsTex"] = mpPhotonCountTex;
        mpCompositePass["gPointSampler"] = mpPointSampler;
        ConstantBuffer::SharedPtr pCompCB = mpCompositePass["PerImageCB"];
        pCompCB["gNumLights"] = mpScene->getLightCount();
        pCompCB["gDebugMode"] = (uint32_t)mDebugMode;
        pCompCB["gInvWvpMat"] = mpCamera->getInvViewProjMatrix();
        pCompCB["gInvPMat"] = glm::inverse(mpCamera->getProjMatrix());
        pCompCB["gCameraPos"] = mpCamera->getPosition(); 
        pCompCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
        pCompCB["dispatchSize"] = int2(mDispatchSize, mDispatchSize);
        pCompCB["gMaxPixelArea"] = mMaxPixelArea; 
        pCompCB["gRayTexScale"] = mRayTexScaleFactor;
        pCompCB["gStatisticsOffset"] = statisticsOffset;
        mpCompositePass->getVars()->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            mpScene->getLight(i)->setIntoProgramVars(mpCompositePass->getVars().get(), pCompCB.get(), "gLightData[" + std::to_string(i) + "]");
        }
        mpCompositePass->execute(pContext, pTargetFbo);
    }
    mFrameCounter++;
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

    RtProgram::SharedPtr photonTraceProgram = mPhotonTraceShaderList.begin()->second.mpPhotonTraceProgram;
#define CAUSTICS_MAP_SIZE 2048
#define TASK_SIZE 4096*4096
    mpRayTaskBuffer = StructuredBuffer::create(mpAnalyseProgram.get(), std::string("gRayTask"), TASK_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPixelInfoBuffer = StructuredBuffer::create(mpUpdateRayDensityProgram.get(), std::string("gPixelInfo"), CAUSTICS_MAP_SIZE* CAUSTICS_MAP_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    //mpPixelInfoBufferDisplay = StructuredBuffer::create(mpUpdateRayDensityProgram.get(), std::string("gPixelInfo"), CAUSTICS_MAP_SIZE * CAUSTICS_MAP_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer = StructuredBuffer::create(photonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), TASK_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer2 = StructuredBuffer::create(photonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), TASK_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpDrawArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gDrawArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg | Resource::BindFlags::ShaderResource);
    mpRayArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gRayArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg);
    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);

    int2 tileDim = getTileDim();
    int avgTileIDCount = 8192;
    mpTileIDInfoBuffer = StructuredBuffer::create(mpAllocateTileProgram[0].get(), std::string("gTileInfo"), tileDim.x * tileDim.y, ResourceBindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    mpIDBuffer = Buffer::create(tileDim.x * tileDim.y * avgTileIDCount * sizeof(uint32_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);
    mpIDCounterBuffer = Buffer::create(sizeof(uint32_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);

    createCausticsMap();

    mpRayDensityTex = Texture::create2D(CAUSTICS_MAP_SIZE, CAUSTICS_MAP_SIZE, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    mpPhotonCountTex = Texture::create1D(width, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

    createGBuffer(width, height, mGBuffer[0]);
    createGBuffer(width, height, mGBuffer[1]);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    Caustics::UniquePtr pRenderer = std::make_unique<Caustics>();
    SampleConfig config;
    config.windowDesc.title = "Caustics";
    config.windowDesc.resizableWindow = true;

    Sample::run(config, pRenderer);
}
