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
//#include "Scene/Model/Model.h"

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
    Gui::Window w(pGui, "Caustics", { 300, 400 }, { 10, 80 });

    w.checkbox("Ray Trace", mRayTrace);

    if (w.button("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileExtensionFilters, filename))
        {
            loadScene(filename, gpFramework->getTargetFbo().get());
            loadShader();
        }
    }

    if (w.button("Load Scene Settings"))
    {
        std::string filename;
        if (openFileDialog(settingFilter, filename))
        {
            loadSceneSetting(filename);
        }
    }
    if (w.button("Save Scene Settings"))
    {
        std::string filename;
        if (saveFileDialog(settingFilter, filename))
        {
            saveSceneSetting(filename);
        }
    }
    if (w.button("Update Shader"))
    {
        loadShader();
    }

    //if (w.group("Display", true))
    {
        auto g = w.group("Display", true);
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
            debugModeList.push_back({ 14, "Ray count Mipmap" });
            debugModeList.push_back({ 15, "Photon Density" });
            debugModeList.push_back({ 16, "Small Photon Color" });
            debugModeList.push_back({ 17, "Small Photon Count" });
            g.dropdown("Composite mode", debugModeList, (uint32_t&)mDebugMode);
        }
        g.var("Max Pixel Value", mMaxPixelArea, 0.0f, 1000000000.f, 5.f);
        g.var("Max Photon Count", mMaxPhotonCount, 0.f, 1000000000.f, 5.f);
        g.var("Ray Count Mipmap", mRayCountMipIdx, 0, 11);
        //pGui->addFloatVar("Max Pixel Value", mMaxPixelArea, 0, 1000000000, 5.f);
        //pGui->addFloatVar("Max Photon Count", mMaxPhotonCount, 0, 1000000000, 5.f);
        //pGui->addIntVar("Ray Count Mipmap", mRayCountMipIdx, 0, 11);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 1, "x1" });
            debugModeList.push_back({ 2, "x2" });
            debugModeList.push_back({ 4, "x4" });
            debugModeList.push_back({ 8, "x8" });
            debugModeList.push_back({ 16, "x16" });
            g.dropdown("Ray Tex Scale", debugModeList, (uint32_t&)mRayTexScaleFactor);
        }
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Photon Trace", true))
    {
        auto g = w.group("Photon Trace", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Fixed Resolution" });
            debugModeList.push_back({ 1, "Adaptive Resolution" });
            debugModeList.push_back({ 3, "Fast Adaptive Resolution" });
            debugModeList.push_back({ 2, "None" });
            g.dropdown("Trace Type", debugModeList, (uint32_t&)mTraceType);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Ray Differential" });
            debugModeList.push_back({ 1, "Ray Cone" });
            debugModeList.push_back({ 2, "None" });
            g.dropdown("Ray Type", debugModeList, (uint32_t&)mPhotonTraceMacro);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 64, "64" });
            debugModeList.push_back({ 128, "128" });
            debugModeList.push_back({ 256, "256" });
            debugModeList.push_back({ 512, "512" });
            debugModeList.push_back({ 1024, "1024" });
            debugModeList.push_back({ 2048, "2048" });
            g.dropdown("Dispatch Size", debugModeList, (uint32_t&)mDispatchSize);
        }
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Avg Square" });
            debugModeList.push_back({ 1, "Avg Length" });
            debugModeList.push_back({ 2, "Max Square" });
            debugModeList.push_back({ 3, "Exact Area" });
            g.dropdown("Area Type", debugModeList, (uint32_t&)mAreaType);
        }
        g.var("Intensity", mIntensity, 0.f, 10.f, 0.1f);
        g.var("Emit size", mEmitSize, 0.f, 1000.f, 1.f);
        g.var("Rough Threshold", mRoughThreshold, 0.f, 1.f, 0.01f);
        g.var("Max Trace Depth", mMaxTraceDepth, 0, 30);
        g.var("IOR Override", mIOROveride, 0.f, 3.f, 0.01f);
        g.checkbox("ID As Color", mColorPhoton);
        g.var("Photon ID Scale", mPhotonIDScale);
        g.var("Min Trace Luminance", mTraceColorThreshold, 0.f, 10.f,0.005f);
        g.var("Min Cull Luminance", mCullColorThreshold, 0.f, 10000.f, 0.01f);
        g.checkbox("Fast Photon Path", mFastPhotonPath);
        g.var("Max Pixel Radius", mMaxPhotonPixelRadius, 0.f, 5000.f, 1.f);
        g.var("Fast Pixel Radius", mFastPhotonPixelRadius, 0.f, 5000.f, 1.f);
        g.var("Fast Draw Count", mFastPhotonDrawCount, 0.f, 50000.f, 0.1f);
        g.var("Color Compress Scale", mSmallPhotonCompressScale, 0.f, 5000.f, 1.f);
        g.checkbox("Shrink Color Payload", mShrinkColorPayload);
        g.checkbox("Shrink Ray Diff Payload", mShrinkRayDiffPayload);
        g.checkbox("Update Photon", mUpdatePhoton);
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Adaptive Resolution", true))
    {
        auto g = w.group("Adaptive Resolution", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Random" });
            debugModeList.push_back({ 1, "Grid" });
            g.dropdown("Sample Placement", debugModeList, (uint32_t&)mSamplePlacement);
        }
        g.var("Luminance Threshold", mPixelLuminanceThreshold, 0.01f, 10.0f, 0.01f);
        g.var("Photon Size Threshold", mMinPhotonPixelSize, 1.f, 1000.0f, 0.1f);
        g.var("Smooth Weight", mSmoothWeight, 0.f, 10.0f, 0.001f);
        g.var("Proportional Gain", mUpdateSpeed, 0.f, 1.f, 0.01f);
        g.var("Variance Gain", mVarianceGain, 0.f, 10.f, 0.0001f);
        g.var("Derivative Gain", mDerivativeGain, -10.f, 10.f, 0.1f);
        g.var("Max Task Per Pixel", mMaxTaskCountPerPixel, 1.0f, 1000000.f, 5.f);
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Smooth Photon", false))
    {
        auto g = w.group("Smooth Photon", true);
        g.checkbox("Remove Isolated Photon", mRemoveIsolatedPhoton);
        g.checkbox("Enable Median Filter", mMedianFilter);
        g.var("Normal Threshold", mNormalThreshold, 0.01f, 1.0f, 0.01f);
        g.var("Distance Threshold", mDistanceThreshold, 0.1f, 100.0f, 0.1f);
        g.var("Planar Threshold", mPlanarThreshold, 0.01f, 10.0f, 0.1f);
        g.var("Trim Direction Threshold", trimDirectionThreshold, 0.f, 1.f);
        g.var("Min Neighbour Count", mMinNeighbourCount, 0, 8);
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Photon Splatting", true))
    {
        auto g = w.group("Photon Splatting", true);
        {
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 0, "Scatter" });
            debugModeList.push_back({ 1, "Gather" });
            debugModeList.push_back({ 2, "None" });
            g.dropdown("Density Estimation", debugModeList, (uint32_t&)mScatterOrGather);
        }
        g.var("Splat size", mSplatSize, 0.f, 100.f, 0.01f);
        g.var("Kernel Power", mKernelPower, 0.01f, 10.f, 0.01f);

        //if(pGui->beginGroup("Scatter Parameters", false))
        {
            auto g2 = w.group("Scatter Parameters", true);
            g2.var("Z Tolerance", mZTolerance, 0.001f, 1.f, 0.001f);
            g2.var("Scatter Normal Threshold", mScatterNormalThreshold, 0.01f, 1.0f, 0.01f);
            g2.var("Scatter Distance Threshold", mScatterDistanceThreshold, 0.1f, 10.0f, 0.1f);
            g2.var("Scatter Planar Threshold", mScatterPlanarThreshold, 0.01f, 10.0f, 0.1f);
            g2.var("Max Anisotropy", mMaxAnisotropy, 1.f, 100.f, 0.1f);
            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({ 0, "Quad" });
                debugModeList.push_back({ 1, "Sphere" });
                g2.dropdown("Photon Geometry", debugModeList, (uint32_t&)mScatterGeometry);
            }
            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({ 0, "Kernel" });
                debugModeList.push_back({ 1, "Solid" });
                debugModeList.push_back({ 2, "Shaded" });
                g2.dropdown("Photon Display Mode", debugModeList, (uint32_t&)mPhotonDisplayMode);
            }

            {
                Gui::DropdownList debugModeList;
                debugModeList.push_back({ 0, "Anisotropic" });
                debugModeList.push_back({ 1, "Isotropic" });
                debugModeList.push_back({ 2, "Photon Mesh" });
                debugModeList.push_back({ 3, "Screen Dot" });
                debugModeList.push_back({ 4, "Screen Dot With Color" });
                g2.dropdown("Photon mode", debugModeList, (uint32_t&)mPhotonMode);
            }
            //pGui->endGroup();
        }

        //if(pGui->beginGroup("Gather Parameters", false))
        {
            auto g2 = w.group("Gather Parameters", true);
            g2.var("Gather Depth Radius", mDepthRadius, 0.f, 10.f, 0.01f);
            g2.var("Gather Min Color", mMinGatherColor, 0.f, 2.f, 0.001f);
            g2.checkbox("Gather Show Tile Count", mShowTileCount);
            g2.var("Gather Tile Count Scale", mTileCountScale, 0, 1000);
            //pGui->endGroup();
        }

        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Temporal Filter", true))
    {
        auto g = w.group("Temporal Filter", true);
        g.checkbox("Enable Temporal Filter", mTemporalFilter);
        g.var("Filter Weight", mFilterWeight, 0.0f, 1.0f, 0.001f);
        g.var("Jitter", mJitter, 0.f, 10.f, 0.01f);
        g.var("Jitter Power", mJitterPower, 0.f, 200.f, 0.01f);
        g.var("Temporal Normal Strength", mTemporalNormalKernel, 0.0001f, 1000.f, 0.01f);
        g.var("Temporal Depth Strength", mTemporalDepthKernel, 0.0001f, 1000.f, 0.01f);
        g.var("Temporal Color Strength", mTemporalColorKernel, 0.0001f, 1000.f, 0.01f);
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Spacial Filter", true))
    {
        auto g = w.group("Spacial Filter", true);
        g.checkbox("Enable Spatial Filter", mSpacialFilter);
        g.var("A trous Pass", mSpacialPasses, 0, 10);
        g.var("Spacial Normal Strength", mSpacialNormalKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Depth Strength", mSpacialDepthKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Color Strength", mSpacialColorKernel, 0.0001f, 100.f, 0.01f);
        g.var("Spacial Screen Kernel", mSpacialScreenKernel, 0.0001f, 100.f, 0.01f);
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Composite", true))
    {
        auto g = w.group("Composite", true);
        {
            int oldResRatio = mCausticsMapResRatio;
            Gui::DropdownList debugModeList;
            debugModeList.push_back({ 1, "x 1" });
            debugModeList.push_back({ 2, "x 1/2" });
            debugModeList.push_back({ 4, "x 1/4" });
            debugModeList.push_back({ 8, "x 1/8" });
            g.dropdown("Caustics Resolution", debugModeList, (uint32_t&)mCausticsMapResRatio);
            if (oldResRatio != mCausticsMapResRatio)
            {
                createCausticsMap();
            }
        }
        g.checkbox("Filter Caustics Map", mFilterCausticsMap);
        g.var("UV Kernel", mUVKernel, 0.0f, 1000.f, 0.1f);
        g.var("Depth Kernel", mZKernel, 0.0f, 1000.f, 0.1f);
        g.var("Normal Kernel", mNormalKernel, 0.0f, 1000.f, 0.1f);
        //pGui->endGroup();
    }
    mLightDirection = vec3(
        cos(mLightAngle.x) * sin(mLightAngle.y),
        cos(mLightAngle.y),
        sin(mLightAngle.x) * sin(mLightAngle.y));
    //if (pGui->beginGroup("Light", true))
    {
        auto g = w.group("Light", true);
        g.var("Light Angle", mLightAngle, -FLT_MAX, FLT_MAX, 0.01f);
        if (mpScene)
        {
            auto light0 = dynamic_cast<DirectionalLight*>(mpScene->getLight(0).get());
            light0->setWorldDirection(mLightDirection);
        }
        g.var("Light Angle Speed", mLightAngleSpeed, -FLT_MAX, FLT_MAX, 0.001f);
        mLightAngle += mLightAngleSpeed*0.01f;
        //pGui->endGroup();
    }

    //if (pGui->beginGroup("Camera"))
    {
        auto g = w.group("Camera", true);
        mpCamera->renderUI(pGui);
        //pGui->endGroup();
    }
}

void Caustics::loadScene(const std::string& filename, const Fbo* pTargetFbo)
{
    //mpScene = RtScene::loadFromFile(filename, RtBuildFlags::None, Model::LoadFlags::None);
    mpScene = Scene::create(filename);
    if (!mpScene) return;

    mpQuad = Scene::create("Caustics/quad.obj");
    mpSphere = Scene::create("Caustics/sphere.obj");


    //Model::SharedPtr pModel = mpScene->getModel(0);
    auto pModel = mpScene->getMesh(0);
    auto bbox = mpScene->getSceneBounds();
    float radius = glm::length(bbox.getSize());// pModel->getRadius();

    mpCamera = mpScene->getCamera();// mpScene->getActiveCamera();
    assert(mpCamera);

    mCamController = FirstPersonCameraController::create(mpCamera);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    Sampler::SharedPtr pSampler = Sampler::create(samplerDesc);
    //pModel->bindSamplerToMaterials(pSampler);
    mpScene->bindSamplerToMaterials(pSampler);

    // Update the controllers
    mCamController->setCameraSpeed(radius * 0.2f);
    auto sceneBBox = mpScene->getSceneBounds();
    float sceneRadius = sceneBBox.getSize().length() * 0.5f;
    //mCamController.setModelParams(mpScene->getCenter(), sceneRadius, sceneRadius);
    float nearZ = 1.f;// std::max(0.1f, pModel->getRadius() / 750.0f);
    float farZ = 1000.f;// radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    mpGaussianKernel = Texture::createFromFile("Caustics/gaussian.png", true, false);
    mpUniformNoise = Texture::createFromFile("Caustics/uniform.png", true, false);
}



Caustics::PhotonTraceShader Caustics::getPhotonTraceShader()
{
    uint flag = photonMacroToFlags();
    auto pIter = mPhotonTraceShaderList.find(flag);
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
        switch (mTraceType)
        {
        case Caustics::TRACE_FIXED:
            desc.addDefine("TRACE_FIXED", "1");
            break;
        case Caustics::TRACE_ADAPTIVE:
            desc.addDefine("TRACE_ADAPTIVE", "1");
            break;
        case Caustics::TRACE_NONE:
            desc.addDefine("TRACE_NONE", "1");
            break;
        case Caustics::TRACE_ADAPTIVE_RAY_MIP_MAP:
            desc.addDefine("TRACE_ADAPTIVE_RAY_MIP_MAP", "1");
            break;
        default:
            break;
        }
        if (mFastPhotonPath)
        {
            desc.addDefine("FAST_PHOTON_PATH", "1");
        }
        if (mShrinkColorPayload)
        {
            desc.addDefine("SMALL_COLOR", "1");
        }
        if (mShrinkRayDiffPayload)
        {
            desc.addDefine("SMALL_RAY_DIFFERENTIAL", "1");
        }
        if (mUpdatePhoton)
        {
            desc.addDefine("UPDATE_PHOTON", "1");
        }

        uint payLoadSize = 80U;
        if (mShrinkColorPayload)
            payLoadSize -= 12U;
        if (mShrinkRayDiffPayload)
            payLoadSize -= 24U;

        auto pPhotonTraceProgram = RtProgram::create(desc, payLoadSize, 8U);
        auto pPhotonTraceState = RtState::create();
        pPhotonTraceState->setProgram(pPhotonTraceProgram);
        auto pPhotonTraceVars = RtProgramVars::create(pPhotonTraceProgram, mpScene);
        mPhotonTraceShaderList[flag] = { pPhotonTraceProgram , pPhotonTraceVars ,pPhotonTraceState };
    }
    return mPhotonTraceShaderList[flag];
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

    mpSmallPhotonTex = Texture::create2D(dim.x, dim.y, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

    auto pPhotonMapTex = Texture::create2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    auto depthTex = Texture::create2D(dim.x, dim.y, ResourceFormat::D24UnormS8, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
    mpCausticsFbo[0] = Fbo::create({ pPhotonMapTex }, depthTex);

    pPhotonMapTex = Texture::create2D(dim.x, dim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    mpCausticsFbo[1] = Fbo::create({ pPhotonMapTex }, depthTex);
}

void Caustics::createGBuffer(int width, int height, GBuffer& gbuffer)
{
    gbuffer.mpDepthTex = Texture::create2D(width, height, ResourceFormat::D24UnormS8, 1, 1, nullptr, Resource::BindFlags::DepthStencil | Resource::BindFlags::ShaderResource);
    gbuffer.mpNormalTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    gbuffer.mpDiffuseTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    gbuffer.mpSpecularTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    gbuffer.mpGPassFbo = Fbo::create({ gbuffer.mpNormalTex , gbuffer.mpDiffuseTex ,gbuffer.mpSpecularTex }, gbuffer.mpDepthTex);//Fbo::create2D(width, height, ResourceFormat::RGBA16Float, ResourceFormat::D24UnormS8);
}

int2 Caustics::getTileDim() const
{
    int2 tileDim;
    tileDim.x = (mpRtOut->getWidth() / mCausticsMapResRatio + mTileSize.x - 1) / mTileSize.x;
    tileDim.y = (mpRtOut->getHeight() / mCausticsMapResRatio + mTileSize.y - 1) / mTileSize.y;
    return tileDim;
}

float Caustics::resolutionFactor()
{
    float2 res(mpRtOut->getWidth(), mpRtOut->getHeight());
    float2 refRes(1920, 1080);
    return glm::length(res) / glm::length(refRes);
}

uint Caustics::photonMacroToFlags()
{
    uint flags = 0;
    flags |= (1 << mPhotonTraceMacro); // 3 bits
    flags |= ((1 << mTraceType) << 3); // 4 bits
    if (mFastPhotonPath)
        flags |= (1 << 7); // 1 bits
    if (mShrinkColorPayload)
        flags |= (1 << 8); // 1 bits
    if (mShrinkRayDiffPayload)
        flags |= (1 << 9); // 1 bits
    if (mUpdatePhoton)
        flags |= (1 << 10); // 1 bits
    return flags;
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

    // generate ray count tex
    mpGenerateRayCountProgram = ComputeProgram::createFromFile("GenerateRayCountMipmap.cs.hlsl", "generateMip0");
    mpGenerateRayCountState = ComputeState::create();
    mpGenerateRayCountState->setProgram(mpGenerateRayCountProgram);
    mpGenerateRayCountVars = ComputeVars::create(mpGenerateRayCountProgram.get());

    // generate ray count mip tex
    mpGenerateRayCountMipProgram = ComputeProgram::createFromFile("GenerateRayCountMipmap.cs.hlsl", "generateMipLevel");
    mpGenerateRayCountMipState = ComputeState::create();
    mpGenerateRayCountMipState->setProgram(mpGenerateRayCountMipProgram);
    mpGenerateRayCountMipVars = ComputeVars::create(mpGenerateRayCountMipProgram.get());

    // smooth photon
    mpSmoothProgram = ComputeProgram::createFromFile("SmoothPhoton.cs.hlsl", "main");
    mpSmoothState = ComputeState::create();
    mpSmoothState->setProgram(mpSmoothProgram);
    mpSmoothVars = ComputeVars::create(mpSmoothProgram.get());

    // allocate tile
    const char* shaderEntries[] = { "OrthogonalizePhoton", "CountTilePhoton","AllocateMemory","StoreTilePhoton" };
    for (int i = 0; i < GATHER_PROCESSING_SHADER_COUNT; i++)
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

    // temporal filter
    mpFilterProgram = ComputeProgram::createFromFile("TemporalFilter.cs.hlsl", "main");
    mpFilterState = ComputeState::create();
    mpFilterState->setProgram(mpFilterProgram);
    mpFilterVars = ComputeVars::create(mpFilterProgram.get());

    // spacial filter
    mpSpacialFilterProgram = ComputeProgram::createFromFile("SpacialFilter.cs.hlsl", "main");
    mpSpacialFilterState = ComputeState::create();
    mpSpacialFilterState->setProgram(mpSpacialFilterProgram);
    mpSpacialFilterVars = ComputeVars::create(mpSpacialFilterProgram.get());

    //mpRtRenderer = RtSceneRenderer::create(mpScene);

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
    loadSceneSetting("Data/init.ini");
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
    if (mTraceType != TRACE_NONE)
    {
        pContext->clearUAV(mpSmallPhotonTex->getUAV().get(), uvec4(0, 0, 0, 0));
        auto photonTraceShader = getPhotonTraceShader();
        //setPhotonTracingCommonVariable(photonTraceShader);
        GraphicsVars* pVars = photonTraceShader.mpPhotonTraceVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        float2 r = getRandomPoint(mFrameCounter) * 2.0f - 1.0f;
        float2 sign(r.x > 0 ? 1 : -1, r.y > 0 ? 1 : -1);
        float2 randomOffset = sign * float2(pow(abs(r.x), mJitterPower), pow(abs(r.y), mJitterPower)) * mJitter;
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(mpRtOut->getWidth(), mpRtOut->getHeight());
        pCB["emitSize"] = mEmitSize;
        pCB["roughThreshold"] = mRoughThreshold;
        pCB["randomOffset"] = mTemporalFilter ? randomOffset : float2(0, 0);
        pCB["rayTaskOffset"] = mDispatchSize * mDispatchSize;
        pCB["coarseDim"] = uint2(mDispatchSize, mDispatchSize);
        pCB["maxDepth"] = mMaxTraceDepth;
        pCB["iorOverride"] = mIOROveride;
        pCB["colorPhotonID"] = (uint32_t)mColorPhoton;
        pCB["photonIDScale"] = mPhotonIDScale;
        pCB["traceColorThreshold"] = mTraceColorThreshold * (512 * 512) / (mDispatchSize * mDispatchSize);
        pCB["cullColorThreshold"] = mCullColorThreshold / 255;
        pCB["gAreaType"] = (uint32_t)mAreaType;
        pCB["gIntensity"] = mIntensity / 1000;
        pCB["gSplatSize"] = mSplatSize;
        pCB["updatePhoton"] = (uint32_t)mUpdatePhoton;
        pCB["gMinDrawCount"] = mFastPhotonDrawCount;
        pCB["gMinScreenRadius"] = mFastPhotonPixelRadius * resolutionFactor();
        pCB["gMaxScreenRadius"] = mMaxPhotonPixelRadius * resolutionFactor();
        pCB["gMipmap"] = int(log(mDispatchSize) / log(2));
        pCB["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
        pCB["cameraPos"] = mpCamera->getPosition();
        auto rayGenVars = photonTraceShader.mpPhotonTraceVars->getRayGenVars();
        rayGenVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
        rayGenVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        rayGenVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
        rayGenVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
        rayGenVars->setTexture("gUniformNoise", mpUniformNoise);
        rayGenVars->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
        rayGenVars->setStructuredBuffer("gRayCountQuadTree", mpRayCountQuadTree);
        rayGenVars->setTexture("gRayDensityTex", mpRayDensityTex);
        rayGenVars->setTexture("gSmallPhotonBuffer", mpSmallPhotonTex);
        rayGenVars->setTexture("gPhotonTexture", causticsFboLast->getColorTexture(0));
        auto hitVars = photonTraceShader.mpPhotonTraceVars->getHitVars(0);
        for (auto& hitVar : hitVars)
        {
            hitVar->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
            hitVar->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
            hitVar->setStructuredBuffer("gDrawArgument", mpDrawArgumentBuffer);
            hitVar->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
        }
        photonTraceShader.mpPhotonTraceState->setMaxTraceRecursionDepth(1);
        uvec3 resolution = mTraceType == TRACE_FIXED ? uvec3(mDispatchSize, mDispatchSize, 1) : uvec3(2048, 4096, 1);
        //mpRtRenderer->renderScene(pContext, photonTraceShader.mpPhotonTraceVars, photonTraceShader.mpPhotonTraceState, resolution, mpCamera.get());
        mpScene->raytrace(pContext, photonTraceShader.mpPhotonTraceState, photonTraceShader.mpPhotonTraceVars, resolution);
    }

    // analysis output
    if ( mUpdatePhoton)
    {
        if(mTraceType == TRACE_ADAPTIVE || mTraceType == TRACE_ADAPTIVE_RAY_MIP_MAP)
        {
            ConstantBuffer::SharedPtr pPerFrameCB = mpUpdateRayDensityVars["PerFrameCB"];
            pPerFrameCB["coarseDim"] = int2(mDispatchSize, mDispatchSize);
            pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
            pPerFrameCB["smoothWeight"] = mSmoothWeight;
            pPerFrameCB["maxTaskPerPixel"] = (int)mMaxTaskCountPerPixel;
            pPerFrameCB["updateSpeed"] = mUpdateSpeed;
            pPerFrameCB["varianceGain"] = mVarianceGain;
            pPerFrameCB["derivativeGain"] = mDerivativeGain;
            mpUpdateRayDensityVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
            mpUpdateRayDensityVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
            mpUpdateRayDensityVars->setTexture("gRayDensityTex", mpRayDensityTex);
            static int groupSize = 16;
            pContext->dispatch(mpUpdateRayDensityState.get(), mpUpdateRayDensityVars.get(), uvec3(mDispatchSize / groupSize, mDispatchSize / groupSize, 1));
        }

        if (mTraceType == TRACE_ADAPTIVE)
        {
            ConstantBuffer::SharedPtr pPerFrameCB = mpAnalyseVars["PerFrameCB"];
            glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
            pPerFrameCB["viewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
            pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
            pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
            pPerFrameCB["normalThreshold"] = mNormalThreshold;
            pPerFrameCB["distanceThreshold"] = mDistanceThreshold;
            pPerFrameCB["planarThreshold"] = mPlanarThreshold;
            pPerFrameCB["samplePlacement"] = (uint32_t)mSamplePlacement;
            pPerFrameCB["pixelLuminanceThreshold"] = mPixelLuminanceThreshold;
            pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
            static float2 offset(0.5, 0.5);
            static float speed = 0.0f;
            pPerFrameCB["randomOffset"] = offset;
            offset += speed;
            mpAnalyseVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
            mpAnalyseVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
            mpAnalyseVars->setStructuredBuffer("gRayTask", mpRayTaskBuffer);
            mpAnalyseVars->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
            mpAnalyseVars->setTexture("gDepthTex", gBuffer->mpGPassFbo->getDepthStencilTexture());
            mpAnalyseVars->setTexture("gRayDensityTex", mpRayDensityTex);
            int2 groupSize(32, 16);
            pContext->dispatch(mpAnalyseState.get(), mpAnalyseVars.get(), uvec3(mDispatchSize / groupSize.x, mDispatchSize / groupSize.y, 1));
        }
        else if (mTraceType == TRACE_ADAPTIVE_RAY_MIP_MAP)
        {
            int startMipLevel = int(log(mDispatchSize) / log(2)) - 1;
            {
                ConstantBuffer::SharedPtr pPerFrameCB = mpGenerateRayCountVars["PerFrameCB"];
                pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
                pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
                pPerFrameCB["mipLevel"] = startMipLevel;
                mpGenerateRayCountVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
                mpGenerateRayCountVars->setTexture("gRayDensityTex", mpRayDensityTex);
                mpGenerateRayCountVars->setStructuredBuffer("gRayCountQuadTree", mpRayCountQuadTree);
                int2 groupSize(8, 8);
                uvec3 blockCount(mDispatchSize / groupSize.x / 2, mDispatchSize / groupSize.y / 2, 1);
                pContext->dispatch(mpGenerateRayCountState.get(), mpGenerateRayCountVars.get(), blockCount);
            }

            for (int mipLevel = startMipLevel - 1, dispatchSize = mDispatchSize / 4; mipLevel >= 0; mipLevel--, dispatchSize >>= 1)
            {
                ConstantBuffer::SharedPtr pPerFrameCB = mpGenerateRayCountMipVars["PerFrameCB"];
                pPerFrameCB["taskDim"] = int2(mDispatchSize, mDispatchSize);
                pPerFrameCB["screenDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
                pPerFrameCB["mipLevel"] = mipLevel;
                mpGenerateRayCountMipVars->setStructuredBuffer("gRayArgument", mpRayArgumentBuffer);
                mpGenerateRayCountMipVars->setTexture("gRayDensityTex", mpRayDensityTex);
                mpGenerateRayCountMipVars->setStructuredBuffer("gRayCountQuadTree", mpRayCountQuadTree);
                int2 groupSize(8, 8);
                uvec3 blockCount((dispatchSize + groupSize.x - 1) / groupSize.x, (dispatchSize + groupSize.y - 1) / groupSize.y, 1);
                pContext->dispatch(mpGenerateRayCountMipState.get(), mpGenerateRayCountMipVars.get(), blockCount);
            }
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
        pPerFrameCB["minPhotonPixelSize"] = mMinPhotonPixelSize * resolutionFactor();
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
    if(mScatterOrGather == DENSITY_ESTIMATION_SCATTER)
    {
        pContext->clearRtv(causticsFbo->getColorTexture(0)->getRTV().get(), vec4(0, 0, 0, 0));
        glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
        glm::mat4 invP = glm::inverse(mpCamera->getProjMatrix());
        ConstantBuffer::SharedPtr pPerFrameCB = mpPhotonScatterVars["PerFrameCB"];
        pPerFrameCB["gWorldMat"] = glm::mat4();
        pPerFrameCB["gWvpMat"] = wvp;
        pPerFrameCB["gInvProjMat"] = invP;
        pPerFrameCB["gEyePosW"] = mpCamera->getPosition();
        pPerFrameCB["gSplatSize"] = mSplatSize;
        pPerFrameCB["gPhotonMode"] = (uint)mPhotonMode;
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
            scatterState->setVao(mpQuad->getVao());
        else
            scatterState->setVao(mpSphere->getVao());
        scatterState->setFbo(causticsFbo);
        if (mPhotonMode == PHOTON_MODE_PHOTON_MESH)
        {
            pContext->drawIndexedInstanced(scatterState.get(), mpPhotonScatterVars.get(), 6, mDispatchSize* mDispatchSize, 0, 0, 0);
        }
        else
        {
            pContext->drawIndexedIndirect(scatterState.get(), mpPhotonScatterVars.get(), 100, mpDrawArgumentBuffer.get(), 0, nullptr, 0);
        }
    }
    else if (mScatterOrGather == DENSITY_ESTIMATION_GATHER)
    {
        int2 tileDim = getTileDim();
        int dimX, dimY;
        if (mTraceType == TRACE_FIXED)
        {
            dimX = mDispatchSize;
            dimY = mDispatchSize;
        }
        else
        {
            int photonCount = MAX_PHOTON_COUNT;
            int sqrtCount = int(sqrt(photonCount));
            dimX = dimY = sqrtCount;
        }
        int blockSize = 32;
        uvec3 dispatchDim[] = {
            uvec3((dimX + blockSize - 1) / blockSize,   (dimY + blockSize - 1) / blockSize, 1),
            uvec3((dimX + blockSize - 1) / blockSize,   (dimY + blockSize - 1) / blockSize, 1),
            uvec3((tileDim.x + mTileSize.x - 1) / mTileSize.x,(tileDim.y + mTileSize.y - 1) / mTileSize.y,1),
            uvec3((dimX + blockSize - 1) / blockSize,   (dimY + blockSize - 1) / blockSize, 1)
        };
        // build tile data
        int2 screenSize(mpRtOut->getWidth() / mCausticsMapResRatio, mpRtOut->getHeight() / mCausticsMapResRatio);
        for (int i = 0; i < GATHER_PROCESSING_SHADER_COUNT; i++)
        {
            auto vars = mpAllocateTileVars[i];
            auto states = mpAllocateTileState[i];
            ConstantBuffer::SharedPtr pPerFrameCB = vars["PerFrameCB"];
            glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
            pPerFrameCB["gViewProjMat"] = wvp;// mpCamera->getViewProjMatrix();
            pPerFrameCB["screenDim"] = screenSize;
            pPerFrameCB["tileDim"] = tileDim;
            pPerFrameCB["gSplatSize"] = mSplatSize;
            pPerFrameCB["minColor"] = mMinGatherColor;
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
        uvec3 dispatchSize(
            (screenSize.x + mTileSize.x - 1) / mTileSize.x,
            (screenSize.y + mTileSize.y - 1) / mTileSize.y, 1);
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
        pPerFrameCB["colorKernel"] = mTemporalColorKernel;
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

    // Spacial filter
    if (mSpacialFilter)
    {
        for (int i = 0; i < mSpacialPasses; i++)
        {
            int2 causticsDim(causticsFbo->getWidth(), causticsFbo->getHeight());
            ConstantBuffer::SharedPtr pPerFrameCB = mpSpacialFilterVars["PerFrameCB"];
            pPerFrameCB["causticsDim"] = causticsDim;
            pPerFrameCB["gBufferDim"] = int2(mpRtOut->getWidth(), mpRtOut->getHeight());
            pPerFrameCB["normalKernel"] = mSpacialNormalKernel;
            pPerFrameCB["depthKernel"] = mSpacialDepthKernel;
            pPerFrameCB["colorKernel"] = mSpacialColorKernel;
            pPerFrameCB["screenKernel"] = mSpacialScreenKernel;
            pPerFrameCB["passID"] = i;
            pPerFrameCB["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
            mpSpacialFilterVars->setTexture("causticsTexThis", causticsFbo->getColorTexture(0));
            mpSpacialFilterVars->setTexture("depthTexThis", gBuffer->mpDepthTex);
            mpSpacialFilterVars->setTexture("normalTexThis", gBuffer->mpNormalTex);
            mpSpacialFilterVars->setTexture("smallPhotonTex", mpSmallPhotonTex);
            static int groupSize = 16;
            uvec3 dim((causticsDim.x + groupSize - 1) / groupSize, (causticsDim.y + groupSize - 1) / groupSize, 1);
            pContext->dispatch(mpSpacialFilterState.get(), mpSpacialFilterVars.get(), dim);
        }
    }

    // Render output
    if(mDebugMode == ShowRayTracing ||
        mDebugMode == ShowAvgScreenArea ||
        mDebugMode == ShowAvgScreenAreaVariance ||
        mDebugMode == ShowCount ||
        mDebugMode == ShowTotalPhoton ||
        mDebugMode == ShowRayTex ||
        mDebugMode == ShowRayCountMipmap ||
        mDebugMode == ShowPhotonDensity ||
        mDebugMode == ShowSmallPhoton ||
        mDebugMode == ShowSmallPhotonCount)
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
        //mpRtRenderer->renderScene(pContext, mpCompositeRTVars, mpCompositeRTState, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1), mpCamera.get());
        mpScene->raytrace(pContext, mpCompositeRTState, mpCompositeRTVars, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1));
    }

    {
        mpCompositePass["gDepthTex"] = gBuffer->mpGPassFbo->getDepthStencilTexture();
        mpCompositePass["gNormalTex"] = gBuffer->mpGPassFbo->getColorTexture(0);
        mpCompositePass["gDiffuseTex"] = gBuffer->mpGPassFbo->getColorTexture(1);
        mpCompositePass["gSpecularTex"] = gBuffer->mpGPassFbo->getColorTexture(2);
        mpCompositePass["gPhotonTex"] = causticsFbo->getColorTexture(0);
        mpCompositePass["gRayCountQuadTree"] = mpRayCountQuadTree;
        mpCompositePass["gRaytracingTex"] = mpRtOut;
        mpCompositePass["gRayTex"] = mpRayDensityTex;
        mpCompositePass["gStatisticsTex"] = mpPhotonCountTex;
        mpCompositePass["gSmallPhotonTex"] = mpSmallPhotonTex;
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
        pCompCB["gMaxPhotonCount"] = mMaxPhotonCount;
        pCompCB["gRayTexScale"] = mRayTexScaleFactor;
        pCompCB["gStatisticsOffset"] = statisticsOffset;
        pCompCB["gRayCountMip"] = mRayCountMipIdx;
        pCompCB["gSmallPhotonColorScale"] = mSmallPhotonCompressScale;
        mpCompositePass->getVars()->setStructuredBuffer("gPixelInfo", mpPixelInfoBuffer);
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            //mpScene->getLight(i)->setIntoProgramVars(mpCompositePass->getVars().get(), pCompCB.get(), "gLightData[" + std::to_string(i) + "]");
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
        mCamController->update();
        if (mRayTrace)
            renderRT(pRenderContext, pTargetFbo);
        else
            mpRasterPass->renderScene(pRenderContext, pTargetFbo);
    }

    TextRenderer::render(pRenderContext, gpFramework->getFrameRate().getMsg(), pTargetFbo, { 20, 20 });
}

bool Caustics::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mCamController->onKeyEvent(keyEvent))
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
    return mCamController->onMouseEvent(mouseEvent);
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
    mpRayTaskBuffer = StructuredBuffer::create(mpAnalyseProgram.get(), std::string("gRayTask"), MAX_PHOTON_COUNT, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPixelInfoBuffer = StructuredBuffer::create(mpUpdateRayDensityProgram.get(), std::string("gPixelInfo"), MAX_CAUSTICS_MAP_SIZE* MAX_CAUSTICS_MAP_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    //mpPixelInfoBufferDisplay = StructuredBuffer::create(mpUpdateRayDensityProgram.get(), std::string("gPixelInfo"), CAUSTICS_MAP_SIZE * CAUSTICS_MAP_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer = StructuredBuffer::create(photonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), MAX_PHOTON_COUNT, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer2 = StructuredBuffer::create(photonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), MAX_PHOTON_COUNT, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpDrawArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gDrawArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg | Resource::BindFlags::ShaderResource);
    mpRayArgumentBuffer = StructuredBuffer::create(mpDrawArgumentProgram.get(), std::string("gRayArgument"), 1, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::IndirectArg);
    mpRayCountQuadTree = StructuredBuffer::create(mpGenerateRayCountProgram.get(), std::string("gRayCountQuadTree"), MAX_CAUSTICS_MAP_SIZE * MAX_CAUSTICS_MAP_SIZE * 2, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);

    int2 tileDim(
        (mpRtOut->getWidth()  + mTileSize.x - 1) / mTileSize.x,
        (mpRtOut->getHeight() + mTileSize.y - 1) / mTileSize.y);
    int avgTileIDCount = 63356;
    mpTileIDInfoBuffer = StructuredBuffer::create(mpAllocateTileProgram[0].get(), std::string("gTileInfo"), tileDim.x * tileDim.y, ResourceBindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    mpIDBuffer = Buffer::create(tileDim.x * tileDim.y * avgTileIDCount * sizeof(uint32_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);
    mpIDCounterBuffer = Buffer::create(sizeof(uint32_t), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None);

    createCausticsMap();

    mpRayDensityTex = Texture::create2D(MAX_CAUSTICS_MAP_SIZE, MAX_CAUSTICS_MAP_SIZE, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
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
