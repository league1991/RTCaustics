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

#define CAUSTICS_MAP_SIZE 1024

static const glm::vec4 kClearColor(0.f, 0.f, 0.f, 1);
static const std::string kDefaultScene = "Caustics/ring.fscene";
static int gDispatchSize = 512;

std::string to_string(const vec3& v)
{
    std::string s;
    s += "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
    return s;
}

void Caustics::onGuiRender(Gui* pGui)
{
    pGui->addCheckBox("Ray Trace", mRayTrace);

    Gui::DropdownList debugModeList;
    debugModeList.push_back({ 0, "Disabled" });
    debugModeList.push_back({ 1, "Depth" });
    debugModeList.push_back({ 2, "Normal" });
    debugModeList.push_back({ 3, "Diffuse" });
    debugModeList.push_back({ 4, "Specular" });
    debugModeList.push_back({ 5, "Photon" });
    debugModeList.push_back({ 6, "World" });
    debugModeList.push_back({ 7, "Roughness" });
    pGui->addDropdown("Debug mode", debugModeList, (uint32_t&)mDebugMode);
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

    pGui->addFloatVar("Emit size", mEmitSize, 0, 1000, 5);
    pGui->addFloatVar("Splat size", mSplatSize, 0, 10, 0.01f);
    if (mpScene)
    {
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            std::string group = "Point Light" + std::to_string(i);
            mpScene->getLight(i)->renderUI(pGui, group.c_str());
        }
    }
    mpCamera->renderUI(pGui);
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
    mCamController.setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, pModel->getRadius() / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());
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

    // photon trace
    RtProgram::Desc photonTraceProgDesc;
    photonTraceProgDesc.addShaderLibrary("PhotonTrace.rt.hlsl");
    photonTraceProgDesc.setRayGen("rayGen");
    photonTraceProgDesc.addHitGroup(0, "primaryClosestHit", "");
    photonTraceProgDesc.addMiss(0, "primaryMiss");
    //photonTraceProgDesc.addHitGroup(1, "", "shadowAnyHit");
    //photonTraceProgDesc.addMiss(1, "shadowMiss");
    mpPhotonTraceProgram = RtProgram::create(photonTraceProgDesc);
    mpPhotonTraceState = RtState::create();
    mpPhotonTraceState->setProgram(mpPhotonTraceProgram);
    mpPhotonTraceState->setMaxTraceRecursionDepth(3);
    mpPhotonTraceVars = RtProgramVars::create(mpPhotonTraceProgram, mpScene);
    //mpPhotonTraceRenderer = RtSceneRenderer::create(mpScene);

    // photon scatter
    BlendState::Desc scatterBlendStateDesc;
    scatterBlendStateDesc.setRtBlend(0, true);
    scatterBlendStateDesc.setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One, BlendState::BlendFunc::One);
    BlendState::SharedPtr scatterBlendState = BlendState::create(scatterBlendStateDesc);
    mpPhotonScatterProgram = GraphicsProgram::createFromFile("PhotonScatter.ps.hlsl", "photonScatterVS", "photonScatterPS");
    DepthStencilState::Desc scatterDSDesc;
    scatterDSDesc.setDepthEnabled(false);
    auto depthStencilState = DepthStencilState::create(scatterDSDesc);
    mpPhotonScatterState = GraphicsState::create();
    mpPhotonScatterState->setProgram(mpPhotonScatterProgram);
    mpPhotonScatterState->setBlendState(scatterBlendState);
    mpPhotonScatterState->setDepthStencilState(depthStencilState);
    mpPhotonScatterVars = GraphicsVars::create(mpPhotonScatterProgram->getReflector());
    //mpPhotonScatterPass = RasterScenePass::create(mpScene, "PhotonScatter.ps.hlsl", "photonScatterVS", "photonScatterPS");

    mpRtRenderer = RtSceneRenderer::create(mpScene);

    mpRasterPass = RasterScenePass::create(mpScene, "Caustics.ps.hlsl", "", "main");

    mpGPass = RasterScenePass::create(mpScene, "GPass.ps.hlsl", "", "gpassPS");

    mpCompositePass = FullScreenPass::create("Composite.ps.hlsl");
}

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

    {
        GraphicsVars* pVars = mpPhotonTraceVars->getGlobalVars().get();
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer("PerFrameCB");
        pCB["invView"] = glm::inverse(mpCamera->getViewMatrix());
        pCB["viewportDims"] = vec2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
        pCB["emitSize"] = mEmitSize;
        //setCommonVars(mpPhotonTraceVars->getGlobalVars().get(), pTargetFbo);
        mSampleIndex++;
    }
}

void Caustics::renderRT(RenderContext* pContext, Fbo::SharedPtr pTargetFbo)
{
    PROFILE("renderRT");
    setPerFrameVars(pTargetFbo.get());

    // gpass
    pContext->clearFbo(mpGPassFbo.get(), vec4(0, 0, 0, 1), 1.0, 0);
    mpGPass->renderScene(pContext, mpGPassFbo);

    // photon tracing
    pContext->clearTexture(mpRtOut.get());
    mpPhotonTraceVars->getRayGenVars()->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
    mpPhotonTraceVars->getRayGenVars()->setTexture("gOutput", mpRtOut);
    auto hitVars = mpPhotonTraceVars->getHitVars(0);
    for (auto& hitVar: hitVars)
    {
        hitVar->setTexture("gOutput", mpRtOut);
        hitVar->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
    }
    mpRtRenderer->renderScene(pContext, mpPhotonTraceVars, mpPhotonTraceState, uvec3(gDispatchSize, gDispatchSize, 1), mpCamera.get());

    // photon scattering
    //pContext->clearTexture(mpCausticsFbo->getColorTexture(0).get());
    //pContext->clearTexture(mpCausticsFbo->getDepthStencilTexture().get(), vec4(1.f, 1.f, 1.f, 1.f));
    //mpPhotonScatterPass->renderScene(pContext, mpCausticsFbo);
    pContext->clearFbo(mpCausticsFbo.get(), vec4(0, 0, 0, 0), 1.0, 0);
    glm::mat4 wvp = mpCamera->getProjMatrix() * mpCamera->getViewMatrix();
    ConstantBuffer::SharedPtr pPerFrameCB = mpPhotonScatterVars["PerFrameCB"];
    pPerFrameCB["gWorldMat"] = glm::mat4();
    pPerFrameCB["gWvpMat"] = wvp;
    pPerFrameCB["gEyePosW"] = mpCamera->getPosition();
    pPerFrameCB["gSplatSize"] = mSplatSize;
    mpPhotonScatterVars->setStructuredBuffer("gPhotonBuffer", mpPhotonBuffer);
    mpPhotonScatterState->setVao(mpQuad->getMesh(0)->getVao());
    mpPhotonScatterState->setFbo(mpCausticsFbo);
    int instanceCount = gDispatchSize * gDispatchSize;
    pContext->drawIndexedInstanced(mpPhotonScatterState.get(), mpPhotonScatterVars.get(), mpQuad->getMesh(0)->getIndexCount(), instanceCount, 0, 0, 0);

    // Render output
    //pContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
    //mpRtVars->getRayGenVars()->setTexture("gOutput", mpRtOut);
    //mpRtRenderer->renderScene(pContext, mpRtVars, mpRtState, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1), mpCamera.get());

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(samplerDesc);
    mpCompositePass["gDepthTex"]   = mpGPassFbo->getDepthStencilTexture();
    mpCompositePass["gNormalTex"]  = mpGPassFbo->getColorTexture(0);
    mpCompositePass["gDiffuseTex"] = mpGPassFbo->getColorTexture(1);
    mpCompositePass["gSpecularTex"]  = mpGPassFbo->getColorTexture(2);
    mpCompositePass["gPhotonTex"] = mpCausticsFbo->getColorTexture(0);
    mpCompositePass["gPointSampler"] = mpPointSampler;
    ConstantBuffer::SharedPtr pCompCB = mpCompositePass["PerImageCB"];
    pCompCB["gNumLights"] = mpScene->getLightCount();
    pCompCB["gDebugMode"] = (uint32_t)mDebugMode;
    pCompCB["gInvWvpMat"] = mpCamera->getInvViewProjMatrix();
    pCompCB["gCameraPos"] = mpCamera->getPosition();
    //ConstantBuffer::SharedPtr pImageCB = mpCompositePass["PerImageCB"];
    //mpCamera->setIntoConstantBuffer(pImageCB.get(), 0);
    for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
    {
        mpScene->getLight(i)->setIntoProgramVars(mpCompositePass->getVars().get(), pCompCB.get(), "gLightData[" + std::to_string(i) + "]");
    }

    mpCompositePass->execute(pContext, pTargetFbo);
    //pContext->blit(mpGPassFbo->getColorTexture(0)->getSRV(), pTargetFbo->getRenderTargetView(0));
    //pContext->blit(mpCausticsFbo->getColorTexture(0)->getSRV(), pTargetFbo->getRenderTargetView(0));
    //pContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
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

    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpPhotonBuffer = StructuredBuffer::create(mpPhotonTraceProgram->getHitProgram(0).get(), std::string("gPhotonBuffer"), CAUSTICS_MAP_SIZE * CAUSTICS_MAP_SIZE, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    //mpCausticsFbo = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
    mpDepthTex = Texture::create2D(width, height, ResourceFormat::D24UnormS8, 1, 1, nullptr, Resource::BindFlags::DepthStencil | Resource::BindFlags::ShaderResource);
    //mpCausticsTex = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    mpCausticsFbo = Fbo::create2D(width, height, ResourceFormat::RGBA16Float, ResourceFormat::D24UnormS8);
    //Fbo::create2D(width, height, ResourceFormat::RGBA16Float);

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
