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
    void onLoad(RenderContext* pRenderContext) override;
    void onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(uint32_t width, uint32_t height) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onGuiRender(Gui* pGui) override;

private:
    bool mRayTrace = true;
    uint32_t mDebugMode = 1;
    //bool mUseDOF = false;
    uint32_t mSampleIndex = 0xdeadbeef;

    Model::SharedPtr mpQuad;
    RtScene::SharedPtr mpScene;
    Camera::SharedPtr mpCamera;
    FirstPersonCameraController mCamController;

    // forward shading pass
    RasterScenePass::SharedPtr mpRasterPass;

    // g-pass
    RasterScenePass::SharedPtr mpGPass;
    Texture::SharedPtr mpNormalTex;
    Texture::SharedPtr mpDiffuseTex;
    Texture::SharedPtr mpSpecularTex;
    Texture::SharedPtr mpDepthTex;
    Fbo::SharedPtr mpGPassFbo;

    // photon trace
    RtProgram::SharedPtr mpPhotonTraceProgram;
    RtProgramVars::SharedPtr mpPhotonTraceVars;
    RtState::SharedPtr mpPhotonTraceState;
    //RtSceneRenderer::SharedPtr mpPhotonTraceRenderer;

    // photon scatter
    //RasterScenePass::SharedPtr mpPhotonScatterPass;
    GraphicsProgram::SharedPtr mpPhotonScatterProgram;
    GraphicsVars::SharedPtr mpPhotonScatterVars;
    GraphicsState::SharedPtr mpPhotonScatterState;
    Fbo::SharedPtr mpCausticsFbo;

    // raytrace
    RtProgram::SharedPtr mpRaytraceProgram;
    RtProgramVars::SharedPtr mpRtVars;
    RtState::SharedPtr mpRtState;
    RtSceneRenderer::SharedPtr mpRtRenderer;
    Texture::SharedPtr mpRtOut;

    // composite pass
    Sampler::SharedPtr mpPointSampler;
    FullScreenPass::SharedPtr mpCompositePass;

    // Caustics map
    //Texture::SharedPtr mpCausticsMap;
    StructuredBuffer::SharedPtr  mpPhotonBuffer;


    void setPerFrameVars(const Fbo* pTargetFbo);
    void renderRT(RenderContext* pContext, Fbo::SharedPtr pTargetFbo);
    void loadScene(const std::string& filename, const Fbo* pTargetFbo);
    void loadShader();
    void setCommonVars(GraphicsVars* pVars, const Fbo* pTargetFbo);
};
