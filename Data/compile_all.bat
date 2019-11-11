
del "PhotonTracing.rt.hlsl.h"
fxc /O3 /T cs_5_1 "PhotonTracing.rt.hlsl" /E TexturedSceneGeom_VS /Fh "PhotonTracing.rt.hlsl.h"

rem del "bin\PhotonScatter_VS.h"
rem fxc /O3 /T vs_5_1 "src/ForwardRendering.hlsl" /E PhotonScatter_VS /Fh "bin/PhotonScatter_VS.h" /Fe "err/PhotonScatter_VS.txt"

rem del "bin\CausticsMapVS.h"
rem fxc /O3 /T vs_5_1 "src/ForwardRendering.hlsl" /E CausticsMapVS /Fh "bin/CausticsMapVS.h" /Fe "err/CausticsMapVS.txt"

rem del "bin\SimpleSceneGeom_VS.h"
rem fxc /O3 /T vs_5_1 "src/ForwardRendering.hlsl" /E SimpleSceneGeom_VS /Fh "bin/SimpleSceneGeom_VS.h" /Fe "err/SimpleSceneGeom_VS.txt"

rem del "bin\CausticsSceneGeom_VS.h"
rem fxc /O3 /T vs_5_1 "src/ForwardRendering.hlsl" /E CausticsSceneGeom_VS /Fh "bin/CausticsSceneGeom_VS.h" /Fe "err/CausticsSceneGeom_VS.txt"

rem del "bin\CausticsMapPS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E CausticsMapPS /Fh "bin/CausticsMapPS.h" /Fe "err/CausticsMapPS.txt"

rem del "bin\CausticsScene_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E CausticsScene_PS /Fh "bin/CausticsScene_PS.h" /Fe "err/CausticsScene_PS.txt"

rem del "bin\SimpleCausticsScene_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E SimpleCausticsScene_PS /Fh "bin/SimpleCausticsScene_PS.h" /Fe "err/SimpleCausticsScene_PS.txt"

rem del "bin\SimpleScene_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E SimpleScene_PS /Fh "bin/SimpleScene_PS.h" /Fe "err/SimpleScene_PS.txt"

rem del "bin\Combine_PS_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E Combine_PS /Fh "bin/Combine_PS.h" /Fe "err/Combine_PS.txt"

rem del "bin\BilateralBlur2_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E BilateralBlur2_PS /Fh "bin/BilateralBlur2_PS.h" /Fe "err/BilateralBlur2_PS.txt"

rem del "bin\BilateralBlur1_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E BilateralBlur1_PS /Fh "bin/BilateralBlur1_PS.h" /Fe "err/BilateralBlur1_PS.txt"

rem del "bin\ScatteredHoleFill1_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E ScatteredHoleFill1_PS /Fh "bin/ScatteredHoleFill1_PS.h" /Fe "err/ScatteredHoleFill1_PS.txt"

rem del "bin\ScatteredHoleFill2_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E ScatteredHoleFill2_PS /Fh "bin/ScatteredHoleFill2_PS.h" /Fe "err/ScatteredHoleFill2_PS.txt"

rem del "bin\PhotonScatter_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E PhotonScatter_PS /Fh "bin/PhotonScatter_PS.h" /Fe "err/PhotonScatter_PS.txt"

rem del "bin\TexturedScene_PS.h"
rem fxc /enable_unbounded_descriptor_tables /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E TexturedScene_PS /Fh "bin/TexturedScene_PS.h" /Fe "err/TexturedScene_PS.txt"

rem del "bin\Screen_Quad_VS.h"
rem fxc /O3 /T vs_5_1 "src/MSAA_Detect.hlsl" /E Screen_Quad_VS /Fh "bin/Screen_Quad_VS.h" /Fe "err/Screen_Quad_VS.txt"

rem del "bin\FXAA_PS.h"
rem fxc /O3 /T ps_5_1 "src/FXAA.hlsl" /E FXAA_PS /Fh "bin/FXAA_PS.h" /Fe "err/FXAA_PS.txt"

rem del "bin\Debug1PS.h"
rem fxc /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E Debug1PS /Fh "bin/Debug1PS.h" /Fe "err/Debug1PS.txt"

rem del "bin\Debug2PS.h"
rem fxc /O3 /T ps_5_1 "src/ForwardRendering.hlsl" /E Debug2PS /Fh "bin/Debug2PS.h" /Fe "err/Debug2PS.txt"

rem del "bin\CountRays_PS.h"
rem fxc /O3 /T ps_5_1 "src/CountRays.hlsl" /E CountRays_PS /Fh "bin/CountRays_PS.h" /Fe "err/CountRays_PS.txt"

rem del "bin\ClearRays_PS.h"
rem fxc /O3 /T ps_5_1 "src/CountRays.hlsl" /E ClearRays_PS /Fh "bin/ClearRays_PS.h" /Fe "err/ClearRays_PS.txt"

rem del "bin\OutputListLength_PS.h"
rem fxc /O3 /T ps_5_1 "src/CountRays.hlsl" /E OutputListLength_PS /Fh "bin/OutputListLength_PS.h" /Fe "err/OutputListLength_PS.txt"

rem del "bin\StencilRed_PS.h"
rem fxc /O3 /T ps_5_1 "src/CountRays.hlsl" /E StencilRed_PS /Fh "bin/StencilRed_PS.h" /Fe "err/StencilRed_PS_PS.txt"

rem del "bin\Screen_Quad_VS.h"
rem fxc /O3 /T vs_5_1 "src/MSAA_Detect.hlsl" /E Screen_Quad_VS /Fh "bin/Screen_Quad_VS.h" /Fe "err/Screen_Quad_VS.txt"

rem del "bin\UntexturedSceneGeom_VS.h"
rem fxc /O3 /T vs_5_1 "src/ForwardRendering.hlsl" /E UntexturedSceneGeom_VS /Fh "bin/UntexturedSceneGeom_VS.h" /Fe "err/UntexturedSceneGeom_VS.txt"

pause

exit

