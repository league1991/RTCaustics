struct Photon
{
    float3 posW;
    float3 normalW;
    float3 color;
    float3 dPdx;
    float3 dPdy;
};

struct RayTask
{
    float2 screenCoord;
    float2 pixelSize;
    int    photonIdx;
};

struct RayArgument
{
    int rayTaskCount;
};
