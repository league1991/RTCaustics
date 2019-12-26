
#define GATHER_TILE_SIZE 16

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
    float  pixelArea;
    int    inFrustum;
};

struct RayArgument
{
    int rayTaskCount;
};

struct IDBlock
{
    int address;
    int count;
};

void getRefractVector(float3 I, float3 N, out float3 R, float eta)
{
    float IN = dot(I, N);
    float RN = -sqrt(1 - eta * eta * (1 - IN * IN));
    float mu = eta * IN - RN;
    R = eta * I - mu * N;
}

// eta = eta i / eta o
bool isTotalInternalReflection(float3 I, float3 N, float eta)
{
    // detect total internal reflection
    float cosI = dot(I, N);
    return cosI * cosI < (1 - 1 / (eta * eta));
}

bool isPhotonAdjecent(Photon photon0, Photon photon1, float normalThreshold, float distanceThreshold, float planarThreshold)
{
    return
        //dot(photon0.normalW, photon1.normalW) < normalThreshold &&
        length(photon0.posW - photon1.posW) < distanceThreshold// &&
        //dot(photon0.normalW, photon0.posW - photon1.posW) < planarThreshold
        ;
}

