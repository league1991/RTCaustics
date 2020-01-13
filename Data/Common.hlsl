
#define GATHER_TILE_SIZE 16

struct Photon
{
    float3 posW;
    //float3 normalW;
    float3 color;
    float3 dPdx;
    float3 dPdy;
};

struct RayTask
{
    float2 screenCoord;
    float pixelSize;
    float intensity;
};

struct PixelInfo
{
    uint screenArea;
    uint screenAreaSq;
    uint count;
    int  photonIdx;
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

float smoothKernel(float x)
{
    x = saturate(x);
    return x * x * (2 * x - 3) + 1;
}

bool isInFrustum(float4 p)
{
    return (p.x >= -p.w) && (p.x <= p.w) && (p.y >= -p.w) && (p.y <= p.w) && (p.z >= 0) && (p.z <= p.w);
}

float linearSample(float a, float b, float u)
{
    if (a == b)
        return u;
    float s = 0.5 * (a + b);
    a /= s;
    b /= s;
    return (a - sqrt(lerp(a * a, b * b, u))) / (a - b);
}

float2 bilinearSample(float a00, float a10, float a01, float a11, float2 uv)
{
    float a0 = a00 + a01;
    float a1 = a10 + a11;
    float u = linearSample(a0, a1, uv.x);
    float b0 = lerp(a00, a10, u);
    float b1 = lerp(a01, a11, u);
    float v = linearSample(b0, b1, uv.y);
    return float2(u, v);
}

float bilinearIntepolation(float a00, float a10, float a01, float a11, float2 uv)
{
    float b0 = lerp(a00, a10, uv.x);
    float b1 = lerp(a01, a11, uv.x);
    return lerp(b0, b1, uv.y);
}

float toViewSpace(float4x4 invProj, float depth)
{
    return (invProj[2][2] * depth + invProj[3][2]) / (invProj[2][3] * depth + invProj[3][3]);
}
