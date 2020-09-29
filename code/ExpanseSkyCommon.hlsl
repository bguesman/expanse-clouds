#ifndef EXPANSE_SKY_COMMON_INCLUDED
#define EXPANSE_SKY_COMMON_INCLUDED



/******************************************************************************/
/***************************** GLOBAL VARIABLES *******************************/
/******************************************************************************/

/* All of these things have to be in a cbuffer, so we can access them across
 * different shaders. */
CBUFFER_START(ExpanseSky)
float _atmosphereThickness;
float _planetRadius;
float _atmosphereRadius;
float4 _groundTint;
/* TODO: currently, we don't use these textures in precomputation. But
 * we could. */
TEXTURECUBE(_groundAlbedoTexture);
bool _hasGroundAlbedoTexture;
TEXTURECUBE(_groundEmissionTexture);
bool _hasGroundEmissionTexture;
float _groundEmissionMultiplier;
float4 _lightPollutionTint;
float _lightPollutionIntensity;
float4x4 _planetRotation;
float _aerosolCoefficient;
float _scaleHeightAerosols;
float _aerosolAnisotropy;
float _aerosolDensity;
float4 _airCoefficients;
float _scaleHeightAir;
float _airDensity;
float4 _ozoneCoefficients;
float _ozoneThickness;
float _ozoneHeight;
float _ozoneDensity;
int _numberOfTransmittanceSamples;
int _numberOfLightPollutionSamples;
int _numberOfScatteringSamples;
int _numberOfGroundIrradianceSamples;
int _numberOfMultipleScatteringSamples;
int _numberOfMultipleScatteringAccumulationSamples;
bool _useImportanceSampling;

/* Clouds geometry. */
float _cloudVolumeLowerRadialBoundary;
float _cloudVolumeUpperRadialBoundary;
float _cloudTextureAngularRange;

/* Clouds noise. */
float4 _basePerlinOctaves;
float _basePerlinOffset;
float _basePerlinScaleFactor;
float4 _baseWorleyOctaves;
float _baseWorleyScaleFactor;
float _baseWorleyBlendFactor;
float4 _structureOctaves;
float _structureScaleFactor;
float4 _detailOctaves;
float _detailScaleFactor;
float _detailNoiseTile;
float4 _coverageOctaves;
float _coverageOffset;
float _coverageScaleFactor;

/* Easier to type... */
#define g _aerosolAnisotropy

/* Redefine colors to float3's for efficiency, since Unity can only set
 * float4's. */
#define _airCoefficientsF3 _airCoefficients.xyz
#define _ozoneCoefficientsF3 _ozoneCoefficients.xyz
#define _groundTintF3 _groundTint.xyz
#define _lightPollutionTintF3 _lightPollutionTint.xyz

/* Set up a sampler for the cubemaps. */
//SAMPLER(SAMPLER_CUBEMAP);

/* Precomputed tables. */

/* Transmittance table. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * mu (y dimension): the zenith angle of the viewing direction. */
TEXTURE2D(_TransmittanceTable);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define TRANSMITTANCE_TABLE_SIZE_H 32
#define TRANSMITTANCE_TABLE_SIZE_MU 128

/* Light pollution table. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * mu (y dimension): the zenith angle of the viewing direction. */
TEXTURE2D(_LightPollutionTableAir);
TEXTURE2D(_LightPollutionTableAerosol);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define LIGHT_POLLUTION_TABLE_SIZE_H 32
#define LIGHT_POLLUTION_TABLE_SIZE_MU 128

/* Single scattering tables. Leverage spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * mu (y dimension): the zenith angle of the viewing direction.
 * mu_l (z dimension): the zenith angle of the light source.
 * nu (w dimension): the azimuth angle of the light source. */
TEXTURE3D(_SingleScatteringTableAir);
TEXTURE3D(_SingleScatteringTableAerosol);
TEXTURE3D(_SingleScatteringTableAirNoShadows);
TEXTURE3D(_SingleScatteringTableAerosolNoShadows);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define SINGLE_SCATTERING_TABLE_SIZE_H 32
#define SINGLE_SCATTERING_TABLE_SIZE_MU 128
#define SINGLE_SCATTERING_TABLE_SIZE_MU_L 32
#define SINGLE_SCATTERING_TABLE_SIZE_NU 32

/* Ground irradiance tables. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * mu (x dimension): dot product between the surface normal and the
 * light direction. */
TEXTURE2D(_GroundIrradianceTableAir);
TEXTURE2D(_GroundIrradianceTableAerosol);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define GROUND_IRRADIANCE_TABLE_SIZE 256

/* Local multiple scattering table. Leverages spherical symmetry of the atmosphere,
 * parameterized by:
 * h (x dimension): the height of the camera.
 * mu_l (y dimension): dot product between the surface normal and the
 * light direction. */
TEXTURE2D(_LocalMultipleScatteringTable);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define MULTIPLE_SCATTERING_TABLE_SIZE_H 32
#define MULTIPLE_SCATTERING_TABLE_SIZE_MU_L 32

/* Global multiple scattering tables. Leverage spherical symmetry of the
 * atmosphere, parameterized by:
 * h (x dimension): the height of the camera.
 * mu (y dimension): the zenith angle of the viewing direction.
 * mu_l (z dimension): the zenith angle of the light source.
 * nu (w dimension): the azimuth angle of the light source. */
TEXTURE3D(_GlobalMultipleScatteringTableAir);
TEXTURE3D(_GlobalMultipleScatteringTableAerosol);
/* Table dimensions. Must match those in ExpanseSkyRenderer.cs. TODO: would
 * be great to be able to set these as constants. I don't want to make them
 * accessible because I haven't set up any way to reallocate these on the
 * fly. */
#define GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_H 32
#define GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_MU 128
#define GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_MU_L 32
#define GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_NU 32

/* Cloud textures. */
TEXTURE3D(_CloudBaseNoiseTable);
#define CLOUD_BASE_NOISE_TABLE_SIZE_X 512
#define CLOUD_BASE_NOISE_TABLE_SIZE_Y 256
#define CLOUD_BASE_NOISE_TABLE_SIZE_Z 512
TEXTURE3D(_CloudDetailNoiseTable);
#define CLOUD_DETAIL_NOISE_TABLE_SIZE_X 128
#define CLOUD_DETAIL_NOISE_TABLE_SIZE_Y 128
#define CLOUD_DETAIL_NOISE_TABLE_SIZE_Z 128
TEXTURE2D(_CloudCurlNoiseTable);
#define CLOUD_CURL_NOISE_TABLE_SIZE_X 128
#define CLOUD_CURL_NOISE_TABLE_SIZE_Y 128
TEXTURE2D(_CloudCoverageTable);
#define CLOUD_COVERAGE_TABLE_SIZE_X 1024
#define CLOUD_COVERAGE_TABLE_SIZE_Y 1024

CBUFFER_END


/* Sampler for tables. */
#ifndef UNITY_SHADER_VARIABLES_INCLUDED
    SAMPLER(s_linear_clamp_sampler);
    SAMPLER(s_trilinear_clamp_sampler);
#endif

/* Some mathematical constants that are good to have pre-computed. */
#define PI 3.1415926535
#define SQRT_PI 1.77245385091
#define E 2.7182818285
#define GOLDEN_RATIO 1.6180339887498948482

#define FLT_EPSILON 0.000001

/******************************************************************************/
/*************************** END GLOBAL VARIABLES *****************************/
/******************************************************************************/





/******************************************************************************/
/***************************** UTILITY FUNCTIONS ******************************/
/******************************************************************************/

float random(float2 uv) {
  return frac(sin(dot(uv,float2(12.9898,78.233)))*43758.5453123);
}

float random_3(float3 uv) {
  return frac(sin(dot(uv,float3(12.9898,78.233, 103.953)))*43758.5453123);
}

float random_3_seeded(float3 uv, float3 seed) {
  return frac(sin(dot(uv,seed))*43758.5453123);
}

float3 random_3_3(float3 uv) {
  return frac(sin(float3(dot(uv,float3(127.1, 311.7, 103.953)),
    dot(uv,float3(269.5,183.3, 119.234)),
    dot(uv,float3(453.1293, 165.932, 53.209))))*43758.5453);
}

float3 random_3_3_seeded(float3 uv, float3 seed1, float3 seed2, float3 seed3) {
  return frac(sin(float3(dot(uv,seed1),
    dot(uv,seed2),
    dot(uv,seed3)))*43758.5453);
}


float random_1(float u) {
  return frac(sin(u * 12.9898)*43758.5453123);
}

float clampCosine(float c) {
  return clamp(c, -1.0, 1.0);
}

float safeSqrt(float s) {
  return sqrt(max(0.0, s));
}

/* Returns minimum non-negative number, given that one number is
 * non-negative. If both numbers are negative, returns a negative number. */
float minNonNegative(float a, float b) {
  return (a < 0.0) ? b : ((b < 0.0) ? a : min(a, b));
}

/* True if a is greater than b within tolerance FLT_EPSILON, false
 * otherwise. */
bool floatGT(float a, float b) {
  return a > b - FLT_EPSILON;
}

/* True if a is less than b within tolerance FLT_EPSILON, false
 * otherwise. */
bool floatLT(float a, float b) {
  return a < b + FLT_EPSILON;
}

float remap(float value, float original_min, float original_max, float new_min, float new_max) {
  return new_min + ((value - original_min) / (original_max - original_min)) * (new_max - new_min);
}

/******************************************************************************/
/*************************** END UTILITY FUNCTIONS ****************************/
/******************************************************************************/


/******************************************************************************/
/****************************** NOISE FUNCTIONS *******************************/
/******************************************************************************/

float roundDown(float p, float m) {
  return (int)(p/m) * m;
}

float roundUp(float p, float m) {
  return ((int)(p/m) + 1) * m;
}

/* Computes perlin noise at uv2 coordinate uv2. grid_res controls the grid
 * resolution for the sample points. */
float perlin(float3 uvw, float3 grid_res, float3 seed) {
  /* We're going to do this all local to upper left corner of the cell uvw is in. */
  float3 top_left = floor(uvw * grid_res);
  /* We need to compute the distance to each of the top corners. */
  float3 offsets[8] = {float3(0, 0, 0), float3(0, 0, 1),
    float3(0, 1, 0), float3(0, 1, 1),
    float3(1, 0, 0), float3(1, 0, 1),
    float3(1, 1, 0), float3(1, 1, 1)};
  float noiseValues[8];
  /* Assign a random value to each point in the cubic cell. */
  for (int i = 0; i < 8; i++) {
    float3 cellCorner = top_left + offsets[i];
    cellCorner.x = float(int(cellCorner.x) % grid_res.x);
    cellCorner.y = float(int(cellCorner.y) % grid_res.y);
    cellCorner.z = float(int(cellCorner.z) % grid_res.z);
    noiseValues[i] = random_3_seeded(cellCorner, seed);
  }

  /* Interpolate between the points according to where uvw is in the cell. */
  float3 p = frac(uvw * grid_res);
  p.x = smoothstep(0, 1, p.x);
  p.y = smoothstep(0, 1, p.y);
  p.z = smoothstep(0, 1, p.z);

  float v01 = lerp(noiseValues[0], noiseValues[1], p.z);
  float v23 = lerp(noiseValues[2], noiseValues[3], p.z);
  float v45 = lerp(noiseValues[4], noiseValues[5], p.z);
  float v67 = lerp(noiseValues[6], noiseValues[7], p.z);

  float v0123 = lerp(v01, v23, p.y);
  float v4567 = lerp(v45, v67, p.y);

  return lerp(v0123, v4567, p.x);
}

float worley(float3 uv, float3 grid_res, float3x3 rotation, float3 seed1, float3 seed2, float3 seed3) {
  /* Tile coordinates. Result is between 0 and 1. */
  /* Experiment: cut grid res in half. */
  grid_res /= 2.0;
  float3 f_uv = frac(grid_res * uv);

  /* Use coordinates of four points in the cell to determine random placements. */
  float3 tl = floor(grid_res * uv);
  float d = FLT_MAX; /* Large number. */
  float3 offsets[4] = {float3(0, 0, 0), float3(0, 0.5, 0), float3(0.5, 0.5, 0), float3(0, 0.5, 0.5)};
  for (int r = 0; r < 4; r++) {
    float3 p = 0.5 * (random_3_3_seeded(tl + offsets[r], seed1, seed2, seed3) + 1);
    /* Compute distance to the point. */
    d = min(d, length(f_uv - p));
  }

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        if (!(i == 0 && j == 0 && k == 0)) {
          float3 tl_neighbor = tl + float3(i, j, k);
          /* Wraparound. */
          float3 tl_neighbor_seed = tl_neighbor - grid_res * floor(tl_neighbor / grid_res);
          for (int r = 0; r < 4; r++) {
            float3 p_neighbor = (0.5 * (1 + random_3_3_seeded(tl_neighbor_seed + offsets[r], seed1, seed2, seed3))) + float3(i, j, k);
            d = min(d, length(p_neighbor - f_uv));
          }
        }
      }
    }
  }

  return d;
}

/******************************************************************************/
/**************************** END NOISE FUNCTIONS *****************************/
/******************************************************************************/


/******************************************************************************/
/********************************* SAMPLING ***********************************/
/******************************************************************************/

/* Given an index and total number of points, generates corresponding
 * point on fibonacci hemi-sphere. */
float3 fibonacciHemisphere(int i, int n) {
  float i_mid = i + 0.5;
  float cos_phi = 1 - i/float(n);
  float sin_phi = safeSqrt(1 - cos_phi * cos_phi);
  float theta = 2 * PI * i / GOLDEN_RATIO;
  float cos_theta = cos(theta);
  float sin_theta = safeSqrt(1 - cos_theta * cos_theta);
  return float3(cos_theta * sin_phi, cos_phi, sin_theta * sin_phi);
}

/* Given an index and total number of points, generates corresponding
 * point on fibonacci sphere. */
float3 fibonacciSphere(int i, int n) {
  float i_mid = i + 0.5;
  float cos_phi = 1 - 2 * i/float(n);
  float sin_phi = safeSqrt(1 - cos_phi * cos_phi);
  float theta = 2 * PI * i / GOLDEN_RATIO;
  float cos_theta = cos(theta);
  float sin_theta = safeSqrt(1 - cos_theta * cos_theta);
  return float3(cos_theta * sin_phi, cos_phi, sin_theta * sin_phi);
}

/* Generates linear location from a sample index.
 * Returns (sample, ds). */
float2 generateLinearSampleFromIndex(int i, int numberOfSamples) {
  return float2((float(i) + 0.5) / float(numberOfSamples),
    1.0 / ((float) numberOfSamples));
}

/* Generates cubed "importance sample" location from a sample index.
 * Returns (sample, ds). */
float2 generateCubicSampleFromIndex(int i, int numberOfSamples) {
  float t_left = float(i) / float(numberOfSamples);
  float t_middle = (float(i) + 0.5) / float(numberOfSamples);
  float t_right = (float(i) + 1.0) / float(numberOfSamples);
  t_left *= t_left * t_left;
  t_middle *= t_middle * t_middle;
  t_right *= t_right * t_right;
  return float2(t_middle, t_right - t_left);
}

/******************************************************************************/
/******************************* END SAMPLING *********************************/
/******************************************************************************/





/******************************************************************************/
/*********************** GEOMETRY INTERSECTION TESTING ************************/
/******************************************************************************/

/* Returns t values of ray intersection with sphere. Third value indicates
 * if there was an intersection at all; if negative, there was no
 * intersection. */
float3 intersectSphere(float3 p, float3 d, float r) {
  float A = dot(d, d);
  float B = 2.f * dot(d, p);
  float C = dot(p, p) - (r * r);
  float det = (B * B) - 4.f * A * C;
  if (floatGT(det, 0.0)) {
    det = safeSqrt(det);
    return float3((-B + det) / (2.f * A), (-B - det) / (2.f * A), 1.0);
  }
  return float3(0, 0, -1.0);
}

/* Struct containing data for ray intersection queries. */
struct IntersectionData {
  float startT, endT;
  bool groundHit, atmoHit;
};

/* Traces a ray starting at point O in direction d. Returns information
 * about where the ray hit on the ground/on the boundary of the atmosphere. */
IntersectionData traceRay(float3 O, float3 d, float planetRadius,
  float atmosphereRadius) {
  /* Perform raw sphere intersections. */
  float3 t_ground = intersectSphere(O, d, planetRadius);
  float3 t_atmo = intersectSphere(O, d, atmosphereRadius);

  IntersectionData toRet = {0, 0, false, false};

  /* We have a hit if the intersection was succesful and if either point
   * is greater than zero (meaning we are in front of the ray, and not
   * behind it). */
  toRet.groundHit = t_ground.z >= 0.0 && (t_ground.x >= 0.0 || t_ground.y >= 0.0);
  toRet.atmoHit = t_atmo.z >= 0.0 && (t_atmo.x >= 0.0 || t_atmo.y >= 0.0);

  if (floatLT(length(O), atmosphereRadius)) {
    /* We are below the atmosphere boundary, and we will start our raymarch
     * at the origin point. */
    toRet.startT = 0;
    if (toRet.groundHit) {
      /* We have hit the ground, and will end our raymarch at the first
       * positive ground hit. */
      toRet.endT = minNonNegative(t_ground.x, t_ground.y);
    } else {
      /* We will end our raymarch at the first positive atmosphere hit. */
      toRet.endT = minNonNegative(t_atmo.x, t_atmo.y);
    }
  } else {
    /* We are outside the atmosphere, and, if we intersect the atmosphere
     * at all, we will start our raymarch at the first atmosphere
     * intersection point. We don't need to be concerned about negative
     * t values, since it's a geometric impossibility to be outside a sphere
     * and intersect both in front of and behind a ray. */
    if (toRet.atmoHit) {
      toRet.startT = min(t_atmo.x, t_atmo.y);
      if (toRet.groundHit) {
        /* If we hit the ground at all, we'll end our ray at the first ground
         * intersection point. */
        toRet.endT = min(t_ground.x, t_ground.y);
      } else {
        /* Otherwise, we'll end our ray at the second atmosphere
         * intersection point. */
        toRet.endT = max(t_atmo.x, t_atmo.y);
      }
    }
    /* If we haven't hit the atmosphere, we leave everything uninitialized,
     * since this ray just goes out into space. */
  }

  return toRet;
}

/******************************************************************************/
/********************* END GEOMETRY INTERSECTION TESTING **********************/
/******************************************************************************/





/******************************************************************************/
/****************************** TEXTURE MAPPING *******************************/
/******************************************************************************/

/* The following implements the strategy used in physically based sky
 * to lerp between 2 4D texture lookups to solve the issue of uv-mapping for
 * a deep texture. */
struct TexCoord4D {
  float x, y, z, w, a;
};

float3 sampleTexture4D(Texture3D tex, TexCoord4D coord) {
  float3 uvw0 = float3(coord.x, coord.y, coord.z);
  float3 uvw1 = float3(coord.x, coord.y, coord.w);
  float3 contrib0 = SAMPLE_TEXTURE3D_LOD(tex, s_linear_clamp_sampler, uvw0, 0).rgb;
  float3 contrib1 = SAMPLE_TEXTURE3D_LOD(tex, s_linear_clamp_sampler, uvw1, 0).rgb;
  return lerp(contrib0, contrib1, coord.a);
}

/* Converts u, v in unit range to a deep texture coordinate (w0, w1, a) with
 * zTexSize rows and zTexCount columns.
 *
 * Returns (w0, w1, a), where w0 and w1 are the locations to sample the
 * texture at and a is the blend amount to use when interpolating between
 * them. Mathematically:
 *
 *         sample(u, v) = a * sample(w0) + (1 - a) * sample(w1)
 *
 */
float3 uvToDeepTexCoord(float u, float v, int zTexSize, int zTexCount) {
  float w = (0.5 + u * (zTexSize - 1)) * (1.0/zTexSize);
  float k = v * (zTexCount - 1);
  float w0 = (floor(k) + w) * (1.0/zTexCount);
  float w1 = (ceil(k) + w) * (1.0/zTexCount);
  float a = frac(k);
  return float3(w0, w1, a);
}

/* Converts deep texture index in range zTexSize * zTexCount to the
 * uv coordinate in unit range that represents the 2D table index for a
 * table with zTexSize rows and zTexCount columns.
 *
 * Returns (u, v).
 */
float2 deepTexIndexToUV(uint deepTexCoord, uint zTexSize, int zTexCount) {
  uint texId = deepTexCoord / zTexSize;
  uint texCoord = deepTexCoord & (zTexSize - 1);
  float u = saturate(texCoord / (float(zTexSize) - 1.0));
  float v = saturate(texId / (float(zTexCount) - 1.0));
  return float2(u, v);
}


/* The following are the map/unmap functions for each individual parameter that
 * the higher-level table coordinate mapping functions rely on. */

/* Maps r, the distance from the planet origin, into range 0-1. Uses
 * mapping from bruneton and neyer. */
float map_r(float r, float atmosphereRadius, float planetRadius) {
  float planetRadiusSq = planetRadius * planetRadius;
  float rho = safeSqrt(r * r - planetRadiusSq);
  float H = safeSqrt(atmosphereRadius * atmosphereRadius - planetRadiusSq);
  return rho / H;
}

/* Unmaps r, the distance from the planet origin, from its mapped uv
 * coordinate. Uses mapping from bruneton and neyer. */
float unmap_r(float u_r, float atmosphereRadius, float planetRadius) {
  float planetRadiusSq = planetRadius * planetRadius;
  float H = safeSqrt(atmosphereRadius * atmosphereRadius - planetRadiusSq);
  float rho = u_r * H;
  return safeSqrt(rho * rho + planetRadiusSq);
}


/* Maps mu, the cosine of the viewing angle, into range 0-1. Uses
 * mapping from bruneton and neyer. */
float map_mu(float r, float mu, float atmosphereRadius, float planetRadius,
  float d, bool groundHit) {
  float planetRadiusSq = planetRadius * planetRadius;
  float rSq = r * r;
  float rho = safeSqrt(r * r - planetRadiusSq);
  float H = safeSqrt(atmosphereRadius * atmosphereRadius - planetRadiusSq);

  float u_mu = 0.0;
  float discriminant = rSq * mu * mu - rSq + planetRadiusSq;
  if (groundHit) {
    float d_min = r - planetRadius;
    float d_max = rho;
    /* Use lower half of [0, 1] range. */
    u_mu = 0.49 - 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    /* Use upper half of [0, 1] range. */
    u_mu = 0.51 + 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  }

  return u_mu;
}

/* Unmaps mu, the cosine of the viewing angle, from its mapped uv
 * coordinate. Uses mapping from bruneton and neyer.  */
float unmap_mu(float u_r, float u_mu, float atmosphereRadius,
  float planetRadius) {
  float planetRadiusSq = planetRadius * planetRadius;
  float H = safeSqrt(atmosphereRadius * atmosphereRadius
    - planetRadiusSq);
  float rho = u_r * H;
  float r = safeSqrt(rho * rho + planetRadiusSq);

  /* Clamp u_mu to valid range. */
  if (floatLT(u_mu, 0.51) && floatGT(u_mu, 0.5)) {
    u_mu = 0.51;
  } else if (floatGT(u_mu, 0.49) && floatLT(u_mu, 0.5)) {
    u_mu = 0.49;
  }

  float mu = 0.0;
  if (floatLT(u_mu, 0.5)) {
    float d_min = r - planetRadius;
    float d_max = rho;
    float d = d_min + (d_max - d_min) * (1.0 - (1.0 / 0.49) * u_mu);
    mu = (d == 0.0) ? -1.0 : clampCosine(-(rho * rho + d * d) / (2 * r * d));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    float d = d_min + (d_max - d_min) * (2.0 * u_mu - 1.02);
    mu = (d == 0.0) ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2 * r * d));
  }

  return mu;
}


/* Maps r and mu together---slightly more efficient than mapping them
 * individually, since they share calculations. */
float2 map_r_mu(float r, float mu, float atmosphereRadius, float planetRadius,
  float d, bool groundHit) {
  float planetRadiusSq = planetRadius * planetRadius;
  float rSq = r * r;
  float rho = safeSqrt(rSq - planetRadiusSq);
  float H = safeSqrt(atmosphereRadius * atmosphereRadius - planetRadiusSq);

  float u_mu = 0.0;
  float discriminant = rSq * mu * mu - rSq + planetRadiusSq;
  if (groundHit) {
    float d_min = r - planetRadius;
    float d_max = rho;
    /* Use lower half of [0, 1] range. */
    u_mu = 0.49 - 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    /* Use upper half of [0, 1] range. */
    u_mu = 0.51 + 0.49 * (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min));
  }

  float u_r = rho / H;

  return float2(u_r, u_mu);
}

/* Unmaps mu and r together---slightly more efficient than unmapping them
 * individually, since they share calculations.  */
float2 unmap_r_mu(float u_r, float u_mu, float atmosphereRadius,
  float planetRadius) {
  float planetRadiusSq = planetRadius * planetRadius;
  float H = safeSqrt(atmosphereRadius * atmosphereRadius - planetRadiusSq);
  float rho = u_r * H;
  float r = safeSqrt(rho * rho + planetRadiusSq);

  /* Clamp u_mu to valid range. */
  if (floatLT(u_mu, 0.51) && floatGT(u_mu, 0.5)) {
    u_mu = 0.51;
  } else if (floatGT(u_mu, 0.49) && floatLT(u_mu, 0.5)) {
    u_mu = 0.49;
  }

  float mu = 0.0;
  if (floatLT(u_mu, 0.5)) {
    float d_min = r - planetRadius;
    float d_max = rho;
    float d = d_min + (d_max - d_min) * (1.0 - (1.0 / 0.49) * u_mu);
    mu = (d == 0.0) ? -1.0 : clampCosine(-(rho * rho + d * d) / (2 * r * d));
  } else {
    float d_min = atmosphereRadius - r;
    float d_max = rho + H;
    float d = d_min + (d_max - d_min) * (2.0 * u_mu - 1.02);
    mu = (d == 0.0) ? 1.0 : clampCosine((H * H - rho * rho - d * d) / (2 * r * d));
  }

  return float2(r, mu);
}


/* Maps mu_l, the cosine of the sun zenith angle, into range 0-1. Uses
 * mapping from bruneton and neyer. */
float map_mu_l(float mu_l) {
  return saturate((1.0 - exp(-3 * mu_l - 0.6)) / (1 - exp(-3.6)));
}

/* Unmaps mu_l, the cosine of the sun zenith angle, from its mapped uv
 * coordinate. Uses mapping from bruneton and neyer. */
float unmap_mu_l(float u_mu_l) {
  return clampCosine((log(1.0 - (u_mu_l * (1 - exp(-3.6)))) + 0.6) / -3.0);
}


/* Maps nu, the cosine of the sun azimuth angle, into range 0-1. Uses
 * mapping from bruneton and neyer. */
float map_nu(float nu) {
  return saturate((1.0 + nu) / 2.0);
}

/* Unmaps nu, the cosine of the sun azimuth angle, from its mapped uv
 * coordinate. Uses mapping from bruneton and neyer. */
float unmap_nu(float u_nu) {
  return clampCosine((u_nu * 2.0) - 1.0);
}



/* The following are uv-mapping and unmapping functions for every table we
 * store. */

/* Returns u_r, u_mu. */
float2 mapTransmittanceCoordinates(float r, float mu, float atmosphereRadius,
  float planetRadius, float d, bool groundHit) {
  return map_r_mu(r, mu, atmosphereRadius, planetRadius, d, groundHit);
}

/* Returns r, mu. */
float2 unmapTransmittanceCoordinates(float u_r, float u_mu,
  float atmosphereRadius, float planetRadius) {
  return unmap_r_mu(u_r, u_mu, atmosphereRadius, planetRadius);
}

/* Returns u_r, u_mu, u_mu_l/u_nu bundled into z. */
TexCoord4D mapSingleScatteringCoordinates(float r, float mu, float mu_l,
  float nu, float atmosphereRadius, float planetRadius, float d,
  bool groundHit) {
  float2 u_r_mu = map_r_mu(r, mu, atmosphereRadius, planetRadius,
    d, groundHit);
  float3 deepTexCoord = uvToDeepTexCoord(map_mu_l(mu_l), map_nu(nu),
    SINGLE_SCATTERING_TABLE_SIZE_MU_L, SINGLE_SCATTERING_TABLE_SIZE_NU);
  TexCoord4D toRet = {u_r_mu.x, u_r_mu.y, deepTexCoord.x,
    deepTexCoord.y, deepTexCoord.z};
  return toRet;
}

/* Returns r, mu, mu_l, and nu. */
float4 unmapSingleScatteringCoordinates(float u_r, float u_mu, float u_mu_l,
  float u_nu, float atmosphereRadius, float planetRadius) {
  float2 r_mu = unmap_r_mu(u_r, u_mu, atmosphereRadius,
    planetRadius);
  return float4(r_mu.x, r_mu.y, unmap_mu_l(u_mu_l), unmap_nu(u_nu));
}

/* Returns u_mu_l, v. V is always zero, since the texture is effectively 1D. */
float2 mapGroundIrradianceCoordinates(float mu_l) {
  return float2(map_mu_l(mu_l), 0);
}

/* Returns r, mu_l. R is always just above the ground. */
float2 unmapGroundIrradianceCoordinates(float u_mu_l, float _planetRadius) {
  return float2(_planetRadius + 0.01, unmap_mu_l(u_mu_l));
}

/* This parameterization was taken from Hillaire's 2020 model. */
/* Returns u_r, u_mu_l. */
float2 mapLocalMultipleScatteringCoordinates(float r, float mu_l,
  float atmosphereRadius, float planetRadius) {
  return float2(map_r(r, atmosphereRadius, planetRadius), map_mu_l(mu_l));
}

/* Returns r, mu_l. */
float2 unmapLocalMultipleScatteringCoordinates(float u_r, float u_mu_l, float atmosphereRadius,
  float planetRadius) {
  return float2(unmap_r(u_r, atmosphereRadius, planetRadius),
    unmap_mu_l(u_mu_l));
}

/* Returns u_r, u_mu, u_mu_l/u_nu bundled into z. */
TexCoord4D mapGlobalMultipleScatteringCoordinates(float r, float mu,
  float mu_l, float nu, float atmosphereRadius, float planetRadius, float d,
  bool groundHit) {
  float2 u_r_mu = map_r_mu(r, mu, atmosphereRadius, planetRadius,
    d, groundHit);
  float3 deepTexCoord = uvToDeepTexCoord(map_mu_l(mu_l), map_nu(nu),
    GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_MU_L,
    GLOBAL_MULTIPLE_SCATTERING_TABLE_SIZE_NU);
  TexCoord4D toRet = {u_r_mu.x, u_r_mu.y, deepTexCoord.x,
    deepTexCoord.y, deepTexCoord.z};
  return toRet;
}

/* Returns r, mu, mu_l, and nu. */
float4 unmapGlobalMultipleScatteringCoordinates(float u_r, float u_mu,
  float u_mu_l, float u_nu, float atmosphereRadius, float planetRadius) {
  float2 r_mu = unmap_r_mu(u_r, u_mu, atmosphereRadius,
    planetRadius);
  return float4(r_mu, unmap_mu_l(u_mu_l), unmap_nu(u_nu));
}


/******************************************************************************/
/**************************** END TEXTURE MAPPING *****************************/
/******************************************************************************/





/******************************************************************************/
/******************************** ATMOSPHERE **********************************/
/******************************************************************************/

/* Computes density at a point for exponentially distributed atmosphere.
 * Assumes the planet is centered at the origin. */
float computeDensityExponential(float3 p, float planetR, float scaleHeight,
  float density) {
  return density * exp((planetR - length(p))/scaleHeight);
}

/* Computes density at a point for tent distributed atmosphere.
 * Assumes the planet is centered at the origin. */
float computeDensityTent(float3 p, float planetR, float height,
  float thickness, float density) {
  return density * max(0.0,
    1.0 - abs(length(p) - planetR - height) / (0.5 * thickness));
}

/* Computes the optical depth for an exponentially distributed layer. */
float computeOpticalDepthExponential(float3 originPoint, float3 samplePoint,
  float planetR, float scaleHeight, float density, int numberOfSamples) {
  // Evaluate integral over curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float acc = 0.0;
  for (int i = 0; i < numberOfSamples; i++) {
    /* Compute where along the ray we're going to sample. */
    float2 t_ds = _useImportanceSampling ?
       (generateCubicSampleFromIndex(i, numberOfSamples)) :
       (generateLinearSampleFromIndex(i, numberOfSamples));

    /* Compute the point we're going to sample at. */
    float3 pt = originPoint + (d * t_ds.x);

    /* Accumulate the density at that point. */
    acc += computeDensityExponential(pt, planetR, scaleHeight, density)
      * t_ds.y * length(d);
  }
  return acc;
}

/* Computes the optical depth for a layer distributed as a tent
 * function at specified height with specified thickness. */
float computeOpticalDepthTent(float3 originPoint, float3 samplePoint,
  float planetR, float height, float thickness, float density,
  int numberOfSamples) {
  // Evaluate integral over curved planet with a midpoint integrator.
  float3 d = samplePoint - originPoint;
  float acc = 0.0;
  for (int i = 0; i < numberOfSamples; i++) {
    /* Compute where along the ray we're going to sample. For the ozone
     * layer, it's much better to use the linearly distributed samples,
     * as opposed to the cubically distributed ones. */
    float2 t_ds = generateLinearSampleFromIndex(i, numberOfSamples);

    /* Compute the point we're going to sample at. */
    float3 pt = originPoint + (d * t_ds.x);

    /* Accumulate the density at that point. */
    acc += computeDensityTent(pt, planetR, height, thickness, density)
      * t_ds.y * length(d);
  }
  return acc;
}

float computeAirPhase(float dot_L_d) {
  return 3.0 / (16.0 * PI) * (1.0 + dot_L_d * dot_L_d);
}

float computeAerosolPhase(float dot_L_d, float g) {
  return 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + dot_L_d * dot_L_d))
    / ((2.0 + g * g) * pow(abs(1.0 + g * g - 2.0 * g * dot_L_d), 1.5));
}

float computeCloudPhase(float dot_L_d, float g0, float g1, float silverIntensity) {
  /**/
  return max(computeAerosolPhase(dot_L_d, g0), silverIntensity * computeAerosolPhase(dot_L_d, 0.99-g1));
}

/******************************************************************************/
/****************************** END ATMOSPHERE ********************************/
/******************************************************************************/



#endif // EXPANSE_SKY_COMMON_INCLUDED
