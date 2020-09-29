/* Much of this is adapted from https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf. */
Shader "HDRP/Sky/ExpanseSky"
{
  HLSLINCLUDE

  #pragma vertex Vert

  #pragma editor_sync_compilation
  #pragma target 4.5
  #pragma only_renderers d3d11 ps4 xboxone vulkan metal switch


  /* Unity. */
  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
  #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonLighting.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/LightDefinition.cs.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Sky/SkyUtils.hlsl"
  #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/LightLoop/CookieSampling.hlsl"

  /* Common functions and global variables. */
  #include "Assets/CustomSky/expanse-clouds/code/ExpanseSkyCommon.hlsl"

/* TODO: had to remove a check that physically based sky was active in
 * the main light loop C# script to get additional light data like light
 * distance. Is there a way to avoid hacking the source? EDIT: also looks
 * like it resets it every time you open the editor. */

/********************************************************************************/
/****************************** UNIFORM VARIABLES *******************************/
/********************************************************************************/

  TEXTURECUBE(_nightSkyTexture);
  bool _hasNightSkyTexture;
  float4 _nightTint;
  float4x4 _nightSkyRotation;
  float _nightIntensity;
  float4 _skyTint;
  float _starAerosolScatterMultiplier;
  float _multipleScatteringMultiplier;
  bool _useAntiAliasing;
  float _ditherAmount;

  /* Celestial bodies. */
  /* Body 1. */
  float _body1LimbDarkening;
  bool _body1ReceivesLight;
  TEXTURECUBE(_body1AlbedoTexture);
  bool _body1HasAlbedoTexture;
  bool _body1Emissive;
  TEXTURECUBE(_body1EmissionTexture);
  bool _body1HasEmissionTexture;
  float4x4 _body1Rotation;
  /* Body 2. */
  float _body2LimbDarkening;
  bool _body2ReceivesLight;
  TEXTURECUBE(_body2AlbedoTexture);
  bool _body2HasAlbedoTexture;
  bool _body2Emissive;
  TEXTURECUBE(_body2EmissionTexture);
  bool _body2HasEmissionTexture;
  float4x4 _body2Rotation;
  /* Body 3. */
  float _body3LimbDarkening;
  bool _body3ReceivesLight;
  TEXTURECUBE(_body3AlbedoTexture);
  bool _body3HasAlbedoTexture;
  bool _body3Emissive;
  TEXTURECUBE(_body3EmissionTexture);
  bool _body3HasEmissionTexture;
  float4x4 _body3Rotation;
  /* Body 4. */
  float _body4LimbDarkening;
  bool _body4ReceivesLight;
  TEXTURECUBE(_body4AlbedoTexture);
  bool _body4HasAlbedoTexture;
  bool _body4Emissive;
  TEXTURECUBE(_body4EmissionTexture);
  bool _body4HasEmissionTexture;
  float4x4 _body4Rotation;

  /* Clouds. */
  float _cloudDensity;
  float _cloudForwardScatteringCoefficient;
  float _cloudBackwardScatteringCoefficient;
  int _numCloudTransmittanceSamples;
  int _numCloudSingleScatteringSamples;


  /* HACK: We only allow 4 celestial bodies now. */
  #define MAX_DIRECTIONAL_LIGHTS 4

  float3   _WorldSpaceCameraPos1;
  float4x4 _ViewMatrix1;
  #undef UNITY_MATRIX_V
  #define UNITY_MATRIX_V _ViewMatrix1


  /* Redefine colors to float3's for efficiency, since Unity can only set
   * float4's. */
  #define _nightTintF3 _nightTint.xyz
  #define _skyTintF3 _skyTint.xyz

/********************************************************************************/
/**************************** END UNIFORM VARIABLES *****************************/
/********************************************************************************/
//
// SerializedDataParameter cloudCoarseMarchFraction;
// SerializedDataParameter cloudDetailMarchFraction;
// SerializedDataParameter cloudVolumeLowerRadialBoundary;
// SerializedDataParameter cloudVolumeUpperRadialBoundary;
// SerializedDataParameter cloudTextureAngularRange;
// SerializedDataParameter cloudFalloffRadius;
// SerializedDataParameter basePerlinOctaves;
// SerializedDataParameter basePerlinOffset;
// SerializedDataParameter basePerlinScaleFactor;
// SerializedDataParameter baseWorleyOctaves;
// SerializedDataParameter baseWorleyScaleFactor;
// SerializedDataParameter structureOctaves;
// SerializedDataParameter structureScaleFactor;
// SerializedDataParameter detailOctaves;
// SerializedDataParameter detailScaleFactor;

  struct Attributes
  {
    uint vertexID : SV_VertexID;
    UNITY_VERTEX_INPUT_INSTANCE_ID
  };

  struct Varyings
  {
    float4 positionCS : SV_POSITION;

    UNITY_VERTEX_OUTPUT_STEREO
  };

  /* Vertex shader just sets vertex position. All the heavy lifting is
   * done in the fragment shader. */
  Varyings Vert(Attributes input)
  {
    Varyings output;
    UNITY_SETUP_INSTANCE_ID(input);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
    output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID, UNITY_RAW_FAR_CLIP_VALUE);

    return output;
  }

  /* Compute the luminance of a Celestial given the illuminance and the cosine
   * of half the angular extent. */
  float3 computeCelestialBodyLuminance(float3 zenithIlluminance, float cosTheta) {
    /* Compute solid angle. */
    float solidAngle = 2.0 * PI * (1.0 - cosTheta);
    return zenithIlluminance / solidAngle;
  }

  float3 limbDarkening(float LdotV, float cosInner, float amount) {
    /* amount = max(FLT_EPS, amount); */
    float centerToEdge = 1.0 - abs((LdotV - cosInner) / (1.0 - cosInner));
    float mu = safeSqrt(1.0 - centerToEdge * centerToEdge);
    float mu2 = mu * mu;
    float mu3 = mu2 * mu;
    float mu4 = mu2 * mu2;
    float mu5 = mu3 * mu2;
    float3 a0 = float3 (0.34685, 0.26073, 0.15248);
    float3 a1 = float3 (1.37539, 1.27428, 1.38517);
    float3 a2 = float3 (-2.04425, -1.30352, -1.49615);
    float3 a3 = float3 (2.70493, 1.47085, 1.99886);
    float3 a4 = float3 (-1.94290, -0.96618, -1.48155);
    float3 a5 = float3 (0.55999, 0.26384, 0.44119);
    return max(0.0, pow(a0 + a1 * mu + a2 * mu2 + a3 * mu3 + a4 * mu4 + a5 * mu5, amount));
  }

  /* Accumulate direct lighting on the ground. */
  float3 shadeGround(float3 groundPoint) {
    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */
    float3 result = float3(0, 0, 0);
    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && (uint) k < MAX_DIRECTIONAL_LIGHTS) {
      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        /* Get the light direction and color. */
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;

        /* Get the ground emission and add it to the direct light.  */
        float3 groundEmission = float3(0, 0, 0);
        if (_hasGroundEmissionTexture) {
          groundEmission = _groundEmissionMultiplier
            * SAMPLE_TEXTURECUBE_LOD(_groundEmissionTexture,
              s_linear_clamp_sampler, mul(normalize(groundPoint), (float3x3)_planetRotation), 0).rgb;
        }
        result += groundEmission;

        /* Get the ground albedo. */
        float3 groundAlbedo = _groundTintF3;
        if (_hasGroundAlbedoTexture) {
          groundAlbedo *= 2.0 * SAMPLE_TEXTURECUBE_LOD(_groundAlbedoTexture,
            s_linear_clamp_sampler, mul(normalize(groundPoint), (float3x3)_planetRotation), 0).rgb;
        }
        groundAlbedo /= PI;

        /* Compute the ground reflectance from the light. */
        float cos_hit_l = dot(normalize(groundPoint), L);
        result += groundAlbedo * lightColor * saturate(cos_hit_l);

        /* Compute the ground reflectance from the sky. */
        float2 groundIrradianceUV = mapGroundIrradianceCoordinates(cos_hit_l);
        float3 groundIrradianceAir =
          SAMPLE_TEXTURE2D_LOD(_GroundIrradianceTableAir,
          s_linear_clamp_sampler, groundIrradianceUV, 0).rgb;
        float3 groundIrradianceAerosol =
          SAMPLE_TEXTURE2D_LOD(_GroundIrradianceTableAerosol,
          s_linear_clamp_sampler, groundIrradianceUV, 0).rgb;
        result += groundAlbedo * lightColor
          * (_skyTintF3 * 2.0 * _airCoefficientsF3 * groundIrradianceAir
            + _aerosolCoefficient * groundIrradianceAerosol);
        k++;
      }
      i++;
    }
    return result;
  }

  /* Shade the celestial body that is closest. Return a negative vector if we
   * didn't hit anything. TODO: this doesn't work for eclipses. */
  float3 shadeClosestCelestialBody(float3 d) {
    /* Define arrays of celestial body variables to make things easier to
     * write. Syntactic sugar, if you will. */
    float celestialBodyLimbDarkening[MAX_DIRECTIONAL_LIGHTS] =
      {_body1LimbDarkening, _body2LimbDarkening, _body3LimbDarkening, _body4LimbDarkening};
    bool celestialBodyReceivesLight[MAX_DIRECTIONAL_LIGHTS] =
      {_body1ReceivesLight, _body2ReceivesLight, _body3ReceivesLight, _body4ReceivesLight};
    TextureCube celestialBodyAlbedoTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1AlbedoTexture, _body2AlbedoTexture, _body3AlbedoTexture, _body4AlbedoTexture};
    bool celestialBodyHasAlbedoTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1HasAlbedoTexture, _body2HasAlbedoTexture, _body3HasAlbedoTexture, _body4HasAlbedoTexture};
    bool celestialBodyEmissive[MAX_DIRECTIONAL_LIGHTS] =
      {_body1Emissive, _body2Emissive, _body3Emissive, _body4Emissive};
    TextureCube celestialBodyEmissionTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1EmissionTexture, _body2EmissionTexture, _body3EmissionTexture, _body4EmissionTexture};
    bool celestialBodyHasEmissionTexture[MAX_DIRECTIONAL_LIGHTS] =
      {_body1HasEmissionTexture, _body2HasEmissionTexture, _body3HasEmissionTexture, _body4HasEmissionTexture};
    float4x4 celestialBodyRotations[MAX_DIRECTIONAL_LIGHTS] =
      {_body1Rotation, _body2Rotation, _body3Rotation, _body4Rotation};

    /* Keep track of the closest body we've seen. */
    float minDist = -1.0;
    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */
    float3 celestialBodyLight = float3(0.0, 0.0, 0.0);
    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && (uint) k < MAX_DIRECTIONAL_LIGHTS) {
      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        float3 L = -normalize(light.forward.xyz);
        float LdotV    = dot(L, d);
        float radInner = 0.5 * light.angularDiameter;
        float cosInner = cos(radInner);
        float cosOuter = cos(radInner + light.flareSize); /* TODO: use for flares in the future. */
        if (LdotV > cosInner + FLT_EPSILON && (minDist < 0 || light.distanceFromCamera < minDist)) {
          /* We can see the celestial body and it's closer than the others
           * we've seen so far. Reset minDist and the celestial body
           * light accumulator. */
          minDist = light.distanceFromCamera;
          celestialBodyLight = float3(0.0, 0.0, 0.0);

          /* We take the approach of allowing a celestial body both to be
           * emissive and to receive light. This is useful for portraying
           * something like city lights on a moon. */

          /* Add the emissive contribution if the celestial body is
           * emissive. */
          if (celestialBodyEmissive[i]) {
            float3 emission = computeCelestialBodyLuminance(light.color, cosInner);
            /* Apply limb darkening. */
            emission *= limbDarkening(LdotV, cosInner, celestialBodyLimbDarkening[i]);
            /* Apply surface tint. */
            emission *= light.surfaceTint.rgb;
            /* Apply emission texture if we have one. */
            if (celestialBodyHasEmissionTexture[i]) {
              /* We have to do some work to compute the intersection. */
              float bodyDist = light.distanceFromCamera;
              float bodyRadius = safeSqrt(1.0 - cosInner * cosInner) * bodyDist;
              float3 planetOriginInBodyFrame = -(L * bodyDist);
              /* Intersect the body at the point we're looking at. */
              float3 bodyIntersection =
              intersectSphere(planetOriginInBodyFrame, d, bodyRadius);
              float3 bodyIntersectionPoint = planetOriginInBodyFrame
                + minNonNegative(bodyIntersection.x, bodyIntersection.y) * d;
              float3 surfaceNormal = normalize(bodyIntersectionPoint);

              emission *=
                SAMPLE_TEXTURECUBE_LOD(celestialBodyEmissionTexture[i],
                s_linear_clamp_sampler, mul(surfaceNormal, (float3x3) celestialBodyRotations[i]), 0).rgb;
            }
            celestialBodyLight += emission;
          }

          /* Add the diffuse contribution if the celestial body receives
           * light. */
          if (celestialBodyReceivesLight[i]) {
            /* We have to do some work to compute the surface normal. */
            float bodyDist = light.distanceFromCamera;
            float bodyRadius = safeSqrt(1.0 - cosInner * cosInner) * bodyDist;
            float3 planetOriginInBodyFrame = -(L * bodyDist);
            float3 bodyIntersection = intersectSphere(planetOriginInBodyFrame, d, bodyRadius);
            float3 bodyIntersectionPoint = planetOriginInBodyFrame + minNonNegative(bodyIntersection.x, bodyIntersection.y) * d;
            float3 bodySurfaceNormal = normalize(bodyIntersectionPoint);

            float3 bodyAlbedo = 2.0 * light.surfaceTint.rgb;
            if (celestialBodyHasAlbedoTexture[i]) {
              bodyAlbedo *= SAMPLE_TEXTURECUBE_LOD(celestialBodyAlbedoTexture[i],
                s_linear_clamp_sampler, mul(bodySurfaceNormal, (float3x3) celestialBodyRotations[i]), 0).rgb;
            }
            bodyAlbedo *= 1.0/PI;

            int j = 0;
            int jk = 0;
            while (j < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
              && jk < MAX_DIRECTIONAL_LIGHTS) {
              DirectionalLightData emissiveLight = _DirectionalLightDatas[j];
              /* Body can't light itself. */
              if (j != i && asint(emissiveLight.distanceFromCamera) >= 0) {
                /* Since both bodies may be pretty far away, we can't just
                 * use the emissive body's direction. We have to take
                 * the difference in body positions. */
                float3 emissivePosition = emissiveLight.distanceFromCamera
                  * -normalize(emissiveLight.forward.xyz);
                float3 bodyPosition = L * bodyDist;
                float3 emissiveDir = normalize(emissivePosition - bodyPosition);
                celestialBodyLight += saturate(dot(emissiveDir, bodySurfaceNormal))
                  * emissiveLight.color * bodyAlbedo;
                jk++;
              }
              j++;
            }
          }
        }
        k++;
      }
      i++;
    }
    /* Return a negative vector if we didn't hit anything. */
    if (minDist < 0) {
      celestialBodyLight = float3(-1.0, -1.0, -1.0);
    }
    return celestialBodyLight;
  }

  /* Accumulate sky color. */
  float3 shadeSky(float r, float mu, float3 startPoint, float t_hit,
    float3 d, bool groundHit) {

    /* Precompute some things outside the loop for efficiency. */
    float3 normalizedStartPoint = normalize(startPoint);

    int i = 0; /* Number of lights we've seen. */
    int k = 0; /* Number of lights that apply to the sky. */

    float3 skyColor = float3(0, 0, 0);

    while (i < min(_DirectionalLightCount, 2 * MAX_DIRECTIONAL_LIGHTS)
      && k < MAX_DIRECTIONAL_LIGHTS) {

      DirectionalLightData light = _DirectionalLightDatas[i];
      /* This lets us know if the light affects the physical sky. */
      if (asint(light.distanceFromCamera) >= 0) {
        float3 L = -normalize(light.forward.xyz);
        float3 lightColor = light.color;

        /* Mu is the zenith angle of the light. */
        float mu_l = clampCosine(dot(normalize(startPoint), L));

        /* Nu is the azimuth angle of the light, relative to the projection of
         * d onto the plane tangent to the surface of the planet at point O. */
        /* Project both L and d onto that plane by removing their "O"
         * component. */
        float3 proj_L = normalize(L - normalizedStartPoint * mu_l);
        float3 proj_d = normalize(d - normalizedStartPoint * dot(normalizedStartPoint, d));
        /* Take their dot product to get the cosine of the angle between them. */
        float nu = clampCosine(dot(proj_L, proj_d));

        TexCoord4D ssCoord = mapSingleScatteringCoordinates(r, mu, mu_l, nu,
          _atmosphereRadius, _planetRadius, t_hit, groundHit);

        float3 singleScatteringContributionAir =
          sampleTexture4D(_SingleScatteringTableAir, ssCoord);

        float3 singleScatteringContributionAerosol =
          sampleTexture4D(_SingleScatteringTableAerosol, ssCoord);

        float dot_L_d = dot(L, d);
        float rayleighPhase = computeAirPhase(dot_L_d);
        float miePhase = computeAerosolPhase(dot_L_d, g);

        float3 finalSingleScattering = (2.0 * _skyTintF3 * _airCoefficientsF3
          * singleScatteringContributionAir * rayleighPhase
          + _aerosolCoefficient * singleScatteringContributionAerosol * miePhase);

        /* Sample multiple scattering. */
        TexCoord4D msCoord = mapGlobalMultipleScatteringCoordinates(r, mu,
          mu_l, nu, _atmosphereRadius, _planetRadius, t_hit,
          groundHit);

        float3 msAir =
          sampleTexture4D(_GlobalMultipleScatteringTableAir, msCoord);

        float3 msAerosol =
          sampleTexture4D(_GlobalMultipleScatteringTableAerosol, msCoord);

        float3 finalMultipleScattering = (2.0 * _skyTintF3
          * _airCoefficientsF3 * msAir
          + _aerosolCoefficient * msAerosol)
          * _multipleScatteringMultiplier;

        skyColor +=
          (finalSingleScattering + finalMultipleScattering) * lightColor;

        k++;

      }

      i++;

    }
    return skyColor;
  }

  /* Accumulate light pollution. */
  float3 shadeLightPollution(float2 uv) {
    float3 lightPollutionAir = SAMPLE_TEXTURE2D(_LightPollutionTableAir,
      s_linear_clamp_sampler, uv).xyz;
    float3 lightPollutionAerosol = SAMPLE_TEXTURE2D(_LightPollutionTableAerosol,
      s_linear_clamp_sampler, uv).xyz;
    /* Consider light pollution to be isotropically scattered, so don't
     * apply phase functions. */
    return ((2.0 * _skyTintF3 * _airCoefficientsF3
      * lightPollutionAir + _aerosolCoefficient * lightPollutionAerosol)
      * _lightPollutionTintF3 * _lightPollutionIntensity).xyz;
  }

  /* Accumulate scattering due to the night sky. */
  float3 shadeNightScattering(float r, float mu, float t_hit, bool groundHit) {
    /* HACK: to get some sort of approximation of rayleigh scattering
     * for the ambient night color of the sky. */
    TexCoord4D ssCoord_night = mapSingleScatteringCoordinates(r, mu, mu, 1.0,
      _atmosphereRadius, _planetRadius, t_hit, groundHit);
     float3 singleScatteringContributionAirNight =
       sampleTexture4D(_SingleScatteringTableAir, ssCoord_night);
     float rayleighPhase_night = computeAirPhase(1.0);

     float3 finalSingleScatteringNight = (2.0 * _skyTintF3 * _airCoefficientsF3
       * singleScatteringContributionAirNight * rayleighPhase_night);

     /* Sample multiple scattering. */
     TexCoord4D msCoord_night = mapGlobalMultipleScatteringCoordinates(r, mu,
       mu, 1, _atmosphereRadius, _planetRadius, t_hit, groundHit);

     float3 msAirNight =
       sampleTexture4D(_GlobalMultipleScatteringTableAir, msCoord_night);

     float3 finalMultipleScatteringNight = (2.0 * _skyTintF3
       * _airCoefficientsF3 * msAirNight)
       * _multipleScatteringMultiplier;

    /* HACK: 1.0/8.0 here is a magic number representing something like the
     * star density. */
    return (finalSingleScatteringNight
      + finalMultipleScatteringNight) * _nightTintF3 * _nightIntensity/8.0;
  }

  /* Map point in world space to cloud texture uvw coordinate. */
  float3 mapCloudUV(float3 p, float2 y_extent, float angularExtent, int xz_tile, int y_tile) {
    /* v is the distance from the planet origin */
    float y = length(p) - _planetRadius;
    y_extent -= _planetRadius;
    float v = (y - y_extent.x) / (y_extent.y - y_extent.x);
    /* We require a small bias here. */
    v = saturate(frac(0.001 + v * y_tile + _cloudVOffset));

    float3 p_norm = normalize(p);

    float u_raw = (1.0 / (2.0 * angularExtent)) * (p_norm.x + angularExtent);
    float u = frac(0.001 + u_raw * xz_tile + _cloudUOffset);

    float w_raw = (1.0 / (2.0 * angularExtent)) * (p_norm.z + angularExtent);
    float w = frac(0.001 + w_raw * xz_tile + _cloudWOffset);

    return saturate(float3(u, v, w));
  }

  /* Sample cloud density textures at specified texture coordinates. */
  float sampleCloudDensity(float3 base_uvw, float3 detail_uvw, float distance) {
    /* Use LOD version of sampling function so that compiler doesn't have to
     * unroll the loop. */
    float4 cloud_base_noise =
      SAMPLE_TEXTURE3D_LOD(_CloudBaseNoiseTable, s_linear_clamp_sampler, base_uvw, 0);
    float3 cloud_detail_noise =
      SAMPLE_TEXTURE3D_LOD(_CloudDetailNoiseTable, s_linear_clamp_sampler, detail_uvw, 0).xyz;
    float3 cloud_coverage =
      SAMPLE_TEXTURE2D_LOD(_CloudCoverageTable, s_linear_clamp_sampler, base_uvw.xz, 0).xyz;

    float lowFreqNoise = cloud_base_noise.x;
    float coverage = dot(cloud_coverage, float3(1, 1, 1)/3.0);
    float density = remap(lowFreqNoise, coverage, 1.0, 0.0, 1.0);

    /* HACK: height gradient. Make parameterized. */
    float heightGradient = base_uvw.y - 0.5;
    if (heightGradient < 0) {
      heightGradient = abs(heightGradient);
      heightGradient = 0.5 - heightGradient;
      heightGradient = saturate(10 * heightGradient);
    } else {
      heightGradient = abs(heightGradient);
      heightGradient = 0.5 - heightGradient;
      heightGradient = saturate(2 * heightGradient + 0.25);
    }
    density = remap(density, 1-heightGradient, 1.0, 0.0, 1.0);

    float structureNoise = _structureNoiseBlendFactor * dot(cloud_base_noise.yzw, float3(1, 1, 1));
    density = remap(density, structureNoise, 1.0, 0.0, 1.0);

    float uberDetail = _detailNoiseBlendFactor * dot(cloud_detail_noise, float3(1, 1, 1));
    density = remap(density, uberDetail, 1.0, 0.0, 1.0);

    float falloff = min(1.0, exp((_cloudFalloffRadius-distance)));
    return max(0.0, falloff * _cloudDensity * density); /* HACK: density hack x10. */
  }

  /* Sample cloud density textures at specified texture coordinates. */
  float sampleCloudDensityLowFrequency(float3 base_uvw, float distance) {
    /* Use LOD version of sampling function so that compiler doesn't have to
     * unroll the loop. */
    float4 cloud_base_noise =
      SAMPLE_TEXTURE3D_LOD(_CloudBaseNoiseTable, s_linear_clamp_sampler, base_uvw, 0);
    float3 cloud_coverage =
      SAMPLE_TEXTURE2D_LOD(_CloudCoverageTable, s_linear_clamp_sampler, base_uvw.xz, 0).xyz;

    float lowFreqNoise = cloud_base_noise.x;
    float coverage = dot(cloud_coverage, float3(1, 1, 1)/3.0);
    float density = remap(lowFreqNoise, coverage, 1.0, 0.0, 1.0);

    /* HACK: height gradient. Make parameterized. */
    float heightGradient = base_uvw.y - 0.5;
    if (heightGradient < 0) {
      heightGradient = abs(heightGradient);
      heightGradient = 0.5 - heightGradient;
      heightGradient = saturate(10 * heightGradient);
    } else {
      heightGradient = abs(heightGradient);
      heightGradient = 0.5 - heightGradient;
      heightGradient = saturate(2 * heightGradient + 0.25);
    }
    density = remap(density, 1-heightGradient, 1.0, 0.0, 1.0);

    /* Just use lowest frequency structure noise. */
    float structureNoise = _structureNoiseBlendFactor * dot(cloud_base_noise.yzw, float3(1, 1, 1));
    density = remap(density, structureNoise, 1.0, 0.0, 1.0);

    float falloff = min(1.0, exp((_cloudFalloffRadius-distance)));
    return max(0.0, falloff * _cloudDensity * density); /* HACK: density hack x10. */
  }

  /* First number is entry t, second number is exit t. If entry t is
   * negative, we are inside the cloud volume or there is no intersection.
   * If the exit t is negative, there is no intersection. */
  float2 intersectCloudVolume(float3 p, float3 d, float2 y_extent) {
    /* Compute ground, lower, and upper intersections. */
    float3 ground_intersection = intersectSphere(p, d, _planetRadius);
    float3 lower_intersection = intersectSphere(p, d, y_extent.x);
    float3 upper_intersection = intersectSphere(p, d, y_extent.y);

    bool ground_hit = (ground_intersection.z > 0
      && (ground_intersection.x > 0 || ground_intersection.y > 0));
    bool upper_hit = (upper_intersection.z > 0
      && (upper_intersection.x > 0 || upper_intersection.y > 0));
    bool lower_hit = (lower_intersection.z > 0
      && (lower_intersection.x > 0 || lower_intersection.y > 0));

    float2 hit = float2(0, 0);

    /* If we hit the ground, we are guaranteed to have hit the other layers. */
    float ground_t = minNonNegative(ground_intersection.x, ground_intersection.y);
    float lower_t = minNonNegative(lower_intersection.x, lower_intersection.y);
    float upper_t = minNonNegative(upper_intersection.x, upper_intersection.y);

    if (ground_hit) {
      if (ground_t < lower_t) {
        /* Ground hit is closest hit. */
        hit = float2(-1, -1);
      } else if (lower_t < ground_t) {
        /* We're in the cloud volume. */
        hit = float2(0, lower_t);
      } else {
        /* We are above the cloud volume looking at the planet. */
        hit = float2(upper_t, lower_t);
      }
    } else {
      /* Things are a little more complicated. */
      if (lower_hit) {
        /* We also have an upper hit, otherwise we'd have hit the
         * ground. The question is, do we have one or two of each?
         * HACK: for now, assume we have only one. There's no way to
         * specify 2 regions to raymarch right now. */
         if (lower_intersection.x > 0 && lower_intersection.y > 0) {
           /* We have two lower hits, so we're grazing the surface
            * of the planet. */
            if (upper_intersection.x > 0 && upper_intersection.y > 0) {
              /* We've also got two upper hits. */
              hit = float2(upper_t, lower_t);
            } else {
              /* We're in the cloud volume, and we have two lower hits
               * and one upper hit. */
              hit = float2(0, lower_t);
            }
         } else {
           /* We have just one lower hit, so we are just looking up
            * through the cloud volume. */
            hit = float2(lower_t, upper_t);
         }
      } else {
        /* We are either... */
        if (!upper_hit) {
          /* In space looking out into space. */
          hit = float2(-1, -1);
        } else if (upper_intersection.x > 0 && upper_intersection.y > 0) {
          /* In space looking through the volume but never hitting the
           * lower part. */
          hit = float2(min(upper_intersection.x, upper_intersection.y),
            max(upper_intersection.x, upper_intersection.y));
        } else {
          /* In the cloud volume looking up. */
          hit = float2(0, upper_t);
        }
      }
    }

    return hit;
  }

  /* Compute cloud transmittance from optical depth and absorption. */
  float cloudTransmittance(float opticalDepth, float absorption) {
    // return saturate(exp(-absorption * opticalDepth));
    return saturate(max(exp(-absorption * opticalDepth), 0.7 * exp(-0.25 * absorption * opticalDepth)));
  }

  /* Shades clouds at point O looking in direction d. First 3 numbers are
   * cloud scattering color, last number is cloud transmittance. */
  float4 shadeClouds(float3 O, float3 d) {

    float2 yExtent = float2(_cloudVolumeLowerRadialBoundary,
      _cloudVolumeUpperRadialBoundary) + _planetRadius;
    float2 intersections = intersectCloudVolume(O, d, yExtent);

    if (intersections.x < 0 && intersections.y < 0) {
      /* We didn't hit the cloud volume. */
      return float4(0, 0, 0, 1);
    }

    // intersections.y = min(5 * _cloudFalloffRadius, intersections.y);
    // if (intersections.x > 5 * _cloudFalloffRadius) {
    //   return float4(0, 0, 0, 1);
    // }

    float3 startPoint = O + intersections.x * d;
    float3 endPoint = O + intersections.y * d;

    float pathLength = length(endPoint - startPoint);

    /* For viewing noise textures only. */
    /* TODO: configurable. */
    bool cloudsDebug = false;
    if (cloudsDebug) {
        float densityAttenuation = 1.0;
        float densityAttenThreshold = 0.20;
        if (d.y < densityAttenThreshold) {
          densityAttenuation = exp(30 * (d.y - densityAttenThreshold));
        }
        float3 cloud_UV_base = mapCloudUV(startPoint, yExtent, _cloudTextureAngularRange, 1, 1);
        float3 cloud_UV_detail = mapCloudUV(startPoint, yExtent, _cloudTextureAngularRange, 10, 10); /* HACK: 10's here. */
        float cloud_density = densityAttenuation * sampleCloudDensity(cloud_UV_base, cloud_UV_detail, length(startPoint - O));
        return float4(cloud_density, cloud_density, cloud_density, 1);
    }

    float t = 0.0; /* Distance we have marched so far. */
    float dt = _cloudCoarseMarchFraction; /* March interval. */
    int numStepsLowDensity = 0;
    float cloudOpticalDepth = 0.0;  /* Accumulated optical depth. */
    float cloudAlpha = 1.0; /* Accumulated alpha value. */
    float3 cloudLighting = float3(0, 0, 0); /* Accumulated lighting. */
    float3 atmosphericBlendFactor = float3(-1, -1, -1);
    float cloud_density = 1.0; /* For determining step size---start out at one to force detail step. */
    while (t < 1.0) {
      /* Modify step size. */
      if (cloud_density >= 0.01 * _cloudDensity) {
        dt = _cloudDetailMarchFraction;
        numStepsLowDensity = 0;
      } else {
        if (numStepsLowDensity == 9) {
          dt = _cloudCoarseMarchFraction;
          numStepsLowDensity = 0;
        } else {
          numStepsLowDensity++;
        }
      }
      dt = min(1 - t, dt);
      /* Generate sample point. */
      float sampleT = pathLength * (t + 0.5 * dt);
      float3 samplePoint = startPoint + sampleT * d;
      float ds = pathLength * dt;

      /* Sample the density at the sample point. */
      float3 cloud_UV_base = mapCloudUV(samplePoint, yExtent, _cloudTextureAngularRange, 1, 1);
      float3 cloud_UV_detail = mapCloudUV(samplePoint, yExtent, _cloudTextureAngularRange, 10, 10);
      cloud_density = sampleCloudDensity(cloud_UV_base, cloud_UV_detail, length(samplePoint - O));

      /* Attenuate density according to curve. */
      float densityAttenuation = 1.0;
      float densityAttenThreshold = 0.25;
      if (d.y < densityAttenThreshold) {
        densityAttenuation = exp(15 * (d.y - densityAttenThreshold));
      }

      /* Accumulate optical depth. */
      cloud_density *= densityAttenuation;
      cloudOpticalDepth += cloud_density * ds;

      /* Accumulate marched distance. */
      t += dt;

      /* Accumulate alpha. */
      float alphaTransmittance = saturate(cloudTransmittance(cloud_density * ds, 1));
      cloudAlpha *= alphaTransmittance;

      /* Light the cloud, if our density is non-zero. */
      if (cloud_density > 0) {
        float densityEstimate = 0.0;

        float singleScatterStepSize = pathLength * dt * 1;
        float3 L = -normalize(_DirectionalLightDatas[0].forward.xyz);
        for (int i = 0; i < _numCloudSingleScatteringSamples; i++) {
          /* Cone sample by adding a random offset. HACK: not really cone sample. */
          float3 coneSample = L * i * singleScatterStepSize;
          // coneSample += 10 * singleScatterStepSize * random_3_3(coneSample);
          float3 lightSamplePoint = samplePoint + coneSample;
          /* Sample the density at the sample point. */
          float3 light_UV_base = mapCloudUV(lightSamplePoint, yExtent, _cloudTextureAngularRange, 1, 1);
          float lowDensitySample = sampleCloudDensityLowFrequency(light_UV_base, length(lightSamplePoint - O));
          densityEstimate += dt * lowDensitySample;
        }

        float lowDensitySampleBase = sampleCloudDensityLowFrequency(cloud_UV_base, length(samplePoint - O));
        float depthProbability = 0.05 + pow(lowDensitySampleBase/_cloudDensity, max(0.0, remap(cloud_UV_base.y, 0.3, 0.85, 0.75, 1.0)));

        /* Accumulate cloud lighting. */
        cloudLighting += depthProbability * ds * cloud_density * (cloudAlpha);
      }

      /* Compute transmittance and write depth if we are above 0.5 and
       * haven't already written depth. */
      // if (atmosphericBlendFactor.x < 0) {
      //   float depthTestTransmittance = saturate(cloudTransmittance(cloudOpticalDepth * (1/t), 1));
      //   if ((1-cloudAlpha) >= 0.5) {
      //     /* HACK: should use atmospheric model, but this is just easier. */
      //     float dist = length(O - samplePoint);
      //     atmosphericBlendFactor = saturate(1 - 1/(1 + exp((-dist/5000)+4)));
      //   }
      // }
    }
    if (atmosphericBlendFactor.x < 0) {
      float dist = length(O - startPoint);
      atmosphericBlendFactor = saturate(1 - 1/(1 + exp((-dist/5000)+6)));
    }

    float cloudT = saturate(cloudTransmittance(cloudOpticalDepth, 1));
    float3 cloudColor = cloudLighting * _DirectionalLightDatas[0].color;
    /* Compute phase. */
    float3 L = -normalize(_DirectionalLightDatas[0].forward.xyz);
    /* HACK: blend is hardcoded. */
    float cloudPhase = computeCloudPhase(dot(L, d), _cloudForwardScatteringCoefficient, _cloudBackwardScatteringCoefficient, 0.5);
    cloudColor *= cloudPhase;

    if (atmosphericBlendFactor.x < 0) {
      /* Our clouds aren't very dense at all. Blend to full cloud. */
      atmosphericBlendFactor = float3(1, 1, 1);
    }

    return float4(cloudColor.x, cloudColor.y, 1-cloudAlpha, saturate(atmosphericBlendFactor.x));
  }

  float3 RenderSky(Varyings input, float exposure, float2 jitter)
  {

    /* Get the origin point and sample direction. */
    float3 O = _WorldSpaceCameraPos1 - float3(0, -_planetRadius, 0);
    float3 d = normalize(-GetSkyViewDirWS(input.positionCS.xy + jitter));

    /* Trace a ray to see what we hit. */
    IntersectionData intersection = traceRay(O, d, _planetRadius,
      _atmosphereRadius);

    /* You might think the start point is just O, but we may be looking
     * at the planet from space, in which case the start point is the point
     * that we hit the atmosphere. */
    float3 startPoint = O + d * intersection.startT;
    float3 endPoint = O + d * intersection.endT;
    float t_hit = intersection.endT - intersection.startT;

    /* Accumulate direct illumination. */
    float3 L0 = float3(0, 0, 0);
    if (intersection.groundHit) {
      L0 += shadeGround(endPoint);
    } else {
      float3 closestCelestialBodyShading = shadeClosestCelestialBody(d);
      if (closestCelestialBodyShading.x < 0) {
        /* We didn't hit anything. Add the stars. */
        float3 starTexture = SAMPLE_TEXTURECUBE_LOD(_nightSkyTexture,
          s_linear_clamp_sampler, mul(d, (float3x3)_nightSkyRotation), 0).rgb;
        L0 += starTexture * _nightTintF3 * _nightIntensity;
      } else {
        /* We did hit a celestial body. Add its shading. */
        L0 += shadeClosestCelestialBody(d);
      }
    }

    /* Compute r and mu for the lookup tables. */
    float r = length(startPoint);
    float mu = clampCosine(dot(normalize(startPoint), d));

    /* Perform the transmittance table lookup to attenuate direct lighting. */
    float2 transmittanceUV = mapTransmittanceCoordinates(r,
      mu, _atmosphereRadius, _planetRadius, t_hit, intersection.groundHit);
    float3 T = SAMPLE_TEXTURE2D(_TransmittanceTable, s_linear_clamp_sampler,
      transmittanceUV).xyz;

    /* HACKY CLOUDS STUFF. HACK. */
    float4 cloud_color_and_t = shadeClouds(O, d);
    float atmosphericBlend = cloud_color_and_t.w;
    float cloudAlpha = cloud_color_and_t.z;
    float cloudColor = cloud_color_and_t.x;

    /* TODO: configurable. */
    bool cloudsDebug = false;
    if (cloudsDebug) {
      return cloudColor;
    }

    float3 finalDirectLighting = (L0 * T);

    /* If we hit the sky or the ground, compute the sky color, air scattering
     * due to the night sky, and the light pollution. */
    float3 skyColor = float3(0, 0, 0);
    float3 nightAirScattering = float3(0, 0, 0);
    float3 lightPollution = float3(0, 0, 0);
    if (intersection.groundHit || intersection.atmoHit) {
      skyColor = shadeSky(r, mu, startPoint, t_hit, d, intersection.groundHit);
      lightPollution = shadeLightPollution(transmittanceUV);
      nightAirScattering = shadeNightScattering(r, mu, t_hit,
        intersection.groundHit);
    }

    float3 sky = finalDirectLighting + skyColor + nightAirScattering + lightPollution;
    float3 clouds = float3(cloudColor, cloudColor, cloudColor);
    float3 skyAndClouds = sky * (1-cloudAlpha) + cloudAlpha * (sky * (1-atmosphericBlend) + (atmosphericBlend) * cloudColor);

    float dither = 1.0 + _ditherAmount * (0.5 - random(d.xy));
    return dither * exposure * skyAndClouds;
  }

  float4 FragBaking(Varyings input) : SV_Target
  {
    return float4(0, 0, 0, 1);
    // return float4(RenderSky(input, 1.0, float2(0, 0)).xyz, 1.0);
  }

  float4 FragRender(Varyings input) : SV_Target
  {
    float MSAA_8X_OFFSETS_X[8] = {1.0/16.0, -1.0/16.0, 5.0/16.0, -3.0/16.0, -5.0/16.0, -7.0/16.0, 3.0/16.0, 7.0/16.0};
    float MSAA_8X_OFFSETS_Y[8] =  {-3.0/16.0, 3.0/16.0, 1.0/16.0, -5.0/16.0, 5.0/16.0, -1.0/16.0, 7.0/16.0, -7.0/16.0};
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
    float exposure = GetCurrentExposureMultiplier();
    float4 skyResult = float4(0.0, 0.0, 0.0, 1.0);
    if (_useAntiAliasing) {
      for (int i = 0; i < 8; i++) {
        skyResult += float4(RenderSky(input, exposure,
          float2(MSAA_8X_OFFSETS_X[i], MSAA_8X_OFFSETS_Y[i])), 1.0);
      }
      skyResult /= 8.0;
      skyResult.w = 1.0;
    } else {
      skyResult = float4(RenderSky(input, exposure, float2(0.0, 0.0)), 1.0);
    }
    return skyResult;
  }

  ENDHLSL

  SubShader
  {
    /* For cubemap. */
    Pass
    {
      ZWrite Off
      ZTest Always
      Blend Off
      Cull Off

      HLSLPROGRAM
      #pragma fragment FragBaking
      ENDHLSL
    }

    /* For fullscreen sky. */
    Pass
    {
      ZWrite Off
      ZTest LEqual
      Blend Off
      Cull Off

      HLSLPROGRAM
      #pragma fragment FragRender
      ENDHLSL
    }
  }
  Fallback Off
}
