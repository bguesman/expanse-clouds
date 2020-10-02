using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

[VolumeComponentMenu("Sky/Expanse Sky")]
// SkyUniqueID does not need to be part of built-in HDRP SkyType enumeration.
// This is only provided to track IDs used by HDRP natively.
// You can use any integer value.
[SkyUniqueID(EXPANSE_SKY_UNIQUE_ID)]
public class ExpanseSky : SkySettings
{
  const int EXPANSE_SKY_UNIQUE_ID = 100003;

  /********************************************************************************/
  /********************************* Parameters ***********************************/
  /********************************************************************************/

  /* Planet Parameters. */
  [Tooltip("The total thickness of the atmosphere, in meters.")]
  public ClampedFloatParameter atmosphereThickness = new ClampedFloatParameter(40000, 0, 200000);

  [Tooltip("The radius of the planet, in meters.")]
  public ClampedFloatParameter planetRadius = new ClampedFloatParameter(6360000, 10, 20000000);

  [Tooltip("The ground albedo as a cubemap texture. The ground is modeled as a Lambertian (completely diffuse) reflector. If no texture is specified, the color of the ground will just be the ground tint.")]
  public CubemapParameter groundAlbedoTexture = new CubemapParameter(null);

  [Tooltip("A color tint to the ground texture. Perfect grey, (128, 128, 128), specifies no tint. If there is no ground texture specified, this is just the color of the ground.")]
  public ColorParameter groundTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("The ground emission as a cubemap texture. Useful for modeling things like city lights. Has no effect on the sky. See \"Light Pollution\" for a way of modeling an emissive ground's effect on the atmosphere.")]
  public CubemapParameter groundEmissionTexture = new CubemapParameter(null);

  [Tooltip("An intensity multiplier on the ground emission texture.")]
  public ClampedFloatParameter groundEmissionMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 100000.0f);

  [Tooltip("The rotation of the planet textures as euler angles. This won't do anything to light directions, star rotations, etc. It is purely for rotating the planet's albedo and emissive textures.")]
  public Vector3Parameter planetRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Night Sky Parameters. */
  [Tooltip("The cubemap texture used to render stars and nebulae.")]
  public CubemapParameter nightSkyTexture = new CubemapParameter(null);

  [Tooltip("The rotation of the night sky texture as euler angles.")]
  public Vector3Parameter nightSkyRotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("A color tint to the night sky texture. Perfect grey, (128, 128, 128), specifies no tint. If there is no night sky texture specified, this is just the color of the night sky.")]
  public ColorParameter nightTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("An intensity multiplier on the night sky texture. Physical luminance values of stars on Earth are very close to zero. However, this rarely plays well with auto-exposure settings.")]
  public ClampedFloatParameter nightIntensity = new ClampedFloatParameter(10.0f, 0.0f, 100000.0f);

  [Tooltip("The color of light pollution from emissive elements on the ground, i.e. city lights, cars, buildings")]
  public ColorParameter lightPollutionTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("The intensity of light pollution from emissive elements on the ground, i.e. city lights, cars, buildings.")]
  public ClampedFloatParameter lightPollutionIntensity = new ClampedFloatParameter(0.0f, 0.0f, 10000.0f);

  /* Aerosol Parameters. */
  [Tooltip("The scattering coefficient for Mie scattering due to aerosols at sea level. The value on Earth is 0.000021.")]
  public FloatParameter aerosolCoefficient = new FloatParameter(0.000021f);

  [Tooltip("The scale height for aerosols. This parameterizes the density falloff of the aerosol layer as it extends toward space. Adjusting this and the aerosol scale height can give the sky a dusty or foggy look.")]
  public ClampedFloatParameter scaleHeightAerosols = new ClampedFloatParameter(1200, 0, 100000);

  [Tooltip("The anisotropy factor for Mie scattering. 0.76 is a reasonable value for Earth. 1.0 specifies fully directional toward the light source, and -1.0 specifies fully directional away from the light source.")]
  public ClampedFloatParameter aerosolAnisotropy = new ClampedFloatParameter(0.76f, -1.0f, 1.0f);

  [Tooltip("The density of aerosols in the atmosphere. 1.0 is the density you would find on Earth. Adjusting this and the aerosol scale height can give the sky a dusty or foggy look.")]
  public ClampedFloatParameter aerosolDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Air Parameters. */
  [Tooltip("The scattering coefficients for wavelength dependent Rayleigh scattering due to air at sea level. Adjusting this can subtly can model changes in the gas composition of the air on the Earth. Adjusting it dramatically will take you into the territory of alien skies.")]
  public Vector3Parameter airCoefficients = new Vector3Parameter(new Vector3(0.0000058f, 0.0000135f, 0.0000331f));

  [Tooltip("The scale height for air. This parameterizes the density falloff of the air layer as it extends toward space.")]
  public ClampedFloatParameter scaleHeightAir = new ClampedFloatParameter(8000, 0, 100000);

  [Tooltip("The density of the air. 1.0 is the density you would find on Earth.")]
  public ClampedFloatParameter airDensity = new ClampedFloatParameter(1.0f, 0.0f, 10.0f);

  /* Ozone Parameters. */
  [Tooltip("The scattering coefficients for wavelength dependent Rayleigh scattering due to the ozone at sea level.")]
  public Vector3Parameter ozoneCoefficients = new Vector3Parameter(new Vector3(0.00000206f, 0.00000498f, 0.000000214f));

  [Tooltip("The thickness of the ozone layer.")]
  public ClampedFloatParameter ozoneThickness = new ClampedFloatParameter(30000, 0, 100000);

  [Tooltip("The height of the ozone layer.")]
  public ClampedFloatParameter ozoneHeight = new ClampedFloatParameter(25000, 0, 100000);

  [Tooltip("Controls the density of the ozone layer. Anywhere between 0.0 and 1.0 is reasonable for a density you would find on Earth. Pushing this higher will deepen the blue of the daytime sky, and further saturate the vibrant colors of sunsets and sunrises.")]
  public ClampedFloatParameter ozoneDensity = new ClampedFloatParameter(0.3f, 0.0f, 10.0f);

  /* Height Fog Parameters. */
  [Tooltip("The scattering coefficients for Mie scattering due to aerosols at sea level. The value on Earth is 0.000021 for all color components. Typically fog scattering is not wavelength-dependent, but it can be useful to model it this way to get a particular look.")]
  public Vector3Parameter heightFogCoefficients = new Vector3Parameter(new Vector3(0.000021f, 0.000021f, 0.000021f));

  [Tooltip("The scale height for height fog. This parameterizes the density falloff of the height fog layer as it extends toward space. Adjusting this and the height fog density can give the player's immediate surroundings a dusty or foggy look.")]
  public ClampedFloatParameter scaleHeightHeightFog = new ClampedFloatParameter(1200, 0, 100000);

  [Tooltip("The anisotropy factor for Mie scattering. 0.76 is a reasonable value for Earth. 1.0 specifies fully directional toward the light source, and -1.0 specifies fully directional away from the light source.")]
  public ClampedFloatParameter heightFogAnisotropy = new ClampedFloatParameter(0.76f, -1.0f, 1.0f);

  [Tooltip("The density of height fog aerosols in the atmosphere. Adjusting this and the height fog scale height can give the area immediately around the player a dusty or foggy look.")]
  public ClampedFloatParameter heightFogDensity = new ClampedFloatParameter(100.0f, 0.0f, 1000.0f);

  [Tooltip("The distance over which the height fog attenuates exponentially.")]
  public ClampedFloatParameter heightFogAttenuationDistance = new ClampedFloatParameter(1000, 0, 100000);

  [Tooltip("The bias to the height fog attenuation---can push the height fog further out or closer in to the player.")]
  public ClampedFloatParameter heightFogAttenuationBias = new ClampedFloatParameter(1000, 0, 100000);

  [Tooltip("Artistic override for tinting the height fog.")]
  public ColorParameter heightFogTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  /* Artistic overrides. */
  [Tooltip("A tint to the overall sky color. Perfect grey, (128, 128, 128), specifies no tint.")]
  public ColorParameter skyTint = new ColorParameter(Color.grey, hdr: false, showAlpha: false, showEyeDropper: true);

  [Tooltip("A multiplier on the multiple scattering contribution. 1.0 is physically accurate. Pushing this above 1.0 can make the daytime sky brighter and more vibrant.")]
  public ClampedFloatParameter multipleScatteringMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);

  /* Celestial Bodies. TODO: as of now, we support up to 4 celestial bodies.
   * Settings such as angular diameter, angular position, distance,
   * color, intensity, and surface tint are specified in the directional light
   * object. But other settings, like limb darkening, cubemap textures,
   * and whether the body is a sun or a moon have no parameter in the
   * directional light itself, and so must be specified here if we aren't
   * going to hack the Unity base code. */
  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body1LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body1ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body1AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body1Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body1EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body1Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body2LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body2ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body2AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body2Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body2EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body2Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body3LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body3ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body3AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body3Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body3EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body3Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  [Tooltip("Darkens the edges of emissive celestial bodies. A value of 1.0 is "
  + "physically accurate for emissive celestial bodies. A value of 0.0 will turn off the effect "
  + "entirely. Higher values will darken more, lower values will darken less.")]
  public ClampedFloatParameter body4LimbDarkening = new ClampedFloatParameter(1.0f, 0.0f, 30.0f);
  [Tooltip("Whether the celestial body receives light from other celestial bodies.")]
  public BoolParameter body4ReceivesLight = new BoolParameter(false);
  [Tooltip("The texture for the surface albedo of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body4AlbedoTexture = new CubemapParameter(null);
  [Tooltip("Whether the celestial body is emissive.")]
  public BoolParameter body4Emissive = new BoolParameter(true);
  [Tooltip("The texture for emission of the celestial body. If no texture is specified, the surface tint (specified in the corresponding directional light) will be used.")]
  public CubemapParameter body4EmissionTexture = new CubemapParameter(null);
  [Tooltip("The rotation of the albedo and emission textures for the celestial body.")]
  public Vector3Parameter body4Rotation = new Vector3Parameter(new Vector3(0.0f, 0.0f, 0.0f));

  /* Sampling. */
  [Tooltip("The number of samples used when computing transmittance lookup tables. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 4 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 32 or higher is recommended.")]
  public ClampedIntParameter numberOfTransmittanceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when computing light pollution. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 8 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 64 or higher is recommended.")]
  public ClampedIntParameter numberOfLightPollutionSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when computing single scattering. With importance sampling turned on, a value of as low as 10 gives near-perfect results on the ground. A value as low as 5 is ok if some visible inaccuracy is tolerable. Without importantance sampling, a value of 32 or higher is recommended.")]
  public ClampedIntParameter numberOfScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples used when sampling the ground irradiance. Importance sampling does not apply here. To get a near-perfect result, around 10 samples is necessary. But it is a fairly subtle effect, so as low as 6 samples gives a decent result.")]
  public ClampedIntParameter numberOfGroundIrradianceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples to use when computing the initial isotropic estimate of multiple scattering. Importance sampling does not apply here. To get a near-perfect result, around 15 samples is necessary. But it is a fairly subtle effect, so as low as 6 samples gives a decent result.")]
  public ClampedIntParameter numberOfMultipleScatteringSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("The number of samples to use when computing the actual accumulated estimate of multiple scattering from the isotropic estimate. The number of samples to use when computing the initial isotropic estimate of multiple scattering. With importance sample, 8 samples gives a near-perfect result. However, multiple scattering is a fairly subtle effect, so as low as 3 samples gives a decent result. Without importance sampling, a value of 32 or higher is necessary for near perfect results, but a value of 4 is sufficient for most needs.")]
  public ClampedIntParameter numberOfMultipleScatteringAccumulationSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Whether or not to use importance sampling. Importance sampling is a sample distribution strategy that increases fidelity given a limited budget of samples. It is recommended to turn it on, as it doesn't decrease fidelity, but does allow for fewer samples to be taken, boosting performance. However, for outer-space perspectives, it can sometimes introduce inaccuracies, so it can be useful to increase sample counts and turn off importance sampling in those cases.")]
  public BoolParameter useImportanceSampling = new BoolParameter(true);

  [Tooltip("Whether or not to use MSAA 8x anti-aliasing. This does negatively affect performance.")]
  public BoolParameter useAntiAliasing = new BoolParameter(true);

  [Tooltip("Amount of dithering used to reduce color banding. If this is too high, noise will be visible.")]
  public ClampedFloatParameter ditherAmount = new ClampedFloatParameter(0.05f, 0.0f, 1.0f);


  /* Clouds geometry. */
  [Tooltip("Lower boundary of cloud volume in the radial (near the player, vertical) direction.")]
  public ClampedFloatParameter cloudVolumeLowerRadialBoundary = new ClampedFloatParameter(1000.0f, 0.0f, 50000.0f);

  [Tooltip("Upper boundary of cloud volume in the radial (near the player, vertical) direction.")]
  public ClampedFloatParameter cloudVolumeUpperRadialBoundary = new ClampedFloatParameter(8000.0f, 0.0f, 50000.0f);

  [Tooltip("Angular range of texture for tiling. Affects both z and x.")]
  public ClampedFloatParameter cloudTextureAngularRange = new ClampedFloatParameter(0.008f, 0.0001f, 0.05f);

  [Tooltip("U offset for texture coordinate. Can be used to animate the clouds in the x direction.")]
  public ClampedFloatParameter cloudUOffset = new ClampedFloatParameter(0.0f, 0.0f, 1.0f);

  [Tooltip("V offset for texture coordinate. Can be used to animate the clouds in the y direction.")]
  public ClampedFloatParameter cloudVOffset = new ClampedFloatParameter(0.0f, 0.0f, 1.0f);

  [Tooltip("W offset for texture coordinate. Can be used to animate the clouds in the z direction.")]
  public ClampedFloatParameter cloudWOffset = new ClampedFloatParameter(0.0f, 0.0f, 1.0f);

  /* Clouds lighting. */
  [Tooltip("Density of the clouds.")]
  public ClampedFloatParameter cloudDensity = new ClampedFloatParameter(0.01f, 0.0f, 1.0f);

  [Tooltip("Radius past which cloud densities will be attenuated exponentially.")]
  public ClampedFloatParameter cloudFalloffRadius = new ClampedFloatParameter(35000.0f, 100.0f, 100000.0f);

  [Tooltip("Density attenuation threshold in meters.")]
  public ClampedFloatParameter densityAttenuationThreshold = new ClampedFloatParameter(35000.0f, 100.0f, 100000.0f);

  [Tooltip("Density attenuation multiplier.")]
  public ClampedFloatParameter densityAttenuationMultiplier = new ClampedFloatParameter(1.0f, 0.0f, 100.0f);

  [Tooltip("Forward scattering coefficient.")]
  public ClampedFloatParameter cloudForwardScattering = new ClampedFloatParameter(0.9f, -1.0f, 1.0f);

  [Tooltip("Backward scattering coefficient.")]
  public ClampedFloatParameter cloudSilverSpread = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);

  [Tooltip("Intensity of silver highlights.")]
  public ClampedFloatParameter silverIntensity = new ClampedFloatParameter(0.5f, 0.0f, 1.0f);

  [Tooltip("Offset of depth probability.")]
  public ClampedFloatParameter depthProbabilityOffset = new ClampedFloatParameter(0.05f, 0.0f, 1.0f);

  [Tooltip("Minimum of depth probability.")]
  public ClampedFloatParameter depthProbabilityMin = new ClampedFloatParameter(0.75f, 0.0f, 3.0f);

  [Tooltip("Maximum of depth probability.")]
  public ClampedFloatParameter depthProbabilityMax = new ClampedFloatParameter(1.0f, 0.0f, 3.0f);

  [Tooltip("Distance over which to apply atmospheric blend.")]
  public ClampedFloatParameter atmosphericBlendDistance = new ClampedFloatParameter(2500.0f, 1.0f, 15000.0f);

  [Tooltip("Bias above which atmospheric blend starts to be applied.")]
  public ClampedFloatParameter atmosphericBlendBias = new ClampedFloatParameter(15000.0f, 0.0f, 100000.0f);

  /* Clouds sampling. */
  [Tooltip("Cloud transmittance samples.")]
  public ClampedIntParameter numCloudTransmittanceSamples = new ClampedIntParameter(10, 1, 256);

  [Tooltip("Cloud single scattering samples.")]
  public ClampedIntParameter numCloudSSSamples = new ClampedIntParameter(4, 1, 64);

  [Tooltip("Fraction along volume ray to march for accumulating coarse estimates.")]
  public ClampedFloatParameter cloudCoarseMarchStepSize = new ClampedFloatParameter(250, 50, 1000);

  [Tooltip("Fraction along volume ray to march for accumulating details.")]
  public ClampedFloatParameter cloudDetailMarchStepSize = new ClampedFloatParameter(100, 50, 1000);

  [Tooltip("Number of steps taken that read zero density before switching back to coarse march step size.")]
  public ClampedIntParameter numZeroStepsBeforeCoarseMarch = new ClampedIntParameter(10, 1, 20);


  /* Clouds noise. */
  [Tooltip("Octaves for FBM used in the base perlin noise.")]
  public Vector4Parameter basePerlinOctaves = new Vector4Parameter(new Vector4(8, 16, 32, 64));
  [Tooltip("Offset to the base perlin noise. Lowering this value will result in patchier, spottier clouds. Raising it will thicken the clouds.")]
  public ClampedFloatParameter basePerlinOffset = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);
  [Tooltip("FBM scale factor for base perlin noise. Octave N will have amplitude scale^N.")]
  public ClampedFloatParameter basePerlinScaleFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);

  [Tooltip("Octaves for FBM used in the base worley noise.")]
  public Vector3Parameter baseWorleyOctaves = new Vector3Parameter(new Vector3(16, 32, 64));
  [Tooltip("FBM scale factor for base worley noise. Octave N will have amplitude scale^N.")]
  public ClampedFloatParameter baseWorleyScaleFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);
  [Tooltip("Blend for base worley noise.")]
  public ClampedFloatParameter baseWorleyBlendFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);

  [Tooltip("Octaves for FBM used in the structure noise.")]
  public Vector3Parameter structureOctaves = new Vector3Parameter(new Vector3(16, 32, 64));
  [Tooltip("FBM scale factor for structure noise. Octave N will have amplitude scale^N.")]
  public ClampedFloatParameter structureScaleFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);
  [Tooltip("How much the higher frequency structure noise blends with the base noise.")]
  public ClampedFloatParameter structureNoiseBlendFactor = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);

  [Tooltip("Octaves for FBM used in the structure noise.")]
  public Vector3Parameter detailOctaves = new Vector3Parameter(new Vector3(8, 16, 32));
  [Tooltip("FBM scale factor for structure noise. Octave N will have amplitude scale^N.")]
  public ClampedFloatParameter detailScaleFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);
  [Tooltip("How much the very high frequency detail noise blends with the base noise.")]
  public ClampedFloatParameter detailNoiseBlendFactor = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);
  [Tooltip("Number of times to tile detail noise.")]
  public ClampedIntParameter detailNoiseTile = new ClampedIntParameter(100, 1, 200);

  [Tooltip("Start of the low height gradient.")]
  public ClampedFloatParameter heightGradientLowStart = new ClampedFloatParameter(0.0f, 0.0f, 1.0f);
  [Tooltip("End of the low height gradient.")]
  public ClampedFloatParameter heightGradientLowEnd = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);
  [Tooltip("Start of the high height gradient.")]
  public ClampedFloatParameter heightGradientHighStart = new ClampedFloatParameter(0.9f, 0.0f, 1.0f);
  [Tooltip("End of the high height gradient.")]
  public ClampedFloatParameter heightGradientHighEnd = new ClampedFloatParameter(1.0f, 0.0f, 1.0f);


  [Tooltip("Octaves for FBM used in the base perlin noise.")]
  public Vector3Parameter coverageOctaves = new Vector3Parameter(new Vector4(8, 16, 32));
  [Tooltip("Offset for coverage map.")]
  public ClampedFloatParameter coverageOffset = new ClampedFloatParameter(0.5f, 0.0f, 1.0f);
  [Tooltip("FBM scale factor for coverage noise. Octave N will have amplitude scale^N.")]
  public ClampedFloatParameter coverageScaleFactor = new ClampedFloatParameter(1.0f, 0.0f, 2.0f);
  [Tooltip("Blend for coverage map.")]
  public ClampedFloatParameter coverageBlendFactor = new ClampedFloatParameter(1.0f, 0.0f, 1.0f);

  /* Clouds debug. */
  [Tooltip("Debug that just shows one layer of noise.")]
  public BoolParameter cloudsDebug = new BoolParameter(false);

  /********************************************************************************/
  /******************************* End Parameters *********************************/
  /********************************************************************************/

  ExpanseSky()
  {
    displayName = "Expanse Sky";
  }

  public override Type GetSkyRendererType()
  {
    return typeof(ExpanseSkyRenderer);
  }

  public override int GetHashCode()
  {
    int hash = base.GetHashCode();
    unchecked
    {
      /* Planet. */
      hash = hash * 23 + atmosphereThickness.value.GetHashCode();
      hash = hash * 23 + planetRadius.value.GetHashCode();
      hash = groundAlbedoTexture.value != null ? hash * 23 + groundAlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + groundTint.value.GetHashCode();
      hash = groundEmissionTexture.value != null ? hash * 23 + groundEmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + groundEmissionMultiplier.value.GetHashCode();
      hash = hash * 23 + lightPollutionTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionIntensity.value.GetHashCode();
      hash = hash * 23 + planetRotation.value.GetHashCode();

      /* Night sky. */
      hash = nightSkyTexture.value != null ? hash * 23 + nightSkyTexture.GetHashCode() : hash;
      hash = hash * 23 + nightSkyRotation.value.GetHashCode();
      hash = hash * 23 + nightTint.value.GetHashCode();
      hash = hash * 23 + nightIntensity.value.GetHashCode();

      /* Aerosols. */
      hash = hash * 23 + aerosolCoefficient.value.GetHashCode();
      hash = hash * 23 + scaleHeightAerosols.value.GetHashCode();
      hash = hash * 23 + aerosolAnisotropy.value.GetHashCode();
      hash = hash * 23 + aerosolDensity.value.GetHashCode();

      /* Air. */
      hash = hash * 23 + airCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightAir.value.GetHashCode();
      hash = hash * 23 + airDensity.value.GetHashCode();

      /* Ozone. */
      hash = hash * 23 + ozoneCoefficients.value.GetHashCode();
      hash = hash * 23 + ozoneThickness.value.GetHashCode();
      hash = hash * 23 + ozoneHeight.value.GetHashCode();
      hash = hash * 23 + ozoneDensity.value.GetHashCode();

      /* Height Fog. */
      hash = hash * 23 + heightFogCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightHeightFog.value.GetHashCode();
      hash = hash * 23 + heightFogAnisotropy.value.GetHashCode();
      hash = hash * 23 + heightFogDensity.value.GetHashCode();
      hash = hash * 23 + heightFogAttenuationDistance.value.GetHashCode();
      hash = hash * 23 + heightFogAttenuationBias.value.GetHashCode();
      hash = hash * 23 + heightFogTint.value.GetHashCode();

      /* Artistic overrides. */
      hash = hash * 23 + skyTint.value.GetHashCode();
      hash = hash * 23 + multipleScatteringMultiplier.value.GetHashCode();

      /* Celestial bodies. */
      hash = hash * 23 + body1LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body1ReceivesLight.value.GetHashCode();
      hash = body1AlbedoTexture.value != null ? hash * 23 + body1AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body1Emissive.value.GetHashCode();
      hash = body1EmissionTexture.value != null ? hash * 23 + body1EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body1Rotation.value.GetHashCode();

      hash = hash * 23 + body2LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body2ReceivesLight.value.GetHashCode();
      hash = body2AlbedoTexture.value != null ? hash * 23 + body2AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body2Emissive.value.GetHashCode();
      hash = body2EmissionTexture.value != null ? hash * 23 + body2EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body2Rotation.value.GetHashCode();

      hash = hash * 23 + body3LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body3ReceivesLight.value.GetHashCode();
      hash = body3AlbedoTexture.value != null ? hash * 23 + body3AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body3Emissive.value.GetHashCode();
      hash = body3EmissionTexture.value != null ? hash * 23 + body3EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body3Rotation.value.GetHashCode();

      hash = hash * 23 + body4LimbDarkening.value.GetHashCode();
      hash = hash * 23 + body4ReceivesLight.value.GetHashCode();
      hash = body4AlbedoTexture.value != null ? hash * 23 + body4AlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + body4Emissive.value.GetHashCode();
      hash = body4EmissionTexture.value != null ? hash * 23 + body4EmissionTexture.GetHashCode() : hash;
      hash = hash * 23 + body4Rotation.value.GetHashCode();

      /* Sampling. */
      hash = hash * 23 + numberOfTransmittanceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfLightPollutionSamples.value.GetHashCode();
      hash = hash * 23 + numberOfScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfGroundIrradianceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringAccumulationSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
      hash = hash * 23 + useAntiAliasing.value.GetHashCode();
      hash = hash * 23 + ditherAmount.value.GetHashCode();

      /* Clouds geometry. */
      hash = hash * 23 + cloudVolumeLowerRadialBoundary.value.GetHashCode();
      hash = hash * 23 + cloudVolumeUpperRadialBoundary.value.GetHashCode();
      hash = hash * 23 + cloudTextureAngularRange.value.GetHashCode();
      hash = hash * 23 + cloudUOffset.value.GetHashCode();
      hash = hash * 23 + cloudVOffset.value.GetHashCode();
      hash = hash * 23 + cloudWOffset.value.GetHashCode();

      /* Clouds lighting. */
      hash = hash * 23 + cloudDensity.value.GetHashCode();
      hash = hash * 23 + cloudFalloffRadius.value.GetHashCode();
      hash = hash * 23 + densityAttenuationThreshold.value.GetHashCode();
      hash = hash * 23 + densityAttenuationMultiplier.value.GetHashCode();
      hash = hash * 23 + cloudForwardScattering.value.GetHashCode();
      hash = hash * 23 + cloudSilverSpread.value.GetHashCode();
      hash = hash * 23 + silverIntensity.value.GetHashCode();
      hash = hash * 23 + depthProbabilityOffset.value.GetHashCode();
      hash = hash * 23 + depthProbabilityMin.value.GetHashCode();
      hash = hash * 23 + depthProbabilityMax.value.GetHashCode();
      hash = hash * 23 + atmosphericBlendDistance.value.GetHashCode();
      hash = hash * 23 + atmosphericBlendBias.value.GetHashCode();

      /* Clouds sampling. */
      hash = hash * 23 + numCloudTransmittanceSamples.value.GetHashCode();
      hash = hash * 23 + numCloudSSSamples.value.GetHashCode();
      hash = hash * 23 + cloudCoarseMarchStepSize.value.GetHashCode();
      hash = hash * 23 + cloudDetailMarchStepSize.value.GetHashCode();
      hash = hash * 23 + numZeroStepsBeforeCoarseMarch.value.GetHashCode();

      /* Clouds noise. */
      hash = hash * 23 + basePerlinOctaves.value.GetHashCode();
      hash = hash * 23 + basePerlinOffset.value.GetHashCode();
      hash = hash * 23 + basePerlinScaleFactor.value.GetHashCode();
      hash = hash * 23 + baseWorleyOctaves.value.GetHashCode();
      hash = hash * 23 + baseWorleyScaleFactor.value.GetHashCode();
      hash = hash * 23 + baseWorleyBlendFactor.value.GetHashCode();
      hash = hash * 23 + structureOctaves.value.GetHashCode();
      hash = hash * 23 + structureScaleFactor.value.GetHashCode();
      hash = hash * 23 + structureNoiseBlendFactor.value.GetHashCode();
      hash = hash * 23 + detailOctaves.value.GetHashCode();
      hash = hash * 23 + detailScaleFactor.value.GetHashCode();
      hash = hash * 23 + detailNoiseBlendFactor.value.GetHashCode();
      hash = hash * 23 + detailNoiseTile.value.GetHashCode();
      hash = hash * 23 + heightGradientLowStart.value.GetHashCode();
      hash = hash * 23 + heightGradientLowEnd.value.GetHashCode();
      hash = hash * 23 + heightGradientHighStart.value.GetHashCode();
      hash = hash * 23 + heightGradientHighEnd.value.GetHashCode();
      hash = hash * 23 + coverageOctaves.value.GetHashCode();
      hash = hash * 23 + coverageOffset.value.GetHashCode();
      hash = hash * 23 + coverageScaleFactor.value.GetHashCode();
      hash = hash * 23 + coverageBlendFactor.value.GetHashCode();

      /* Clouds debug. */
      hash = hash * 23 + cloudsDebug.value.GetHashCode();
    }
    return hash;
  }

  public int GetPrecomputationHashCode()
  {
    int hash = base.GetHashCode();
    unchecked
    {
      hash = hash * 23 + atmosphereThickness.value.GetHashCode();
      hash = hash * 23 + planetRadius.value.GetHashCode();
      hash = groundAlbedoTexture.value != null ? hash * 23 + groundAlbedoTexture.GetHashCode() : hash;
      hash = hash * 23 + groundTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionTint.value.GetHashCode();
      hash = hash * 23 + lightPollutionIntensity.value.GetHashCode();
      hash = hash * 23 + planetRotation.value.GetHashCode();
      hash = hash * 23 + aerosolCoefficient.value.GetHashCode();
      hash = hash * 23 + scaleHeightAerosols.value.GetHashCode();
      hash = hash * 23 + aerosolAnisotropy.value.GetHashCode();
      hash = hash * 23 + aerosolDensity.value.GetHashCode();
      hash = hash * 23 + heightFogCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightHeightFog.value.GetHashCode();
      hash = hash * 23 + heightFogAnisotropy.value.GetHashCode();
      hash = hash * 23 + heightFogDensity.value.GetHashCode();
      hash = hash * 23 + heightFogAttenuationDistance.value.GetHashCode();
      hash = hash * 23 + heightFogAttenuationBias.value.GetHashCode();
      hash = hash * 23 + airCoefficients.value.GetHashCode();
      hash = hash * 23 + scaleHeightAir.value.GetHashCode();
      hash = hash * 23 + airDensity.value.GetHashCode();
      hash = hash * 23 + ozoneCoefficients.value.GetHashCode();
      hash = hash * 23 + ozoneThickness.value.GetHashCode();
      hash = hash * 23 + ozoneHeight.value.GetHashCode();
      hash = hash * 23 + ozoneDensity.value.GetHashCode();
      hash = hash * 23 + numberOfTransmittanceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfLightPollutionSamples.value.GetHashCode();
      hash = hash * 23 + numberOfScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfGroundIrradianceSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringSamples.value.GetHashCode();
      hash = hash * 23 + numberOfMultipleScatteringAccumulationSamples.value.GetHashCode();
      hash = hash * 23 + useImportanceSampling.value.GetHashCode();
    }
    return hash;
  }

  public int GetCloudPrecomputationHashCode()
  {
    int hash = base.GetHashCode();
    unchecked
    {
      /* Clouds geometry. */
      hash = hash * 23 + cloudVolumeLowerRadialBoundary.value.GetHashCode();
      hash = hash * 23 + cloudVolumeUpperRadialBoundary.value.GetHashCode();
      hash = hash * 23 + cloudTextureAngularRange.value.GetHashCode();

      /* Clouds noise. */
      hash = hash * 23 + basePerlinOctaves.value.GetHashCode();
      hash = hash * 23 + basePerlinOffset.value.GetHashCode();
      hash = hash * 23 + basePerlinScaleFactor.value.GetHashCode();
      hash = hash * 23 + baseWorleyOctaves.value.GetHashCode();
      hash = hash * 23 + baseWorleyScaleFactor.value.GetHashCode();
      hash = hash * 23 + baseWorleyBlendFactor.value.GetHashCode();
      hash = hash * 23 + structureOctaves.value.GetHashCode();
      hash = hash * 23 + structureScaleFactor.value.GetHashCode();
      hash = hash * 23 + detailOctaves.value.GetHashCode();
      hash = hash * 23 + detailScaleFactor.value.GetHashCode();
      hash = hash * 23 + detailNoiseTile.value.GetHashCode();
      hash = hash * 23 + coverageOctaves.value.GetHashCode();
      hash = hash * 23 + coverageOffset.value.GetHashCode();
      hash = hash * 23 + coverageScaleFactor.value.GetHashCode();
    }
    return hash;
  }
}
