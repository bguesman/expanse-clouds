using UnityEditor.Rendering;
using UnityEngine.Rendering.HighDefinition;
using UnityEditor.Rendering.HighDefinition;

// [CanEditMultipleObjects]
[VolumeComponentEditor(typeof(ExpanseSky))]
class ExpanseSkyEditor : SkySettingsEditor
{
    /* Planet. */
    SerializedDataParameter atmosphereThickness;
    SerializedDataParameter planetRadius;
    SerializedDataParameter groundAlbedoTexture;
    SerializedDataParameter groundTint;
    SerializedDataParameter groundEmissionTexture;
    SerializedDataParameter groundEmissionMultiplier;
    SerializedDataParameter planetRotation;

    /* Night sky. */
    SerializedDataParameter nightSkyTexture;
    SerializedDataParameter nightSkyRotation;
    SerializedDataParameter nightTint;
    SerializedDataParameter nightIntensity;
    SerializedDataParameter lightPollutionTint;
    SerializedDataParameter lightPollutionIntensity;

    /* Aerosols. */
    SerializedDataParameter aerosolCoefficient;
    SerializedDataParameter scaleHeightAerosols;
    SerializedDataParameter aerosolAnisotropy;
    SerializedDataParameter aerosolDensity;

    /* Air. */
    SerializedDataParameter airCoefficients;
    SerializedDataParameter scaleHeightAir;
    SerializedDataParameter airDensity;

    /* Ozone. */
    SerializedDataParameter ozoneCoefficients;
    SerializedDataParameter ozoneThickness;
    SerializedDataParameter ozoneHeight;
    SerializedDataParameter ozoneDensity;

    /* Artistic Overrides. */
    SerializedDataParameter skyTint;
    SerializedDataParameter multipleScatteringMultiplier;

    /* Body 1. */
    SerializedDataParameter body1LimbDarkening;
    SerializedDataParameter body1ReceivesLight;
    SerializedDataParameter body1AlbedoTexture;
    SerializedDataParameter body1Emissive;
    SerializedDataParameter body1EmissionTexture;
    SerializedDataParameter body1Rotation;
    /* Body 2. */
    SerializedDataParameter body2LimbDarkening;
    SerializedDataParameter body2ReceivesLight;
    SerializedDataParameter body2AlbedoTexture;
    SerializedDataParameter body2Emissive;
    SerializedDataParameter body2EmissionTexture;
    SerializedDataParameter body2Rotation;
    /* Body 3. */
    SerializedDataParameter body3LimbDarkening;
    SerializedDataParameter body3ReceivesLight;
    SerializedDataParameter body3AlbedoTexture;
    SerializedDataParameter body3Emissive;
    SerializedDataParameter body3EmissionTexture;
    SerializedDataParameter body3Rotation;
    /* Body 4. */
    SerializedDataParameter body4LimbDarkening;
    SerializedDataParameter body4ReceivesLight;
    SerializedDataParameter body4AlbedoTexture;
    SerializedDataParameter body4Emissive;
    SerializedDataParameter body4EmissionTexture;
    SerializedDataParameter body4Rotation;

    /* Sampling and Rendering. */
    SerializedDataParameter numberOfTransmittanceSamples;
    SerializedDataParameter numberOfLightPollutionSamples;
    SerializedDataParameter numberOfScatteringSamples;
    SerializedDataParameter numberOfGroundIrradianceSamples;
    SerializedDataParameter numberOfMultipleScatteringSamples;
    SerializedDataParameter numberOfMultipleScatteringAccumulationSamples;
    SerializedDataParameter useImportanceSampling;
    SerializedDataParameter useAntiAliasing;
    SerializedDataParameter ditherAmount;

    /* Clouds geometry. */
    SerializedDataParameter cloudVolumeLowerRadialBoundary;
    SerializedDataParameter cloudVolumeUpperRadialBoundary;
    SerializedDataParameter cloudTextureAngularRange;
    SerializedDataParameter cloudUOffset;
    SerializedDataParameter cloudVOffset;
    SerializedDataParameter cloudWOffset;

    /* Clouds lighting. */
    SerializedDataParameter cloudDensity;
    SerializedDataParameter cloudFalloffRadius;
    SerializedDataParameter densityAttenuationThreshold;
    SerializedDataParameter densityAttenuationMultiplier;
    SerializedDataParameter cloudForwardScattering;
    SerializedDataParameter cloudSilverSpread;
    SerializedDataParameter silverIntensity;
    SerializedDataParameter depthProbabilityOffset;
    SerializedDataParameter depthProbabilityMin;
    SerializedDataParameter depthProbabilityMax;
    SerializedDataParameter atmosphericBlendDistance;
    SerializedDataParameter atmosphericBlendBias;

    /* Clouds sampling. */
    SerializedDataParameter numCloudTransmittanceSamples;
    SerializedDataParameter numCloudSSSamples;
    SerializedDataParameter cloudCoarseMarchFraction;
    SerializedDataParameter cloudDetailMarchFraction;
    SerializedDataParameter numZeroStepsBeforeCoarseMarch;


    /* Clouds noise. */
    SerializedDataParameter basePerlinOctaves;
    SerializedDataParameter basePerlinOffset;
    SerializedDataParameter basePerlinScaleFactor;
    SerializedDataParameter baseWorleyOctaves;
    SerializedDataParameter baseWorleyScaleFactor;
    SerializedDataParameter baseWorleyBlendFactor;
    SerializedDataParameter structureOctaves;
    SerializedDataParameter structureScaleFactor;
    SerializedDataParameter structureNoiseBlendFactor;
    SerializedDataParameter detailOctaves;
    SerializedDataParameter detailScaleFactor;
    SerializedDataParameter detailNoiseBlendFactor;
    SerializedDataParameter detailNoiseTile;
    SerializedDataParameter heightGradientLowStart;
    SerializedDataParameter heightGradientLowEnd;
    SerializedDataParameter heightGradientHighStart;
    SerializedDataParameter heightGradientHighEnd;
    SerializedDataParameter coverageOctaves;
    SerializedDataParameter coverageOffset;
    SerializedDataParameter coverageScaleFactor;
    SerializedDataParameter coverageBlendFactor;

    /* Clouds debug. */
    SerializedDataParameter cloudsDebug;

    public override void OnEnable()
    {
        base.OnEnable();

        m_CommonUIElementsMask = (uint)SkySettingsUIElement.UpdateMode;

        var o = new PropertyFetcher<ExpanseSky>(serializedObject);

        atmosphereThickness = Unpack(o.Find(x => x.atmosphereThickness));
        planetRadius = Unpack(o.Find(x => x.planetRadius));
        groundAlbedoTexture = Unpack(o.Find(x => x.groundAlbedoTexture));
        groundTint = Unpack(o.Find(x => x.groundTint));
        groundEmissionTexture = Unpack(o.Find(x => x.groundEmissionTexture));
        groundEmissionMultiplier = Unpack(o.Find(x => x.groundEmissionMultiplier));
        planetRotation = Unpack(o.Find(x => x.planetRotation));

        /* Night sky. */
        nightSkyTexture = Unpack(o.Find(x => x.nightSkyTexture));
        nightSkyRotation = Unpack(o.Find(x => x.nightSkyRotation));
        nightTint = Unpack(o.Find(x => x.nightTint));
        nightIntensity = Unpack(o.Find(x => x.nightIntensity));
        lightPollutionTint = Unpack(o.Find(x => x.lightPollutionTint));
        lightPollutionIntensity = Unpack(o.Find(x => x.lightPollutionIntensity));

        /* Aerosols. */
        aerosolCoefficient = Unpack(o.Find(x => x.aerosolCoefficient));
        scaleHeightAerosols = Unpack(o.Find(x => x.scaleHeightAerosols));
        aerosolAnisotropy = Unpack(o.Find(x => x.aerosolAnisotropy));
        aerosolDensity = Unpack(o.Find(x => x.aerosolDensity));

        /* Air. */
        airCoefficients = Unpack(o.Find(x => x.airCoefficients));
        scaleHeightAir = Unpack(o.Find(x => x.scaleHeightAir));
        airDensity = Unpack(o.Find(x => x.airDensity));

        /* Ozone. */
        ozoneCoefficients = Unpack(o.Find(x => x.ozoneCoefficients));
        ozoneThickness = Unpack(o.Find(x => x.ozoneThickness));
        ozoneHeight = Unpack(o.Find(x => x.ozoneHeight));
        ozoneDensity = Unpack(o.Find(x => x.ozoneDensity));

        /* Artistic Overrides. */
        skyTint = Unpack(o.Find(x => x.skyTint));
        multipleScatteringMultiplier = Unpack(o.Find(x => x.multipleScatteringMultiplier));

        /* Body 1. */
        body1LimbDarkening = Unpack(o.Find(x => x.body1LimbDarkening));
        body1ReceivesLight = Unpack(o.Find(x => x.body1ReceivesLight));
        body1AlbedoTexture = Unpack(o.Find(x => x.body1AlbedoTexture));
        body1Emissive = Unpack(o.Find(x => x.body1Emissive));
        body1EmissionTexture = Unpack(o.Find(x => x.body1EmissionTexture));
        body1Rotation = Unpack(o.Find(x => x.body1Rotation));
        /* Body 2. */
        body2LimbDarkening = Unpack(o.Find(x => x.body2LimbDarkening));
        body2ReceivesLight = Unpack(o.Find(x => x.body2ReceivesLight));
        body2AlbedoTexture = Unpack(o.Find(x => x.body2AlbedoTexture));
        body2Emissive = Unpack(o.Find(x => x.body2Emissive));
        body2EmissionTexture = Unpack(o.Find(x => x.body2EmissionTexture));
        body2Rotation = Unpack(o.Find(x => x.body2Rotation));
        /* Body 3. */
        body3LimbDarkening = Unpack(o.Find(x => x.body3LimbDarkening));
        body3ReceivesLight = Unpack(o.Find(x => x.body3ReceivesLight));
        body3AlbedoTexture = Unpack(o.Find(x => x.body3AlbedoTexture));
        body3Emissive = Unpack(o.Find(x => x.body3Emissive));
        body3EmissionTexture = Unpack(o.Find(x => x.body3EmissionTexture));
        body3Rotation = Unpack(o.Find(x => x.body3Rotation));
        /* Body 4. */
        body4LimbDarkening = Unpack(o.Find(x => x.body4LimbDarkening));
        body4ReceivesLight = Unpack(o.Find(x => x.body4ReceivesLight));
        body4AlbedoTexture = Unpack(o.Find(x => x.body4AlbedoTexture));
        body4Emissive = Unpack(o.Find(x => x.body4Emissive));
        body4EmissionTexture = Unpack(o.Find(x => x.body4EmissionTexture));
        body4Rotation = Unpack(o.Find(x => x.body4Rotation));

        /* Sampling and Rendering. */
        numberOfTransmittanceSamples = Unpack(o.Find(x => x.numberOfTransmittanceSamples));
        numberOfLightPollutionSamples = Unpack(o.Find(x => x.numberOfLightPollutionSamples));
        numberOfScatteringSamples = Unpack(o.Find(x => x.numberOfScatteringSamples));
        numberOfGroundIrradianceSamples = Unpack(o.Find(x => x.numberOfGroundIrradianceSamples));
        numberOfMultipleScatteringSamples = Unpack(o.Find(x => x.numberOfMultipleScatteringSamples));
        numberOfMultipleScatteringAccumulationSamples = Unpack(o.Find(x => x.numberOfMultipleScatteringAccumulationSamples));
        useImportanceSampling = Unpack(o.Find(x => x.useImportanceSampling));
        useAntiAliasing = Unpack(o.Find(x => x.useAntiAliasing));
        ditherAmount = Unpack(o.Find(x => x.ditherAmount));

        /* Clouds. */
        cloudDensity = Unpack(o.Find(x => x.cloudDensity));
        cloudForwardScattering = Unpack(o.Find(x => x.cloudForwardScattering));
        cloudSilverSpread = Unpack(o.Find(x => x.cloudSilverSpread));
        numCloudTransmittanceSamples = Unpack(o.Find(x => x.numCloudTransmittanceSamples));
        numCloudSSSamples = Unpack(o.Find(x => x.numCloudSSSamples));

        cloudCoarseMarchFraction = Unpack(o.Find(x => x.cloudCoarseMarchFraction));
        cloudDetailMarchFraction = Unpack(o.Find(x => x.cloudDetailMarchFraction));
        cloudVolumeLowerRadialBoundary = Unpack(o.Find(x => x.cloudVolumeLowerRadialBoundary));
        cloudVolumeUpperRadialBoundary = Unpack(o.Find(x => x.cloudVolumeUpperRadialBoundary));
        cloudTextureAngularRange = Unpack(o.Find(x => x.cloudTextureAngularRange));
        cloudFalloffRadius = Unpack(o.Find(x => x.cloudFalloffRadius));
        cloudUOffset = Unpack(o.Find(x => x.cloudUOffset));
        cloudVOffset = Unpack(o.Find(x => x.cloudVOffset));
        cloudWOffset = Unpack(o.Find(x => x.cloudWOffset));
        structureNoiseBlendFactor = Unpack(o.Find(x => x.structureNoiseBlendFactor));
        detailNoiseBlendFactor = Unpack(o.Find(x => x.detailNoiseBlendFactor));
        basePerlinOctaves = Unpack(o.Find(x => x.basePerlinOctaves));
        basePerlinOffset = Unpack(o.Find(x => x.basePerlinOffset));
        basePerlinScaleFactor = Unpack(o.Find(x => x.basePerlinScaleFactor));
        baseWorleyOctaves = Unpack(o.Find(x => x.baseWorleyOctaves));
        baseWorleyScaleFactor = Unpack(o.Find(x => x.baseWorleyScaleFactor));
        structureOctaves = Unpack(o.Find(x => x.structureOctaves));
        structureScaleFactor = Unpack(o.Find(x => x.structureScaleFactor));
        detailOctaves = Unpack(o.Find(x => x.detailOctaves));
        detailScaleFactor = Unpack(o.Find(x => x.detailScaleFactor));

        /* Clouds geometry. */
        cloudVolumeLowerRadialBoundary = Unpack(o.Find(x => x.cloudVolumeLowerRadialBoundary));
        cloudVolumeUpperRadialBoundary = Unpack(o.Find(x => x.cloudVolumeUpperRadialBoundary));
        cloudTextureAngularRange = Unpack(o.Find(x => x.cloudTextureAngularRange));
        cloudUOffset = Unpack(o.Find(x => x.cloudUOffset));
        cloudVOffset = Unpack(o.Find(x => x.cloudVOffset));
        cloudWOffset = Unpack(o.Find(x => x.cloudWOffset));

        /* Clouds lighting. */
        cloudDensity = Unpack(o.Find(x => x.cloudDensity));
        cloudFalloffRadius = Unpack(o.Find(x => x.cloudFalloffRadius));
        densityAttenuationThreshold = Unpack(o.Find(x => x.densityAttenuationThreshold));
        densityAttenuationMultiplier = Unpack(o.Find(x => x.densityAttenuationMultiplier));
        cloudForwardScattering = Unpack(o.Find(x => x.cloudForwardScattering));
        cloudSilverSpread = Unpack(o.Find(x => x.cloudSilverSpread));
        silverIntensity = Unpack(o.Find(x => x.silverIntensity));
        depthProbabilityOffset = Unpack(o.Find(x => x.depthProbabilityOffset));
        depthProbabilityMin = Unpack(o.Find(x => x.depthProbabilityMin));
        depthProbabilityMax = Unpack(o.Find(x => x.depthProbabilityMax));
        atmosphericBlendDistance = Unpack(o.Find(x => x.atmosphericBlendDistance));
        atmosphericBlendBias = Unpack(o.Find(x => x.atmosphericBlendBias));

        /* Clouds sampling. */
        numCloudTransmittanceSamples = Unpack(o.Find(x => x.numCloudTransmittanceSamples));
        numCloudSSSamples = Unpack(o.Find(x => x.numCloudSSSamples));
        cloudCoarseMarchFraction = Unpack(o.Find(x => x.cloudCoarseMarchFraction));
        cloudDetailMarchFraction = Unpack(o.Find(x => x.cloudDetailMarchFraction));
        numZeroStepsBeforeCoarseMarch = Unpack(o.Find(x => x.numZeroStepsBeforeCoarseMarch));


        /* Clouds noise. */
        basePerlinOctaves = Unpack(o.Find(x => x.basePerlinOctaves));
        basePerlinOffset = Unpack(o.Find(x => x.basePerlinOffset));
        basePerlinScaleFactor = Unpack(o.Find(x => x.basePerlinScaleFactor));
        baseWorleyOctaves = Unpack(o.Find(x => x.baseWorleyOctaves));
        baseWorleyScaleFactor = Unpack(o.Find(x => x.baseWorleyScaleFactor));
        baseWorleyBlendFactor = Unpack(o.Find(x => x.baseWorleyBlendFactor));
        structureOctaves = Unpack(o.Find(x => x.structureOctaves));
        structureScaleFactor = Unpack(o.Find(x => x.structureScaleFactor));
        structureNoiseBlendFactor = Unpack(o.Find(x => x.structureNoiseBlendFactor));
        detailOctaves = Unpack(o.Find(x => x.detailOctaves));
        detailScaleFactor = Unpack(o.Find(x => x.detailScaleFactor));
        detailNoiseBlendFactor = Unpack(o.Find(x => x.detailNoiseBlendFactor));
        detailNoiseTile = Unpack(o.Find(x => x.detailNoiseTile));
        heightGradientLowStart = Unpack(o.Find(x => x.heightGradientLowStart));
        heightGradientLowEnd = Unpack(o.Find(x => x.heightGradientLowEnd));
        heightGradientHighStart = Unpack(o.Find(x => x.heightGradientHighStart));
        heightGradientHighEnd = Unpack(o.Find(x => x.heightGradientHighEnd));
        coverageOctaves = Unpack(o.Find(x => x.coverageOctaves));
        coverageOffset = Unpack(o.Find(x => x.coverageOffset));
        coverageScaleFactor = Unpack(o.Find(x => x.coverageScaleFactor));
        coverageBlendFactor = Unpack(o.Find(x => x.coverageBlendFactor));

        /* Clouds debug. */
        cloudsDebug = Unpack(o.Find(x => x.cloudsDebug));
    }

    public override void OnInspectorGUI()
    {
      // var titleStyle : GUIStyle;
      UnityEngine.GUIStyle titleStyle = new UnityEngine.GUIStyle();
      titleStyle.fontSize = 16;
      UnityEngine.GUIStyle subtitleStyle = new UnityEngine.GUIStyle();
      subtitleStyle.fontSize = 12;

      /* Planet. */
      UnityEditor.EditorGUILayout.LabelField("Planet", titleStyle);
      PropertyField(atmosphereThickness);
      PropertyField(planetRadius);
      PropertyField(groundAlbedoTexture);
      PropertyField(groundTint);
      PropertyField(groundEmissionTexture);
      PropertyField(groundEmissionMultiplier);
      PropertyField(planetRotation);

      /* Night sky. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Night Sky", titleStyle);
      PropertyField(nightSkyTexture);
      PropertyField(nightSkyRotation);
      PropertyField(nightTint);
      PropertyField(nightIntensity);
      PropertyField(lightPollutionTint);
      PropertyField(lightPollutionIntensity);

      /* Aerosols. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Aerosol Layer", titleStyle);
      PropertyField(aerosolCoefficient);
      PropertyField(scaleHeightAerosols);
      PropertyField(aerosolAnisotropy);
      PropertyField(aerosolDensity);

      /* Air. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Air Layer", titleStyle);
      PropertyField(airCoefficients);
      PropertyField(scaleHeightAir);
      PropertyField(airDensity);

      /* Ozone. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Ozone Layer", titleStyle);
      PropertyField(ozoneCoefficients);
      PropertyField(ozoneThickness);
      PropertyField(ozoneHeight);
      PropertyField(ozoneDensity);

      /* Artistic Overrides. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Artistic Overrides", titleStyle);
      PropertyField(skyTint);
      PropertyField(multipleScatteringMultiplier);

      /* Body 1. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Bodies", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 1", subtitleStyle);
      PropertyField(body1LimbDarkening);
      PropertyField(body1ReceivesLight);
      PropertyField(body1AlbedoTexture);
      PropertyField(body1Emissive);
      PropertyField(body1EmissionTexture);
      PropertyField(body1Rotation);
      /* Body 2. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 2", subtitleStyle);
      PropertyField(body2LimbDarkening);
      PropertyField(body2ReceivesLight);
      PropertyField(body2AlbedoTexture);
      PropertyField(body2Emissive);
      PropertyField(body2EmissionTexture);
      PropertyField(body2Rotation);
      /* Body 3. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 3", subtitleStyle);
      PropertyField(body3LimbDarkening);
      PropertyField(body3ReceivesLight);
      PropertyField(body3AlbedoTexture);
      PropertyField(body3Emissive);
      PropertyField(body3EmissionTexture);
      PropertyField(body3Rotation);
      /* Body 4. */
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Celestial Body 4", subtitleStyle);
      PropertyField(body4LimbDarkening);
      PropertyField(body4ReceivesLight);
      PropertyField(body4AlbedoTexture);
      PropertyField(body4Emissive);
      PropertyField(body4EmissionTexture);
      PropertyField(body4Rotation);

      /* Sampling and Rendering. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Sampling and Rendering", titleStyle);
      PropertyField(numberOfTransmittanceSamples);
      PropertyField(numberOfLightPollutionSamples);
      PropertyField(numberOfScatteringSamples);
      PropertyField(numberOfGroundIrradianceSamples);
      PropertyField(numberOfMultipleScatteringSamples);
      PropertyField(numberOfMultipleScatteringAccumulationSamples);
      PropertyField(useImportanceSampling);
      PropertyField(useAntiAliasing);
      PropertyField(ditherAmount);

      /* Clouds. */
      UnityEditor.EditorGUILayout.LabelField("", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("Clouds (Experimental)", titleStyle);
      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Geometry", subtitleStyle);
      PropertyField(cloudVolumeLowerRadialBoundary);
      PropertyField(cloudVolumeUpperRadialBoundary);
      PropertyField(cloudTextureAngularRange);
      PropertyField(cloudUOffset);
      PropertyField(cloudVOffset);
      PropertyField(cloudWOffset);

      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Lighting", subtitleStyle);
      PropertyField(cloudDensity);
      PropertyField(cloudFalloffRadius);
      PropertyField(densityAttenuationThreshold);
      PropertyField(densityAttenuationMultiplier);
      PropertyField(cloudForwardScattering);
      PropertyField(cloudSilverSpread);
      PropertyField(silverIntensity);
      PropertyField(depthProbabilityOffset);
      PropertyField(depthProbabilityMin);
      PropertyField(depthProbabilityMax);
      PropertyField(atmosphericBlendDistance);
      PropertyField(atmosphericBlendBias);

      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Sampling", subtitleStyle);
      PropertyField(numCloudTransmittanceSamples);
      PropertyField(numCloudSSSamples);
      PropertyField(cloudCoarseMarchFraction);
      PropertyField(cloudDetailMarchFraction);
      PropertyField(numZeroStepsBeforeCoarseMarch);


      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Noise", subtitleStyle);
      PropertyField(basePerlinOctaves);
      PropertyField(basePerlinOffset);
      PropertyField(basePerlinScaleFactor);
      PropertyField(baseWorleyOctaves);
      PropertyField(baseWorleyScaleFactor);
      PropertyField(baseWorleyBlendFactor);
      PropertyField(structureOctaves);
      PropertyField(structureScaleFactor);
      PropertyField(structureNoiseBlendFactor);
      PropertyField(detailOctaves);
      PropertyField(detailScaleFactor);
      PropertyField(detailNoiseBlendFactor);
      PropertyField(detailNoiseTile);
      PropertyField(heightGradientLowStart);
      PropertyField(heightGradientLowEnd);
      PropertyField(heightGradientHighStart);
      PropertyField(heightGradientHighEnd);
      PropertyField(coverageOctaves);
      PropertyField(coverageOffset);
      PropertyField(coverageScaleFactor);
      PropertyField(coverageBlendFactor);

      UnityEditor.EditorGUILayout.LabelField("", subtitleStyle);
      UnityEditor.EditorGUILayout.LabelField("Debug", subtitleStyle);
      PropertyField(cloudsDebug);

      base.CommonSkySettingsGUI();
    }
}
