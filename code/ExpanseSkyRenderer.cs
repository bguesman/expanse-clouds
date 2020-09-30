using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

class ExpanseSkyRenderer : SkyRenderer
{

    /********************************************************************************/
    /**************************** Shader Variable ID's ******************************/
    /********************************************************************************/

    public static readonly int _atmosphereThicknessID = Shader.PropertyToID("_atmosphereThickness");
    public static readonly int _atmosphereRadiusID = Shader.PropertyToID("_atmosphereRadius");
    public static readonly int _planetRadiusID = Shader.PropertyToID("_planetRadius");
    public static readonly int _groundAlbedoTextureID = Shader.PropertyToID("_groundAlbedoTexture");
    public static readonly int _hasGroundAlbedoTextureID = Shader.PropertyToID("_hasGroundAlbedoTexture");
    public static readonly int _groundTintID = Shader.PropertyToID("_groundTint");
    public static readonly int _groundEmissionTextureID = Shader.PropertyToID("_groundEmissionTexture");
    public static readonly int _hasGroundEmissionTextureID = Shader.PropertyToID("_hasGroundEmissionTexture");
    public static readonly int _lightPollutionTintID = Shader.PropertyToID("_lightPollutionTint");
    public static readonly int _lightPollutionIntensityID = Shader.PropertyToID("_lightPollutionIntensity");
    public static readonly int _planetRotationID = Shader.PropertyToID("_planetRotation");
    public static readonly int _groundEmissionMultiplierID = Shader.PropertyToID("_groundEmissionMultiplier");
    public static readonly int _nightSkyTextureID = Shader.PropertyToID("_nightSkyTexture");
    public static readonly int _hasNightSkyTextureID = Shader.PropertyToID("_hasNightSkyTexture");
    public static readonly int _nightSkyRotationID = Shader.PropertyToID("_nightSkyRotation");
    public static readonly int _nightTintID = Shader.PropertyToID("_nightTint");
    public static readonly int _nightIntensityID = Shader.PropertyToID("_nightIntensity");
    public static readonly int _aerosolCoefficientID = Shader.PropertyToID("_aerosolCoefficient");
    public static readonly int _scaleHeightAerosolsID = Shader.PropertyToID("_scaleHeightAerosols");
    public static readonly int _aerosolAnisotropyID = Shader.PropertyToID("_aerosolAnisotropy");
    public static readonly int _aerosolDensityID = Shader.PropertyToID("_aerosolDensity");
    public static readonly int _airCoefficientsID = Shader.PropertyToID("_airCoefficients");
    public static readonly int _scaleHeightAirID = Shader.PropertyToID("_scaleHeightAir");
    public static readonly int _airDensityID = Shader.PropertyToID("_airDensity");
    public static readonly int _ozoneCoefficientsID = Shader.PropertyToID("_ozoneCoefficients");
    public static readonly int _ozoneThicknessID = Shader.PropertyToID("_ozoneThickness");
    public static readonly int _ozoneHeightID = Shader.PropertyToID("_ozoneHeight");
    public static readonly int _ozoneDensityID = Shader.PropertyToID("_ozoneDensity");
    public static readonly int _skyTintID = Shader.PropertyToID("_skyTint");
    public static readonly int _multipleScatteringMultiplierID = Shader.PropertyToID("_multipleScatteringMultiplier");
    public static readonly int _limbDarkeningID = Shader.PropertyToID("_limbDarkening");
    public static readonly int _numberOfTransmittanceSamplesID = Shader.PropertyToID("_numberOfTransmittanceSamples");
    public static readonly int _numberOfLightPollutionSamplesID = Shader.PropertyToID("_numberOfLightPollutionSamples");
    public static readonly int _numberOfScatteringSamplesID = Shader.PropertyToID("_numberOfScatteringSamples");
    public static readonly int _numberOfGroundIrradianceSamplesID = Shader.PropertyToID("_numberOfGroundIrradianceSamples");
    public static readonly int _numberOfMultipleScatteringSamplesID = Shader.PropertyToID("_numberOfMultipleScatteringSamples");
    public static readonly int _numberOfMultipleScatteringAccumulationSamplesID = Shader.PropertyToID("_numberOfMultipleScatteringAccumulationSamples");
    public static readonly int _useImportanceSamplingID = Shader.PropertyToID("_useImportanceSampling");
    public static readonly int _useAntiAliasingID = Shader.PropertyToID("_useAntiAliasing");
    public static readonly int _ditherAmountID = Shader.PropertyToID("_ditherAmount");

    /* Celestial bodies. */
    /* Body 1. */
    public static readonly int _body1LimbDarkeningID = Shader.PropertyToID("_body1LimbDarkening");
    public static readonly int _body1ReceivesLightID = Shader.PropertyToID("_body1ReceivesLight");
    public static readonly int _body1AlbedoTextureID = Shader.PropertyToID("_body1AlbedoTexture");
    public static readonly int _body1HasAlbedoTextureID = Shader.PropertyToID("_body1HasAlbedoTexture");
    public static readonly int _body1EmissiveID = Shader.PropertyToID("_body1Emissive");
    public static readonly int _body1EmissionTextureID = Shader.PropertyToID("_body1EmissionTexture");
    public static readonly int _body1HasEmissionTextureID = Shader.PropertyToID("_body1HasEmissionTexture");
    public static readonly int _body1RotationID = Shader.PropertyToID("_body1Rotation");
    /* Body 2. */
    public static readonly int _body2LimbDarkeningID = Shader.PropertyToID("_body2LimbDarkening");
    public static readonly int _body2ReceivesLightID = Shader.PropertyToID("_body2ReceivesLight");
    public static readonly int _body2AlbedoTextureID = Shader.PropertyToID("_body2AlbedoTexture");
    public static readonly int _body2HasAlbedoTextureID = Shader.PropertyToID("_body2HasAlbedoTexture");
    public static readonly int _body2EmissiveID = Shader.PropertyToID("_body2Emissive");
    public static readonly int _body2EmissionTextureID = Shader.PropertyToID("_body2EmissionTexture");
    public static readonly int _body2HasEmissionTextureID = Shader.PropertyToID("_body2HasEmissionTexture");
    public static readonly int _body2RotationID = Shader.PropertyToID("_body2Rotation");
    /* Body 3. */
    public static readonly int _body3LimbDarkeningID = Shader.PropertyToID("_body3LimbDarkening");
    public static readonly int _body3ReceivesLightID = Shader.PropertyToID("_body3ReceivesLight");
    public static readonly int _body3AlbedoTextureID = Shader.PropertyToID("_body3AlbedoTexture");
    public static readonly int _body3HasAlbedoTextureID = Shader.PropertyToID("_body3HasAlbedoTexture");
    public static readonly int _body3EmissiveID = Shader.PropertyToID("_body3Emissive");
    public static readonly int _body3EmissionTextureID = Shader.PropertyToID("_body3EmissionTexture");
    public static readonly int _body3HasEmissionTextureID = Shader.PropertyToID("_body3HasEmissionTexture");
    public static readonly int _body3RotationID = Shader.PropertyToID("_body3Rotation");
    /* Body 4. */
    public static readonly int _body4LimbDarkeningID = Shader.PropertyToID("_body4LimbDarkening");
    public static readonly int _body4ReceivesLightID = Shader.PropertyToID("_body4ReceivesLight");
    public static readonly int _body4AlbedoTextureID = Shader.PropertyToID("_body4AlbedoTexture");
    public static readonly int _body4HasAlbedoTextureID = Shader.PropertyToID("_body4HasAlbedoTexture");
    public static readonly int _body4EmissiveID = Shader.PropertyToID("_body4Emissive");
    public static readonly int _body4EmissionTextureID = Shader.PropertyToID("_body4EmissionTexture");
    public static readonly int _body4HasEmissionTextureID = Shader.PropertyToID("_body4HasEmissionTexture");
    public static readonly int _body4RotationID = Shader.PropertyToID("_body4Rotation");

    public static readonly int _WorldSpaceCameraPos1ID = Shader.PropertyToID("_WorldSpaceCameraPos1");
    public static readonly int _ViewMatrix1ID = Shader.PropertyToID("_ViewMatrix1");
    public static readonly int _PixelCoordToViewDirWS = Shader.PropertyToID("_PixelCoordToViewDirWS");

    /* Clouds geometry. */
    public static readonly int _cloudVolumeLowerRadialBoundaryID = Shader.PropertyToID("_cloudVolumeLowerRadialBoundary");
    public static readonly int _cloudVolumeUpperRadialBoundaryID = Shader.PropertyToID("_cloudVolumeUpperRadialBoundary");
    public static readonly int _cloudTextureAngularRangeID = Shader.PropertyToID("_cloudTextureAngularRange");
    public static readonly int _cloudUOffsetID = Shader.PropertyToID("_cloudUOffset");
    public static readonly int _cloudVOffsetID = Shader.PropertyToID("_cloudVOffset");
    public static readonly int _cloudWOffsetID = Shader.PropertyToID("_cloudWOffset");

    /* Clouds lighting. */
    public static readonly int _cloudDensityID = Shader.PropertyToID("_cloudDensity");
    public static readonly int _cloudFalloffRadiusID = Shader.PropertyToID("_cloudFalloffRadius");
    public static readonly int _densityAttenuationThresholdID = Shader.PropertyToID("_densityAttenuationThreshold");
    public static readonly int _densityAttenuationMultiplierID = Shader.PropertyToID("_densityAttenuationMultiplier");
    public static readonly int _cloudForwardScatteringID = Shader.PropertyToID("_cloudForwardScattering");
    public static readonly int _cloudSilverSpreadID = Shader.PropertyToID("_cloudSilverSpread");
    public static readonly int _silverIntensityID = Shader.PropertyToID("_silverIntensity");
    public static readonly int _depthProbabilityOffsetID = Shader.PropertyToID("_depthProbabilityOffset");
    public static readonly int _depthProbabilityMinID = Shader.PropertyToID("_depthProbabilityMin");
    public static readonly int _depthProbabilityMaxID = Shader.PropertyToID("_depthProbabilityMax");
    public static readonly int _atmosphericBlendDistanceID = Shader.PropertyToID("_atmosphericBlendDistance");
    public static readonly int _atmosphericBlendBiasID = Shader.PropertyToID("_atmosphericBlendBias");

    /* Clouds sampling. */
    public static readonly int _numCloudTransmittanceSamplesID = Shader.PropertyToID("_numCloudTransmittanceSamples");
    public static readonly int _numCloudSSSamplesID = Shader.PropertyToID("_numCloudSSSamples");
    public static readonly int _cloudCoarseMarchStepSizeID = Shader.PropertyToID("_cloudCoarseMarchStepSize");
    public static readonly int _cloudDetailMarchStepSizeID = Shader.PropertyToID("_cloudDetailMarchStepSize");
    public static readonly int _numZeroStepsBeforeCoarseMarchID = Shader.PropertyToID("_numZeroStepsBeforeCoarseMarch");

    /* Clouds noise. */
    public static readonly int _basePerlinOctavesID = Shader.PropertyToID("_basePerlinOctaves");
    public static readonly int _basePerlinOffsetID = Shader.PropertyToID("_basePerlinOffset");
    public static readonly int _basePerlinScaleFactorID = Shader.PropertyToID("_basePerlinScaleFactor");
    public static readonly int _baseWorleyOctavesID = Shader.PropertyToID("_baseWorleyOctaves");
    public static readonly int _baseWorleyScaleFactorID = Shader.PropertyToID("_baseWorleyScaleFactor");
    public static readonly int _baseWorleyBlendFactorID = Shader.PropertyToID("_baseWorleyBlendFactor");
    public static readonly int _structureOctavesID = Shader.PropertyToID("_structureOctaves");
    public static readonly int _structureScaleFactorID = Shader.PropertyToID("_structureScaleFactor");
    public static readonly int _structureNoiseBlendFactorID = Shader.PropertyToID("_structureNoiseBlendFactor");
    public static readonly int _detailOctavesID = Shader.PropertyToID("_detailOctaves");
    public static readonly int _detailScaleFactorID = Shader.PropertyToID("_detailScaleFactor");
    public static readonly int _detailNoiseBlendFactorID = Shader.PropertyToID("_detailNoiseBlendFactor");
    public static readonly int _detailNoiseTileID = Shader.PropertyToID("_detailNoiseTile");
    public static readonly int _heightGradientLowStartID = Shader.PropertyToID("_heightGradientLowStart");
    public static readonly int _heightGradientLowEndID = Shader.PropertyToID("_heightGradientLowEnd");
    public static readonly int _heightGradientHighStartID = Shader.PropertyToID("_heightGradientHighStart");
    public static readonly int _heightGradientHighEndID = Shader.PropertyToID("_heightGradientHighEnd");
    public static readonly int _coverageOctavesID = Shader.PropertyToID("_coverageOctaves");
    public static readonly int _coverageOffsetID = Shader.PropertyToID("_coverageOffset");
    public static readonly int _coverageScaleFactorID = Shader.PropertyToID("_coverageScaleFactor");
    public static readonly int _coverageBlendFactorID = Shader.PropertyToID("_coverageBlendFactor");

    /* Clouds debug. */
    public static readonly int _cloudsDebugID = Shader.PropertyToID("_cloudsDebug");

    /* Sky Tables. */
    public static readonly int _transmittanceTableID = Shader.PropertyToID("_TransmittanceTable");
    public static readonly int _lightPollutionTableAirID = Shader.PropertyToID("_LightPollutionTableAir");
    public static readonly int _lightPollutionTableAerosolID = Shader.PropertyToID("_LightPollutionTableAerosol");
    public static readonly int _groundIrradianceTableAirID = Shader.PropertyToID("_GroundIrradianceTableAir");
    public static readonly int _groundIrradianceTableAerosolID = Shader.PropertyToID("_GroundIrradianceTableAerosol");
    public static readonly int _singleScatteringTableAirID = Shader.PropertyToID("_SingleScatteringTableAir");
    public static readonly int _singleScatteringTableAerosolID = Shader.PropertyToID("_SingleScatteringTableAerosol");
    public static readonly int _singleScatteringTableAirNoShadowsID = Shader.PropertyToID("_SingleScatteringTableAirNoShadows");
    public static readonly int _singleScatteringTableAerosolNoShadowsID = Shader.PropertyToID("_SingleScatteringTableAerosolNoShadows");
    public static readonly int _localMultipleScatteringTableID = Shader.PropertyToID("_LocalMultipleScatteringTable");
    public static readonly int _globalMultipleScatteringTableAirID = Shader.PropertyToID("_GlobalMultipleScatteringTableAir");
    public static readonly int _globalMultipleScatteringTableAerosolID = Shader.PropertyToID("_GlobalMultipleScatteringTableAerosol");

    /* Cloud Tables. */
    public static readonly int _cloudBaseNoiseTableID = Shader.PropertyToID("_CloudBaseNoiseTable");
    public static readonly int _cloudDetailNoiseTableID = Shader.PropertyToID("_CloudDetailNoiseTable");
    public static readonly int _cloudCurlNoiseTableID = Shader.PropertyToID("_CloudCurlNoiseTable");
    public static readonly int _cloudCoverageTableID = Shader.PropertyToID("_CloudCoverageTable");

    /********************************************************************************/
    /************************** End Shader Variable ID's ****************************/
    /********************************************************************************/


    /********************************************************************************/
    /************************************ Sky ***************************************/
    /********************************************************************************/

    /* Dimensions of precomputed tables.  */
    const int TransmittanceTableSizeH = 32;
    const int TransmittanceTableSizeMu = 128;

    const int LightPollutionTableSizeH = 32;
    const int LightPollutionTableSizeMu = 128;

    const int SingleScatteringTableSizeH = 32;
    const int SingleScatteringTableSizeMu = 128;
    const int SingleScatteringTableSizeMuL = 32;
    const int SingleScatteringTableSizeNu = 32;

    const int GroundIrradianceTableSize = 256;

    const int LocalMultipleScatteringTableSizeH = 32;
    const int LocalMultipleScatteringTableSizeMuL = 32;

    const int GlobalMultipleScatteringTableSizeH = 32;
    const int GlobalMultipleScatteringTableSizeMu = 128;
    const int GlobalMultipleScatteringTableSizeMuL = 32;
    const int GlobalMultipleScatteringTableSizeNu = 32;

    /* Transmittance table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * mu (y dimension): the zenith angle of the viewing direction. */
    RTHandle[]                   m_TransmittanceTables;

    /* Light pollution table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * mu (y dimension): the zenith angle of the viewing direction. */
    RTHandle[]                   m_LightPollutionTables;

    /* Single scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * mu (y dimension): the zenith angle of the viewing direction.
     * mu_l (z dimension): the zenith angle of the light source.
     * nu (w dimension): the azimuth angle of the light source. */
    RTHandle[]                   m_SingleScatteringTables;

    /* Ground irradiance table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * mu (x dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_GroundIrradianceTables;

    /* Multiple scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * mu_l (y dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_LocalMultipleScatteringTables;

    /* Multiple scattering table. Leverages spherical symmetry of the atmosphere,
     * parameterized by:
     * h (x dimension): the height of the camera.
     * mu_l (y dimension): dot product between the surface normal and the
     * light direction. */
    RTHandle[]                   m_GlobalMultipleScatteringTables;

    RTHandle AllocateTransmittanceTable(int index)
    {
        var table = RTHandles.Alloc(TransmittanceTableSizeH,
                                    TransmittanceTableSizeMu,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("TransmittanceTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateLightPollutionTable(int index)
    {
        var table = RTHandles.Alloc(LightPollutionTableSizeH,
                                    LightPollutionTableSizeMu,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("LightPollutionTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateGroundIrradianceTable(int index)
    {
        var table = RTHandles.Alloc(GroundIrradianceTableSize,
                                    1,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("GroundIrradianceTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateSingleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(SingleScatteringTableSizeH,
                                    SingleScatteringTableSizeMu,
                                    SingleScatteringTableSizeMuL *
                                    SingleScatteringTableSizeNu,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("SingleScatteringTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateLocalMultipleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(LocalMultipleScatteringTableSizeH,
                                    LocalMultipleScatteringTableSizeMuL,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("LocalMultipleScatteringTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateGlobalMultipleScatteringTable(int index)
    {
        var table = RTHandles.Alloc(GlobalMultipleScatteringTableSizeH,
                                    GlobalMultipleScatteringTableSizeMu,
                                    GlobalMultipleScatteringTableSizeMuL *
                                    GlobalMultipleScatteringTableSizeNu,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("GlobalMultipleScatteringTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    ComputeShader GetExpanseSkyPrecomputeShader()
    {
        return Resources.Load<ComputeShader>("ExpanseSkyPrecompute");
    }

    /********************************************************************************/
    /********************************** End Sky *************************************/
    /********************************************************************************/



    /********************************************************************************/
    /********************************** Clouds **************************************/
    /********************************************************************************/
    const int CloudBaseNoiseTableSizeX = 512;
    const int CloudBaseNoiseTableSizeY = 256;
    const int CloudBaseNoiseTableSizeZ = 512;

    const int CloudDetailNoiseTableSizeX = 128;
    const int CloudDetailNoiseTableSizeY = 128;
    const int CloudDetailNoiseTableSizeZ = 128;

    const int CloudCurlNoiseTableSizeX = 128;
    const int CloudCurlNoiseTableSizeY = 128;

    const int CloudCoverageTableSizeX = 1024;
    const int CloudCoverageTableSizeY = 1024;

    /* Cloud base noise table. */
    RTHandle[]                   m_CloudBaseNoiseTable;

    /* Cloud detail noise table. */
    RTHandle[]                   m_CloudDetailNoiseTable;

    /* Cloud curl noise table. */
    RTHandle[]                   m_CloudCurlNoiseTable;

    /* Cloud coverage table. */
    RTHandle[]                   m_CloudCoverageTable;

    RTHandle AllocateCloudBaseNoiseTable(int index)
    {
        var table = RTHandles.Alloc(CloudBaseNoiseTableSizeX,
                                    CloudBaseNoiseTableSizeY,
                                    CloudBaseNoiseTableSizeZ,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("CloudBaseNoiseTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateCloudDetailNoiseTable(int index)
    {
        var table = RTHandles.Alloc(CloudDetailNoiseTableSizeX,
                                    CloudDetailNoiseTableSizeY,
                                    CloudDetailNoiseTableSizeZ,
                                    dimension: TextureDimension.Tex3D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("CloudDetailNoiseTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateCloudCurlNoiseTable(int index)
    {
        var table = RTHandles.Alloc(CloudCurlNoiseTableSizeX,
                                    CloudCurlNoiseTableSizeY,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("CloudCurlNoiseTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    RTHandle AllocateCloudCoverageTable(int index)
    {
        var table = RTHandles.Alloc(CloudCoverageTableSizeX,
                                    CloudCoverageTableSizeX,
                                    dimension: TextureDimension.Tex2D,
                                    colorFormat: s_ColorFormat,
                                    enableRandomWrite: true,
                                    name: string.Format("CloudCoverageTable{0}", index));

        Debug.Assert(table != null);

        return table;
    }

    ComputeShader GetExpanseCloudPrecomputeShader()
    {
        return Resources.Load<ComputeShader>("ExpanseCloudPrecompute");
    }

    /********************************************************************************/
    /******************************** End Clouds ************************************/
    /********************************************************************************/



    static ComputeShader         s_PrecomputeCS;

    static ComputeShader         s_CloudPrecomputeCS;

    static GraphicsFormat s_ColorFormat = GraphicsFormat.R16G16B16A16_SFloat;

    /* Use the same strategy as Physically Based Sky to determine when we
     * need to update our tables---compute a hash of the relevant parameters
     * every frame and check for differences. */
    int m_LastPrecomputationParamHash;
    int m_LastCloudPrecomputationParamHash;

    // Renders a cubemap into a render texture (can be cube or 2D)
    Material m_ExpanseSkyMaterial;
    MaterialPropertyBlock m_PropertyBlock = new MaterialPropertyBlock();
    private static int m_RenderCubemapID = 0; // FragBaking
    private static int m_RenderFullscreenSkyID = 1; // FragRender

    public override void Build()
    {
      /* Shaders. */
      m_ExpanseSkyMaterial = CoreUtils.CreateEngineMaterial(GetExpanseSkyShader());
      s_PrecomputeCS = GetExpanseSkyPrecomputeShader();
      s_CloudPrecomputeCS = GetExpanseCloudPrecomputeShader();

      Debug.Assert(m_ExpanseSkyMaterial != null);
      Debug.Assert(s_PrecomputeCS != null);
      Debug.Assert(s_CloudPrecomputeCS != null);

      /* Precomputed sky tables. */
      m_TransmittanceTables = new RTHandle[2];
      m_TransmittanceTables[0] = AllocateTransmittanceTable(0);
      m_TransmittanceTables[1] = AllocateTransmittanceTable(1);

      m_LightPollutionTables = new RTHandle[2];
      m_LightPollutionTables[0] = AllocateLightPollutionTable(0);
      m_LightPollutionTables[1] = AllocateLightPollutionTable(1);

      m_GroundIrradianceTables = new RTHandle[2];
      m_GroundIrradianceTables[0] = AllocateGroundIrradianceTable(0);
      m_GroundIrradianceTables[1] = AllocateGroundIrradianceTable(1);

      m_SingleScatteringTables = new RTHandle[4];
      m_SingleScatteringTables[0] = AllocateSingleScatteringTable(0);
      m_SingleScatteringTables[1] = AllocateSingleScatteringTable(1);
      m_SingleScatteringTables[2] = AllocateSingleScatteringTable(2);
      m_SingleScatteringTables[3] = AllocateSingleScatteringTable(3);

      m_LocalMultipleScatteringTables = new RTHandle[2];
      m_LocalMultipleScatteringTables[0] = AllocateLocalMultipleScatteringTable(0);
      m_LocalMultipleScatteringTables[1] = AllocateLocalMultipleScatteringTable(1);

      m_GlobalMultipleScatteringTables = new RTHandle[2];
      m_GlobalMultipleScatteringTables[0] = AllocateGlobalMultipleScatteringTable(0);
      m_GlobalMultipleScatteringTables[1] = AllocateGlobalMultipleScatteringTable(1);

      /* Precomputed cloud noise tables. */
      m_CloudBaseNoiseTable = new RTHandle[1];
      m_CloudBaseNoiseTable[0] = AllocateCloudBaseNoiseTable(0);

      m_CloudDetailNoiseTable = new RTHandle[1];
      m_CloudDetailNoiseTable[0] = AllocateCloudDetailNoiseTable(0);

      m_CloudCurlNoiseTable = new RTHandle[1];
      m_CloudCurlNoiseTable[0] = AllocateCloudCurlNoiseTable(0);

      m_CloudCoverageTable = new RTHandle[1];
      m_CloudCoverageTable[0] = AllocateCloudCoverageTable(0);
    }

    // Project dependent way to retrieve a shader.
    Shader GetExpanseSkyShader()
    {
        return Shader.Find("HDRP/Sky/ExpanseSky");
    }

    public override void Cleanup()
    {
        CoreUtils.Destroy(m_ExpanseSkyMaterial);
        RTHandles.Release(m_TransmittanceTables[0]);
        m_TransmittanceTables[0] = null;
        RTHandles.Release(m_TransmittanceTables[1]);
        m_TransmittanceTables[1] = null;

        RTHandles.Release(m_LightPollutionTables[0]);
        m_LightPollutionTables[0] = null;
        RTHandles.Release(m_LightPollutionTables[1]);
        m_LightPollutionTables[1] = null;

        RTHandles.Release(m_SingleScatteringTables[0]);
        m_SingleScatteringTables[0] = null;
        RTHandles.Release(m_SingleScatteringTables[1]);
        m_SingleScatteringTables[1] = null;
        RTHandles.Release(m_SingleScatteringTables[2]);
        m_SingleScatteringTables[2] = null;
        RTHandles.Release(m_SingleScatteringTables[3]);
        m_SingleScatteringTables[3] = null;

        RTHandles.Release(m_GroundIrradianceTables[0]);
        m_GroundIrradianceTables[0] = null;
        RTHandles.Release(m_GroundIrradianceTables[1]);
        m_GroundIrradianceTables[1] = null;

        RTHandles.Release(m_LocalMultipleScatteringTables[0]);
        m_LocalMultipleScatteringTables[0] = null;
        RTHandles.Release(m_LocalMultipleScatteringTables[1]);
        m_LocalMultipleScatteringTables[1] = null;

        RTHandles.Release(m_GlobalMultipleScatteringTables[0]);
        m_GlobalMultipleScatteringTables[0] = null;
        RTHandles.Release(m_GlobalMultipleScatteringTables[1]);
        m_GlobalMultipleScatteringTables[1] = null;

        RTHandles.Release(m_CloudBaseNoiseTable[0]);
        m_CloudBaseNoiseTable[0] = null;

        RTHandles.Release(m_CloudDetailNoiseTable[0]);
        m_CloudDetailNoiseTable[0] = null;

        RTHandles.Release(m_CloudCurlNoiseTable[0]);
        m_CloudCurlNoiseTable[0] = null;
    }

    void SetGlobalSkyConstants(CommandBuffer cmd, BuiltinSkyParameters builtinParams) {

        var expanseSky = builtinParams.skySettings as ExpanseSky;

        /* Set the parameters that need to be used for sky table
         * precomputation, aka, everything in ExpanseSkyCommon.hlsl. */

        /* Planet. */
        cmd.SetGlobalFloat(_atmosphereThicknessID, expanseSky.atmosphereThickness.value);
        cmd.SetGlobalFloat(_planetRadiusID, expanseSky.planetRadius.value);
        cmd.SetGlobalFloat(_atmosphereRadiusID, expanseSky.atmosphereThickness.value + expanseSky.planetRadius.value);
        cmd.SetGlobalVector(_groundTintID, expanseSky.groundTint.value);
        if (expanseSky.groundAlbedoTexture.value != null) {
          cmd.SetGlobalTexture(_groundAlbedoTextureID, expanseSky.groundAlbedoTexture.value);
        }
        cmd.SetGlobalFloat(_hasGroundAlbedoTextureID, (expanseSky.groundAlbedoTexture.value == null) ? 0f : 1f);
        cmd.SetGlobalVector(_lightPollutionTintID, expanseSky.lightPollutionTint.value);
        cmd.SetGlobalFloat(_lightPollutionIntensityID, expanseSky.lightPollutionIntensity.value);

        Quaternion planetRotation = Quaternion.Euler(expanseSky.planetRotation.value.x,
                                                     expanseSky.planetRotation.value.y,
                                                     expanseSky.planetRotation.value.z);
        cmd.SetGlobalMatrix(_planetRotationID, Matrix4x4.Rotate(planetRotation));

        /* Aerosols. */
        cmd.SetGlobalFloat(_aerosolCoefficientID, expanseSky.aerosolCoefficient.value);
        cmd.SetGlobalFloat(_scaleHeightAerosolsID, expanseSky.scaleHeightAerosols.value);
        cmd.SetGlobalFloat(_aerosolAnisotropyID, expanseSky.aerosolAnisotropy.value);
        cmd.SetGlobalFloat(_aerosolDensityID, expanseSky.aerosolDensity.value);

        /* Air. */
        cmd.SetGlobalVector(_airCoefficientsID, expanseSky.airCoefficients.value);
        cmd.SetGlobalFloat(_scaleHeightAirID, expanseSky.scaleHeightAir.value);
        cmd.SetGlobalFloat(_airDensityID, expanseSky.airDensity.value);

        /* Ozone. */
        cmd.SetGlobalVector(_ozoneCoefficientsID, expanseSky.ozoneCoefficients.value);
        cmd.SetGlobalFloat(_ozoneThicknessID, expanseSky.ozoneThickness.value);
        cmd.SetGlobalFloat(_ozoneHeightID, expanseSky.ozoneHeight.value);
        cmd.SetGlobalFloat(_ozoneDensityID, expanseSky.ozoneDensity.value);

        /* Sampling. */
        cmd.SetGlobalInt(_numberOfTransmittanceSamplesID, expanseSky.numberOfTransmittanceSamples.value);
        cmd.SetGlobalInt(_numberOfLightPollutionSamplesID, expanseSky.numberOfLightPollutionSamples.value);
        cmd.SetGlobalInt(_numberOfScatteringSamplesID, expanseSky.numberOfScatteringSamples.value);
        cmd.SetGlobalInt(_numberOfGroundIrradianceSamplesID, expanseSky.numberOfGroundIrradianceSamples.value);
        cmd.SetGlobalInt(_numberOfMultipleScatteringSamplesID, expanseSky.numberOfMultipleScatteringSamples.value);
        cmd.SetGlobalInt(_numberOfMultipleScatteringAccumulationSamplesID, expanseSky.numberOfMultipleScatteringAccumulationSamples.value);
        cmd.SetGlobalFloat(_useImportanceSamplingID, expanseSky.useImportanceSampling.value ? 1f : 0f);

        /* Set the texture for the actual sky shader. */
        cmd.SetGlobalTexture(_transmittanceTableID, m_TransmittanceTables[0]);
        cmd.SetGlobalTexture(_lightPollutionTableAirID, m_LightPollutionTables[0]);
        cmd.SetGlobalTexture(_lightPollutionTableAerosolID, m_LightPollutionTables[1]);
        cmd.SetGlobalTexture(_groundIrradianceTableAirID, m_GroundIrradianceTables[0]);
        cmd.SetGlobalTexture(_groundIrradianceTableAerosolID, m_GroundIrradianceTables[1]);
        cmd.SetGlobalTexture(_singleScatteringTableAirID, m_SingleScatteringTables[0]);
        cmd.SetGlobalTexture(_singleScatteringTableAerosolID, m_SingleScatteringTables[1]);
        cmd.SetGlobalTexture(_singleScatteringTableAirNoShadowsID, m_SingleScatteringTables[2]);
        cmd.SetGlobalTexture(_singleScatteringTableAerosolNoShadowsID, m_SingleScatteringTables[3]);
        cmd.SetGlobalTexture(_localMultipleScatteringTableID, m_LocalMultipleScatteringTables[0]);
        cmd.SetGlobalTexture(_globalMultipleScatteringTableAirID, m_GlobalMultipleScatteringTables[0]);
        cmd.SetGlobalTexture(_globalMultipleScatteringTableAerosolID, m_GlobalMultipleScatteringTables[1]);
    }

    void SetGlobalCloudConstants(CommandBuffer cmd, BuiltinSkyParameters builtinParams) {

        var expanseSky = builtinParams.skySettings as ExpanseSky;

        /* Clouds. */
        /* Clouds geometry. */
        cmd.SetGlobalFloat(_cloudVolumeLowerRadialBoundaryID, expanseSky.cloudVolumeLowerRadialBoundary.value);
        cmd.SetGlobalFloat(_cloudVolumeUpperRadialBoundaryID, Mathf.Max(expanseSky.cloudVolumeLowerRadialBoundary.value+1,
          expanseSky.cloudVolumeUpperRadialBoundary.value));
        cmd.SetGlobalFloat(_cloudTextureAngularRangeID, expanseSky.cloudTextureAngularRange.value);

        /* Clouds noise. */
        cmd.SetGlobalVector(_basePerlinOctavesID, expanseSky.basePerlinOctaves.value);
        cmd.SetGlobalFloat(_basePerlinOffsetID, expanseSky.basePerlinOffset.value);
        cmd.SetGlobalFloat(_basePerlinScaleFactorID, expanseSky.basePerlinScaleFactor.value);
        cmd.SetGlobalVector(_baseWorleyOctavesID, expanseSky.baseWorleyOctaves.value);
        cmd.SetGlobalFloat(_baseWorleyScaleFactorID, expanseSky.baseWorleyScaleFactor.value);
        cmd.SetGlobalFloat(_baseWorleyBlendFactorID, expanseSky.baseWorleyBlendFactor.value);
        cmd.SetGlobalVector(_structureOctavesID, expanseSky.structureOctaves.value);
        cmd.SetGlobalFloat(_structureScaleFactorID, expanseSky.structureScaleFactor.value);
        cmd.SetGlobalVector(_detailOctavesID, expanseSky.detailOctaves.value);
        cmd.SetGlobalFloat(_detailScaleFactorID, expanseSky.detailScaleFactor.value);
        cmd.SetGlobalFloat(_detailNoiseTileID, expanseSky.detailNoiseTile.value);
        cmd.SetGlobalVector(_coverageOctavesID, expanseSky.coverageOctaves.value);
        cmd.SetGlobalFloat(_coverageOffsetID, expanseSky.coverageOffset.value);
        cmd.SetGlobalFloat(_coverageScaleFactorID, expanseSky.coverageScaleFactor.value);

        /* Set the cloud noise textures. */
        cmd.SetGlobalTexture(_cloudBaseNoiseTableID, m_CloudBaseNoiseTable[0]);
        cmd.SetGlobalTexture(_cloudDetailNoiseTableID, m_CloudDetailNoiseTable[0]);
        cmd.SetGlobalTexture(_cloudCurlNoiseTableID, m_CloudCurlNoiseTable[0]);
        cmd.SetGlobalTexture(_cloudCoverageTableID, m_CloudCoverageTable[0]);
    }

    void SetPrecomputeTextures() {
      /********/
      /* Sky. */
      /********/
      int transmittanceKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_TRANSMITTANCE");
      /* Set the texture for the compute shader. */
      s_PrecomputeCS.SetTexture(transmittanceKernelHandle, "_TransmittanceTableRW",
        m_TransmittanceTables[0]);

      int lightPollutionKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_LIGHT_POLLUTION");
      /* Set the texture for the compute shader. */
      s_PrecomputeCS.SetTexture(lightPollutionKernelHandle, "_LightPollutionTableAirRW",
        m_LightPollutionTables[0]);
      s_PrecomputeCS.SetTexture(lightPollutionKernelHandle, "_LightPollutionTableAerosolRW",
        m_LightPollutionTables[1]);

      int groundIrradianceKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_GROUND_IRRADIANCE");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(groundIrradianceKernelHandle, "_GroundIrradianceTableAirRW",
        m_GroundIrradianceTables[0]);
      s_PrecomputeCS.SetTexture(groundIrradianceKernelHandle, "_GroundIrradianceTableAerosolRW",
        m_GroundIrradianceTables[1]);

      int singleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_SINGLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAirRW",
        m_SingleScatteringTables[0]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAerosolRW",
        m_SingleScatteringTables[1]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAirNoShadowsRW",
        m_SingleScatteringTables[2]);
      s_PrecomputeCS.SetTexture(singleScatteringKernelHandle, "_SingleScatteringTableAerosolNoShadowsRW",
        m_SingleScatteringTables[3]);

      int localMultipleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_LOCAL_MULTIPLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(localMultipleScatteringKernelHandle, "_LocalMultipleScatteringTableRW",
        m_LocalMultipleScatteringTables[0]);

      int globalMultipleScatteringKernelHandle =
        s_PrecomputeCS.FindKernel("COMPUTE_GLOBAL_MULTIPLE_SCATTERING");
      /* Set the textures for the compute shader. */
      s_PrecomputeCS.SetTexture(globalMultipleScatteringKernelHandle, "_GlobalMultipleScatteringTableAirRW",
        m_GlobalMultipleScatteringTables[0]);
      s_PrecomputeCS.SetTexture(globalMultipleScatteringKernelHandle, "_GlobalMultipleScatteringTableAerosolRW",
        m_GlobalMultipleScatteringTables[1]);
    }

    void SetPrecomputeCloudTextures() {
      int cloudBaseNoiseKernelHandle =
        s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_BASE_NOISE");
      /* Set the textures for the compute shader. */
      s_CloudPrecomputeCS.SetTexture(cloudBaseNoiseKernelHandle, "_CloudBaseNoiseTableRW",
        m_CloudBaseNoiseTable[0]);

      int cloudDetailNoiseKernelHandle =
        s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_DETAIL_NOISE");
      /* Set the textures for the compute shader. */
      s_CloudPrecomputeCS.SetTexture(cloudDetailNoiseKernelHandle, "_CloudDetailNoiseTableRW",
        m_CloudDetailNoiseTable[0]);

      int cloudCurlNoiseKernelHandle =
        s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_CURL_NOISE");
      /* Set the textures for the compute shader. */
      s_CloudPrecomputeCS.SetTexture(cloudCurlNoiseKernelHandle, "_CloudCurlNoiseTableRW",
        m_CloudCurlNoiseTable[0]);

      int cloudCoverageKernelHandle =
        s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_COVERAGE");
      /* Set the textures for the compute shader. */
      s_CloudPrecomputeCS.SetTexture(cloudCoverageKernelHandle, "_CloudCoverageTableRW",
        m_CloudCoverageTable[0]);
    }

    void PrecomputeTables(CommandBuffer cmd)
    {
      using (new ProfilingSample(cmd, "Precompute Expanse Sky Tables"))
      {
        int transmittanceKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_TRANSMITTANCE");
        cmd.DispatchCompute(s_PrecomputeCS, transmittanceKernelHandle,
          TransmittanceTableSizeH / 4, TransmittanceTableSizeMu / 4, 1);

        int lightPollutionKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_LIGHT_POLLUTION");
        cmd.DispatchCompute(s_PrecomputeCS, lightPollutionKernelHandle,
          LightPollutionTableSizeH / 4, LightPollutionTableSizeMu / 4, 1);

        int singleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_SINGLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, singleScatteringKernelHandle,
          SingleScatteringTableSizeH / 4, SingleScatteringTableSizeMu / 4,
          (SingleScatteringTableSizeMuL * SingleScatteringTableSizeNu) / 4);

        int localMultipleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_LOCAL_MULTIPLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, localMultipleScatteringKernelHandle,
          LocalMultipleScatteringTableSizeH / 4,
          LocalMultipleScatteringTableSizeH / 4, 1);

        int globalMultipleScatteringKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_GLOBAL_MULTIPLE_SCATTERING");
        cmd.DispatchCompute(s_PrecomputeCS, globalMultipleScatteringKernelHandle,
          GlobalMultipleScatteringTableSizeH / 4,
          GlobalMultipleScatteringTableSizeMu / 4,
          (GlobalMultipleScatteringTableSizeMuL * GlobalMultipleScatteringTableSizeNu) / 4);

        int groundIrradianceKernelHandle =
          s_PrecomputeCS.FindKernel("COMPUTE_GROUND_IRRADIANCE");
        cmd.DispatchCompute(s_PrecomputeCS, groundIrradianceKernelHandle,
          GroundIrradianceTableSize / 4, 1, 1);
      }
    }

    void PrecomputeCloudTables(CommandBuffer cmd)
    {
      using (new ProfilingSample(cmd, "Precompute Expanse Cloud Tables"))
      {
        int cloudBaseNoiseKernelHandle =
          s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_BASE_NOISE");
        cmd.DispatchCompute(s_CloudPrecomputeCS, cloudBaseNoiseKernelHandle,
          CloudBaseNoiseTableSizeX / 4, CloudBaseNoiseTableSizeY / 4, CloudBaseNoiseTableSizeZ / 4);

        int cloudDetailNoiseKernelHandle =
          s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_DETAIL_NOISE");
        cmd.DispatchCompute(s_CloudPrecomputeCS, cloudDetailNoiseKernelHandle,
          CloudDetailNoiseTableSizeX / 4, CloudDetailNoiseTableSizeY / 4, CloudDetailNoiseTableSizeZ / 4);

        int cloudCurlNoiseKernelHandle =
          s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_CURL_NOISE");
        cmd.DispatchCompute(s_CloudPrecomputeCS, cloudCurlNoiseKernelHandle,
          CloudCurlNoiseTableSizeX / 4, CloudCurlNoiseTableSizeY / 4, 1);

        int cloudCoverageKernelHandle =
          s_CloudPrecomputeCS.FindKernel("COMPUTE_CLOUD_COVERAGE");
        cmd.DispatchCompute(s_CloudPrecomputeCS, cloudCoverageKernelHandle,
          CloudCoverageTableSizeX / 4, CloudCoverageTableSizeY / 4, 1);
      }
    }

    protected override bool Update(BuiltinSkyParameters builtinParams)
    {
      var expanseSky = builtinParams.skySettings as ExpanseSky;
      int currentPrecomputationHash = expanseSky.GetPrecomputationHashCode();
      if (currentPrecomputationHash != m_LastPrecomputationParamHash) {
        SetGlobalSkyConstants(builtinParams.commandBuffer, builtinParams);

        SetPrecomputeTextures();

        PrecomputeTables(builtinParams.commandBuffer);

        m_LastPrecomputationParamHash = currentPrecomputationHash;
      }

      int currentCloudPrecomputationHash = expanseSky.GetCloudPrecomputationHashCode();
      if (currentCloudPrecomputationHash != m_LastCloudPrecomputationParamHash) {
        SetGlobalCloudConstants(builtinParams.commandBuffer, builtinParams);

        SetPrecomputeCloudTextures();

        PrecomputeCloudTables(builtinParams.commandBuffer);

        m_LastCloudPrecomputationParamHash = currentCloudPrecomputationHash;
      }

      return builtinParams.skySettings.updateMode != EnvironmentUpdateMode.Realtime;
    }

    /* As in the unity physical sky, renderSunDisk is unsupported, because we
     * instead use the celestial body system. */
    public override void RenderSky(BuiltinSkyParameters builtinParams, bool renderForCubemap, bool renderSunDisk)
    {
        /* TODO: Unity says this is obselete, but finding a way to
         * profile would actually be very cool. */
        using (new ProfilingSample(builtinParams.commandBuffer, "Draw Expanse Sky"))
        {
            var expanseSky = builtinParams.skySettings as ExpanseSky;

            int passID = renderForCubemap ? m_RenderCubemapID : m_RenderFullscreenSkyID;

            /* Set the shader uniform variables. */
            if (expanseSky.nightSkyTexture.value != null) {
              m_PropertyBlock.SetTexture(_nightSkyTextureID, expanseSky.nightSkyTexture.value);
            }
            m_PropertyBlock.SetFloat(_hasNightSkyTextureID, (expanseSky.nightSkyTexture.value == null) ? 0f : 1f);

            if (expanseSky.groundEmissionTexture.value != null) {
              m_PropertyBlock.SetTexture(_groundEmissionTextureID, expanseSky.groundEmissionTexture.value);
            }
            m_PropertyBlock.SetFloat(_hasGroundEmissionTextureID, (expanseSky.groundEmissionTexture.value == null) ? 0f : 1f);

            m_PropertyBlock.SetFloat(_groundEmissionMultiplierID, expanseSky.groundEmissionMultiplier.value);

            Quaternion nightSkyRotation = Quaternion.Euler(expanseSky.nightSkyRotation.value.x,
                                                         expanseSky.nightSkyRotation.value.y,
                                                         expanseSky.nightSkyRotation.value.z);
            m_PropertyBlock.SetMatrix(_nightSkyRotationID, Matrix4x4.Rotate(nightSkyRotation));
            m_PropertyBlock.SetVector(_nightTintID, expanseSky.nightTint.value);
            m_PropertyBlock.SetFloat(_nightIntensityID, expanseSky.nightIntensity.value);
            m_PropertyBlock.SetVector(_skyTintID, expanseSky.skyTint.value);
            m_PropertyBlock.SetFloat(_multipleScatteringMultiplierID, expanseSky.multipleScatteringMultiplier.value);
            m_PropertyBlock.SetFloat(_useAntiAliasingID, expanseSky.useAntiAliasing.value ? 1f : 0f);
            m_PropertyBlock.SetFloat(_ditherAmountID, expanseSky.ditherAmount.value * 0.25f);

            /* Celestial bodies. */
            /* Body 1. */
            m_PropertyBlock.SetFloat(_body1LimbDarkeningID, expanseSky.body1LimbDarkening.value);
            m_PropertyBlock.SetFloat(_body1ReceivesLightID, expanseSky.body1ReceivesLight.value ? 1f : 0f);
            if (expanseSky.body1AlbedoTexture.value != null) {
              m_PropertyBlock.SetTexture(_body1AlbedoTextureID, expanseSky.body1AlbedoTexture.value);
            }
            m_PropertyBlock.SetFloat(_body1HasAlbedoTextureID, (expanseSky.body1EmissionTexture.value == null) ? 0f : 1f);
            m_PropertyBlock.SetFloat(_body1EmissiveID, expanseSky.body1Emissive.value ? 1f : 0f);
            if (expanseSky.body1EmissionTexture.value != null) {
              m_PropertyBlock.SetTexture(_body1EmissionTextureID, expanseSky.body1EmissionTexture.value);
            }
            m_PropertyBlock.SetFloat(_body1HasEmissionTextureID, (expanseSky.body1EmissionTexture.value == null) ? 0f : 1f);
            Quaternion body1Rotation = Quaternion.Euler(expanseSky.body1Rotation.value.x,
                                                         expanseSky.body1Rotation.value.y,
                                                         expanseSky.body1Rotation.value.z);
            m_PropertyBlock.SetMatrix(_body1RotationID, Matrix4x4.Rotate(body1Rotation));

            /* Body 2. */
            m_PropertyBlock.SetFloat(_body2LimbDarkeningID, expanseSky.body2LimbDarkening.value);
            m_PropertyBlock.SetFloat(_body2ReceivesLightID, expanseSky.body2ReceivesLight.value ? 1f : 0f);
            if (expanseSky.body2AlbedoTexture.value != null) {
              m_PropertyBlock.SetTexture(_body2AlbedoTextureID, expanseSky.body2AlbedoTexture.value);
            }
            m_PropertyBlock.SetFloat(_body2HasAlbedoTextureID, (expanseSky.body2AlbedoTexture.value == null) ? 0f : 1f);
            m_PropertyBlock.SetFloat(_body2EmissiveID, expanseSky.body2Emissive.value ? 1f : 0f);
            if (expanseSky.body2EmissionTexture.value != null) {
              m_PropertyBlock.SetTexture(_body2EmissionTextureID, expanseSky.body2EmissionTexture.value);
            }
            m_PropertyBlock.SetFloat(_body2HasEmissionTextureID, (expanseSky.body2EmissionTexture.value == null) ? 0f : 1f);
            Quaternion body2Rotation = Quaternion.Euler(expanseSky.body2Rotation.value.x,
                                                         expanseSky.body2Rotation.value.y,
                                                         expanseSky.body2Rotation.value.z);
            m_PropertyBlock.SetMatrix(_body2RotationID, Matrix4x4.Rotate(body2Rotation));

            /* Body 3. */
            m_PropertyBlock.SetFloat(_body3LimbDarkeningID, expanseSky.body3LimbDarkening.value);
            m_PropertyBlock.SetFloat(_body3ReceivesLightID, expanseSky.body3ReceivesLight.value ? 1f : 0f);
            if (expanseSky.body3AlbedoTexture.value != null) {
              m_PropertyBlock.SetTexture(_body3AlbedoTextureID, expanseSky.body3AlbedoTexture.value);
            }
            m_PropertyBlock.SetFloat(_body3HasAlbedoTextureID, (expanseSky.body3AlbedoTexture.value == null) ? 0f : 1f);
            m_PropertyBlock.SetFloat(_body3EmissiveID, expanseSky.body3Emissive.value ? 1f : 0f);
            if (expanseSky.body3EmissionTexture.value != null) {
              m_PropertyBlock.SetTexture(_body3EmissionTextureID, expanseSky.body3EmissionTexture.value);
            }
            m_PropertyBlock.SetFloat(_body3HasEmissionTextureID, (expanseSky.body3EmissionTexture.value == null) ? 0f : 1f);
            Quaternion body3Rotation = Quaternion.Euler(expanseSky.body3Rotation.value.x,
                                                         expanseSky.body3Rotation.value.y,
                                                         expanseSky.body3Rotation.value.z);
            m_PropertyBlock.SetMatrix(_body3RotationID, Matrix4x4.Rotate(body3Rotation));

            /* Body 4. */
            m_PropertyBlock.SetFloat(_body4LimbDarkeningID, expanseSky.body4LimbDarkening.value);
            m_PropertyBlock.SetFloat(_body4ReceivesLightID, expanseSky.body4ReceivesLight.value ? 1f : 0f);
            if (expanseSky.body4AlbedoTexture.value != null) {
              m_PropertyBlock.SetTexture(_body4AlbedoTextureID, expanseSky.body4AlbedoTexture.value);
            }
            m_PropertyBlock.SetFloat(_body4HasAlbedoTextureID, (expanseSky.body4AlbedoTexture.value == null) ? 0f : 1f);
            m_PropertyBlock.SetFloat(_body4EmissiveID, expanseSky.body4Emissive.value ? 1f : 0f);
            if (expanseSky.body4EmissionTexture.value != null) {
              m_PropertyBlock.SetTexture(_body4EmissionTextureID, expanseSky.body4EmissionTexture.value);
            }
            m_PropertyBlock.SetFloat(_body4HasEmissionTextureID, (expanseSky.body4EmissionTexture.value == null) ? 0f : 1f);
            Quaternion body4Rotation = Quaternion.Euler(expanseSky.body4Rotation.value.x,
                                                         expanseSky.body4Rotation.value.y,
                                                         expanseSky.body4Rotation.value.z);
            m_PropertyBlock.SetMatrix(_body4RotationID, Matrix4x4.Rotate(body4Rotation));

            /* Clouds geometry. */
            m_PropertyBlock.SetFloat(_cloudUOffsetID, expanseSky.cloudUOffset.value);
            m_PropertyBlock.SetFloat(_cloudVOffsetID, expanseSky.cloudVOffset.value);
            m_PropertyBlock.SetFloat(_cloudWOffsetID, expanseSky.cloudWOffset.value);

            /* Clouds lighting. */
            m_PropertyBlock.SetFloat(_cloudDensityID, expanseSky.cloudDensity.value);
            m_PropertyBlock.SetFloat(_cloudFalloffRadiusID, expanseSky.cloudFalloffRadius.value);
            m_PropertyBlock.SetFloat(_densityAttenuationThresholdID, expanseSky.densityAttenuationThreshold.value);
            m_PropertyBlock.SetFloat(_densityAttenuationMultiplierID, expanseSky.densityAttenuationMultiplier.value);
            m_PropertyBlock.SetFloat(_cloudForwardScatteringID, expanseSky.cloudForwardScattering.value);
            m_PropertyBlock.SetFloat(_cloudSilverSpreadID, expanseSky.cloudSilverSpread.value);
            m_PropertyBlock.SetFloat(_silverIntensityID, expanseSky.silverIntensity.value);
            m_PropertyBlock.SetFloat(_depthProbabilityOffsetID, expanseSky.depthProbabilityOffset.value);
            m_PropertyBlock.SetFloat(_depthProbabilityMinID, expanseSky.depthProbabilityMin.value);
            m_PropertyBlock.SetFloat(_depthProbabilityMaxID, expanseSky.depthProbabilityMax.value);
            m_PropertyBlock.SetFloat(_atmosphericBlendDistanceID, expanseSky.atmosphericBlendDistance.value);
            m_PropertyBlock.SetFloat(_atmosphericBlendBiasID, expanseSky.atmosphericBlendBias.value);

            /* Clouds sampling. */
            m_PropertyBlock.SetInt(_numCloudTransmittanceSamplesID, expanseSky.numCloudTransmittanceSamples.value);
            m_PropertyBlock.SetInt(_numCloudSSSamplesID, expanseSky.numCloudSSSamples.value);
            m_PropertyBlock.SetFloat(_cloudCoarseMarchStepSizeID, expanseSky.cloudCoarseMarchStepSize.value);
            m_PropertyBlock.SetFloat(_cloudDetailMarchStepSizeID, expanseSky.cloudDetailMarchStepSize.value);
            m_PropertyBlock.SetInt(_numZeroStepsBeforeCoarseMarchID, expanseSky.numZeroStepsBeforeCoarseMarch.value);

            /* Clouds noise. */
            m_PropertyBlock.SetFloat(_structureNoiseBlendFactorID, expanseSky.structureNoiseBlendFactor.value);
            m_PropertyBlock.SetFloat(_detailNoiseBlendFactorID, expanseSky.detailNoiseBlendFactor.value);
            m_PropertyBlock.SetFloat(_heightGradientLowStartID, expanseSky.heightGradientLowStart.value);
            m_PropertyBlock.SetFloat(_heightGradientLowEndID, expanseSky.heightGradientLowEnd.value);
            m_PropertyBlock.SetFloat(_heightGradientHighStartID, expanseSky.heightGradientHighStart.value);
            m_PropertyBlock.SetFloat(_heightGradientHighEndID, expanseSky.heightGradientHighEnd.value);
            m_PropertyBlock.SetFloat(_coverageBlendFactorID, expanseSky.coverageBlendFactor.value);

            /* Clouds debug. */
            m_PropertyBlock.SetFloat(_cloudsDebugID, expanseSky.cloudsDebug.value ? 1f : 0f);

            /* Builtins. */
            m_PropertyBlock.SetVector(_WorldSpaceCameraPos1ID, builtinParams.worldSpaceCameraPos);
            m_PropertyBlock.SetMatrix(_ViewMatrix1ID, builtinParams.viewMatrix);
            m_PropertyBlock.SetMatrix(_PixelCoordToViewDirWS, builtinParams.pixelCoordToViewDirMatrix);

            CoreUtils.DrawFullScreen(builtinParams.commandBuffer, m_ExpanseSkyMaterial, m_PropertyBlock, passID);
        }
    }
}
