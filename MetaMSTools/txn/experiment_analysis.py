from pydantic import Field

from ..ms_tools import (
    AdductDetector,
    AdductDetectorConfig,
    Centrizer,
    CentrizerConfig,
    FeatureFinder,
    FeatureFinderConfig,
    FeatureLinker,
    FeatureLinkerConfig,
    RTAligner,
    RTAlignerConfig,
    TICSmoother,
    TICSmootherConfig,
    MSTool,
    MSToolConfig,
    OpenMSDataWrapper,
)
from .reference_analysis import ReferenceFeatureFinderConfig,ReferenceFeatureFinder

class ExperimentAnalysisConfig(MSToolConfig):

    use_tic_smoother: bool = Field(
        default=True,
        description="是否使用TIC平滑"
    )
    tic_smoother_config: TICSmootherConfig = Field(
        default=TICSmootherConfig(),
        description="TIC平滑的配置"
    )
    use_centrizer: bool = Field(
        default=False,
        description="是否使用质心化校正"
    )
    centrizer_config: CentrizerConfig = Field(
        default=CentrizerConfig(),
        description="质心化校正的配置"
    )
    feature_finder_config: FeatureFinderConfig = Field(
        default=FeatureFinderConfig(),
        description="特征提取的配置"
    )
    use_rt_aligner: bool = Field(
        default=True,
        description="是否使用RT对齐"
    )
    rt_aligner_config: RTAlignerConfig = Field(
        default=RTAlignerConfig(),
        description="RT对齐的配置"
    )
    reference_analysis_config: ReferenceFeatureFinderConfig = Field(
        default=ReferenceFeatureFinderConfig(),
        description="参考文件的分析配置，用于RT对齐"
    )
    use_adduct_detector: bool = Field(
        default=True,
        description="是否使用加合物检测"
    )
    adduct_detector_config: AdductDetectorConfig = Field(
        default=AdductDetectorConfig(),
        description="加合物检测的配置"
    )
    feature_linker_config: FeatureLinkerConfig = Field(
        default=FeatureLinkerConfig(),
        description="特征链接的配置（如果只有一个输入文件，则不会进行特征链接）"
    )

class ExperimentAnalysis(MSTool):

    config_type = ExperimentAnalysisConfig
    config: ExperimentAnalysisConfig

    def __call__(
        self,
        exp_file_paths: list[str],
        ref_file_paths: list[str],
    ):
        pass
