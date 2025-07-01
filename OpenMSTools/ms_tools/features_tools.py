from typing import ClassVar, Literal

import dask
import dask.bag as db
import pyopenms as oms
from pydantic import Field

from .ABCs import (
    MSTool,
    MSToolConfig,
    OpenMSDataWrapper,
    OpenMSMethodConfig,
    OpenMSMethodParam,
)


class MassTraceDetectionConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.MassTraceDetection]] = oms.MassTraceDetection

    mass_error_ppm: float = Field(default=5.0, description="允许的质量偏差（以ppm为单位）")
    noise_threshold_int: float = Field(
        default=1000.0, description="强度阈值，低于该阈值的峰将被视为噪声"
    )
    chrom_peak_snr: float = Field(
        default=3.0, description="超过噪声阈值的峰的最小强度（信噪比），高于该阈值的峰将被视为峰值"
    )
    reestimate_mt_sd: Literal["true", "false"] = Field(
        default="true", description="启用动态重新估计质量踪迹的m/z方差"
    )
    quant_method: Literal["area", "median", "max_height"] = Field(
        default="area",
        description="质量踪迹的量化方法。对于LC数据，推荐使用'area'，对于直接注射数据，推荐使用'median'。'max_height'只使用质量踪迹中最强的峰",
    )
    trace_termination_criterion: Literal["outlier", "sample_rate"] = Field(
        default="outlier",
        description="质量踪迹的终止准则。\
            在'outlier'模式下，如果预定义数量的连续异常值被发现，质量踪迹的扩展将被取消（请参见trace_termination_outliers参数）。\
            在'sample_rate'模式下，如果找到的峰与访问的谱图之间的比率低于'min_sample_rate'阈值，则质量踪迹在两个方向上的扩展将停止。",
    )
    trace_termination_outliers: int = Field(
        default=5,
        description="质量踪迹在单个方向上扩展时，如果达到此数量的连续谱图没有可检测的峰，则扩展将被取消",
    )
    min_sample_rate: float = Field(default=0.5, description="质量踪迹中必须包含峰的最小扫描比例")
    min_trace_length: float = Field(default=5.0, description="质量踪迹的最小预期长度（以秒为单位）")
    max_trace_length: float = Field(
        default=-1.0,
        description="质量踪迹的最大预期长度（以秒为单位）。设置为负值以禁用最大长度检查",
    )
    max_traces: int = Field(default=0, description="最大质量踪迹数量。设置为0以禁用最大数量检查")

class ElutionPeakDetectionConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.ElutionPeakDetection]] = oms.ElutionPeakDetection

    chrom_fwhm: float = Field(default=5.0, description="预期色谱峰的半高全宽（FWHM）（以秒为单位）")
    chrom_peak_snr: float = Field(default=3.0, description="最小信号噪声，质量踪迹应具有的信号噪声")
    width_filtering: Literal["off", "fixed", "auto"] = Field(
        default="fixed",
        description="启用过滤不合理的峰宽。\
            fixed:过滤掉超出[min_fwhm, max_fwhm]间隔的质量踪迹（相应地设置参数）。\
            auto:使用峰宽分布的5和95百分位数进行过滤。",
    )
    min_fwhm: float = Field(
        default=1.0,
        description="最小色谱峰半高全宽（FWHM）（以秒为单位）。如果参数width_filtering为off或auto，则忽略此参数",
    )
    max_fwhm: float = Field(
        default=60.0,
        description="最大色谱峰半高全宽（FWHM）（以秒为单位）。如果参数width_filtering为off或auto，则忽略此参数",
    )
    masstrace_snr_filtering: Literal["false", "true"] = Field(
        default="false", description="在平滑后应用基于信号噪声比的后过滤"
    )
    noise_threshold: float = Field(
        default=1000.0,
        description="噪声阈值。如果masstrace_snr_filtering为true，则使用此阈值进行后过滤。",
    )

class FeatureFindingElements(OpenMSMethodParam):

    C: bool = Field(default=True, description="是否假设样本中存在C元素")
    H: bool = Field(default=True, description="是否假设样本中存在H元素")
    N: bool = Field(default=True, description="是否假设样本中存在N元素")
    O: bool = Field(default=True, description="是否假设样本中存在O元素")
    P: bool = Field(default=True, description="是否假设样本中存在P元素")
    S: bool = Field(default=True, description="是否假设样本中存在S元素")

    def dump2openms(self) -> dict[Literal['elements'], str]:
        return {
            "elements": "".join(
                [
                    "C" if self.C else "",
                    "H" if self.H else "",
                    "N" if self.N else "",
                    "O" if self.O else "",
                    "P" if self.P else "",
                    "S" if self.S else "",
                ]
            )
        }

class FeatureFindingMetaboConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.FeatureFindingMetabo]] = oms.FeatureFindingMetabo

    local_rt_range: float = Field(default=30.0, description="RT范围，用于查找共洗脱质量踪迹")
    local_mz_range: float = Field(default=6.5, description="MZ范围，用于查找同位素质量踪迹")
    charge_lower_bound: int = Field(default=1, description="考虑的最低电荷状态")
    charge_upper_bound: int = Field(default=3, description="考虑的最高电荷状态")
    chrom_fwhm: float = Field(default=5.0, description="预期色谱峰的半高全宽（FWHM）（以秒为单位）")
    report_summed_ints: Literal["false", "true"] = Field(
        default="false",
        description="设置为true以使用所有质量踪迹的强度总和，而不是使用单个质量踪迹的单同位素强度。",
    )
    enable_RT_filtering: Literal["false", "true"] = Field(
        default="true", description="在组装质量踪迹时要求足够的保留时间重叠。禁用直接注射数据。"
    )
    isotope_filtering_model: Literal[
        "metabolites (2% RMS)", "metabolites (5% RMS)", "peptides", "none"
    ] = Field(
        default="metabolites (5% RMS)", description="根据测量质量或MS设备的性能选择适当的噪声模型。"
    )
    mz_scoring_13C: Literal["false", "true"] = Field(
        default="false",
        description="使用13C同位素峰位置（~1.003355 Da）作为同位素质量踪迹的预期偏移（强烈推荐脂质组学！）。\
            禁用一般代谢物（如Kenar等2014，MCP中所述）。",
    )
    use_smoothed_intensities: Literal["false", "true"] = Field(
        default="true", description="使用LOESS强度而不是原始强度。"
    )
    report_convex_hulls: Literal["false", "true"] = Field(
        default="true",
        description="每个报告的特征增加基础质量踪迹的凸包（显著增加featureXML文件大小）。",
    )
    report_chromatograms: Literal["false", "true"] = Field(
        default="true", description="为每个报告的特征添加色谱图（输出在mzml中）。"
    )
    remove_single_traces: Literal["false", "true"] = Field(
        default="true", description="删除未组装的踪迹（单个踪迹）。"
    )
    mz_scoring_by_elements: Literal["false", "true"] = Field(
        default="false",
        description="使用假设元素的m/z范围检测同位素峰。计算假设元素的同位素峰的预期m/z范围。如果启用，这将忽略'mz_scoring_13C'",
    )
    elements: FeatureFindingElements = Field(
        default=FeatureFindingElements(), description="假设样本中存在的元素（这会影响同位素检测）。"
    )

class FeatureFinderConfig(MSToolConfig):

    mass_trace_detection: MassTraceDetectionConfig = Field(
        default=MassTraceDetectionConfig(), description="质量踪迹检测配置"
    )
    elution_peak_detection: ElutionPeakDetectionConfig = Field(
        default=ElutionPeakDetectionConfig(), description="色谱峰检测配置"
    )
    feature_finding_metabo: FeatureFindingMetaboConfig = Field(
        default=FeatureFindingMetaboConfig(), description="质量特征检测配置"
    )

class FeatureFinder(MSTool):

    config_type = FeatureFinderConfig
    config: FeatureFinderConfig

    def __init__(self, config: FeatureFinderConfig | None = None):
        super().__init__(config)
        self.mass_trace_dectecotr = oms.MassTraceDetection()
        self.elution_peak_detector = oms.ElutionPeakDetection()
        self.feature_finder = oms.FeatureFindingMetabo()

    def dectect_mass_traces(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        self.mass_trace_dectecotr.setParameters(self.config.mass_trace_detection.param)

        if len(data.exps) == 0:
            return data

        def run_dectect_mass_traces(exp):
            mass_traces = []
            self.mass_trace_dectecotr.run(exp, mass_traces, 0)
            return mass_traces

        if len(data.exps) == 1:
            data.mass_traces = [run_dectect_mass_traces(data.exps[0])]
        else:
            inputs_bag = db.from_sequence(data.exps)
            outputs_bag = inputs_bag.map(run_dectect_mass_traces)
            data.mass_traces = outputs_bag.compute(scheduler="threads")

        return data

    def detect_elution_peaks(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        self.elution_peak_detector.setParameters(self.config.elution_peak_detection.param)

        if len(data.mass_traces) == 0:
            return data

        def run_detect_elution_peaks(mass_traces):
            mass_traces_split = []
            mass_traces_final = []
            self.elution_peak_detector.detectPeaks(mass_traces, mass_traces_split)
            if self.config.elution_peak_detection.width_filtering == "auto":
                self.elution_peak_detector.filterByPeakWidth(mass_traces_split, mass_traces_final)
            else:
                mass_traces_final = mass_traces_split
            return mass_traces_final

        if len(data.mass_traces) == 1:
            data.mass_traces = [run_detect_elution_peaks(data.mass_traces[0])]
        else:
            inputs_bag = db.from_sequence(data.mass_traces)
            outputs_bag = inputs_bag.map(run_detect_elution_peaks)
            data.mass_traces = outputs_bag.compute(scheduler="threads")

        return data

    def find_features(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        self.feature_finder.setParameters(self.config.feature_finding_metabo.param)

        if len(data.mass_traces) == 0:
            return data

        def run_find_features(mass_traces):
            features = oms.FeatureMap()
            chroms = []
            self.feature_finder.run(mass_traces, features, chroms)
            return features,chroms

        if len(data.mass_traces) == 1:
            feature_map,chromatogram_peaks = run_find_features(data.mass_traces[0])
            data.features = [feature_map]
            data.chromatogram_peaks = [chromatogram_peaks]
        else:
            inputs_bag = db.from_sequence(data.mass_traces)
            outputs_bag = inputs_bag.map(run_find_features)
            features_bag = outputs_bag.pluck(0)
            chromatograms_bag = outputs_bag.pluck(1)
            data.features,data.chromatogram_peaks = dask.compute(features_bag,chromatograms_bag,scheduler="threads")

        return data

    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        data = self.dectect_mass_traces(data)
        data = self.detect_elution_peaks(data)
        data = self.find_features(data)

        return data
