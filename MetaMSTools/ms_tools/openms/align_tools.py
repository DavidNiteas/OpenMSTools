from typing import ClassVar, Literal

import dask.bag as db
import pyopenms as oms
from pydantic import Field

from ...snaps.MassDataModule.data_module.configs import OpenMSMethodConfig
from ...snaps.MassDataModule.data_module.openms_module import OpenMSMethodParamWrapper
from ...snaps.MassDataModule.data_module.wrappers import OpenMSDataWrapper
from ..ABCs import MSTool


class SuperimposerConfig(OpenMSMethodParamWrapper):

    wrapper_name = "superimposer"

    mz_pair_max_distance: float = Field(
        default=0.5, ge=0.0, description="m/z对的最大距离，这个条件适用于哈希表中的配对"
    )
    rt_pair_distance_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="在每个Map中，考虑用于配对的对必须在保留时间间隔的至少这个分数内（即max - min）",
    )
    num_used_points: int = Field(
        default=2000,
        ge=-1,
        description="在每个Map中，考虑用于配对的点数，使用强度选择。使用所有点，设置为-1。",
    )
    scaling_bucket_size: float = Field(
        default=5.0e-03,
        ge=0.0,
        description="RT间隔的缩放，在配对聚类期间，RT间隔被哈希到大小为这个值的桶中。一个很好的选择是重复运行时预期的误差的一小部分。",
    )
    shift_bucket_size: float = Field(
        default=3.0,
        ge=0.0,
        description="在配对聚类期间，RT间隔的较低（分别较高）端被哈希到大小为这个值的桶中。一个很好的选择是连续的MS扫描之间的时间。",
    )
    max_shift: float = Field(
        default=1000.0,
        ge=0.0,
        description="在配对聚类期间考虑的最大偏移（以秒为单位）。这个值适用于两个方向。",
    )
    max_scaling: float = Field(
        default=2.0, ge=1.0, description="在配对聚类期间考虑的最大缩放。最小缩放是这个值的倒数。"
    )
    dump_buckets: str = Field(
        default="",
        description="哈希表桶的基文件名，如果非空，则将哈希表桶转储到这个文件中。每次调用都会自动附加一个序列号。",
    )
    dump_pairs: str = Field(
        default="",
        description="哈希表对的基文件名，如果非空，则将哈希表对转储到这个文件中（非常大！）。每次调用都会自动附加一个序列号。",
    )

class DistanceRTConfig(OpenMSMethodParamWrapper):

    wrapper_name = "distance_RT"

    max_difference: float = Field(
        default=100.0, ge=0.0, description="RT距离的最大差异（以秒为单位）"
    )
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化RT差异（[0-1]，相对于max_difference）的指数。使用1或2会很快，其他值会非常慢。",
    )
    weight: float = Field(default=1.0, ge=0.0, description="RT距离的权重")

class DistanceMZConfig(OpenMSMethodParamWrapper):

    wrapper_name = "distance_MZ"

    max_difference: float = Field(default=0.3, ge=0.0, description="m/z距离的最大差异")
    unit: Literal["Da", "ppm"] = Field(default="Da", description="m/z距离的单位")
    exponent: float = Field(
        default=2.0,
        ge=0.0,
        description="归一化m/z差异（[0-1]，相对于max_difference）的指数。使用1或2会很快，其他值会非常慢。",
    )
    weight: float = Field(default=1.0, ge=0.0, description="m/z距离的权重")

class DistanceIntensityConfig(OpenMSMethodParamWrapper):

    wrapper_name = "distance_intensity"

    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化强度差异（[0-1]，相对于max_difference）的指数。使用1或2会很快，其他值会非常慢。",
    )
    weight: float = Field(default=0.0, ge=0.0, description="强度距离的权重")
    log_transform: Literal["enabled", "disabled"] = Field(
        default="disabled",
        description="是否对强度进行对数变换，\
            如果禁用，d = |int_f2 - int_f1| / int_max。\
            如果启用，d = |log(int_f2 + 1) - log(int_f1 + 1)| / log(int_max + 1))",
    )

class PairfinderConfig(OpenMSMethodParamWrapper):

    wrapper_name = "pairfinder"

    second_nearest_gap: float = Field(
        default=2.0,
        ge=1.0,
        description="在两个Map中，考虑用于配对的点必须在第二近邻的间隙大于这个值时才配对。",
    )
    use_identifications: Literal["true", "false"] = Field(
        default="false",
        description="是否从不配对不同注释的特征（没有ID的特征总是匹配；只考虑每个注释的最佳命中）",
    )
    ignore_charge: Literal["true", "false"] = Field(
        default="false", description="是否忽略电荷状态（或至少一个未知电荷'0'）"
    )
    ignore_adduct: Literal["true", "false"] = Field(
        default="true", description="是否忽略加合物（或至少一个没有加合物注释的特征）"
    )
    distance_RT: DistanceRTConfig = Field(
        default=DistanceRTConfig(), description="RT距离的参数设置"
    )
    distance_MZ: DistanceMZConfig = Field(
        default=DistanceMZConfig(), description="m/z距离的参数设置"
    )
    distance_intensity: DistanceIntensityConfig = Field(
        default=DistanceIntensityConfig(), description="强度距离的参数设置"
    )

class RTAlignerConfig(OpenMSMethodConfig):

    openms_method: ClassVar[
        type[oms.MapAlignmentAlgorithmPoseClustering]
    ] = oms.MapAlignmentAlgorithmPoseClustering

    max_num_peaks_considered: int = Field(
        default=1000, ge=-1, description="最大峰值数，-1表示使用所有峰值"
    )
    superimposer: SuperimposerConfig = Field(
        default=SuperimposerConfig(), description="对齐的参数设置"
    )
    pairfinder: PairfinderConfig = Field(default=PairfinderConfig(), description="配对的参数设置")

class RTAligner(MSTool):

    config_type = RTAlignerConfig
    config: RTAlignerConfig

    def __init__(self, config: RTAlignerConfig | None = None):
        super().__init__(config)
        self.openms_aligner = oms.MapAlignmentAlgorithmPoseClustering()
        self.transformer = oms.MapAlignmentTransformer()

    def infer_trafo(self, data: OpenMSDataWrapper) -> oms.TransformationDescription:

        self.openms_aligner.setParameters(self.config.param)
        self.openms_aligner.setReference(data.ref_feature_for_align)

        if len(data.features) == 0:
            return data

        def run_infer_trafo(inputs):
            feature_map, trafo = inputs
            self.openms_aligner.align(feature_map,trafo)
            return trafo

        if len(data.features) == 1:
            inputs = (data.features[0], oms.TransformationDescription())
            data.trafos = [run_infer_trafo(inputs)]
        else:
            features_bag = db.from_sequence(data.features)
            inputs_bag = features_bag.map(lambda x: (x, oms.TransformationDescription()))
            outputs_bag = inputs_bag.map(run_infer_trafo)
            data.trafos = outputs_bag.compute(scheduler="threads")

        return data

    def align_features(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        if len(data.features) == 0:
            return data

        def run_align_features(inputs):
            feature_map, trafo = inputs
            self.transformer.transformRetentionTimes(feature_map, trafo, True)
            return feature_map

        if len(data.features) == 1:
            inputs = (data.features[0], data.trafos[0])
            data.features = [run_align_features(inputs)]
        else:
            inputs_bag = db.from_sequence(zip(data.features, data.trafos))
            outputs_bag = inputs_bag.map(run_align_features)
            data.features = outputs_bag.compute(scheduler="threads")

        return data

    def align_exps(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        if len(data.exps) == 0:
            return data

        def run_align_exps(inputs):
            exp, trafo = inputs
            self.transformer.transformRetentionTimes(exp, trafo, True)
            return exp

        if len(data.exps) == 1:
            inputs = (data.exps[0], data.trafos[0])
            data.exps = [run_align_exps(inputs)]
        else:
            inputs_bag = db.from_sequence(zip(data.exps, data.trafos))
            outputs_bag = inputs_bag.map(run_align_exps)
            data.exps = outputs_bag.compute(scheduler="threads")

        return data

    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        data = self.infer_trafo(data)
        data = self.align_features(data)
        data = self.align_exps(data)

        return data
