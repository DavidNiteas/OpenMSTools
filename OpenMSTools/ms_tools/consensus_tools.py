from pydantic import Field
import pyopenms as oms
import dask.bag as db
from typing_extensions import Self
from typing import ClassVar, Type, Literal, Optional, Union, Dict
from .ABCs import OpenMSMethodParamWrapper, ConvertMethodConfig, OpenMSMethodConfig, MSTool, OpenMSDataWrapper

class QTDistanceRTConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_RT"
    
    max_difference: float = Field(default=100.0, ge=0.0, description="RT距离的最大差异")
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化RT差异（[0-1]，相对于max_difference）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="RT距离的权重")

class QTDistanceMZConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_MZ"
    
    max_difference: float = Field(default=0.3, ge=0.0, description="m/z距离的最大差异")
    unit: Literal["Da", "ppm"] = Field(default="Da", description="m/z距离的单位")
    exponent: float = Field(
        default=2.0,
        ge=0.0,
        description="归一化m/z差异（[0-1]，相对于max_difference）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="m/z距离的权重")

class QTDistanceIntensityConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_intensity"
    
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化强度差异（[0-1]，相对于max_difference）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=0.0, ge=0.0, description="强度距离的权重")
    log_transform: Literal["enabled", "disabled"] = Field(
        default="disabled",
        description="是否对强度进行对数变换，如果禁用，d = |int_f2 - int_f1| / int_max。如果启用，d = |log(int_f2 + 1) - log(int_f1 + 1)| / log(int_max + 1))",
    )

class FeatureGroupingAlgorithmQTConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.FeatureGroupingAlgorithmQT]] = oms.FeatureGroupingAlgorithmQT
    
    use_identifications: Literal["true", "false"] = Field(
        default="false",
        description="是否从不配对不同注释的特征（没有ID的特征总是匹配；只考虑每个注释的最佳命中）",
    )
    nr_partitions: int = Field(
        default=100,
        ge=1,
        description="m/z空间中应该使用的分区数量（更多分区意味着更快的运行时间和更高效的内存执行）",
    )
    min_nr_diffs_per_bin: int = Field(
        default=50,
        ge=5,
        description="如果使用ID：在RT区域中，使用匹配ID的差异数量来计算未IDed特征的链接公差。RT区域将扩展直到达到该数量。",
    )
    min_IDscore_forTolCalc: float = Field(
        default=1.0, description="如果使用ID：假设可靠匹配的ID的最小得分。检查当前的得分类型！"
    )
    noID_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="如果使用ID：对于归一化的距离，缺失ID的惩罚是多少？0 = 没有偏差，1 = ID在最大公差内时总是被优先考虑（即使更远）",
    )
    ignore_charge: Literal["true", "false"] = Field(
        default="false", description="是否忽略电荷状态（或至少一个未知电荷'0'）"
    )
    ignore_adduct: Literal["true", "false"] = Field(
        default="true", description="是否忽略加合物（或至少一个没有加合物注释的特征）"
    )
    distance_RT: QTDistanceRTConfig = Field(
        default=QTDistanceRTConfig(), description="RT距离的参数设置"
    )
    distance_MZ: QTDistanceMZConfig = Field(
        default=QTDistanceMZConfig(), description="m/z距离的参数设置"
    )
    distance_intensity: QTDistanceIntensityConfig = Field(
        default=QTDistanceIntensityConfig(), description="强度距离的参数设置"
    )
    
class WarpConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "warp"
    
    enabled: Literal["true", "false"] = Field(
        default="true", description="在Grouping之前，是否使用LOWESS变换内部扭曲特征的RT"
    )
    rt_tol: float = Field(default=100.0, ge=0.0, description="RT公差窗口宽度（秒）")
    mz_tol: float = Field(default=5.0, ge=0.0, description="m/z公差（ppm或Da）")
    max_pairwise_log_fc: float = Field(
        default=0.5,
        description="兼容性图构建期间，两个兼容信号之间的最大绝对log10倍数变化。如果绝对log倍数变化超过此限制，则来自不同图谱的两个信号将不会在兼容性图中通过边连接（但是，它们可能仍然最终出现在同一连通分量中）。注意：这不限制连接阶段的倍数变化，仅在RT对齐期间限制，我们试图找到高质量的对齐锚点。将此值设置为 < 0 会禁用 FC 检查。",
    )
    min_rel_cc_size: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="仅考虑包含至少max(2, (warp_min_occur * number_of_input_maps))输入映射的兼容特征的连接组件",
    )
    max_nr_conflicts: int = Field(
        default=0,
        ge=-1,
        description="每个相关组分中允许的冲突数量（特征来自同一映射），-1表示允许任何数量的冲突",
    )

class LinkConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "link"
    
    rt_tol: float = Field(default=30.0, ge=0.0, description="RT公差窗口宽度（秒）")
    mz_tol: float = Field(default=10.0, ge=0.0, description="m/z公差（ppm或Da）")
    charge_merging: Literal["Identical", "With_charge_zero", "Any"] = Field(
        default="With_charge_zero",
        description="是否允许电荷不匹配（Identical），允许链接电荷零（即未知电荷状态）与每个电荷状态，或者忽略电荷（Any）",
    )
    adduct_merging: Literal["Identical", "With_unknown_adducts", "Any"] = Field(
        default="Any",
        description="是否仅允许相同加合物进行链接（Identical），也允许链接加合物为零的特征，或者忽略加合物（Any）",
    )

class KDDistanceRTConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_RT"
    
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化RT差异（[0-1]，相对于max_difference）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="RT距离的权重")

class KDDistanceMZConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_MZ"
    
    exponent: float = Field(
        default=2.0,
        ge=0.0,
        description="归一化m/z差异（[0-1]，相对于max_difference）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="m/z距离的权重")

class KDDistanceIntensityConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_intensity"
    
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化强度差异（[0-1]）的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="强度距离的权重")
    log_transform: Literal["enabled", "disabled"] = Field(
        default="enabled", description="是否对强度进行对数变换"
    )

class LOWESSConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "LOWESS"
    
    span: float = Field(
        default=0.666666666666667,
        ge=0.0,
        le=1.0,
        description="用于每个局部回归的点数（f）的分数，决定了平滑量。选择此参数在0.2到0.8之间通常会得到良好的拟合。",
    )
    num_iterations: int = Field(default=3, ge=0, description="鲁棒拟合的迭代次数")
    delta: float = Field(
        default=-1.0,
        description="非负值时可用于节省计算（推荐值为输入范围的0.01，例如对于从1000秒到2000秒的数据，可以设置为10）。设置为负值会自动执行此操作。",
    )
    interpolation_type: Literal["linear", "cspline", "akima"] = Field(
        default="cspline", description="用于插值的插值方法"
    )
    extrapolation_type: Literal["two-point-linear", "four-point-linear", "global-linear"] = Field(
        default="four-point-linear", description="用于外插的插值方法"
    )

class FeatureGroupingAlgorithmKDConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.FeatureGroupingAlgorithmKD]] = oms.FeatureGroupingAlgorithmKD
    
    warp: WarpConfig = Field(default=WarpConfig(), description="Warp配置")
    link: LinkConfig = Field(default=LinkConfig(), description="Link配置")
    distance_RT: KDDistanceRTConfig = Field(default=KDDistanceRTConfig(), description="RT距离配置")
    distance_MZ: KDDistanceMZConfig = Field(default=KDDistanceMZConfig(), description="m/z距离配置")
    distance_intensity: KDDistanceIntensityConfig = Field(
        default=KDDistanceIntensityConfig(), description="强度距离配置"
    )
    LOWESS: LOWESSConfig = Field(default=LOWESSConfig(), description="LOWESS配置")
    
class ULBDistanceRTConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_RT"
    
    max_difference: float = Field(
        default=100.0, ge=0.0, description="永远不配对RT距离大于max_difference的特征"
    )
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="归一化RT差异([0-1],相对于max_difference)的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="RT距离的权重")

class ULBDistanceMZConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_MZ"
    
    max_difference: float = Field(
        default=0.3, ge=0.0, description="永远不配对m/z距离大于max_difference的特征"
    )
    unit: Literal["Da", "ppm"] = Field(default="Da", description="max_difference的单位")
    exponent: float = Field(
        default=2.0,
        ge=0.0,
        description="归一化([0-1],相对于max_difference)的m/z差异的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=1.0, ge=0.0, description="m/z距离的权重")

class ULBDistanceIntensityConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "distance_intensity"
    
    exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="相对强度([0-1])的差异的指数，使用1或2会很快，其他值会非常慢",
    )
    weight: float = Field(default=0.0, ge=0.0, description="强度距离的权重")
    log_transform: Literal["enabled", "disabled"] = Field(
        default="disabled",
        description="是否对强度进行对数变换，如果禁用，d = |int_f2 - int_f1| / int_max。如果启用，d = |log(int_f2 + 1) - log(int_f1 + 1)| / log(int_max + 1))",
    )

class FeatureGroupingAlgorithmUnlabeledConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.FeatureGroupingAlgorithmUnlabeled]] = oms.FeatureGroupingAlgorithmUnlabeled
    
    second_nearest_gap: float = Field(
        default=2.0,
        ge=1.0,
        description="只配对距离第二近邻(for both sides)大于second_nearest_gap的特征",
    )
    use_identifications: Literal["true", "false"] = Field(
        default="false", description="从不配对注释不同的特征(相同注释的特征总是配对)."
    )
    ignore_charge: Literal["true", "false"] = Field(
        default="false",
        description="false [default]: 配对要求相等的电荷状态(或至少一个未知电荷'0'); true: 配对不考虑电荷状态",
    )
    ignore_adduct: Literal["true", "false"] = Field(
        default="true",
        description="true [default]: 配对要求相等的加合物(或至少一个没有加合物注释的特征); true: 配对不考虑加合物",
    )
    distance_RT: ULBDistanceRTConfig = Field(
        default=ULBDistanceRTConfig(), description="RT距离的参数设置"
    )
    distance_MZ: ULBDistanceMZConfig = Field(
        default=ULBDistanceMZConfig(), description="m/z距离的参数设置"
    )
    distance_intensity: ULBDistanceIntensityConfig = Field(
        default=ULBDistanceIntensityConfig(), description="强度距离的参数设置"
    )
    
class FeatureGroupingAlgorithmLabeledConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.FeatureGroupingAlgorithmLabeled]] = oms.FeatureGroupingAlgorithmLabeled
    
    rt_estimate: Literal["true", "false"] = Field(
        default="true",
        description="是否估计RT距离和偏差，如果为true，则使用高斯分布拟合RT距离和偏差，如果为false，则使用rt_pair_dist, rt_dev_low和rt_dev_high定义RT距离和偏差，注意：此选项仅适用于具有显著数量的配对数据集！",
    )
    rt_pair_dist: float = Field(default=-20.0, description="RT距离（以秒为单位）")
    rt_dev_low: float = Field(default=15.0, ge=0.0, description="RT偏差的最小值")
    rt_dev_high: float = Field(default=15.0, ge=0.0, description="RT偏差的最大值")
    mz_pair_dists: list[float] = Field(default=[4.0], description="m/z距离（以Th为单位）的最优值")
    mz_dev: float = Field(default=0.05, ge=0.0, description="m/z偏差的最大值")
    mrm: Literal["true", "false"] = Field(default="false", description="是否使用MRM数据")
    
class FeatureLinkerConfig(ConvertMethodConfig):
    
    configs_type: ClassVar[
        Dict[
            Literal[
                "QT","KD","Unlabeled","Labeled"
            ],
            Union[
                Type[FeatureGroupingAlgorithmQTConfig],
                Type[FeatureGroupingAlgorithmKDConfig],
                Type[FeatureGroupingAlgorithmUnlabeledConfig],
                Type[FeatureGroupingAlgorithmLabeledConfig],
            ]
        ]
    ] = {
        "QT": FeatureGroupingAlgorithmQTConfig,
        "KD": FeatureGroupingAlgorithmKDConfig,
        "Unlabeled": FeatureGroupingAlgorithmUnlabeledConfig,
        "Labeled": FeatureGroupingAlgorithmLabeledConfig,
    }
    
    method_name: Literal[
        "QT","KD","Unlabeled","Labeled"
    ] = Field(default="QT", description="指定使用的特征分组算法")
    
    configs: Dict[
        Literal[
            "QT","KD","Unlabeled","Labeled"
        ],
        Union[
            FeatureGroupingAlgorithmQTConfig,
            FeatureGroupingAlgorithmKDConfig,
            FeatureGroupingAlgorithmUnlabeledConfig,
            FeatureGroupingAlgorithmLabeledConfig,
        ]
    ] = Field(
        default={
            "QT": FeatureGroupingAlgorithmQTConfig(),
            "KD": FeatureGroupingAlgorithmKDConfig(),
            "Unlabeled": FeatureGroupingAlgorithmUnlabeledConfig(),
            "Labeled": FeatureGroupingAlgorithmLabeledConfig(),
        }
    )
    
    @property
    def config(self) -> Union[
        FeatureGroupingAlgorithmQTConfig,
        FeatureGroupingAlgorithmKDConfig,
        FeatureGroupingAlgorithmUnlabeledConfig,
        FeatureGroupingAlgorithmLabeledConfig,
    ]:
        return self.configs[self.method_name]
    
    @config.setter
    def config(self, value: Union[
        FeatureGroupingAlgorithmQTConfig,
        FeatureGroupingAlgorithmKDConfig,
        FeatureGroupingAlgorithmUnlabeledConfig,
        FeatureGroupingAlgorithmLabeledConfig,
    ]):
        self.configs[self.method_name] = value

class FeatureLinker(MSTool):
    
    config_type: FeatureLinkerConfig = FeatureLinkerConfig
    
    def __init__(self, config: Optional[FeatureLinkerConfig] = None):
        super().__init__(config)
        self.config: FeatureLinkerConfig
        self.feature_linker = self.config.config.openms_method()
        
    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:
        consensus_map = oms.ConsensusMap()
        file_descriptions = consensus_map.getColumnHeaders()
        for i,(exp_name, feature_map) in enumerate(zip(data.exp_names, data.features)):
            file_description = file_descriptions.get(i, oms.ColumnHeader())
            file_description.filename = exp_name
            file_description.size = feature_map.size()
            file_description.unique_id = feature_map.getUniqueId()
            file_descriptions[i] = file_description
        consensus_map.setColumnHeaders(file_descriptions)
        self.feature_linker.group(data.features, consensus_map)
        data.consensus_map = consensus_map
        return data