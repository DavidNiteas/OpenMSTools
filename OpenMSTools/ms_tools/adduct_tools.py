from pydantic import Field, model_validator
from abc import ABC, abstractmethod
import pyopenms as oms
from decimal import Decimal
from .ABCs import TomlConfig, OpenMSMethodParam, OpenMSMethodConfig, MSTool, OpenMSDataWrapper
import dask.bag as db
from typing_extensions import Self
from typing import ClassVar,Type,Literal,List,Dict

class AdductConfig(ABC, TomlConfig):
    
    def validate_probabilities(self) -> bool:
        charged_total = Decimal("0.0")
        adducts = []
        for i in self.adducts:
            name, charge, val = i.split(":")
            adducts.append((name, charge, Decimal(val)))
            if charge != "0":
                charged_total += Decimal(val)
        if charged_total == 1.0:
            return True
        else:
            return False
    
    @model_validator(mode="after")
    def validate_probabilities_sum_to_one(self) -> Self:
        """the total probability of all charged adducts needs to be 1"""
        charged_total = Decimal("0.0")
        adducts = []
        for i in self.adducts:
            name, charge, val = i.split(":")
            adducts.append((name, charge, Decimal(val)))
            if charge != "0":
                charged_total += Decimal(val)
        if charged_total == 1.0:
            return self
        else:
            for name, charge, value in adducts:
                if charge == "0":
                    continue
                setattr(self, name.replace("-", "_"), value / charged_total)
            return self

    @property
    @abstractmethod
    def adducts(self) -> List[str]: ...

class AdductPositiveModeConfig(AdductConfig):
    
    H: float = Field(default=0.4, description="[M+H]⁺，质子化分子（最常见）")
    Na: float = Field(default=0.1, description="[M+Na]⁺，钠加合物的概率")
    K: float = Field(default=0.1, description="[M+K]⁺，钾加合物的概率")
    NH4: float = Field(default=0.2, description="[M+NH₄]⁺，铵加合物的概率")
    H_1O_1: float = Field(default=0.1, description="[M+H-H₂O]⁺，质子化后失水的概率")
    H_3O_2: float = Field(default=0.1, description="[M-H3-O2]⁺，质子化后失去两个水的概率")
    H_2O: float = Field(default=0.1, description="[M-H₂O]，中性丢失水的概率")

    @property
    def adducts(self) -> List[str]:
        adducts = {
            "H": (self.H, "+"),
            "Na": (self.Na, "+"),
            "K": (self.K, "+"),
            "NH4": (self.NH4, "+"),
            "H-1O-1": (self.H_1O_1, "+"),
            "H-3O-2": (self.H_3O_2, "+"),
            "H-2O": (self.H_2O, "0"),
        }
        return [f"{key}:{charge}:{value}" for key, (value, charge) in adducts.items()]
    
class AdductNegativeModeConfig(AdductConfig):
    
    H_1: float = Field(default=1.0, description="[M-H]⁻，去质子化分子（最常见）")
    Cl: float = Field(default=0.0, description="[M+Cl]⁻，氯离子加合物的概率")
    HCO2: float = Field(default=0.0, description="[M+HCOO]⁻，甲酸根加合物的概率")
    C2H3O2: float = Field(default=0.0, description="[M+CH₃COO]⁻，乙酸根加合物的概率")
    H_3O_1: float = Field(default=0.0, description="[M-H-H₂O]⁻，去质子化后失水的概率")
    H_1C_1O_2: float = Field(default=0.0, description="[M-H-CO₂]⁻，去质子化后失去二氧化碳的概率")
    CH2O2: float = Field(default=0.5, description="[M-CH₂O₂]，中性丢失甲酸根的概率")

    @property
    def adducts(self) -> List[str]:
        adducts = {
            "H-1": (self.H_1, "-"),
            "Cl": (self.Cl, "-"),
            "HCO2": (self.HCO2, "-"),
            "C2H3O2": (self.C2H3O2, "-"),
            "H-3O-1": (self.H_3O_1, "-"),
            "H-1C-1O-2": (self.H_1C_1O_2, "-"),
            "CH2O2": (self.CH2O2, "0"),
        }
        return [f"{key}:{charge}:{value}" for key, (value, charge) in adducts.items()]

class AdductsModeConfig(OpenMSMethodParam):
    
    positive: AdductPositiveModeConfig = Field(
        default=AdductPositiveModeConfig(),
        description="正向加成的概率分布。"
    )
    negative: AdductNegativeModeConfig = Field(
        default=AdductNegativeModeConfig(),
        description="负向加成的概率分布。"
    )
    mode: Literal["pos", "neg"] = Field("pos", description="选择正负向的加成")
    
    def dump2openms(self) -> Dict[Literal['potential_adducts'],List[str]]:
        param_dict = {}
        match self.mode:
            case 'pos':
                param_dict['potential_adducts'] = self.positive.adducts
            case 'neg':
                param_dict['potential_adducts'] = self.negative.adducts
        return param_dict

class AdductDetectorConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.MetaboliteFeatureDeconvolution]] = oms.MetaboliteFeatureDeconvolution
    
    charge_min: int = Field(default=1, description="最小可能的电荷")
    charge_max: int = Field(default=3, description="最大可能的电荷")
    charge_span_max: int = Field(
        default=3,
        description="单个分析物的最大电荷范围，即观察q1=[5,6,7]意味着span=3。设置为1只会找到相同电荷的加合物变体",
    )
    q_try: Literal["feature", "heuristic", "all"] = Field(
        default="feature",
        description="尝试每个特征的不同电荷值，根据上述设置('heuristic' [不测试所有电荷，只测试可能的电荷]或'all')，或保留特征电荷不变('feature')",
    )
    retention_max_diff: float = Field(
        default=1.0, description="任何两个特征之间的最大允许RT差异，如果它们的关联应被确定"
    )
    retention_max_diff_local: float = Field(
        default=1.0,
        description="在考虑加合物偏移后，两个共特征之间的最大允许RT差异（如果没有加合物偏移，此值应等于'retention_max_diff'，否则应更小！）",
    )
    mass_max_diff: float = Field(
        default=0.05,
        description="每个特征的最大允许质量差异。定义一个对称的质量公差窗口，当查看可能的特征对时，允许的特征差异被合并考虑。对于ppm公差，每个窗口基于各自的观察特征m/z（而不是导致观察到的m/z的假设实验m/z）！",
    )
    unit: Literal["Da", "ppm"] = Field(default="Da", description="'max_difference'参数的单位")
    max_neutrals: int = Field(
        default=1,
        description="允许的最大中性加合物数量（q=0）。在'potential_adducts'参数中添加它们！",
    )
    use_minority_bound: Literal["true", "false"] = Field(
        default="true", description="通过概率阈值修剪考虑的加合物过渡。"
    )
    max_minority_bound: int = Field(default=3, description="通过概率阈值修剪考虑的加合物过渡。")
    min_rt_overlap: float = Field(
        default=0.66, description="两个特征的凸包RT交集与两个特征的并集（如果给出CHs）的最小重叠"
    )
    intensity_filter: Literal["true", "false"] = Field(
        default="false",
        description="启用强度过滤，仅允许两个具有相同电荷的特征之间存在边，如果一个特征的加合物较少，则其强度必须小于另一个特征。不适用于不同电荷的特征。",
    )
    negative_mode: Literal["true", "false"] = Field(default="false", description="启用负离子模式。")
    default_map_label: str = Field(
        default="decharged features", description="默认映射标签，所有特征默认放入输出共识文件中"
    )
    verbose_level: int = Field(default=0, description="处理期间给出的调试信息的数量。")
    potential_adducts: AdductsModeConfig = Field(
        default=AdductsModeConfig(),
        description="可能的加合物列表，可以是正负向的，也可以是混合的。"
    )
    
    @model_validator(mode="after")
    def adducts_mode_validator(cls, v: Self) -> Self:
        if v.negative_mode == "true" and v.potential_adducts.mode == "pos":
            raise ValueError("Cannot use negative mode with positive mode adducts")
        if v.negative_mode == "false" and v.potential_adducts.mode == "neg":
            raise ValueError("Cannot use positive mode with negative mode adducts")
        return v

class AdductDetector(MSTool):
    
    config_type = AdductDetectorConfig
    config: AdductDetectorConfig
    
    def __init__(self, config = None):
        super().__init__(config)
        self.openms_adduct_detector = oms.MetaboliteFeatureDeconvolution()
    
    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:
        
        self.openms_adduct_detector.setParameters(self.config.param)
        
        if len(data.features) == 0:
            return data
        
        def run_adduct_detector(inputs):
            feature_in, feature_map_out, groups, edges = inputs
            self.openms_adduct_detector.compute(feature_in, feature_map_out, groups, edges)
            return feature_map_out
        
        if len(data.features) == 1:
            inputs = (data.features[0], oms.FeatureMap(), oms.ConsensusMap(), oms.ConsensusMap())
            data.features[0] = run_adduct_detector(inputs)
        else:
            features_bag = db.from_sequence(data.features)
            inputs_bag = features_bag.map(lambda x: (x, oms.FeatureMap(), oms.ConsensusMap(), oms.ConsensusMap()))
            outputs_bag = inputs_bag.map(run_adduct_detector)
            data.features = outputs_bag.compute(scheduler="threads")
            
        return data