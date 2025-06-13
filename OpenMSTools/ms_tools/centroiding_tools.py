from pydantic import Field, model_validator
import pyopenms as oms
import dask.bag as db
from typing_extensions import Self
from typing import ClassVar, Type, Literal, Optional, Union
from .ABCs import OpenMSMethodParamWrapper, OpenMSMethodConfig, MSTool, OpenMSDataWrapper

class SignalToNoiseConfig(OpenMSMethodParamWrapper):
    
    wrapper_name = "SignalToNoise"
    
    max_intensity: float = Field(
        default=-1,
        ge=-1,
        description="直方图构建所考虑的最大强度。默认情况下，它将自动计算（参见 auto_mode）。仅当您清楚自己在做什么（并将“auto_mode”更改为“-1”）时才提供此参数！所有等于或高于“max_intensity”的强度都将被添加到最后一个直方图 bin 中。如果选择的“max_intensity”太小，噪声估计也可能太小。如果选择的太大，bin 将变得相当大（您可以通过增加“bin_count”来解决这个问题，但这会增加运行时）。一般来说，与 MeanIterative-S/N 相比，Median-S/N 估计器对人工设置的 max_intensity 更具鲁棒性。",
    )
    auto_max_stdev_factor: float = Field(
        default=3.0,
        ge=0.0,
        le=999.0,
        description="用于“最大强度”估计的参数（如果“auto_mode”== 0）：平均值 + “auto_max_stdev_factor” * 标准差",
    )
    auto_max_percentile: int = Field(
        default=95,
        ge=0,
        le=100,
        description="用于“最大强度”估计的参数（如果“auto_mode”== 1）：auto_max_percentile 百分位",
    )
    auto_mode: int = Field(
        default=0,
        ge=-1,
        le=1,
        description="用于确定最大强度的方法：-1 --> 使用“max_intensity”；0 --> “auto_max_stdev_factor” 方法（默认）；1 --> “auto_max_percentile” 方法",
    )
    win_len: float = Field(
        default=200.0, ge=1.0, description="用于直方图构建的窗口长度（以 Thomson 为单位）"
    )
    bin_count: int = Field(default=30, ge=3, description="用于直方图构建的 bin 数量")
    min_required_elements: int = Field(
        default=10, ge=1, description="窗口中所需的最小元素数量（否则认为稀疏）"
    )
    noise_for_empty_window: float = Field(default=1.0e20, description="用于稀疏窗口的噪声值")
    write_log_messages: Literal["true", "false"] = Field(
        default="true", description="在稀疏窗口或右端直方图 bin 中写入日志消息"
    )

class CentrizerConfig(OpenMSMethodConfig):
    
    openms_method: ClassVar[Type[oms.PeakPickerHiRes]] = oms.PeakPickerHiRes
    
    signal_to_noise: float = Field(
        default=0.0,
        ge=0.0,
        description="最小信号噪声比，用于峰值检测。默认值为 0.0，表示禁用信号噪声比估计。",
    )
    spacing_difference_gap: float = Field(
        default=4.0,
        ge=0.0,
        description="峰值扩展的间隔差值。默认值为 4.0，表示间隔差值为 4 倍最小间隔。",
    )
    spacing_difference: float = Field(
        default=1.5,
        ge=0.0,
        description="峰值扩展的间隔差值。默认值为 1.5，表示间隔差值为 1.5 倍最小间隔。",
    )
    missing: int = Field(
        default=1, ge=0, description="峰值扩展的缺失值。默认值为 1，表示允许 1 个缺失值。"
    )
    ms_levels: Union[int, list[int]] = Field(
        default=[1, 2],
        description="用于峰值检测的质谱级别。默认值为 [1, 2]，表示仅在 MS1 和 MS2 级别进行峰值检测。",
    )
    report_FWHM: Literal["true", "false"] = Field(
        default="false", description="是否报告峰的 FWHM 值。默认值为 False，表示不报告 FWHM 值。"
    )
    report_FWHM_unit: Literal["relative", "absolute"] = Field(
        default="relative",
        description="FWHM 的单位。默认值为 'relative'，表示 FWHM 以相对单位表示。",
    )
    SignalToNoise: SignalToNoiseConfig = Field(
        default=SignalToNoiseConfig(), description="用于信号噪声估计的参数"
    )
    
    @property
    def is_legal_ms_levels(self) -> bool:
        if isinstance(self.ms_levels, int):
            return self.ms_levels >= 1
        elif isinstance(self.ms_levels, list):
            return all(i >= 1 for i in self.ms_levels)
        else:
            return False

    @model_validator(mode="after")
    def validate_ms_levels(self) -> Self:
        if isinstance(self.ms_levels, int):
            assert self.ms_levels >= 1, f"ms_levels 必须 >= 1, 当前:{self.ms_levels}"
        elif isinstance(self.ms_levels, list):
            assert all(
                i >= 1 for i in self.ms_levels 
            ), f"ms_levels 必须 >= 1, 当前:{self.ms_levels}"
        else:
            raise ValueError("ms_levels 必须是整数或列表")
        return self
    
class Centrizer(MSTool):
    
    config_type = CentrizerConfig
    config: CentrizerConfig
    
    def __init__(self, config: Optional[CentrizerConfig] = None):
        super().__init__(config)
        self.peak_picker = oms.PeakPickerHiRes()
        
    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        if len(data.exps) == 0:
            return data
        
        def run_centrization(exp_in):
            exp_out = oms.MSExperiment()
            self.peak_picker.pickExperiment(exp_in, exp_out, True)
            return exp_out
        
        if len(data.exps) == 1:
            data.exps = [run_centrization(data.exps[0])]
        else:
            inputs_bag = db.from_sequence(data.exps)
            outputs_bag = inputs_bag.map(run_centrization)
            data.exps = outputs_bag.compute(scheduler="threads")
            
        return data