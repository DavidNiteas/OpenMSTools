from .base_tools import (
    oms,InitProcessAndParamObj,
    oms_exps,
    get_kv_pairs
)
from .async_tools import trio
from .ms_exp_tools import copy_ms_experiments
from typing import Optional

async def PeakPickerHiRes_pickExperiment_coroutine(
    centroided_exp:oms.MSExperiment,
    inputs_exp:oms.MSExperiment,
    process_obj:oms.PeakPickerHiRes,
):
    process_obj.pickExperiment(centroided_exp,inputs_exp,True)
    
async def mz_centroiding_step(
    centroided_exps:oms_exps,
    inputs_exps:oms_exps,
    process_obj:oms.PeakPickerHiRes,
):
    async with trio.open_nursery() as nursery:
        for key,centroided_exp in get_kv_pairs(centroided_exps):
            nursery.start_soon(PeakPickerHiRes_pickExperiment_coroutine, centroided_exp, inputs_exps[key], process_obj)            

@InitProcessAndParamObj(
    oms.PeakPickerHiRes
)
def mz_centroiding(
    ms_exps:oms_exps,
    process_obj:Optional[oms.PeakPickerHiRes] = None,
    param_obj:Optional[oms.Param] = None,
    **kwargs,
) -> oms_exps:
    """
    对质谱实验数据进行质心提取处理。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含需要进行质心提取的原始数据。
    - process_obj (Optional[oms.Centroiding]): 可选的质心提取处理对象。如果为None，则使用默认的质心提取参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - signal_to_noise (float): 信噪比，默认值为0.0。
        - spacing_difference_gap (float): 间距差异的阈值，默认值为4.0。
        - spacing_difference (float): 间距差异，默认值为1.5。
        - missing (int): 缺失数据的处理方式，默认值为1。
        - ms_levels (list): MS级别的列表，默认值为空。
        - report_FWHM (str): 是否报告全宽半高（FWHM），默认为'false'。
        - report_FWHM_unit (str): FWHM的单位，默认为'relative'。
        - SignalToNoise:max_intensity (float): 信噪比的最大强度，默认值为-1。
        - SignalToNoise:auto_max_stdev_factor (float): 自动最大标准差因子，默认值为3.0。
        - SignalToNoise:auto_max_percentile (int): 自动最大百分位，默认值为95。
        - SignalToNoise:auto_mode (int): 自动模式，默认值为0。
        - SignalToNoise:win_len (float): 窗口长度，默认值为200.0。
        - SignalToNoise:bin_count (int): 直方图的分箱数量，默认值为30。
        - SignalToNoise:min_required_elements (int): 最小要求元素数量，默认值为10。
        - SignalToNoise:noise_for_empty_window (float): 空窗口噪声值，默认值为1e+20。
        - SignalToNoise:write_log_messages (str): 是否记录日志信息，默认为'true'。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经质心提取处理后的质谱实验数据对象。
    """
    if isinstance(ms_exps, oms.MSExperiment):
        inputs_exps = [ms_exps]
    else:
        inputs_exps = ms_exps
    centroided_exps = copy_ms_experiments(inputs_exps, use_blank=True)
    trio.run(mz_centroiding_step, centroided_exps, inputs_exps, process_obj)
    if isinstance(ms_exps, oms.MSExperiment):
        centroided_exps = centroided_exps[0]
    return centroided_exps