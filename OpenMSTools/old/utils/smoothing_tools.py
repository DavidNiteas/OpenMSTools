from .base_tools import (
    oms,InitProcessAndParamObj,
    oms_exps,
    get_kv_pairs
)
from .async_tools import trio
from .ms_exp_tools import copy_ms_experiments
from typing import Union, Optional

async def GaussFilter_filterExperiment_coroutine(
    smoothed_exp:oms.MSExperiment,
    process_obj:oms.GaussFilter,
):
    process_obj.filterExperiment(smoothed_exp)
    
async def mz_smooting_step(
    smoothed_exps:oms_exps,
    process_obj:oms.GaussFilter,
):
    async with trio.open_nursery() as nursery:
        for _,smoothed_exp in get_kv_pairs(smoothed_exps):
            nursery.start_soon(GaussFilter_filterExperiment_coroutine, smoothed_exp, process_obj)

@InitProcessAndParamObj(
    oms.GaussFilter
)
def mz_smoothing(
    ms_exps: oms_exps,
    process_obj: Optional[oms.GaussFilter] = None,
    param_obj: Union[oms.Param, dict, None] = None,
    **kwargs,
) -> oms_exps:
    """
    对质谱实验数据进行高斯平滑处理。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含需要平滑处理的原始数据。
    - process_obj (Optional[oms.GaussFilter]): 可选的高斯滤波处理对象。如果为None，则使用默认的高斯滤波参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - gaussian_width (float): 高斯滤波器的宽度，默认值为0.2。
        - ppm_tolerance (float): 允许的PPM容差，默认值为10.0。
        - use_ppm_tolerance (str): 是否使用PPM容差，默认为'false'。
        - write_log_messages (str): 是否记录日志信息，默认为'false'。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经平滑处理后的质谱实验数据对象。
    """
    smoothed_exps = copy_ms_experiments(ms_exps)
    if isinstance(smoothed_exps, oms.MSExperiment):
        smoothed_exps = [smoothed_exps]
    trio.run(mz_smooting_step, smoothed_exps, process_obj)
    if isinstance(ms_exps, oms.MSExperiment):
        smoothed_exps = smoothed_exps[0]
    return smoothed_exps

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