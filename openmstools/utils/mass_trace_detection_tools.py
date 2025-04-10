from .base_tools import (
    oms,InitProcessAndParamObj,
    oms_exps,
    get_kv_pairs
)
from .async_tools import trio
from typing import Optional, Union, List, Dict

async def mass_trace_detection_coroutine(
    ms_exp:oms.MSExperiment,
    mass_traces:list,
    process_obj:oms.MassTraceDetection,
    max_traces:int = 0,
):
    process_obj.run(ms_exp, mass_traces, max_traces)
    # process_obj.run(ms_exp, mass_traces)
    
async def mass_trace_detection_step(
    ms_exps:oms_exps,
    mass_traces:Union[List[list],Dict[str,list]],
    process_obj:oms.MassTraceDetection,
    max_traces:int = 0,
):
    async with trio.open_nursery() as nursery:
        for key,ms_exp in get_kv_pairs(ms_exps):
            nursery.start_soon(mass_trace_detection_coroutine, ms_exp, mass_traces[key], process_obj, max_traces)

@InitProcessAndParamObj(
    oms.MassTraceDetection,
    defualt_params={
        'mass_error_ppm': 3.0,
        "noise_threshold_int": 1.0e04
    },
)
def mass_traces_detection(
    ms_exps:oms_exps,
    process_obj:Optional[oms.MassTraceDetection] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    max_traces:int = 0,
    **kwargs,
) -> Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]]:
    """
    检测质谱实验数据中的质量踪迹。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含待检测的质量踪迹数据。
    - process_obj (Optional[oms.MassTraceDetection]): 可选的质量踪迹检测处理对象。如果为None，则使用默认的检测参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - mass_error_ppm (float): 质量误差，单位为 ppm（默认值为 3.0）。
        - noise_threshold_int (float): 噪声阈值，以强度表示（默认值为 1e4）。
        - chrom_peak_snr (float): 色谱峰信噪比（默认值为 3.0）。
        - reestimate_mt_sd (str): 是否重新估计质量踪迹的标准差，取值为 'true' 或 'false'（默认值为 'true'）。
        - quant_method (str): 量化方法，支持 'area'（默认值为 'area'）。
        - trace_termination_criterion (str): 质量踪迹终止标准，支持 'outlier'（默认值为 'outlier'）。
        - trace_termination_outliers (int): 终止质量踪迹时允许的最大离群点数量（默认值为 5）。
        - min_sample_rate (float): 最小样本率，用于质量踪迹检测（默认值为 0.5）。
        - min_trace_length (float): 检测到的质量踪迹的最小长度（默认值为 5.0）。
        - max_trace_length (float): 检测到的质量踪迹的最大长度（若为 -1.0，则不限制）（默认值为 -1.0）。
    - max_traces (int): 最大可检测的质量踪迹数量（默认值为 0，表示不限制）。
    - **kwargs: 其他可选参数。

    返回：
    List[oms.Kernel_MassTrace]: 检测到的质量踪迹列表，包含在返回结果中的所有质量踪迹对象。
    """
    if isinstance(ms_exps, oms.MSExperiment):
        input_exps = [ms_exps]
    else:
        input_exps = ms_exps
    if isinstance(input_exps, dict):
        mass_traces = {k:[] for k in input_exps.keys()}
    else:
        mass_traces = [[] for i in range(len(input_exps))]
    trio.run(mass_trace_detection_step, input_exps, mass_traces, process_obj, max_traces)
    if isinstance(ms_exps, oms.MSExperiment):
        mass_traces = mass_traces[0]
    return mass_traces