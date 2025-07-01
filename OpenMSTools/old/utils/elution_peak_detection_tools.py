from typing import Dict, List, Optional, Union

from .async_tools import trio
from .base_tools import InitProcessAndParamObj, get_kv_pairs, oms


async def elution_peak_detection_coroutine(
    mass_traces:list,
    mass_traces_deconvol:list,
    process_obj:oms.ElutionPeakDetection,
):
    process_obj.detectPeaks(mass_traces,mass_traces_deconvol)

async def elution_peak_detection_step(
    mass_traces:Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]],
    mass_traces_deconvol:Union[list,List[list],Dict[str,list]],
    process_obj:oms.ElutionPeakDetection,
):
    async with trio.open_nursery() as nursery:
        for key,mass_trace in get_kv_pairs(mass_traces):
            nursery.start_soon(elution_peak_detection_coroutine, mass_trace, mass_traces_deconvol[key], process_obj)

@InitProcessAndParamObj(
    oms.ElutionPeakDetection,
    defualt_params={
        'width_filtering': 'fixed',
        'max_fwhm': 15.0,
    }
)
def elution_peak_detection(
    mass_traces:Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]],
    process_obj:Optional[oms.ElutionPeakDetection] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    **kwargs,
) -> Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]]:
    """
    检测质谱数据中的洗脱峰。

    参数：
    - mass_traces (List[oms.Kernel_MassTrace]): 输入的质量踪迹列表，包含待检测的洗脱峰数据。
    - process_obj (Optional[oms.ElutionPeakDetection]): 可选的洗脱峰检测处理对象。如果为None，则使用默认的检测参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - chrom_fwhm (float): 色谱峰全宽半高（FWHM），用于峰检测（默认值为 5.0）。
        - chrom_peak_snr (float): 色谱峰的信噪比，用于筛选（默认值为 3.0）。
        - width_filtering (str): 峰宽过滤方法，支持 'fixed'（默认值为 'fixed'）。
        - min_fwhm (float): 峰的最小全宽半高（默认值为 1.0）。
        - max_fwhm (float): 峰的最大全宽半高（默认值为 60.0）。
        - masstrace_snr_filtering (str): 是否对质量踪迹进行信噪比过滤，取值为 'true' 或 'false'（默认值为 'false'）。
    - **kwargs: 其他可选参数。

    返回：
    List[oms.Kernel_MassTrace]: 检测到的洗脱峰列表，包含在返回结果中的所有洗脱峰对象。
    """
    if isinstance(mass_traces[0] if isinstance(mass_traces, list) else None, oms.Kernel_MassTrace):
        input_traces = [mass_traces]
        sigle = True
    else:
        input_traces = mass_traces
        sigle = False
    if isinstance(input_traces, dict):
        mass_traces_deconvol = {k:[] for k in input_traces.keys()}
    else:
        mass_traces_deconvol = [[] for i in range(len(input_traces))]
    trio.run(elution_peak_detection_step, input_traces, mass_traces_deconvol, process_obj)
    if sigle:
        mass_traces_deconvol = mass_traces_deconvol[0]
    return mass_traces_deconvol
