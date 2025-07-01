from typing import Dict, List, Optional, Tuple, Union

from .async_tools import trio
from .base_tools import InitProcessAndParamObj, get_kv_pairs, oms


async def feature_finding_metabo_coroutine(
    mass_traces_deconvol:List[oms.Kernel_MassTrace],
    feature_map:oms.FeatureMap,
    chromatograms:list,
    process_obj:oms.FeatureFindingMetabo,
):
    process_obj.run(mass_traces_deconvol, feature_map, chromatograms)
    feature_map.setUniqueIds()

async def feature_finding_metabo_step(
    mass_traces_deconvol:Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]],
    feature_maps:Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    chromatograms:Union[List[list],Dict[str,list]],
    process_obj:oms.FeatureFindingMetabo,
):
    async with trio.open_nursery() as nursery:
        for key,mass_trace_deconvol in get_kv_pairs(mass_traces_deconvol):
            nursery.start_soon(feature_finding_metabo_coroutine, mass_trace_deconvol, feature_maps[key], chromatograms[key], process_obj)

@InitProcessAndParamObj(
    oms.FeatureFindingMetabo,
    defualt_params={
        'remove_single_traces':'true',
        'report_chromatograms':'true',
        'isotope_filtering_model':'none',
        'local_rt_range':10.0,
        'report_convex_hulls':'true',
        'charge_lower_bound':1,
        'charge_upper_bound':1,
    },
)
def mapping_features(
    mass_traces_deconvol:Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]],
    process_obj:Optional[oms.FeatureFindingMetabo] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    **kwargs,
) -> Tuple[
    Union[oms.FeatureMap,List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    Union[List[oms.ChromatogramPeak],List[List[oms.ChromatogramPeak]],Dict[str,List[oms.ChromatogramPeak]]],
]:
    """
    检测质谱数据中的洗脱峰。

    参数：
    - mass_traces (List[oms.Kernel_MassTrace]): 输入的质量踪迹列表，包含待检测的洗脱峰数据。
    - process_obj (Optional[oms.ElutionPeakDetection]): 可选的洗脱峰检测处理对象。如果为None，则使用默认的检测参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - local_rt_range (float): 局部保留时间范围，用于峰检测（默认值为 10.0）。
        - local_mz_range (float): 局部 m/z 范围，用于峰检测（默认值为 6.5）。
        - charge_lower_bound (int): 允许的最低电荷状态（默认值为 1）。
        - charge_upper_bound (int): 允许的最高电荷状态（默认值为 1）。
        - chrom_fwhm (float): 色谱峰全宽半高（FWHM），用于峰检测（默认值为 5.0）。
        - report_summed_ints (str): 是否报告累加强度，取值为 'true' 或 'false'（默认值为 'false'）。
        - enable_RT_filtering (str): 是否启用保留时间过滤，取值为 'true' 或 'false'（默认值为 'true'）。
        - isotope_filtering_model (str): 同位素过滤模型（默认值为 'none'）。
        - mz_scoring_13C (str): 是否对 13C 进行 m/z 评分，取值为 'true' 或 'false'（默认值为 'false'）。
        - use_smoothed_intensities (str): 是否使用平滑强度，取值为 'true' 或 'false'（默认值为 'true'）。
        - report_convex_hulls (str): 是否报告凸包，取值为 'true' 或 'false'（默认值为 'true'）。
        - report_chromatograms (str): 是否报告色谱图，取值为 'true' 或 'false'（默认值为 'true'）。
        - remove_single_traces (str): 是否移除单一质量踪迹，取值为 'true' 或 'false'（默认值为 'true'）。
        - mz_scoring_by_elements (str): 是否根据元素进行 m/z 评分，取值为 'true' 或 'false'（默认值为 'false'）。
        - elements (str): 要评分的元素，默认值为 'CHNOPS'。
    - **kwargs: 其他可选参数。

    返回：
    - oms.FeatureMap: 质量特征映射表。
    - List[oms.ChromatogramPeak]: 检测到的洗脱峰列表，包含在返回结果中的所有洗脱峰对象。
    """
    if isinstance(mass_traces_deconvol[0] if isinstance(mass_traces_deconvol, list) else None, oms.Kernel_MassTrace):
        input_traces = [mass_traces_deconvol]
        sigle = True
    else:
        input_traces = mass_traces_deconvol
        sigle = False
    if isinstance(input_traces, dict):
        feature_maps = {k:oms.FeatureMap() for k in input_traces.keys()}
        chromatograms = {k:[] for k in input_traces.keys()}
    else:
        feature_maps = [oms.FeatureMap() for i in range(len(input_traces))]
        chromatograms = [[] for i in range(len(input_traces))]
    trio.run(feature_finding_metabo_step, input_traces, feature_maps, chromatograms, process_obj)
    if sigle:
        feature_maps = feature_maps[0]
        chromatograms = chromatograms[0]
    return feature_maps, chromatograms

def set_primary_ms_run_path(
    feature_maps:List[oms.FeatureMap],
    file_names:List[str],
) -> None:
    for feature_map,file_name in zip(feature_maps,file_names):
        feature_map.setPrimaryMSRunPath(
            [file_name.encode()]
        )
