from .base_tools import (
    oms,InitProcessAndParamObj,
    get_kv_pairs
)
from .async_tools import trio
from .ms_exp_tools import copy_ms_experiments
from typing import Optional, Union, List, Dict, Tuple

async def align_feature_map_coroutine(
    feature_map:oms.FeatureMap,
    trafo:oms.TransformationDescription,
    process_obj:oms.MapAlignmentAlgorithmPoseClustering,
    key:str,
):
    try:
        process_obj.align(feature_map, trafo)
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)
    except RuntimeError as e:
        print(f"Alignment failed for feature map {key}: {e}")
        raise e
        
async def align_feature_map_step(
    feature_maps:Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    trafos:Union[List[oms.TransformationDescription],Dict[str,oms.TransformationDescription]],
    process_obj:oms.MapAlignmentAlgorithmPoseClustering,
):
    async with trio.open_nursery() as nursery:
        for key,feature_map in get_kv_pairs(feature_maps):
            if trafos[key] is not None:
                nursery.start_soon(align_feature_map_coroutine, feature_map, trafos[key], process_obj, key)
                
def align_feature_maps(
    feature_maps:Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    ref_index:Union[int,str,None],
    process_obj:oms.MapAlignmentAlgorithmPoseClustering,
) -> Tuple[
    Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    Union[List[oms.TransformationDescription],Dict[str,oms.TransformationDescription]]
]:
    trafos = {}
    for key,feature_map in get_kv_pairs(feature_maps):
        if key != ref_index:
            trafos[key] = oms.TransformationDescription()
        else:
            trafos[key] = None
    trio.run(align_feature_map_step, feature_maps, trafos, process_obj)
    if isinstance(feature_maps, list):
        trafos = list(trafos.values())
    return feature_maps, trafos

async def align_ms_exp_coroutine(
    ms_exp:oms.MSExperiment,
    trafo:oms.TransformationDescription,
    key:str,
):
    try:
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(ms_exp, trafo, True)
    except RuntimeError as e:
        print(f"Alignment failed for MS experiment {key}: {e}")
        raise e
    
async def align_ms_exp_step(
    ms_exps:Union[List[oms.MSExperiment],Dict[str,List[oms.MSExperiment]]],
    trafos:Union[List[oms.TransformationDescription],Dict[str,List[oms.TransformationDescription]]],
):
    async with trio.open_nursery() as nursery:
        for key,ms_exp in get_kv_pairs(ms_exps):
            if trafos[key] is not None:
                nursery.start_soon(align_ms_exp_coroutine, ms_exp, trafos[key], key)
                
def align_ms_exps(
    ms_exps:Union[List[oms.MSExperiment],Dict[str,List[oms.MSExperiment]]],
    trafos:Union[List[oms.TransformationDescription],Dict[str,List[oms.TransformationDescription]]],
) -> Union[List[oms.MSExperiment],Dict[str,List[oms.MSExperiment]]]:
    aligned_exps = copy_ms_experiments(ms_exps)
    trio.run(align_ms_exp_step, aligned_exps, trafos)
    return aligned_exps
        
@InitProcessAndParamObj(
    oms.MapAlignmentAlgorithmPoseClustering,
    defualt_params={
        'max_num_peaks_considered':-1,
        'pairfinder:distance_MZ:max_difference':10.0,
        'pairfinder:distance_MZ:unit':'ppm',
        'pairfinder:ignore_charge':'true',
    },
)
def align(
    ms_exps: Union[List[oms.MSExperiment], Dict[str, List[oms.MSExperiment]]],
    feature_maps: Union[List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
    ref_map: Union[oms.FeatureMap, int, str],
    process_obj: Optional[oms.MapAlignmentAlgorithmPoseClustering] = None,
    param_obj: Union[oms.Param, dict, None] = None,
    **kwargs,
) -> Tuple[
    Union[List[oms.MSExperiment], Dict[str, List[oms.MSExperiment]]],
    Union[List[oms.TransformationDescription], Dict[str, List[oms.TransformationDescription]]],
]:
    """
    对多个质谱实验进行对齐。

    该函数使用指定的特征图和参考特征图或其索引，利用 MapAlignmentAlgorithmPoseClustering 对象对一系列质谱实验进行对齐。

    参数：
    - ms_exps (Union[List[oms.MSExperiment], Dict[str, List[oms.MSExperiment]]]): 需要对齐的质谱实验列表或字典。
    - feature_maps (Union[List[oms.FeatureMap], Dict[str, oms.FeatureMap]]): 对应的特征图列表或字典。
    - ref_map (Union[oms.FeatureMap, int, str]): 参考特征图或其索引，所有其他特征图将与此图对齐。
    - process_obj (Optional[oms.MapAlignmentAlgorithmPoseClustering], optional): 可选的处理对象，默认为 None。
    - param_obj (Union[oms.Param, dict, None], optional): 可选的参数对象或字典，默认为 None。
        - max_num_peaks_considered (int): 考虑的最大峰值数量，设为 -1 表示无限制。
        - superimposer:mz_pair_max_distance (float): 质谱对齐时，最大质荷比（m/z）对的距离（默认值为 0.5）。
        - superimposer:rt_pair_distance_fraction (float): 反应时间（RT）对距离的分数，决定对齐的容许范围（默认值为 0.1）。
        - superimposer:num_used_points (int): 用于对齐计算的点的数量（默认值为 2000）。
        - superimposer:scaling_bucket_size (float): 缩放桶的大小，管理缩放参数（默认值为 0.005）。
        - superimposer:shift_bucket_size (float): 位移桶的大小，用于校正位移（默认值为 3.0）。
        - superimposer:max_shift (float): 允许的最大位移（默认值为 1000.0）。
        - superimposer:max_scaling (float): 允许的最大缩放比例（默认值为 2.0）。
        - superimposer:dump_buckets (str): 用于调试的桶转储选项（默认值为空字符串）。
        - superimposer:dump_pairs (str): 用于调试的对转储选项（默认值为空字符串）。
        - pairfinder:second_nearest_gap (float): 第二近邻之间的距离差距（默认值为 2.0）。
        - pairfinder:use_identifications (str): 是否使用鉴定信息，取值为 'true' 或 'false'（默认值为 'false'）。
        - pairfinder:ignore_charge (str): 是否忽略电荷信息，取值为 'true' 或 'false'（默认值为 'true'）。
        - pairfinder:ignore_adduct (str): 是否忽略附加物，取值为 'true' 或 'false'（默认值为 'true'）。
        - pairfinder:distance_RT:max_difference (float): 反应时间（RT）最大差异（默认值为 100.0）。
        - pairfinder:distance_RT:exponent (float): 反应时间距离的指数权重（默认值为 1.0）。
        - pairfinder:distance_RT:weight (float): 反应时间距离的权重（默认值为 1.0）。
        - pairfinder:distance_MZ:max_difference (float): 质荷比（m/z）最大差异（默认值为 10.0）。
        - pairfinder:distance_MZ:unit (str): 质荷比单位，通常为 'ppm'（默认值为 'ppm'）。
        - pairfinder:distance_MZ:exponent (float): 质荷比距离的指数权重（默认值为 2.0）。
        - pairfinder:distance_MZ:weight (float): 质荷比距离的权重（默认值为 1.0）。
        - pairfinder:distance_intensity:exponent (float): 强度距离的指数权重（默认值为 1.0）。
        - pairfinder:distance_intensity:weight (float): 强度距离的权重（默认值为 0.0）。
        - pairfinder:distance_intensity:log_transform (str): 强度距离的对数变换选项，取值为 'enabled' 或 'disabled'（默认值为 'disabled'）。
    - **kwargs: 其他可选参数。

    返回：
    Tuple[Union[List[oms.MSExperiment], Dict[str, List[oms.MSExperiment]]], Union[List[oms.TransformationDescription], Dict[str, List[oms.TransformationDescription]]]]:
        - 对齐后的质谱实验列表或字典。
        - 每个特征图的变换描述列表或字典，参考特征图的变换描述为 None。
    """
    if isinstance(ref_map, oms.FeatureMap):
        process_obj.setReference(ref_map)
        ref_index = None
    else:
        process_obj.setReference(feature_maps[ref_map])
        ref_index = ref_map
    feature_maps,trafos = align_feature_maps(feature_maps, ref_index, process_obj)
    aligned_exps = align_ms_exps(ms_exps, trafos)
    return aligned_exps, trafos