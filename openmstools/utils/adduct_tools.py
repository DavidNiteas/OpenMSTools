from .base_tools import (
    oms,InitProcessAndParamObj,
    get_kv_pairs
)
from .async_tools import trio
from typing import Optional, Union, List, Dict, Tuple

async def detect_adduct_coroutine(
    feature_map_in:oms.FeatureMap,
    feature_map_out:oms.FeatureMap,
    consensus_group:oms.ConsensusFeature,
    consensus_edge:oms.ConsensusFeature,
    process_obj:oms.MetaboliteFeatureDeconvolution,
):
    process_obj.compute(feature_map_in, feature_map_out, consensus_group, consensus_edge)
    
async def detect_adduct_step(
    feature_maps_in:Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    feature_maps_out:Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
    consensus_groups:Union[List[oms.ConsensusFeature],Dict[str,List[oms.ConsensusFeature]]],
    consensus_edges:Union[List[oms.ConsensusFeature],Dict[str,List[oms.ConsensusFeature]]],
    process_obj:oms.MetaboliteFeatureDeconvolution,
):
    async with trio.open_nursery() as nursery:
        for key,feature_map_in in get_kv_pairs(feature_maps_in):
            nursery.start_soon(
                detect_adduct_coroutine, 
                feature_map_in, feature_maps_out[key], 
                consensus_groups[key], consensus_edges[key], 
                process_obj,
            )
            
@InitProcessAndParamObj(
    oms.MetaboliteFeatureDeconvolution,
    defualt_params={
        "potential_adducts":[
            "H:+:0.4",
            "Na:+:0.1",
            "K:+:0.1",
            "NH4:+:0.2",
            "H-1O-1:+:0.1",
            "H-3O-2:+:0.1",
        ],
        'charge_min':1,
        'charge_max':1,
        'charge_span_max':1,
        'retention_max_diff':3.0,
        'retention_max_diff_local':3.0,
    },
)
def adduct_detection(
    feature_maps: Union[oms.FeatureMap,List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
    process_obj: Optional[oms.MapAlignmentAlgorithmPoseClustering] = None,
    param_obj: Union[oms.Param, dict, None] = None,
    **kwargs,
) -> Tuple[
    Union[List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
    Union[List[oms.ConsensusFeature], Dict[str, List[oms.ConsensusFeature]]],
    Union[List[oms.ConsensusFeature], Dict[str, List[oms.ConsensusFeature]]],
]:
    """
    检测并识别代谢物的加合物特征。

    本函数将在输入的特征图上运行加合物检测步骤，并返回处理结果，包括特征图、共识特征和边缘共识特征。

    参数:
    - feature_maps: 输入的特征图，可以是单个特征图、特征图列表或特征图字典。
    - process_obj: 处理对象，用于映射对齐算法，默认为None。
    - param_obj: 参数对象或字典，包含检测加合物所需的参数设置，默认为None。默认参数如下：
        - potential_adducts: 可能的加合物及其相对强度，以<加合物>:<电荷>:<出现概率>的形式输入（概率和应为1）
            - 默认的可能加合物列表为：
                - H:+:0.4
                - Na:+:0.1
                - K:+:0.1
                - NH4:+:0.2
                - H-1O-1:+:0.1
                - H-3O-2:+:0.1
        - charge_min: 最小电荷数，默认为1
        - charge_max: 最大电荷数，默认为1
        - charge_span_max: 最大电荷范围，默认为1
        - retention_max_diff: 最大保留时间差，默认为3.0
        - retention_max_diff_local: 局部最大保留时间差，默认为3.0
        - mass_max_diff: 最大质量差，默认为0.05
        - unit: 质量单位，默认为' Da'
        - max_neutrals: 最大中性物质数量，默认为1
        - use_minority_bound: 是否使用少数界限，默认为'true'
        - max_minority_bound: 最大少数界限，默认为3
        - min_rt_overlap: 最小保留时间重叠比例，默认为0.66
        - intensity_filter: 是否应用强度过滤，默认为'false'
        - negative_mode: 是否为负模式，默认为'false'
        - default_map_label: 默认映射标签，默认为'decharged features'
        - verbose_level: 输出详细程度，默认为0
    - kwargs: 其他可选参数。

    返回:
    - output_maps: 处理后的特征图，可以是列表或字典形式。
    - groups: 共识特征的列表或字典。
    - edges: 边缘共识特征的列表或字典。
    """
    if isinstance(feature_maps[0] if isinstance(feature_maps, list) else None, oms.FeatureMap):
        input_maps = [feature_maps]
        sigle = True
    else:
        input_maps = feature_maps
        sigle = False
    if isinstance(input_maps, dict):
        output_maps = {k:oms.FeatureMap() for k in input_maps.keys()}
        groups = {k:oms.ConsensusMap() for k in input_maps.keys()}
        edges = {k:oms.ConsensusMap() for k in input_maps.keys()}
    else:
        output_maps = [oms.FeatureMap() for i in range(len(input_maps))]
        groups = [oms.ConsensusMap() for i in range(len(input_maps))]
        edges = [oms.ConsensusMap() for i in range(len(input_maps))]
    trio.run(detect_adduct_step, input_maps, output_maps, groups, edges, process_obj)
    if sigle:
        output_maps = output_maps[0]
        groups = groups[0]
        edges = edges[0]
    return output_maps, groups, edges