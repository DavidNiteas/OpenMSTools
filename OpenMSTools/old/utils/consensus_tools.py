from .base_tools import (
    oms,InitProcessAndParamObj,
    get_kv_pairs
)
from typing import Optional, Union, List, Dict, Tuple

@InitProcessAndParamObj(
    oms.FeatureGroupingAlgorithmQT,
    defualt_params={
        'ignore_charge':'true'
    }
)
def link_features(
    feature_maps: Union[List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
    process_obj: Optional[oms.FeatureGroupingAlgorithmQT] = None,
    param_obj: Union[oms.Param, dict, None] = None,
    **kwargs,
) -> oms.ConsensusMap:
    """
    将特征图链接到一个一致性图，并对特征进行分组。

    参数:
    - feature_maps: 特征图的列表或字典，包含要链接的特征。
    - process_obj: 可选的特征分组算法对象，默认为None。如果提供，则使用该对象进行特征分组。
    - param_obj: 可选的参数对象，默认为None。如果提供，则根据该参数进行特征分组的配置。
        - use_identifications: 是否使用标识，默认为 'false'
        - nr_partitions: 分区的数量，默认为 100
        - min_nr_diffs_per_bin: 每个箱子中的最小差异数量，默认为 50
        - min_IDscore_forTolCalc: 容忍计算的最小ID分数，默认为 1.0
        - noID_penalty: 无ID惩罚，默认为 0.0
        - ignore_charge: 是否忽略电荷，默认为 'false'
        - ignore_adduct: 是否忽略附加物，默认为 'true'
        - distance_RT:max_difference: RT距离的最大差异，默认为 100.0
        - distance_RT:exponent: RT距离的指数，默认为 1.0
        - distance_RT:weight: RT距离的权重，默认为 1.0
        - distance_MZ:max_difference: MZ距离的最大差异，默认为 0.3
        - distance_MZ:unit: MZ距离的单位，默认为 'Da'
        - distance_MZ:exponent: MZ距离的指数，默认为 2.0
        - distance_MZ:weight: MZ距离的权重，默认为 1.0
        - distance_intensity:exponent: 强度距离的指数，默认为 1.0
        - distance_intensity:weight: 强度距离的权重，默认为 0.0
        - distance_intensity:log_transform: 强度距离的对数变换，默认为 'disabled'
    - **kwargs: 其他可选关键字参数。

    返回:
    - oms.ConsensusMap: 返回一个包含链接特征和分组的共识图对象。
    """
    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()
    i = 0
    feature_map_list = []
    for key, feature_map in get_kv_pairs(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_description.filename = str(key)
        file_description.size = feature_map.size()
        file_description.unique_id = feature_map.getUniqueId()
        file_descriptions[i] = file_description
        feature_map_list.append(feature_map)
        i += 1
    consensus_map.setColumnHeaders(file_descriptions)
    process_obj.group(feature_map_list, consensus_map)
    return consensus_map