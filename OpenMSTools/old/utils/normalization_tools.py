from .base_tools import (
    oms,InitProcessAndParamObj,
    oms_exps,
    get_kv_pairs
)
from .async_tools import trio
from .ms_exp_tools import copy_ms_experiments
from typing import Optional, Union

async def Normalizer_filterPeakMap_coroutine(
    normalized_exp:oms.MSExperiment,
    process_obj:oms.Normalizer,
):
    process_obj.filterPeakMap(normalized_exp)
    
async def intensity_normalization_step(
    normalized_exps:oms_exps,
    process_obj:oms.Normalizer,
):
    async with trio.open_nursery() as nursery:
        for _,normalized_exp in get_kv_pairs(normalized_exps):
            nursery.start_soon(Normalizer_filterPeakMap_coroutine, normalized_exp, process_obj)

@InitProcessAndParamObj(
    oms.Normalizer,
)
def intensity_normalization(
    ms_exps: oms_exps,
    process_obj: Optional[oms.Normalizer] = None,
    param_obj: Union[oms.Param, dict, None] = None,
    **kwargs,
) -> oms_exps:
    """
    对质谱实验数据进行强度归一化处理。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含需要进行归一化处理的原始数据。
    - process_obj (Optional[oms.Normalizer]): 可选的归一化处理对象。如果为None，则使用默认的归一化参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - method (str): 归一化方法，默认值为'to_one'，将会单独对每一个谱图进行归一化；可选'to_TIC'，将会以总离子流作为标准。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经强度归一化处理后的质谱实验数据对象。
    """
    normalized_exp = copy_ms_experiments(ms_exps)
    if isinstance(normalized_exp, oms.MSExperiment):
        normalized_exp = [normalized_exp]
    trio.run(intensity_normalization_step, normalized_exp, process_obj)
    if isinstance(ms_exps, oms.MSExperiment):
        normalized_exp = normalized_exp[0]
    return normalized_exp