from .base_tools import (
    oms,InitProcessAndParamObj,
    oms_exps,
    get_kv_pairs
)
from .async_tools import trio
from .ms_exp_tools import copy_ms_experiments
from typing import Optional, Union, Literal

async def SpectraMerger_mergeSpectraBlockWise_coroutine(
    merged_exp:oms.MSExperiment,
    process_obj:oms.SpectraMerger,
):
    process_obj.mergeSpectraBlockWise(merged_exp)
    
async def merge_ms1_by_block_step(
    merged_exps:oms_exps,
    process_obj:oms.SpectraMerger,
):
    async with trio.open_nursery() as nursery:
        for _,merged_exp in get_kv_pairs(merged_exps):
            nursery.start_soon(SpectraMerger_mergeSpectraBlockWise_coroutine, merged_exp, process_obj)

@InitProcessAndParamObj(
    oms.SpectraMerger,
)
def merge_ms1_by_block(
    ms_exps:oms_exps,
    process_obj:Optional[oms.SpectraMerger] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    **kwargs,
) -> oms_exps:
    """
    将质谱实验数据中的 MS1 谱图按块合并。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含待合并的 MS1 谱图数据。
    - process_obj (Optional[oms.SpectraMerger]): 可选的谱图合并处理对象。如果为None，则使用默认的合并参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - mz_binning_width (float): m/z 归类宽度，单位为 ppm（默认值为 5.0）。
        - mz_binning_width_unit (str): m/z 归类宽度单位（默认值为 'ppm'）。
        - sort_blocks (str): 块排序方式，默认值为 'RT_ascending'（保留时间升序）。
        - average_gaussian:spectrum_type (str): 平均高斯谱类型（默认值为 'automatic'）。
        - average_gaussian:ms_level (int): 平均高斯方法适用的质谱级别（默认值为 1）。
        - average_gaussian:rt_FWHM (float): 平均高斯的保留时间全宽半高（默认值为 5.0）。
        - average_gaussian:cutoff (float): 平均高斯方法的截止值（默认值为 0.01）。
        - average_gaussian:precursor_mass_tol (float): 平均高斯的前体质量容差（默认值为 0.0）。
        - average_gaussian:precursor_max_charge (int): 平均高斯方法中前体的最大电荷（默认值为 1）。
        - average_tophat:spectrum_type (str): 平均顶帽谱类型（默认值为 'automatic'）。
        - average_tophat:ms_level (int): 平均顶帽方法适用的质谱级别（默认值为 1）。
        - average_tophat:rt_range (float): 平均顶帽的保留时间范围（默认值为 5.0）。
        - average_tophat:rt_unit (str): 平均顶帽的保留时间单位（默认值为 'scans'）。
        - block_method:ms_levels (list): 块方法适用的质谱级别列表（默认值为 [1]）。
        - block_method:rt_block_size (int): 块方法中的保留时间块大小（默认值为 5）。
        - block_method:rt_max_length (float): 块方法中的最大保留时间长度（默认值为 0.0）。
        - precursor_method:mz_tolerance (float): 前体方法的 m/z 容差（默认值为 0.0001）。
        - precursor_method:mass_tolerance (float): 前体方法的质量容差（默认值为 0.0）。
        - precursor_method:rt_tolerance (float): 前体方法的保留时间容差（默认值为 5.0）。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经块合并处理后的质谱实验数据对象，合并完成的 MS1 谱图将包含在返回的对象中。
    """
    merged_exps = copy_ms_experiments(ms_exps)
    if isinstance(merged_exps, oms.MSExperiment):
        merged_exps = [merged_exps]
    trio.run(merge_ms1_by_block_step, merged_exps, process_obj)
    if isinstance(ms_exps, oms.MSExperiment):
        merged_exps = merged_exps[0]
    return merged_exps

async def SpectraMerger_mergeSpectraPrecursors_coroutine(
    merged_exp:oms.MSExperiment,
    process_obj:oms.SpectraMerger,
):
    process_obj.mergeSpectraPrecursors(merged_exp)
    
async def merge_ms2_by_precursors_step(
    merged_exp:oms_exps,
    process_obj:oms.SpectraMerger,
):
    async with trio.open_nursery() as nursery:
        for _,merged_exp in get_kv_pairs(merged_exp):
            nursery.start_soon(SpectraMerger_mergeSpectraPrecursors_coroutine, merged_exp, process_obj)

@InitProcessAndParamObj(
    oms.SpectraMerger,
)
def merge_ms2_by_precursors(
    ms_exp:oms_exps,
    process_obj:Optional[oms.SpectraMerger] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    **kwargs,
) -> oms_exps:
    '''
    将质谱实验数据中的 MS2 谱图按母离子和保留时间进行合并。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含待合并的 MS1 谱图数据。
    - process_obj (Optional[oms.SpectraMerger]): 可选的谱图合并处理对象。如果为None，则使用默认的合并参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - mz_binning_width (float): m/z 归类宽度，单位为 ppm（默认值为 5.0）。
        - mz_binning_width_unit (str): m/z 归类宽度单位（默认值为 'ppm'）。
        - sort_blocks (str): 块排序方式，默认值为 'RT_ascending'（保留时间升序）。
        - average_gaussian:spectrum_type (str): 平均高斯谱类型（默认值为 'automatic'）。
        - average_gaussian:ms_level (int): 平均高斯方法适用的质谱级别（默认值为 1）。
        - average_gaussian:rt_FWHM (float): 平均高斯的保留时间全宽半高（默认值为 5.0）。
        - average_gaussian:cutoff (float): 平均高斯方法的截止值（默认值为 0.01）。
        - average_gaussian:precursor_mass_tol (float): 平均高斯的前体质量容差（默认值为 0.0）。
        - average_gaussian:precursor_max_charge (int): 平均高斯方法中前体的最大电荷（默认值为 1）。
        - average_tophat:spectrum_type (str): 平均顶帽谱类型（默认值为 'automatic'）。
        - average_tophat:ms_level (int): 平均顶帽方法适用的质谱级别（默认值为 1）。
        - average_tophat:rt_range (float): 平均顶帽的保留时间范围（默认值为 5.0）。
        - average_tophat:rt_unit (str): 平均顶帽的保留时间单位（默认值为 'scans'）。
        - block_method:ms_levels (list): 块方法适用的质谱级别列表（默认值为 [1]）。
        - block_method:rt_block_size (int): 块方法中的保留时间块大小（默认值为 5）。
        - block_method:rt_max_length (float): 块方法中的最大保留时间长度（默认值为 0.0）。
        - precursor_method:mz_tolerance (float): 前体方法的 m/z 容差（默认值为 0.0001）。
        - precursor_method:mass_tolerance (float): 前体方法的质量容差（默认值为 0.0）。
        - precursor_method:rt_tolerance (float): 前体方法的保留时间容差（默认值为 5.0）。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经合并处理后的质谱实验数据对象，合并完成的 MS2 谱图将包含在返回的对象中。
    '''
    merged_exp = copy_ms_experiments(ms_exp)
    if isinstance(merged_exp, oms.MSExperiment):
        merged_exp = [merged_exp]
    trio.run(merge_ms2_by_precursors_step, merged_exp, process_obj)
    if isinstance(ms_exp, oms.MSExperiment):
        merged_exp = merged_exp[0]
    return merged_exp

async def SpectraMerger_average_coroutine(
    averaged_exp:oms.MSExperiment,
    process_obj:oms.SpectraMerger,
    average_method:Literal['gaussian', 'tophat']
):
    process_obj.average(averaged_exp, average_method)
    
async def spec_averaging_step(
    averaged_exps:oms_exps,
    process_obj:oms.SpectraMerger,
    average_method:Literal['gaussian', 'tophat']
):
    async with trio.open_nursery() as nursery:
        for _,averaged_exp in get_kv_pairs(averaged_exps):
            nursery.start_soon(SpectraMerger_average_coroutine, averaged_exp, process_obj, average_method)

@InitProcessAndParamObj(
    oms.SpectraMerger,
)
def spec_averaging(
    ms_exps:oms_exps,
    process_obj:Optional[oms.SpectraMerger] = None,
    param_obj:Union[oms.Param,dict,None] = None,
    average_method:Literal['gaussian', 'tophat'] = 'gaussian',
    **kwargs,
) -> oms_exps:
    '''
    对给定光谱的相邻光谱的峰值强度进行平均。平均后的光谱数量不会改变，因为它是针对每个单独的输入光谱进行的。

    参数：
    - ms_exp (oms.MSExperiment): 输入的质谱实验数据对象，包含待合并的 MS1 谱图数据。
    - process_obj (Optional[oms.SpectraMerger]): 可选的谱图合并处理对象。如果为None，则使用默认的合并参数。
    - param_obj (Union[oms.Param, dict, None]): 可选的参数对象或字典，包含以下默认参数：
        - mz_binning_width (float): m/z 归类宽度，单位为 ppm（默认值为 5.0）。
        - mz_binning_width_unit (str): m/z 归类宽度单位（默认值为 'ppm'）。
        - sort_blocks (str): 块排序方式，默认值为 'RT_ascending'（保留时间升序）。
        - average_gaussian:spectrum_type (str): 平均高斯谱类型（默认值为 'automatic'）。
        - average_gaussian:ms_level (int): 平均高斯方法适用的质谱级别（默认值为 1）。
        - average_gaussian:rt_FWHM (float): 平均高斯的保留时间全宽半高（默认值为 5.0）。
        - average_gaussian:cutoff (float): 平均高斯方法的截止值（默认值为 0.01）。
        - average_gaussian:precursor_mass_tol (float): 平均高斯的前体质量容差（默认值为 0.0）。
        - average_gaussian:precursor_max_charge (int): 平均高斯方法中前体的最大电荷（默认值为 1）。
        - average_tophat:spectrum_type (str): 平均顶帽谱类型（默认值为 'automatic'）。
        - average_tophat:ms_level (int): 平均顶帽方法适用的质谱级别（默认值为 1）。
        - average_tophat:rt_range (float): 平均顶帽的保留时间范围（默认值为 5.0）。
        - average_tophat:rt_unit (str): 平均顶帽的保留时间单位（默认值为 'scans'）。
        - block_method:ms_levels (list): 块方法适用的质谱级别列表（默认值为 [1]）。
        - block_method:rt_block_size (int): 块方法中的保留时间块大小（默认值为 5）。
        - block_method:rt_max_length (float): 块方法中的最大保留时间长度（默认值为 0.0）。
        - precursor_method:mz_tolerance (float): 前体方法的 m/z 容差（默认值为 0.0001）。
        - precursor_method:mass_tolerance (float): 前体方法的质量容差（默认值为 0.0）。
        - precursor_method:rt_tolerance (float): 前体方法的保留时间容差（默认值为 5.0）。
    - average_method (Literal['gaussian', 'tophat']): 选择使用的平均方法，可选 'gaussian' 或 'tophat'。
    - **kwargs: 其他可选参数。

    返回：
    oms.MSExperiment: 经平均处理后的质谱实验数据对象。
    '''
    averaged_exps = copy_ms_experiments(ms_exps)
    if isinstance(averaged_exps, oms.MSExperiment):
        averaged_exps = [averaged_exps]
    trio.run(spec_averaging_step, averaged_exps, process_obj, average_method)
    if isinstance(ms_exps, oms.MSExperiment):
        averaged_exps = averaged_exps[0]
    return averaged_exps