from __future__ import annotations
import pyopenms as oms
import numpy as np
import pandas as pd
import hashlib
import trio
from rich.console import Console
from rich.progress import Progress, ProgressColumn, GetTimeCallable, TaskID, track
from typing import List,Tuple,Dict,Union,Optional,Callable,Any,Literal,Hashable

oms_exps = Union[oms.MSExperiment,List[oms.MSExperiment],Dict[str,oms.MSExperiment]]

oms_process_inputs = Union[
    oms_exps,
    Union[List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]]],
    Union[List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
]
oms_process_obj = Union[
    oms.GaussFilter,oms.PeakPickerHiRes,oms.Normalizer,oms.SpectraMerger,
    oms.MassTraceDetection,oms.ElutionPeakDetection,oms.FeatureFindingMetabo,
    oms.MapAlignmentAlgorithmPoseClustering,
]
oms_process_outputs = Union[
    oms_exps,oms.ConsensusMap,
    List[oms.Kernel_MassTrace],List[List[oms.Kernel_MassTrace]],Dict[str,List[oms.Kernel_MassTrace]],
    Tuple[
        Union[oms.FeatureMap,List[oms.FeatureMap],Dict[str,oms.FeatureMap]],
        Union[List[oms.ChromatogramPeak],List[List[oms.ChromatogramPeak]],Dict[str,List[oms.ChromatogramPeak]]],
    ],
    Tuple[
        Union[List[oms.MSExperiment], Dict[str, List[oms.MSExperiment]]],
        Union[List[oms.TransformationDescription], Dict[str, List[oms.TransformationDescription]]],
    ],
    Tuple[
        Union[List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
        Union[List[oms.ConsensusFeature], Dict[str, List[oms.ConsensusFeature]]],
        Union[List[oms.ConsensusFeature], Dict[str, List[oms.ConsensusFeature]]],
    ]
]

def get_kv_pairs(item:Union[list,dict],use_progress:bool=False,**kwargs):
    if isinstance(item,list):
        if use_progress:
            return enumerate(track(item,**kwargs))
        else:
            return enumerate(item)
    else:
        if use_progress:
            return track(item.items(),**kwargs)
        else:
            return item.items()
        
class ProgressManager(Progress):
    
    def __init__(
        self,
        *columns: Union[str, ProgressColumn],
        console: Optional[Console] = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30.0,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: Optional[GetTimeCallable] = None,
        disable: bool = False,
        expand: bool = False,
    ):
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )
        self.task_name_id_map = {}
        
    @property
    def Name2ID(self) -> Dict[Hashable,TaskID]:
        return self.task_name_id_map

    def add_task(
        self,
        task_name: Hashable,
        description: Optional[str] = None,
        start: bool = True,
        total: Optional[float] = None,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> None:
        if description is None:
            description = f'{str(task_name)}:'
        task_id = super().add_task(
            description,
            start=start,
            total=total,
            completed=completed,
            visible=visible,
            **fields,
        )
        self.Name2ID[task_name] = task_id
        
    def update(
        self,
        task_name: Hashable,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if task_name not in self.Name2ID:
            self.add_task(task_name)
        task_id = self.Name2ID[task_name]
        super().update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )
        
    def update_total(
        self,
        task_name: Hashable,
        advance: float = 1,
        completed: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if task_name in self.Name2ID:
            if completed is None:
                if self._tasks[self.Name2ID[task_name]].total is None:
                    self.update(task_name, total=advance, description=description)
                else:
                    self.update(task_name, total=self._tasks[self.Name2ID[task_name]].total+advance, description=description)
            else:
                self.update(task_name, total=completed, description=description)

def use_coroutine(func):
    
    async def coroutine(
        key: Union[Hashable, int], 
        data_dict: Dict[Hashable,Dict[int,Any]], 
        args: Tuple[Any], 
        kwargs: Dict[str,Any],
        progress: Union[ProgressManager,None] = None,
    ):
        data_dict[key] = func(*args, **kwargs)
        
        if progress is not None:
            progress.update(func.__name__, advance=1)
            
    coroutine.__inner_func_name__ = func.__name__
        
    return coroutine

async def start_coroutine(
    func: Callable, 
    func_inps: Union[
        Dict[Hashable, Tuple[tuple,dict]],
        List[Tuple[tuple,dict]],
    ],
    data_dict: Dict[Hashable,Any],
    progress: Union[ProgressManager,None] = None,
):
    async with trio.open_nursery() as nursery:
        for key, (args, kwargs) in get_kv_pairs(func_inps):
            nursery.start_soon(func, key, data_dict, args, kwargs, progress)
            
def run_coroutine(
    func: Callable,
    func_inps: Union[
        Dict[Hashable, Tuple[tuple,dict]],
        List[Tuple[tuple,dict]],
    ],
    data_dict: Optional[Dict[Hashable,Dict[int,Any]]] = None,
    coroutine: Optional[
        Callable[
            [
                Callable,
                Union[
                    Dict[Hashable, Tuple[tuple,dict]],
                    List[Tuple[tuple,dict]],
                ],
                Dict[Hashable,Any],
            ]
        ]
    ] = None,
    use_progress: bool = True,
    progress: Union[Progress,None] = None,
    description: str = None,
) -> Dict:
    need_close = False
    if data_dict is None:
        data_dict = {}
    if coroutine is None:
        coroutine = start_coroutine
    if use_progress:
        if progress is None:
            progress = ProgressManager()
            progress.add_task(task_name=func.__inner_func_name__,total=len(func_inps),description=description)
            progress.start()
            need_close = True
    else:
        progress = None
    trio.run(coroutine, func, func_inps, data_dict, progress)
    if need_close:
        progress.stop()
    return data_dict

class InitProcessAndParamObj:
    
    func_obj_map = {}
    
    param_cache = {}
    
    def __init__(
        self, 
        process_cls,
        defualt_params:dict = {},
        **kwargs,
    ):
        self.process_cls = process_cls
        self.defualt_params = defualt_params
        self.defualt_params.update(kwargs)

    def __call__(
        self, func: Callable[
            [oms_process_inputs, oms_process_obj, Union[oms.Param,dict,None]],oms_process_outputs
        ]
    ) -> Callable[
            [oms_process_inputs, oms_process_obj, Union[oms.Param,dict,None]],oms_process_outputs
        ]:
        
        wrapper = self.get_wrapper(func)
        
        InitProcessAndParamObj.func_obj_map[wrapper] = self
        InitProcessAndParamObj.param_cache[self.process_cls] = {}

        return wrapper
    
    def get_wrapper(
        self,
        func: Callable[
            [oms_process_inputs, oms_process_obj, Union[oms.Param,dict,None]],oms_process_outputs
        ],
    ) -> Callable[
            [oms_process_inputs, oms_process_obj, Union[oms.Param,dict,None]],oms_process_outputs
        ]:
            def base_wrapper(
                inps: oms_process_inputs,
                process_obj: Optional[oms_process_obj] = None,
                param_obj: Union[oms.Param,dict,None] = None,
                **kwargs
            ) -> oms_process_outputs:
                
                new_param_dict = self.get_param_dict(param_obj, kwargs)
                param_key = InitProcessAndParamObj.calulate_param_hash(new_param_dict)
                if param_key in InitProcessAndParamObj.param_cache[self.process_cls]:
                    process_obj = InitProcessAndParamObj.param_cache[self.process_cls][param_key]
                else:
                    process_obj = self.init_process_obj(new_param_dict, process_obj, param_obj)
                    InitProcessAndParamObj.param_cache[self.process_cls][param_key] = process_obj
                    
                return func(inps, process_obj, param_obj, **kwargs)
            
            def align_wrapper(
                exp_list: List[oms.MSExperiment],
                feature_maps: List[oms.FeatureMap],
                ref_map: Union[oms.FeatureMap,int,str],
                process_obj: Optional[oms.MapAlignmentAlgorithmPoseClustering] = None,
                param_obj: Union[oms.Param,dict,None] = None,
                **kwargs
            ) -> Tuple[List[oms.MSExperiment],List[oms.TransformationDescription]]:
                new_param_dict = self.get_param_dict(param_obj, kwargs)
                process_obj = self.init_process_obj(new_param_dict, process_obj, param_obj)
                return func(exp_list, feature_maps, ref_map, process_obj, param_obj, **kwargs)
            
            if self.process_cls == oms.MapAlignmentAlgorithmPoseClustering:
                return align_wrapper
            else:
                return base_wrapper
            
    def get_param_dict(
        self,
        param_obj:Union[oms.Param,dict,None],
        kwargs:dict,
    ) -> dict:
    
        new_param_dict ={}
        new_param_dict.update(self.defualt_params)
        if isinstance(param_obj,dict):
            new_param_dict.update(param_obj)
        new_param_dict.update(kwargs)
        return new_param_dict
    
    def init_process_obj(
        self,
        new_param_dict: dict,
        process_obj: Optional[oms_process_obj] = None,
        param_obj: Union[oms.Param,dict,None] = None,
    ) -> oms_process_obj:
        if not isinstance(process_obj, self.process_cls):
            process_obj = self.process_cls()
        if not isinstance(param_obj, oms.Param):
            new_param_obj = process_obj.getParameters()
        for param_name, param_value in new_param_dict.items():
            new_param_obj.setValue(param_name, param_value)
        process_obj.setParameters(new_param_obj)
        return process_obj
    
    @staticmethod
    def calulate_param_hash(new_param_dict: dict) -> int:
        params_set = set()
        for k,v in new_param_dict.items():
            if isinstance(v, list):
                v = frozenset(v)
            params_set.add((k,v))
        return hash(frozenset(params_set))
    
    @property
    def ProcessCls(self) -> oms_process_obj:
        return self.process_cls
    
    @property
    def DefaultParams(self) -> dict:
        return self.defualt_params
    
    @property
    def ProcessClsDefaultParams(self) -> oms.Param:
        return self.process_cls().getParameters()
    
    @property
    def ProcessClsDefaultParamsDict(self) -> dict:
        return dict(self.ProcessClsDefaultParams.items())
    
    @property
    def ProcessDefaultParamsDict(self) -> dict:
        params = {k.decode('utf-8'):v for k,v in self.ProcessClsDefaultParamsDict.items()}
        params.update(self.DefaultParams)
        return params
    
def calculate_mzfile_hash(file_path:str,hash_type:str = "md5"):
    hash_object = hashlib.new(hash_type)
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()

def load_mzml_file(file_path: str) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)
    return exp

def load_mzxml_file(file_path: str) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.MzXMLFile().load(file_path, exp)
    return exp

def load_exp_ms_file(file_path: str) -> oms.MSExperiment:
    if file_path.lower().endswith(".mzml"):
        return load_mzml_file(file_path)
    elif file_path.lower().endswith(".mzxml"):
        return load_mzxml_file(file_path)
    else:
        raise ValueError("Unsupported file format")

@use_coroutine
def load_exp_ms_file_coroutine(file_path: str):
    return load_exp_ms_file(file_path)

def load_exp_ms_files(
    file_path_list: list,
    key_type: Literal[
        "index",
        "file_path",
        "file_name",
        "md5", "sha1", "sha256"
    ] = "file_path",
    use_progress: bool = True,
) -> Tuple[
    Dict[Hashable, oms.MSExperiment], 
    Dict[Hashable, str],
]:
    inps = {file_path:((file_path,),{}) for file_path in file_path_list}
    exps = run_coroutine(load_exp_ms_file_coroutine, inps, use_progress=use_progress, description="Loading MSExperiment files")
    result_dict = {}
    path_map = {}
    if key_type == "file_path":
        result_dict = exps
        path_map = dict(zip(file_path_list, file_path_list))
    elif key_type == "file_name":
        for file_path, exp in exps.items():
            file_name = file_path.split("/")[-1]
            result_dict[file_name] = exp
            path_map[file_name] = file_path
    elif key_type == "index":
        for i, file_path in enumerate(exps.keys()):
            result_dict[i] = exps[file_path]
            path_map[i] = file_path
    else:
        for file_path, exp in exps.items():
            hash_value = calculate_mzfile_hash(file_path, key_type)
            result_dict[hash_value] = exp
            path_map[hash_value] = file_path
    return result_dict, path_map

async def copy_ms_experiment_step(ms_exp: oms.MSExperiment, key:str, result_dict:dict, use_blank:bool=False):
    if use_blank:
        result_dict[key] = oms.MSExperiment()
    else:
        result_dict[key] = oms.MSExperiment(ms_exp)
    
async def copy_ms_experiments_step(ms_exps: Dict[str, oms.MSExperiment], result_dict:dict, use_blank:bool=False):
    async with trio.open_nursery() as nursery:
        for key, ms_exp in ms_exps.items():
            nursery.start_soon(copy_ms_experiment_step, ms_exp, key, result_dict, use_blank)

def copy_ms_experiments(ms_exps: oms_exps, use_blank:bool=False) -> oms_exps:
    if isinstance(ms_exps, oms.MSExperiment):
        return oms.MSExperiment(ms_exps)
    result_dict = {}
    if isinstance(ms_exps, dict):
        trio.run(copy_ms_experiments_step, ms_exps, result_dict, use_blank)
        return result_dict
    else:
        trio.run(copy_ms_experiment_step, dict(zip(range(len(ms_exps)), ms_exps)), result_dict, use_blank)
        return list(result_dict.values())
    
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
def detect_mass_traces(
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

def merge_exps(ms_exps: oms_exps) -> oms.MSExperiment:
    merged_exp = oms.MSExperiment()
    for _,exp in get_kv_pairs(ms_exps):
        for spec in exp.getSpectra():
            merged_exp.addSpectrum(spec)
    return merged_exp

def set_primary_ms_run_path(
    feature_maps:List[oms.FeatureMap],
    file_names:List[str],
) -> None:
    for feature_map,file_name in zip(feature_maps,file_names):
        feature_map.setPrimaryMSRunPath(
            [file_name.encode()]
        )
        

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
def detect_adduct(
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

def ppm2da(mz: float, ppm: float) -> float:
    """
    将 ppm 值转化为以 dalton 为单位的值。
    :param mz: 质量数值（质荷比）
    :param ppm: 每百万的偏差值
    :return: 转换后的值
    """
    return np.multiply(mz, np.divide(ppm, 1e6))

def id_dict_to_str(id_dict: Dict[str, int]) -> str:
    """
    将字典格式的标识符转换为字符串格式。
    :param id_dict: 包含控制器类型、编号和扫描信息的字典
    :return: 格式化后的字符串
    """
    return "SPEC[T{}N{}S{}]".format(
                id_dict['controllerType'] if 'controllerType' in id_dict else '?',
                id_dict['controllerNumber'] if 'controllerNumber' in id_dict else '?',
                id_dict['scan'],
            )

def unzip_results(
    result_dict: Dict[Hashable, Dict[int, Any]]
) -> Dict[Tuple[Hashable, int], Any]:
    """
    将嵌套字典展开，将内部字典的键与外部字典的键组合为新字典的键。
    :param result_dict: 嵌套字典，外部键为哈希类型，内部键为整数
    :return: 展开的字典，键为元组（外部键，内部键），值为内部字典的值
    """
    unzipped_dict = {}
    for key, value in result_dict.items():
        for index, item in value.items():
            unzipped_dict[(key, index)] = item
    return unzipped_dict

def inverse_dict(d: dict) -> dict:
    """
    反转字典的键和值。
    :param d: 原始字典
    :return: 反转后的字典，值变为键，键变为值
    """
    return {v: k for k, v in d.items()}
    
class AsyncBase:
    
    def __init__(self):
        self.progress = ProgressManager()
    
    @property
    def PathTag(self,) -> Dict[Hashable, Union[oms.MSExperiment, oms.FeatureMap]]:
        pass
    
    @property
    def PathTagLenFunc(self) -> Callable[[Hashable], int]:
        pass
    
    @staticmethod
    def use_coroutine(func) -> Callable:
        
        async def coroutine(
            self, 
            access_path: Tuple[Hashable, int], 
            data_dict: Dict[Hashable,Dict[int,Any]], 
            progress: ProgressManager,
            *args, 
            **kwargs,
        ):
            exp_id, item_id = access_path
            if exp_id not in data_dict:
                data_dict[exp_id] = {}
            data_dict[exp_id][item_id] = func(self, access_path, *args, **kwargs)
            progress.update(func.__name__,advance=1)
            
        coroutine.__inner_func_name__ = func.__name__
            
        return coroutine
    
    def unzip_access_path(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ],
    ) -> List[Tuple[Hashable, int]]:
        access_path_seq = []
        if access_path is None:
            exp_ids,spec_ids = None,None
        elif not isinstance(access_path, (list,tuple)):
            exp_ids,spec_ids = access_path,None
        elif isinstance(access_path,tuple):
            exp_ids,spec_ids = access_path
        else:
            return access_path
        if exp_ids is None:
            exp_ids = list(self.PathTag.keys())
        elif not isinstance(exp_ids, list):
            exp_ids = [exp_ids]
        if spec_ids is None:
            for exp_id in exp_ids:
                if exp_id in self.PathTag:
                    spec_ids = list(range(self.PathTagLenFunc(exp_id)))
                    for spec_id in spec_ids:
                        access_path_seq.append((exp_id, spec_id))
        else:
            if not isinstance(spec_ids, list):
                spec_ids = [spec_ids]
            for exp_id in exp_ids:
                if exp_id in self.PathTag:
                    lens = self.PathTagLenFunc(exp_id)
                    for spec_id in spec_ids:
                        if 0 <= spec_id < lens:
                            access_path_seq.append((exp_id, spec_id))
        return access_path_seq
                
    async def start_coroutine(
        self, 
        func: Callable, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ],
        data_dict: Dict[Hashable,Dict[int,Any]],
        args: Tuple,
        kwargs: Dict,
    ):
        access_path_list = self.unzip_access_path(access_path)
        self.progress.add_task(task_name=func.__inner_func_name__, total=len(access_path_list))
        async with trio.open_nursery() as nursery:
            for access_path in access_path_list:
                nursery.start_soon(func, access_path, data_dict, self.progress, *args, **kwargs)
    
    def run_coroutine(
        self,
        func: Callable,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ],
        data_dict: Optional[Dict[Hashable,Dict[int,Any]]] = None,
        args: Tuple = (),
        kwargs: Dict = {},
    ) -> Dict:
        if data_dict is None:
            data_dict = {}
        with self.progress:
            trio.run(self.start_coroutine, func, access_path, data_dict, args, kwargs)
        return data_dict
    
class ToleranceBase:
    
    mz_tolerance_type = 'ppm'
    mz_tolerance = 3.0
    rt_tolerance = 6.0
    
    @property
    def MZ_Atols(self) -> Tuple[float,float]:
        if self.mz_tolerance_type == 'ppm':
            mz_atol,mz_rtol = 0, self.mz_tolerance * 1e-6
        else:
            mz_atol,mz_rtol = self.mz_tolerance, 0
        return mz_atol,mz_rtol

class MSExperiments(AsyncBase,ToleranceBase):
    
    def __init__(
        self, 
        exps: Union[oms.MSExperiment, List[oms.MSExperiment], Dict[str, oms.MSExperiment]],
        precursor_search_tolerance: Tuple[float,float] = (5.0,5.0),
        precursor_search_tolerance_type: Literal['ppm','Da'] = 'ppm',
        mz_tolerance: float = 3.0,
        rt_tolerance: float = 6.0,
        mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    ):
        self.progress = ProgressManager()
        if isinstance(exps, oms.MSExperiment):
            self.exps = {0:exps}
        elif isinstance(exps, list):
            self.exps = {i:exp for i,exp in enumerate(exps)}
        elif isinstance(exps, MSExperiments):
            self.exps = exps.exps
        else:
            self.exps = exps
        self.init_spec_uid_access_map()
        self.init_ms2_search_info()
        self.precursor_search_tl, self.precursor_search_tr = precursor_search_tolerance
        self.precursor_search_tolerance_type = precursor_search_tolerance_type
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mz_tolerance_type = mz_tolerance_type
        
    def init_spec_uid_access_map(self):
        self.spec_uid_access_map = inverse_dict(unzip_results(self.SPEC_UIDs()))
    
    def init_ms2_search_info(self):
        # self.ms_access = np.array(list(self.SPEC_UID_ACCESS_MAP.values()),dtype=object)
        all_PIs = self.precursor_mzs()
        self.ms2_access: Dict[Hashable, Union[List[Tuple[Hashable, int]],np.ndarray]] = {}
        self.ms2_PI: Dict[Hashable, Union[List[float],np.ndarray]] = {}
        self.ms2_RT: Dict[Hashable, Union[List[float],np.ndarray]] = {}
        for exp_id, PIs in all_PIs.items():
            self.ms2_access[exp_id] = []
            self.ms2_PI[exp_id] = []
            self.ms2_RT[exp_id] = []
            for i, PI in PIs.items():
                if PI is not None:
                    access_path = (exp_id, i)
                    self.ms2_access[exp_id].append(access_path)
                    self.ms2_PI[exp_id].append(PI)
                    self.ms2_RT[exp_id].append(self.SpecRT(access_path))
            self.ms2_access[exp_id] = np.array(self.ms2_access[exp_id],dtype=object)
            self.ms2_PI[exp_id] = np.array(self.ms2_PI[exp_id])
            self.ms2_RT[exp_id] = np.array(self.ms2_RT[exp_id])
        print()
        
    @property
    def EXPS(self) -> Dict[Hashable, oms.MSExperiment]:
        return self.exps
    
    @property
    def SPEC_UID_ACCESS_MAP(self) -> Dict[Tuple[Hashable, int], str]:
        return self.spec_uid_access_map
    
    @property
    def PathTag(self,) -> Dict[Hashable, oms.MSExperiment]:
        return self.EXPS
    
    @property
    def PathTagLenFunc(self) -> Callable[[Hashable], int]:
        return self.EXP_len
    
    # @property
    # def MSAccess(self) -> np.ndarray:
    #     return self.ms_access
    
    @property
    def MS2PIs(self) -> Dict[Hashable, np.ndarray]:
        return self.ms2_PI
    
    @property
    def MS2RTs(self) -> Dict[Hashable, np.ndarray]:
        return self.ms2_RT
    
    @property
    def MS2Access(self) -> Dict[Hashable, np.ndarray]:
        return self.ms2_access
    
    def search_ms2(
        self,exp_id:Hashable,mz:float,RT:float,
    ) -> Optional[List[Tuple[Hashable,int]]]:
        mz_atol,mz_rtol = self.MZ_Atols
        PI_mask = np.isclose(self.MS2PIs[exp_id], mz, atol=mz_atol, rtol=mz_rtol)
        RT_mask = np.isclose(self.MS2RTs[exp_id], RT, atol=self.rt_tolerance,  rtol=0)
        mask = PI_mask & RT_mask
        if mask.any():
            tag_index = np.argwhere(mask).reshape(-1)
            access_path = [tuple(access) for access in self.MS2Access[exp_id][tag_index]]
            return access_path
        else:
            return None
        
    def search_ms2_by_range(
        self,
        exp_id: Hashable,
        mz_range: Tuple[float,float],
        RT_range: Tuple[float,float],
    ) -> Optional[List[Tuple[Hashable,int]]]:
        PI_mask = (self.MS2PIs[exp_id] >= (mz_range[0] - 0.0001)) & (self.MS2PIs[exp_id] <= (mz_range[1] + 0.0001))
        RT_mask = (self.MS2RTs[exp_id] >= (RT_range[0] - 0.0001)) & (self.MS2RTs[exp_id] <= (RT_range[1] + 0.0001))
        mask = PI_mask & RT_mask
        if mask.any():
            tag_index = np.argwhere(mask).reshape(-1)
            access_path = [tuple(access) for access in self.MS2Access[exp_id][tag_index]]
            return access_path
        else:
            return None
    
    def __len__(self) -> int:
        return len(self.exps)
    
    def EXP(self, exp_id: Hashable) -> Optional[oms.MSExperiment]:
        return self.EXPS.get(exp_id, None)
    
    def EXP_len(self, exp_id: Hashable) -> Optional[int]:
        exp = self.EXP(exp_id)
        if exp is not None:
            return exp.size()
        else:
            return None
    
    def SPECS(self, exp_id: Hashable) -> Optional[List[oms.MSSpectrum]]:
        exp = self.EXP(exp_id)
        if exp is not None:
            return exp.getSpectra()
        else:
            return None
        
    def SPEC(self, access_path: Tuple[Hashable, int]) -> Optional[oms.MSSpectrum]:
        exp_id, spec_id = access_path
        exp = self.EXP(exp_id)
        if exp is not None:
            return exp.getSpectrum(spec_id)
        else:
            return None
        
    def __getitem__(self, access_path: Tuple[Hashable, int]) -> oms.MSSpectrum:
        exp_id, spec_id = access_path
        exp = self.EXPS[exp_id]
        spec = exp.getSpectrum(spec_id)
        if spec is None:
            raise KeyError(f"Spectrum {spec_id} not found in experiment {exp_id}.")
        return spec
        
    def SPEC_ID(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        ids_dict = self.SpecIDDict(access_path)
        if ids_dict is not None:
            return id_dict_to_str(ids_dict)
        else:
            return None
    
    @AsyncBase.use_coroutine
    def SPEC_ID_coroutine(
        self,
        access_path: Tuple[Hashable, int],
    ) -> None:
        return self.SPEC_ID(access_path)
    
    def SPEC_IDs(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, str]]:
        return self.run_coroutine(
            self.SPEC_ID_coroutine,
            access_path,
        )
        
    def SPEC_UID(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        exp_id, spec_index = access_path
        spec_id = self.SPEC_ID(access_path)
        if spec_id is not None:
            return "{}:{}".format(exp_id, spec_id)
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SPEC_UID_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SPEC_UID(access_path)
    
    def SPEC_UIDs(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, str]]:
        return self.run_coroutine(
            self.SPEC_UID_coroutine,
            access_path,
        )
        
    def SpecPeaksTIC(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.calculateTIC()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecPeaksTIC_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecPeaksTIC(access_path)
    
    def SpecPeaksTICs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecPeaksTIC_coroutine, access_path)
        
    def findHighestInWindow(
        self, 
        access_path: Tuple[Hashable, int], 
        mz:float,
        tolerance_left: float = 1.0, 
        tolerance_right: float = 1.0,
        tolerance_unit: Literal['ppm', 'Da'] = 'ppm',
    ) -> Optional[int]:
        if tolerance_unit == 'ppm':
            tolerance_left = ppm2da(mz, tolerance_left)
            tolerance_right = ppm2da(mz, tolerance_right)
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.findHighestInWindow(mz, tolerance_left, tolerance_right)
        else:
            return None
        
    def findNearest(
        self,
        access_path: Tuple[Hashable, int],
        mz: float,
        tolerance_or_left: Optional[float] = None,
        tolerance_unit: Literal['ppm', 'Da'] = 'ppm',
        tolerance_right: Optional[float] = None,
    ) -> Optional[int]:
        if tolerance_unit == 'ppm':
            if tolerance_or_left is not None:
                tolerance_or_left = ppm2da(mz, tolerance_or_left)
            if tolerance_right is not None:
                tolerance_right = ppm2da(mz, tolerance_right)
        spec = self.SPEC(access_path)
        if spec is not None:
            if tolerance_or_left is None:
                return spec.findNearest(mz)
            elif tolerance_right is None:
                return spec.findNearest(mz, tolerance_or_left)
            else:
                return spec.findNearest(mz, tolerance_or_left, tolerance_right)
        else:
            return None
    
    def DriftTime(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            dt = spec.getDriftTime()
            if dt == -1:
                return None
            else:
                return dt
        else:
            return None
        
    @AsyncBase.use_coroutine
    def DriftTime_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.DriftTime(access_path)

    def DriftTimes(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.DriftTime_coroutine, access_path)
        
    def MSLevel(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMSLevel()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def MSLevel_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.MSLevel(access_path)
        
    def MSLevels(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, int]]:
        return self.run_coroutine(self.MSLevel_coroutine, access_path)
        
    def SpecMaxIntensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMaxIntensity()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecMaxIntensity_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecMaxIntensity(access_path)
    
    def SpecMaxIntensities(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecMaxIntensity_coroutine, access_path)
        
    def SpecMaxMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMaxMZ()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecMaxMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecMaxMZ(access_path)
        
    def SpecMaxMZS(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecMaxMZ_coroutine, access_path)
        
    def SpecMinIntensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMinIntensity()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecMinIntensity_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecMinIntensity(access_path)
        
    def SpecMinIntensities(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecMinIntensity_coroutine, access_path)
        
    def SpecMinMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMinMZ()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecMinMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecMinMZ(access_path)
        
    def SpecMinMZS(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecMinMZ_coroutine, access_path)
        
    def SpecTIC(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMetaValue('total ion current')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecTIC_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecTIC(access_path)
        
    def SpecTICs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecTIC_coroutine, access_path)
        
    def SpecBasePeakMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMetaValue('base peak m/z')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecBasePeakMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecBasePeakMZ(access_path)
        
    def SpecBasePeakMZS(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecBasePeakMZ_coroutine, access_path)
        
    def SpecBasePeakIntensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMetaValue('base peak intensity')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecBasePeakIntensity_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecBasePeakIntensity(access_path)
        
    def SpecBasePeakIntensities(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecBasePeakIntensity_coroutine, access_path)
        
    def SpecLowestObservedMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMetaValue('lowest observed m/z')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecLowestObservedMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecLowestObservedMZ(access_path)
        
    def SpecLowestObservedMZS(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecLowestObservedMZ_coroutine, access_path)
        
    def SpecHighestObservedMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getMetaValue('highest observed m/z')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecHighestObservedMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecHighestObservedMZ(access_path)
        
    def SpecHighestObservedMZS(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecHighestObservedMZ_coroutine, access_path)
        
    def MassCondition(
        self, 
        access_path: Tuple[Hashable, int],
        re_type: Literal['str', 'dict'] = 'dict',
    ) -> Optional[Dict[
        Literal[
            'instrument_type',
            'polarity',
            'peak_type',
            'ionization_method',
            'scan_mode',
            'ms_level',
            'cleavage_info',
            'scan_range',
        ],str
    ]]:
        exp_id, spec_id = access_path
        spec = self.SPEC((exp_id, spec_id))
        if spec is not None:
            filter_string = spec.getMetaValue('filter string')
            if re_type == 'str':
                return filter_string
            if isinstance(filter_string, str):
                items = filter_string.split(' ')
                if "ms2" in items:
                    conditions = {
                        'instrument_type': items[0],
                        'polarity': items[1],
                        'peak_type': items[2],
                        'ionization_method': items[3],
                        'scan_mode': items[5],
                        'ms_level': items[6] if items[6] != "ms" else "ms1",
                        'cleavage_info': items[7],
                        'scan_range': items[8],
                    }
                else:
                    conditions = {
                        'instrument_type': items[0],
                        'polarity': items[1],
                        'peak_type': items[2],
                        'ionization_method': items[3],
                        'scan_mode': items[4],
                        'ms_level': items[5] if items[5] != "ms" else "ms1",
                        'scan_range': items[6],
                    }
                return conditions
        return None
    
    @AsyncBase.use_coroutine
    def MassCondition_coroutine(
        self, 
        access_path: Tuple[Hashable, int],
        re_type: Literal['str', 'dict'] = 'dict',
    ) -> None:
        return self.MassCondition(access_path, re_type)
        
    def MassConditions(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
        re_type: Literal['str', 'dict'] = 'dict',
    ) -> Dict[Hashable, Dict[int, Dict[
        Literal[
            'instrument_type',
            'polarity',
            'peak_type',
            'ionization_method',
            'scan_mode',
            'ms_level',
            'cleavage_info',
            'scan_range',
        ],str
    ]]]:
        return self.run_coroutine(self.MassCondition_coroutine, access_path, re_type)
    
    # def SpecOriginalRT(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
    #     spec = self.SPEC(access_path)
    #     if spec is not None:
    #         return spec.getMetaValue('original_rt')
    #     else:
    #         return None
        
    def SpecRT(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getRT()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecRT_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecRT(access_path)
        
    def SpecRTs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.SpecRT_coroutine, access_path)
        
    def SpecIDDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Optional[
        Dict[
            Literal[
                'controllerType',
                'controllerNumber',
                'scan',
            ],int
        ]
    ]:
        spec = self.SPEC(access_path)
        if spec is not None:
            id_string = spec.getNativeID()
            if isinstance(id_string, str):
                ids = {}
                key_value_pairs = id_string.split()
                for pair in key_value_pairs:
                    key, value = pair.split('=')
                    ids[key] = int(value)
                return ids
        return None
    
    @AsyncBase.use_coroutine
    def SpecIDDict_coroutine(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> None:
        return self.SpecIDDict(access_path)
        
    def SpecIDsDict(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict[
        Literal[
            'controllerType',
            'controllerNumber',
            'scan',
        ],int
    ]]]:
        return self.run_coroutine(self.SpecIDDict_coroutine, access_path)
    
    def is_centroid_spec(
        self,
        access_path: Tuple[Hashable, int],
    ) -> Optional[bool]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return bool(spec.getType())
        else:
            return None
        
    def SpecDataFrame(
        self, 
        access_path: Tuple[Hashable, int], 
        export_meta_values: bool = True,
    ) -> Optional[pd.DataFrame]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return oms._dataframes._MSSpectrumDF.get_df(spec,export_meta_values)
        else:
            return None
        
    def SpecPeaks(self, access_path: Tuple[Hashable, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        spec = self.SPEC(access_path)
        if spec is not None:
            mzs, intensities = spec.get_peaks()
            return mzs, intensities
        else:
            return None
        
    @AsyncBase.use_coroutine
    def SpecPeaks_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecPeaks(access_path)
        
    def SpecPeaksArrays(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        return self.run_coroutine(self.SpecPeaks_coroutine, access_path)
        
    def is_sorted_spec(
        self,
        access_path: Tuple[Hashable, int],
    ) -> Optional[bool]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.isSorted()
        else:
            return None
        
    def PeakCount(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.size()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def PeakCount_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.PeakCount(access_path)
        
    def PeakCounts(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, int]]:
        return self.run_coroutine(self.PeakCount_coroutine, access_path)
        
    def Precursors(self, access_path: Tuple[Hashable, int]) -> Optional[List[oms.Precursor]]:
        spec = self.SPEC(access_path)
        if spec is not None:
            return spec.getPrecursors()
        else:
            return None
        
    def Precursor(self, access_path: Tuple[Hashable, int, int]) -> Optional[oms.Precursor]:
        precursors = self.Precursors(access_path)
        if isinstance(precursors, list) and len(precursors) > 0:
            return precursors[0]
        else:
            return None
    
    def precursor_mz(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        precursor = self.Precursor(access_path)
        if precursor is not None:
            return precursor.getMZ()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def precursor_mz_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.precursor_mz(access_path)
        
    def precursor_mzs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.precursor_mz_coroutine, access_path)
    
    def precursor_intensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        precursor = self.Precursor(access_path)
        if precursor is not None:
            return precursor.getIntensity()
        else:
            return None
    
    def precursor_charge(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        precursor = self.Precursor(access_path)
        if precursor is not None:
            return precursor.getCharge()
        else:
            return None

    def precursor_ref_ids(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Optional[Dict[
        Literal['controllerType','controllerNumber','scan'],int
    ]]:
        precursor = self.Precursor(access_path)
        if precursor is not None:
            ref_string = precursor.getMetaValue('spectrum_ref')
            if not isinstance(ref_string, str):
                return None
            else:
                precursor_ref = {}
                key_value_pairs = ref_string.split()
                for pair in key_value_pairs:
                    key, value = pair.split('=')
                    precursor_ref[key] = int(value)
                return precursor_ref
        else:
            return None
        
    def collision_energy(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        precursor = self.Precursor(access_path)
        if precursor is not None:
            return float(precursor.getMetaValue('collision energy'))
        else:
            return None
        
    def PrecursorDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> None:
        precursor_dict = {}
        exp_id, spec_id = access_path
        ref_dict = self.precursor_ref_ids(access_path)
        if ref_dict is not None:
            PI_SpecID = id_dict_to_str(ref_dict)
            precursor_dict['PI_SpecID'] = PI_SpecID
            precursor_dict['PI'] = self.precursor_mz(access_path)
            PI_intensity = self.precursor_intensity(access_path)
            if PI_intensity == 0 and (exp_id,PI_SpecID) in self.SPEC_UID_ACCESS_MAP:
                PI_spec_access = self.SPEC_UID_ACCESS_MAP[(exp_id,PI_SpecID)]
                peak_index = self.findHighestInWindow(
                    PI_spec_access,
                    precursor_dict['PI'],
                    self.precursor_search_tl,
                    self.precursor_search_tr,
                    self.precursor_search_tolerance_type,
                )
                if peak_index is not None:
                    PI_intensity = self.SpecPeaks(PI_spec_access)[1][peak_index]
            precursor_dict['PI_intensity'] = PI_intensity
            precursor_dict['charge'] = self.precursor_charge(access_path)
            precursor_dict['collision_energy'] = self.collision_energy(access_path)
        return precursor_dict
    
    @AsyncBase.use_coroutine
    def PrecursorDict_coroutine(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> None:
        return self.PrecursorDict(access_path)
    
    def PrecursorsDict(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict[
        Literal['PI_SpecID','PI','PI_intensity','charge','collision_energy'],
        Union[str,float,int],
    ]]]:
        return self.run_coroutine(
            self.PrecursorDict_coroutine,
            access_path,
        )
        
    def SpecDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Dict[
        Literal[
            'EXP_ID','SPEC_INDEX','ID',
            'RT','mz','intens',
            'basepeak_mz','basepeak_intens','peak_counts',
            'instrument_type','polarity','peak_type','ionization_method',
            'scan_mode','ms_level','cleavage_info','scan_range',
            'PI','PI_intensity','charge','collision_energy',
        ],Union[str,float,int,np.ndarray]
    ]:
        exp_id,spec_index = access_path
        mass_conditions = self.MassCondition(access_path)
        precursor_dict = self.PrecursorDict(access_path)
        RT = self.SpecRT(access_path)
        if 'PI_SpecID' in precursor_dict:
            PI_SpecID = precursor_dict['PI_SpecID']
            if (exp_id,PI_SpecID) in self.SPEC_UID_ACCESS_MAP:
                PI_spec_access = self.SPEC_UID_ACCESS_MAP[(exp_id,PI_SpecID)]
                RT = self.SpecRT(PI_spec_access)
        mz,intens = self.SpecPeaks(access_path)
        spec_dict = {
            "EXP_ID":exp_id,
            "SPEC_INDEX":spec_index,
            "SPEC_ID":self.SPEC_ID(access_path),
            "RT":RT,
            "mz":mz,
            "intens":intens,
            "basepeak_mz":self.SpecBasePeakMZ(access_path),
            "basepeak_intens":self.SpecBasePeakIntensity(access_path),
            "peak_counts":self.PeakCount(access_path),
            **mass_conditions,
            **precursor_dict,
        }
        return spec_dict
    
    @AsyncBase.use_coroutine
    def SpecDict_coroutine(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> None:
        return self.SpecDict(access_path)
    
    def SpecsDict(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict[
        Literal[
            'EXP_ID','SPEC_INDEX','ID',
            'RT','mz','intens',
            'basepeak_mz','basepeak_intens','peak_counts',
            'instrument_type','polarity','peak_type','ionization_method',
            'scan_mode','ms_level','cleavage_info','scan_range',
            'PI','PI_intensity','charge','collision_energy',
        ],Union[str,float,int,np.ndarray]
    ]]]:
        return self.run_coroutine(
            self.SpecDict_coroutine,
            access_path,
        )
        
    def SpecsDataFrame(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> pd.DataFrame:
        specs_df = {}
        specs_dict = self.SpecsDict(access_path)
        for exp_id, exp_dict in specs_dict.items():
            for spec_id, spec_dict in exp_dict.items():
                specs_df["{}:{}".format(exp_id,spec_dict['SPEC_ID'])] = spec_dict
        specs_df = pd.DataFrame(specs_df).transpose().sort_values(by=['EXP_ID','SPEC_INDEX'])
        specs_df.drop(['cleavage_info'],axis=1,inplace=True)
        specs_df.index.name = 'spec_id'
        return specs_df
    
    @staticmethod
    def split_specs_dataframe_by_mslevel(specs_df:pd.DataFrame) -> Dict[int,pd.DataFrame]:
        ms_level_dfs = {}
        for i,ms_level_df in specs_df.groupby('ms_level'):
            ms_level = int(ms_level_df['ms_level'].iloc[0].strip('ms'))
            ms_level_dfs[ms_level] = ms_level_df
        return ms_level_dfs
    
    def SpecsDataFrames(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[int,pd.DataFrame]:
        return MSExperiments.split_specs_dataframe_by_mslevel(self.SpecsDataFrame(access_path))
    
class PeakConvexHull:
    
    def __init__(
        self,
        merged_hull: oms.ConvexHull2D,
        sub_hulls: List[oms.ConvexHull2D],
    ):
        self.merged_hull = self.oms2py(merged_hull)
        self.sub_hulls = [self.oms2py(sub_hull) for sub_hull in sub_hulls]
        
    def __len__(self) -> int:
        return 1 + len(self.sub_hulls)
        
    def __getitem__(self, index: int) -> Dict[
        Literal[
            'RT_start',
            'RT_end',
            'MZ_start',
            'MZ_end',
            'points_array',
        ],Union[float,np.ndarray]
    ]:
        if index == 0:
            return self.merged_hull
        else:
            return self.sub_hulls[index-1]
        
    @property
    def MergedHull(self) -> Dict[
        Literal[
            'RT_start',
            'RT_end',
            'MZ_start',
            'MZ_end',
            'points_array',
        ],Union[float,np.ndarray]
    ]:
        return self.merged_hull
        
    @property
    def MainHull(self) -> Dict[
        Literal[
            'RT_start',
            'RT_end',
            'MZ_start',
            'MZ_end',
            'points_array',
        ],Union[float,np.ndarray]
    ]:
        return self.sub_hulls[0]
    
    @property
    def TailHulls(self) -> List[Dict[
        Literal[
            'RT_start',
            'RT_end',
            'MZ_start',
            'MZ_end',
            'points_array',
        ],Union[float,np.ndarray]
    ]]:
        return self.sub_hulls[1:]
        
    def to_dict(self) -> Dict[
        Literal['merged','main','tail'],Dict[
            Literal[
                'RT_start',
                'RT_end',
                'MZ_start',
                'MZ_end',
                'points_array',
        ],Union[float,np.ndarray]
    ]]:
        return {
            "merged":self.MergedHull,
            "main":self.MainHull,
            "tail":self.TailHulls,
        }
        
    @staticmethod
    def oms2py(
        hull_obj: oms.ConvexHull2D,
    ) -> Dict[
        Literal[
            'RT_start',
            'RT_end',
            'MZ_start',
            'MZ_end',
            'points_array',
        ],Union[float,np.ndarray]
    ]:
        (RT_start,MZ_start),(RT_end,MZ_end) = hull_obj.getBoundingBox2D()
        points_array = hull_obj.getHullPoints()
        return {
            "RT_start":RT_start,
            "RT_end":RT_end,
            "MZ_start":MZ_start,
            "MZ_end":MZ_end,
            "points_array":points_array,
        }
        
class FeatureMaps(AsyncBase,ToleranceBase):
    
    def __init__(
        self, 
        feature_maps: Union[oms.FeatureMap,List[oms.FeatureMap], Dict[str, oms.FeatureMap]],
        consensus_map: Optional[oms.ConsensusMap] = None,
        consensus_threshold: float = 0.0,
        mz_tolerance: float = 3.0,
        rt_tolerance: float = 6.0,
        mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    ):
        self.progress = ProgressManager()
        if isinstance(feature_maps, oms.FeatureMap):
            self.feature_maps = {0:feature_maps}
        elif isinstance(feature_maps, list):
            self.feature_maps = {i:fm for i,fm in enumerate(feature_maps)}
        elif feature_maps is None:
            self.feature_maps = None
        elif isinstance(feature_maps, FeatureMaps):
            self.feature_maps = feature_maps.FEATURE_MAPS
        else:
            self.feature_maps = feature_maps
        self.init_cluster_id_access_map()
        self.init_search_info()
        self.consensus_map = consensus_map
        self.consensus_map_df = None
        self.consensus_threshold = consensus_threshold
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mz_tolerance_type = mz_tolerance_type
        
    def init_cluster_id_access_map(self):
        self.cluster_id_access_map = inverse_dict(unzip_results(self.ClusterIDs()))
                
    # def init_search_info(self):
    #     self.cluster_access = np.array(list(self.CLUSTER_ID_ACCESS_MAP.values()),dtype=object)
    #     self.cluster_mzs = np.array(list(unzip_results(self.FeatureMZs()).values()))
    #     self.cluster_RTs = np.array(list(unzip_results(self.FeatureRTs()).values()))
        
    def init_search_info(self):
        self.cluster_access: Dict[Hashable, Union[List[Tuple[Hashable, int]]],np.ndarray] = {}
        self.cluster_search_hulls: Dict[Hashable, Dict[str, Union[np.ndarray,List[float]]]] = {}
        for exp_id, main_range_dict in self.ClusterMainRanges().items():
            self.cluster_search_hulls[exp_id] = {
                'RT_start':[],
                'RT_end':[],
                'MZ_start':[],
                'MZ_end':[],
            }
            self.cluster_access[exp_id] = []
            for i, ((RT_start,MZ_start),(RT_end,MZ_end)) in main_range_dict.items():
                self.cluster_search_hulls[exp_id]['RT_start'].append(RT_start)
                self.cluster_search_hulls[exp_id]['RT_end'].append(RT_end)
                self.cluster_search_hulls[exp_id]['MZ_start'].append(MZ_start)
                self.cluster_search_hulls[exp_id]['MZ_end'].append(MZ_end)
                self.cluster_access[exp_id].append((exp_id,i))
            self.cluster_search_hulls[exp_id]['RT_start'] = np.array(self.cluster_search_hulls[exp_id]['RT_start'])
            self.cluster_search_hulls[exp_id]['RT_end'] = np.array(self.cluster_search_hulls[exp_id]['RT_end'])
            self.cluster_search_hulls[exp_id]['MZ_start'] = np.array(self.cluster_search_hulls[exp_id]['MZ_start'])
            self.cluster_search_hulls[exp_id]['MZ_end'] = np.array(self.cluster_search_hulls[exp_id]['MZ_end'])
            self.cluster_access[exp_id] = np.array(self.cluster_access[exp_id],dtype=object)
    
    @property
    def FEATURE_MAPS(self) -> Dict[Hashable, oms.FeatureMap]:
        return self.feature_maps
    
    @property
    def CLUSTER_ID_ACCESS_MAP(self) -> Dict[Tuple[Hashable, int], int]:
        return self.cluster_id_access_map
    
    @property
    def CONSENSUS_MAP(self) -> Optional[oms.ConsensusMap]:
        return self.consensus_map
    
    @CONSENSUS_MAP.setter
    def CONSENSUS_MAP(self, value: oms.ConsensusMap) -> None:
        self.consensus_map = value
    
    @property
    def has_consensus_map(self) -> bool:
        return self.consensus_map is not None
    
    @property
    def PathTag(self,) -> Dict[Hashable, oms.FeatureMap]:
        return self.FEATURE_MAPS
    
    @property
    def PathTagLenFunc(self) -> Callable[[Hashable], int]:
        return self.FeatureMap_len
    
    @property
    def ConsensusMapDataFrame(self) -> pd.DataFrame:
        if self.has_consensus_map:
            if self.consensus_map_df is None:
                self.consensus_map_df = self.get_consensus_map_df()
                return self.consensus_map_df
            else:
                return self.consensus_map_df
        else:
            return pd.DataFrame()
        
    @ConsensusMapDataFrame.setter
    def ConsensusMapDataFrame(self, value: pd.DataFrame) -> None:
        self.consensus_map_df = value
        
    @property
    def ClusterAccess(self) -> Dict[Hashable,np.ndarray]:
        return self.cluster_access
    
    # @property
    # def ClusterMZs(self) -> np.ndarray:
    #     return self.cluster_mzs
    
    # @property
    # def ClusterRTs(self) -> np.ndarray:
    #     return self.cluster_RTs
    
    @property
    def ClusterSearchHulls(self) -> Dict[Hashable, Dict[str, np.ndarray]]:
        return self.cluster_search_hulls
    
    # def search_cluster(
    #     self,mz:float,RT:float
    # ) -> Optional[Tuple[Hashable,int]]:
    #     mz_atol,mz_rtol = self.MZ_Atols
    #     mz_mask = np.isclose(self.ClusterMZs, mz, atol=mz_atol, rtol=mz_rtol)
    #     rt_mask = np.isclose(self.ClusterRTs, RT, atol=self.rt_tolerance,  rtol=0)
    #     mask = mz_mask & rt_mask
    #     if mask.any():
    #         tag_index = np.argwhere(mask).reshape(-1)
    #         if len(tag_index) > 1:
    #             distance = np.abs(np.subtract(self.ClusterRTs, RT))
    #             best_match = np.argmin(distance[tag_index])
    #             tag_index = tag_index[best_match]
    #         else:
    #             tag_index = tag_index[0]
    #         access_path = tuple(self.ClusterAccess[tag_index])
    #         return access_path
    #     else:
    #         return None
        
    def search_cluster(
        self,exp_id:Hashable,mz:float,RT:float,
    ) -> Optional[Tuple[Hashable,int]]:
        RT_mask = ((self.ClusterSearchHulls[exp_id]['RT_start'] - 0.0001) <= RT) & (self.ClusterSearchHulls[exp_id]['RT_end'] + 0.0001 >= RT)
        MZ_mask = ((self.ClusterSearchHulls[exp_id]['MZ_start'] - 0.0001) <= mz) & (self.ClusterSearchHulls[exp_id]['MZ_end'] + 0.0001 >= mz)
        mask = RT_mask & MZ_mask
        if mask.any():
            tag_index = np.argwhere(mask).reshape(-1)
            tag_index = tag_index[0]
            access_path = self.ClusterAccess[exp_id][tag_index]
            return access_path
        else:
            return None
            
    def get_consensus_map_df(self) -> pd.DataFrame:
        consensus_df = self.CONSENSUS_MAP.get_df().reset_index(drop=True)
        consensus_df = consensus_df[consensus_df['quality'] > self.consensus_threshold]
        return consensus_df
    
    def search_consensus_map(
        self,
        mz: float,
        RT: float,
        exp_id: Hashable,
    ) -> Optional[Tuple[int,float,float,float]]:
        if self.has_consensus_map:
            mz_atol,mz_rtol = self.MZ_Atols
            mz_mask = np.isclose(self.ConsensusMapDataFrame['mz'], mz, atol=mz_atol, rtol=mz_rtol)
            rt_mask = np.isclose(self.ConsensusMapDataFrame['RT'], RT, atol=self.rt_tolerance,  rtol=0)
            intens_mask = self.ConsensusMapDataFrame[str(exp_id)] > 0
            mask = pd.Series(mz_mask,index=self.ConsensusMapDataFrame.index) & pd.Series(rt_mask,index=self.ConsensusMapDataFrame.index) & intens_mask
            if mask.any():
                tag_df = self.ConsensusMapDataFrame[mask]
                tag_df = tag_df.sort_values(by=[str(exp_id)],ascending=False)
                tag = tag_df.iloc[0]
                return (
                    tag.name,
                    tag['mz'],
                    tag['RT'],
                    tag['quality'],
                )
                
    def search_consensus_map_by_range(
        self,
        mz_range: Tuple[float,float],
        RT_range: Tuple[float,float],
        exp_id: Hashable,
    ):
        if self.has_consensus_map:
            mz_mask = (self.ConsensusMapDataFrame['mz'] >= mz_range[0]) & (self.ConsensusMapDataFrame['mz'] <= mz_range[1])
            rt_mask = (self.ConsensusMapDataFrame['RT'] >= RT_range[0]) & (self.ConsensusMapDataFrame['RT'] <= RT_range[1])
            intens_mask = self.ConsensusMapDataFrame[str(exp_id)] > 0
            mask = pd.Series(mz_mask,index=self.ConsensusMapDataFrame.index) & pd.Series(rt_mask,index=self.ConsensusMapDataFrame.index) & intens_mask
            if mask.any():
                tag_df = self.ConsensusMapDataFrame[mask]
                tag_df = tag_df.sort_values(by=[str(exp_id)],ascending=False)
                tag = tag_df.iloc[0]
                return (
                    tag.name,
                    tag['mz'],
                    tag['RT'],
                    tag['quality'],
                )
    
    def FeatureMap_len(self, exp_id: Hashable) -> Optional[int]:
        feature_map = self.FEATURE_MAPS.get(exp_id, None)
        if feature_map is not None:
            return feature_map.size()
        else:
            return None
    
    def Feature(self, access_path: Tuple[Hashable, int]) -> Optional[oms.Feature]:
        map_id, feature_id = access_path
        feature_map = self.FEATURE_MAPS.get(map_id, None)
        if feature_map is not None:
            return feature_map[feature_id]
        else:
            return None
    
    @AsyncBase.use_coroutine
    def Feature_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.Feature(access_path)
    
    def Features(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, oms.Feature]]:
        return self.run_coroutine(self.Feature_coroutine, access_path)
        
    def __getitem__(self, access_path: Tuple[Hashable, int]) -> Optional[oms.Feature]:
        map_id, feature_id = access_path
        feature_map = self.FEATURE_MAPS[map_id]
        return feature_map[feature_id]
    
    def FeatureID(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getUniqueId()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureID_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureID(access_path)
    
    def FeatureIDs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, int]]:
        return self.run_coroutine(self.FeatureID_coroutine, access_path)
    
    def ClusterID(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        exp_id, feature_id = access_path
        feature_id = self.FeatureID(access_path)
        if feature_id is not None:
            return "{}:CLUST[{}]".format(exp_id, feature_id)
        else:
            return None
        
    @AsyncBase.use_coroutine
    def ClusterID_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.ClusterID(access_path)
    
    def ClusterIDs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, str]]:
        return self.run_coroutine(self.ClusterID_coroutine, access_path)
    
    def FeatureCharge(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getCharge()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureCharge_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureCharge(access_path)
    
    def FeatureCharges(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, int]]:
        return self.run_coroutine(self.FeatureCharge_coroutine, access_path)
        
    def FeatureRT(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getRT()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureRT_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureRT(access_path)
    
    def FeatureRTs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureRT_coroutine, access_path)
        
    def FeatureMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMZ()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMZ(access_path)
    
    def FeatureMZs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureMZ_coroutine, access_path)
        
    def FeatureIntensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getIntensity()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureIntensity_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureIntensity(access_path)
    
    def FeatureIntensitys(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureIntensity_coroutine, access_path)
        
    def FeatureQuality(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        exp_id, feature_id = access_path
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getQuality(feature_id)
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureQuality_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureQuality(access_path)
    
    def FeatureQualitys(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureQuality_coroutine, access_path)
        
    def FeatureConvexHull(self, access_path: Tuple[Hashable, int]) -> Optional[PeakConvexHull]:
        feature = self.Feature(access_path)
        if feature is not None:
            merged_hull = feature.getConvexHull()
            sub_hulls = feature.getConvexHulls()
            return PeakConvexHull(merged_hull, sub_hulls)
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureConvexHull_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureConvexHull(access_path)
    
    def FeatureConvexHulls(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, PeakConvexHull]]:
        return self.run_coroutine(self.FeatureConvexHull_coroutine, access_path)
    
    def ClusterMergedRange(self, access_path: Tuple[Hashable, int]) -> Tuple[Tuple[float,float],Tuple[float,float]]:
        feature = self.Feature(access_path)
        if feature is not None:
            hull = feature.getConvexHull()
            if hull is not None:
                return hull.getBoundingBox2D()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def ClusterMergedRange_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.ClusterMergedRange(access_path)
    
    def ClusterMergedRanges(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Tuple[Tuple[float,float],Tuple[float,float]]]]:
        return self.run_coroutine(self.ClusterMergedRange_coroutine, access_path)
    
    def ClusterMainRange(self, access_path: Tuple[Hashable, int]) -> Tuple[Tuple[float,float],Tuple[float,float]]:
        feature = self.Feature(access_path)
        if feature is not None:
            hulls = feature.getConvexHulls()
            if len(hulls) > 0:
                return hulls[0].getBoundingBox2D()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def ClusterMainRange_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.ClusterMainRange(access_path)
    
    def ClusterMainRanges(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, Tuple[Tuple[float,float],Tuple[float,float]]]]:
        return self.run_coroutine(self.ClusterMainRange_coroutine, access_path)
    
    def FeatureLabel(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('label')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureLabel_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureLabel(access_path)
    
    def FeatureLabels(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, str]]:
        return self.run_coroutine(self.FeatureLabel_coroutine, access_path)
        
    def FeatureFWHM(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('FWHM')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureFWHM_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureFWHM(access_path)
    
    def FeatureFWHMs(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureFWHM_coroutine, access_path)
        
    def FeatureMaxHeight(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('max_height')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMaxHeight_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMaxHeight(access_path)

    def FeatureMaxHeights(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureMaxHeight_coroutine, access_path)

    def FeatureNumOfMasstraces(self, access_path: Tuple[Hashable, int]) -> Optional[int]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('num_of_masstraces')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureNumOfMasstraces_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureNumOfMasstraces(access_path)

    def FeatureNumOfMasstracesList(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, int]]:
        return self.run_coroutine(self.FeatureNumOfMasstraces_coroutine, access_path)

    def FeatureMasstraceIntensity(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('masstrace_intensity')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMasstraceIntensity_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMasstraceIntensity(access_path)

    def FeatureMasstraceIntensities(
        self, 
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureMasstraceIntensity_coroutine, access_path)

    def FeatureMasstraceCentroidRT(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('masstrace_centroid_rt')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMasstraceCentroidRT_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMasstraceCentroidRT(access_path)

    def FeatureMasstraceCentroidRTs(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureMasstraceCentroidRT_coroutine, access_paths)

    def FeatureMasstraceCentroidMZ(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('masstrace_centroid_mz')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMasstraceCentroidMZ_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMasstraceCentroidMZ(access_path)

    def FeatureMasstraceCentroidMZs(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureMasstraceCentroidMZ_coroutine, access_paths)

    def FeatureIsotopeDistances(self, access_path: Tuple[Hashable, int]) -> Optional[List[float]]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('isotope_distances')
        else:
            return None

    @AsyncBase.use_coroutine
    def FeatureIsotopeDistances_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureIsotopeDistances(access_path)

    def FeatureIsotopeDistancesList(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, List[float]]]:
        return self.run_coroutine(self.FeatureIsotopeDistances_coroutine, access_paths)

    def FeatureLegalIsotopePattern(self, access_path: Tuple[Hashable, int]) -> Optional[List[str]]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('legal_isotope_pattern')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureLegalIsotopePattern_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureLegalIsotopePattern(access_path)

    def FeatureLegalIsotopePatterns(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, List[str]]]:
        return self.run_coroutine(self.FeatureLegalIsotopePattern_coroutine, access_paths)

    def FeatureDCChargeAdducts(self, access_path: Tuple[Hashable, int]) -> Optional[List[str]]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('dc_charge_adducts')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureDCChargeAdducts_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureDCChargeAdducts(access_path)

    def FeatureDCChargeAdductsList(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, List[str]]]:
        return self.run_coroutine(self.FeatureDCChargeAdducts_coroutine, access_paths)

    def FeatureDCChargeAdductMass(self, access_path: Tuple[Hashable, int]) -> Optional[float]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('dc_charge_adduct_mass')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureDCChargeAdductMass_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureDCChargeAdductMass(access_path)

    def FeatureDCChargeAdductMasses(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, float]]:
        return self.run_coroutine(self.FeatureDCChargeAdductMass_coroutine, access_paths)

    def FeatureGroup(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('Group')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureGroup_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureGroup(access_path)

    def FeatureGroups(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, str]]:
        return self.run_coroutine(self.FeatureGroup_coroutine, access_paths)

    def FeatureIsUngroupedWithCharge(self, access_path: Tuple[Hashable, int]) -> Optional[bool]:
        feature = self.Feature(access_path)
        if feature is not None:
            return feature.getMetaValue('is_ungrouped_with_charge')
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureIsUngroupedWithCharge_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureIsUngroupedWithCharge(access_path)

    def FeatureIsUngroupedWithCharges(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, bool]]:
        return self.run_coroutine(self.FeatureIsUngroupedWithCharge_coroutine, access_paths)
    
    def ClusterDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Dict[
        Literal[
            'EXP_ID','cluster_Index','cluster_ID',
            'cluster_RT','cluster_MZ','cluster_intens','cluster_quality',
            'adduct','adduct_mz','charge',
            'legal_isotope_pattern','isotope_distances',
            'convex_hull','FWHM','vertex_height',
            'masstraces_num','masstrace_intensities',
            'masstrace_centroid_rt','masstrace_centroid_mz',
            'consensus_group','consensus_RT','consensus_MZ','consensus_quality'
        ],Union[
            float,int,str,List[float],List[int],Dict[
                                            Literal[
                                                'RT_start',
                                                'RT_end',
                                                'MZ_start',
                                                'MZ_end',
                                                'points_array',
                                            ],Union[float,np.ndarray]
                                        ]]
    ]:
        feature = self.Feature(access_path)
        if feature is None:
            return {}
        exp_id, feature_id = access_path
        cluster_RT = self.FeatureRT(access_path)
        cluster_MZ = self.FeatureMZ(access_path)
        cluster_intens = self.FeatureIntensity(access_path)
        cluster_quality = self.FeatureQuality(access_path)
        adduct = self.FeatureDCChargeAdducts(access_path)
        adduct_mz = self.FeatureDCChargeAdductMass(access_path)
        charge = self.FeatureCharge(access_path)
        legal_isotope_pattern = self.FeatureLegalIsotopePattern(access_path)
        isotope_distances = self.FeatureIsotopeDistances(access_path)
        convex_hull = self.FeatureConvexHull(access_path).to_dict()
        FWHM = self.FeatureFWHM(access_path)
        vertex_height = self.FeatureMaxHeight(access_path)
        masstraces_num = self.FeatureNumOfMasstraces(access_path)
        masstrace_intensities = self.FeatureMasstraceIntensity(access_path)
        masstrace_centroid_rt = self.FeatureMasstraceCentroidRT(access_path)
        masstrace_centroid_mz = self.FeatureMasstraceCentroidMZ(access_path)
        consensus_tuple = self.search_consensus_map(
            cluster_MZ,cluster_RT,exp_id,
        )
        if consensus_tuple is not None:
            consensus_group, consensus_MZ, consensus_RT, consensus_quality = consensus_tuple
        else:
            consensus_group, consensus_MZ, consensus_RT, consensus_quality = -1, -1, -1, -1
        cluster_dict = {
            'EXP_ID':exp_id,
            'cluster_Index':feature_id,
            'cluster_ID':'CLUST[{}]'.format(self.FeatureID(access_path)),
            'cluster_RT':cluster_RT,
            'cluster_MZ':cluster_MZ,
            'cluster_intens':cluster_intens,
            'cluster_quality':cluster_quality,
            'adduct':adduct,
            'adduct_mz':adduct_mz,
            'charge':charge,
            'legal_isotope_pattern':legal_isotope_pattern,
            'isotope_distances':isotope_distances,
            'convex_hull':convex_hull,
            'FWHM':FWHM,
            'vertex_height':vertex_height,
            'masstraces_num':masstraces_num,
            'masstrace_intensities':masstrace_intensities,
            'masstrace_centroid_rt':masstrace_centroid_rt,
            'masstrace_centroid_mz':masstrace_centroid_mz,
            'consensus_group':consensus_group,
            'consensus_RT':consensus_RT,
            'consensus_MZ':consensus_MZ,
            'consensus_quality':consensus_quality,
        }
        return cluster_dict
    
    @AsyncBase.use_coroutine
    def ClusterDict_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.ClusterDict(access_path)

    def ClustersDict(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict]]:
        return self.run_coroutine(self.ClusterDict_coroutine, access_paths)
    
    def ClusterDataFrame(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Optional[pd.DataFrame]:
        clusters_df = {}
        clusters_dict = self.ClustersDict(access_paths)
        for exp_id, map_dict in clusters_dict.items():
            for feature_index, cluster_dict in map_dict.items():
                clusters_df["{}:{}".format(exp_id,cluster_dict['cluster_ID'])] = cluster_dict
        clusters_df = pd.DataFrame(clusters_df).transpose().sort_values(by=['EXP_ID','cluster_Index'])
        clusters_df.index.name = 'cluster_id'
        return clusters_df
    
class OpenMSDataWrapper():
    
    def __init__(
        self,
        exps: Union[oms.MSExperiment, List[oms.MSExperiment], Dict[str, oms.MSExperiment]],
        feature_maps: Union[oms.FeatureMap,List[oms.FeatureMap], Dict[str, oms.FeatureMap],None] = None,
        consensus_map: Optional[oms.ConsensusMap] = None,
        precursor_search_tolerance: Tuple[float,float] = (5.0,5.0),
        precursor_search_tolerance_type: Literal['ppm','Da'] = 'ppm',
        consensus_threshold: float = 0.0,
        mz_tolerance: float = 3.0,
        rt_tolerance: float = 6.0,
        mz_tolerance_type: Literal['ppm','Da'] = 'ppm',
    ) -> None:
        self.precursor_search_tl, self.precursor_search_tr = precursor_search_tolerance
        self.precursor_search_tolerance_type = precursor_search_tolerance_type
        self.exps = MSExperiments(
            exps,
            precursor_search_tolerance,
            precursor_search_tolerance_type,
        )
        if feature_maps is not None:
            self.feature_maps = FeatureMaps(
                feature_maps,consensus_map,
                consensus_threshold,
                mz_tolerance,
                rt_tolerance,
                mz_tolerance_type,
            )
        else:
            self.feature_maps = None
        
    @property
    def EXPS(self) -> MSExperiments:
        return self.exps
    
    @property
    def FEATURE_MAPS(self) -> FeatureMaps:
        return self.feature_maps
    
    @property
    def CONSENSUS_MAP(self) -> Optional[oms.ConsensusMap]:
        return self.FEATURE_MAPS.CONSENSUS_MAP
    
    @property
    def has_feature_maps(self) -> bool:
        if self.FEATURE_MAPS is None:
            return False
        else:
            return True
    
    @property
    def has_consensus_map(self) -> bool:
        if self.has_feature_maps:
            return self.FEATURE_MAPS.has_consensus_map
        return False
    
    def SpecDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Dict[
        Literal[
            'EXP_ID','SPEC_INDEX','ID',
            'RT','mz','intens',
            'basepeak_mz','basepeak_intens','peak_counts',
            'instrument_type','polarity','peak_type','ionization_method',
            'scan_mode','ms_level','cleavage_info','scan_range',
            'PI','PI_intensity','charge','collision_energy',
            'cluster_ID','consensus_group',
        ],Union[str,float,int,np.ndarray]
    ]:
        exp_id, spec_index = access_path
        spec_dict = self.EXPS.SpecDict(access_path)
        if spec_dict['ms_level'] != 'ms1':
            if self.has_feature_maps:
                cluster_access_path = self.FEATURE_MAPS.search_cluster(
                    exp_id=exp_id,mz=spec_dict['PI'],RT=spec_dict['RT'],
                )
                if cluster_access_path is not None:
                    spec_dict['cluster_ID'] = self.FEATURE_MAPS.ClusterID(cluster_access_path)
                    if self.has_consensus_map:
                        main_range = self.FEATURE_MAPS.ClusterMainRange(cluster_access_path)
                        if main_range is not None:
                            (cluster_RT_start,cluster_MZ_start),(cluster_RT_end,cluster_MZ_end) = main_range
                            consensus_tuple = self.FEATURE_MAPS.search_consensus_map_by_range(
                                (cluster_MZ_start,cluster_MZ_end),
                                (cluster_RT_start,cluster_RT_end),
                                exp_id,
                            )
                            if consensus_tuple is not None:
                                spec_dict['consensus_group'] = consensus_tuple[0]
                            else:
                                spec_dict['consensus_group'] = -1
        return spec_dict
    
    @AsyncBase.use_coroutine
    def SpecDict_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.SpecDict(access_path)

    def SpecsDict(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict]]:
        return self.EXPS.run_coroutine(self.SpecDict_coroutine, access_paths)
    
    def SpecsDataFrame(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> pd.DataFrame:
        return MSExperiments.SpecsDataFrame(self,access_path)
    
    def SpecsDataFrames(
        self,
        access_path: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable],Hashable,None], 
                Union[List[int],int,None]
            ],
            Hashable,
            None,
        ] = None,
    ) -> Dict[int, pd.DataFrame]:
        return MSExperiments.SpecsDataFrames(self,access_path)
    
    def ClusterDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Dict[
        Literal[
            'EXP_ID','cluster_Index','cluster_ID',
            'cluster_RT','cluster_MZ','cluster_intens','cluster_quality',
            'adduct','adduct_mz','charge',
            'legal_isotope_pattern','isotope_distances',
            'convex_hull','FWHM','vertex_height',
            'masstraces_num','masstrace_intensities',
            'masstrace_centroid_rt','masstrace_centroid_mz',
            'consensus_group','consensus_RT','consensus_MZ','consensus_quality',
            'SubordinateSpecIDs',
        ],Union[
            float,int,str,List[float],List[int],Dict[
                                            Literal[
                                                'RT_start',
                                                'RT_end',
                                                'MZ_start',
                                                'MZ_end',
                                                'points_array',
                                            ],Union[float,np.ndarray]
                                        ],List[str]]
    ]:
        exp_id, cluster_index = access_path
        cluster_dict = self.FEATURE_MAPS.ClusterDict(access_path)
        access_path_list = self.EXPS.search_ms2_by_range(
            exp_id,
            (cluster_dict['convex_hull']['main']['MZ_start'],cluster_dict['convex_hull']['main']['MZ_end']),
            (cluster_dict['convex_hull']['main']['RT_start'],cluster_dict['convex_hull']['main']['RT_end']),
        )
        if access_path_list is not None:
            SubordinateSpecAccessPaths = [
                access_path for access_path in access_path_list 
                if access_path[0] == exp_id
            ]
            cluster_dict['SubordinateSpecIDs'] = [
                    self.EXPS.SPEC_UID(tag_access_path) 
                    for tag_access_path in SubordinateSpecAccessPaths
            ]
        return cluster_dict
    
    @AsyncBase.use_coroutine
    def ClusterDict_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.ClusterDict(access_path)

    def ClustersDict(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Dict[Hashable, Dict[int, Dict]]:
        return self.FEATURE_MAPS.run_coroutine(self.ClusterDict_coroutine, access_paths)
    
    def ClusterDataFrame(
        self, 
        access_paths: Union[
            List[Tuple[Hashable, int]],
            Tuple[
                Union[List[Hashable], Hashable, None], 
                Union[List[int], int, None]
            ],
            Hashable,
            None
        ] = None,
    ) -> Optional[pd.DataFrame]:
        return FeatureMaps.ClusterDataFrame(self,access_paths)
    
    def to_dataframes(self) -> Dict[int, pd.DataFrame]:
        df_dict = self.SpecsDataFrames()
        df_dict[0] = self.ClusterDataFrame()
        return df_dict

if __name__ == '__main__':
    
    #使用示例
    file_paths = [
        '/mnt/data/daiql/MetaEngine/demo/test_exp_data/Metabolomics_1.mzML',
        '/mnt/data/daiql/MetaEngine/demo/test_exp_data/Metabolomics_2.mzML',
    ]
    ref_path = '/mnt/data/daiql/MetaEngine/demo/test_exp_data/Metabolomics_1.mzML'
    exps,path_map = load_exp_ms_files(file_paths,'md5')
    key_map = inverse_dict(path_map)
    mass_traces = detect_mass_traces(exps)
    mass_traces_deconvol = elution_peak_detection(mass_traces)
    feature_maps, chrom_outs = mapping_features(mass_traces_deconvol)
    feature_maps, groups, edges = detect_adduct(feature_maps)
    aligned_exps, trafos = align(exps,feature_maps,key_map[ref_path])
    feature_links = link_features(feature_maps)
    ms_wrapper = OpenMSDataWrapper(
        exps,feature_maps,feature_links
    )
    ms_datas = ms_wrapper.to_dataframes()
    print('done.')