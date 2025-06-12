from ..utils.async_tools import trio, ProgressManager
from ..utils.base_tools import oms
import numpy as np
from typing import Dict, Tuple, Hashable, Any, Union, List, Optional, Callable

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