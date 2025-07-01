import hashlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import pyopenms as oms
from rich.progress import track

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

def calculate_mzfile_hash(file_path:str,hash_type:str = "md5"):
    hash_object = hashlib.new(hash_type)
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()

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
