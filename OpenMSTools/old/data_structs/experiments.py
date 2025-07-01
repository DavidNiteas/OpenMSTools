from collections.abc import Hashable
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.base_tools import oms
from .structs_tools import (
    AsyncBase,
    ProgressManager,
    ToleranceBase,
    id_dict_to_str,
    inverse_dict,
    ppm2da,
    unzip_results,
)


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
            return f"{exp_id}:{spec_id}"
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
            precursor_dict['PI_SpecID'] = f"{exp_id}:{PI_SpecID}"
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
        specs_df.index.name = 'spec_uid'
        return specs_df

    @staticmethod
    def split_specs_dataframe_by_mslevel(specs_df:pd.DataFrame) -> Dict[int,pd.DataFrame]:
        ms_level_dfs = {}
        for i,ms_level_df in specs_df.groupby('ms_level'):
            ms_level = int(ms_level_df['ms_level'].iloc[0].strip('ms'))
            ms_level_dfs[ms_level] = ms_level_df.dropna(axis=1, how='all').drop(columns=['ms_level'],inplace=False)
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
