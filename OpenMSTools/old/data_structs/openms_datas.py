from .experiments import MSExperiments
from .features import FeatureMaps
from ..utils.base_tools import oms
from .structs_tools import AsyncBase
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Hashable, Union, List, Optional, Literal

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
            'feature_id','consensus_group',
        ],Union[str,float,int,np.ndarray]
    ]:
        exp_id, spec_index = access_path
        spec_dict = self.EXPS.SpecDict(access_path)
        if spec_dict['ms_level'] != 'ms1':
            if self.has_feature_maps:
                feature_access_path = self.FEATURE_MAPS.search_feature(
                    exp_id=exp_id,mz=spec_dict['PI'],RT=spec_dict['RT'],
                )
                if feature_access_path is not None:
                    spec_dict['feature_id'] = self.FEATURE_MAPS.FeatureStringID(feature_access_path)
                    if self.has_consensus_map:
                        main_range = self.FEATURE_MAPS.FeatureMainRange(feature_access_path)
                        if main_range is not None:
                            (feature_RT_start,feature_MZ_start),(feature_RT_end,feature_MZ_end) = main_range
                            consensus_tuple = self.FEATURE_MAPS.search_consensus_map_by_range(
                                (feature_MZ_start,feature_MZ_end),
                                (feature_RT_start,feature_RT_end),
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
    
    def FeatureDict(
        self, 
        access_path: Tuple[Hashable, int],
    ) -> Dict[
        Literal[
            'EXP_ID','Feature_Index','feature_id',
            'feature_RT','feature_MZ','feature_intens','feature_quality',
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
        exp_id, feature_index = access_path
        feature_dict = self.FEATURE_MAPS.FeatureDict(access_path)
        access_path_list = self.EXPS.search_ms2_by_range(
            exp_id,
            (feature_dict['convex_hull']['main']['MZ_start'],feature_dict['convex_hull']['main']['MZ_end']),
            (feature_dict['convex_hull']['main']['RT_start'],feature_dict['convex_hull']['main']['RT_end']),
        )
        if access_path_list is not None:
            SubordinateSpecAccessPaths = [
                access_path for access_path in access_path_list 
                if access_path[0] == exp_id
            ]
            feature_dict['SubordinateSpecIDs'] = [
                    self.EXPS.SPEC_UID(tag_access_path) 
                    for tag_access_path in SubordinateSpecAccessPaths
            ]
        return feature_dict
    
    @AsyncBase.use_coroutine
    def FeatureDict_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureDict(access_path)

    def FeaturesDict(
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
        return self.FEATURE_MAPS.run_coroutine(self.FeatureDict_coroutine, access_paths)
    
    def FeaturesDataFrame(
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
        return FeatureMaps.FeaturesDataFrame(self,access_paths)
    
    def to_dataframes(self) -> Dict[int, pd.DataFrame]:
        df_dict = self.SpecsDataFrames()
        df_dict[0] = self.FeaturesDataFrame()
        return df_dict