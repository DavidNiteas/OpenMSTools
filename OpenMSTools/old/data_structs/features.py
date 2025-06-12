from ..utils.base_tools import oms
from .structs_tools import (
    AsyncBase,ToleranceBase,ProgressManager,
    inverse_dict,unzip_results
)
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Hashable, Union, List, Optional, Callable, Literal

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
        self.init_feature_id_access_map()
        self.init_search_info()
        self.consensus_map = consensus_map
        self.consensus_map_df = None
        self.consensus_threshold = consensus_threshold
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.mz_tolerance_type = mz_tolerance_type
        
    def init_feature_id_access_map(self):
        self.feature_id_access_map = inverse_dict(unzip_results(self.FeatureStringIDs()))
        
    def init_search_info(self):
        self.feature_access: Dict[Hashable, Union[List[Tuple[Hashable, int]]],np.ndarray] = {}
        self.feature_search_hulls: Dict[Hashable, Dict[str, Union[np.ndarray,List[float]]]] = {}
        for exp_id, main_range_dict in self.FeatureMainRanges().items():
            self.feature_search_hulls[exp_id] = {
                'RT_start':[],
                'RT_end':[],
                'MZ_start':[],
                'MZ_end':[],
            }
            self.feature_access[exp_id] = []
            for i, ((RT_start,MZ_start),(RT_end,MZ_end)) in main_range_dict.items():
                self.feature_search_hulls[exp_id]['RT_start'].append(RT_start)
                self.feature_search_hulls[exp_id]['RT_end'].append(RT_end)
                self.feature_search_hulls[exp_id]['MZ_start'].append(MZ_start)
                self.feature_search_hulls[exp_id]['MZ_end'].append(MZ_end)
                self.feature_access[exp_id].append((exp_id,i))
            self.feature_search_hulls[exp_id]['RT_start'] = np.array(self.feature_search_hulls[exp_id]['RT_start'])
            self.feature_search_hulls[exp_id]['RT_end'] = np.array(self.feature_search_hulls[exp_id]['RT_end'])
            self.feature_search_hulls[exp_id]['MZ_start'] = np.array(self.feature_search_hulls[exp_id]['MZ_start'])
            self.feature_search_hulls[exp_id]['MZ_end'] = np.array(self.feature_search_hulls[exp_id]['MZ_end'])
            self.feature_access[exp_id] = np.array(self.feature_access[exp_id],dtype=object)
    
    @property
    def FEATURE_MAPS(self) -> Dict[Hashable, oms.FeatureMap]:
        return self.feature_maps
    
    @property
    def FEATURE_ID_ACCESS_MAP(self) -> Dict[Tuple[Hashable, int], int]:
        return self.feature_id_access_map
    
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
    def FeatureAccess(self) -> Dict[Hashable,np.ndarray]:
        return self.feature_access
    
    @property
    def FeatureSearchHulls(self) -> Dict[Hashable, Dict[str, np.ndarray]]:
        return self.feature_search_hulls
        
    def search_feature(
        self,exp_id:Hashable,mz:float,RT:float,
    ) -> Optional[Tuple[Hashable,int]]:
        RT_mask = ((self.FeatureSearchHulls[exp_id]['RT_start'] - 0.0001) <= RT) & (self.FeatureSearchHulls[exp_id]['RT_end'] + 0.0001 >= RT)
        MZ_mask = ((self.FeatureSearchHulls[exp_id]['MZ_start'] - 0.0001) <= mz) & (self.FeatureSearchHulls[exp_id]['MZ_end'] + 0.0001 >= mz)
        mask = RT_mask & MZ_mask
        if mask.any():
            tag_index = np.argwhere(mask).reshape(-1)
            tag_index = tag_index[0]
            access_path = self.FeatureAccess[exp_id][tag_index]
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
    
    def FeatureStringID(self, access_path: Tuple[Hashable, int]) -> Optional[str]:
        exp_id, feature_id = access_path
        feature_id = self.FeatureID(access_path)
        if feature_id is not None:
            return "{}:Feature[{}]".format(exp_id, feature_id)
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureStringID_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureStringID(access_path)
    
    def FeatureStringIDs(
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
        return self.run_coroutine(self.FeatureStringID_coroutine, access_path)
    
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
    
    def FeatureMergedRange(self, access_path: Tuple[Hashable, int]) -> Tuple[Tuple[float,float],Tuple[float,float]]:
        feature = self.Feature(access_path)
        if feature is not None:
            hull = feature.getConvexHull()
            if hull is not None:
                return hull.getBoundingBox2D()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMergedRange_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMergedRange(access_path)
    
    def FeatureMergedRanges(
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
        return self.run_coroutine(self.FeatureMergedRange_coroutine, access_path)
    
    def FeatureMainRange(self, access_path: Tuple[Hashable, int]) -> Tuple[Tuple[float,float],Tuple[float,float]]:
        feature = self.Feature(access_path)
        if feature is not None:
            hulls = feature.getConvexHulls()
            if len(hulls) > 0:
                return hulls[0].getBoundingBox2D()
        else:
            return None
        
    @AsyncBase.use_coroutine
    def FeatureMainRange_coroutine(self, access_path: Tuple[Hashable, int]) -> None:
        return self.FeatureMainRange(access_path)
    
    def FeatureMainRanges(
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
        return self.run_coroutine(self.FeatureMainRange_coroutine, access_path)
    
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
        feature_RT = self.FeatureRT(access_path)
        feature_MZ = self.FeatureMZ(access_path)
        feature_intens = self.FeatureIntensity(access_path)
        feature_quality = self.FeatureQuality(access_path)
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
            feature_MZ,feature_RT,exp_id,
        )
        if consensus_tuple is not None:
            consensus_group, consensus_MZ, consensus_RT, consensus_quality = consensus_tuple
        else:
            consensus_group, consensus_MZ, consensus_RT, consensus_quality = -1, -1, -1, -1
        feature_dict = {
            'EXP_ID':exp_id,
            'Feature_Index':feature_id,
            'openms_feature_id':self.FeatureID(access_path),
            'feature_RT':feature_RT,
            'feature_MZ':feature_MZ,
            'feature_intens':feature_intens,
            'feature_quality':feature_quality,
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
        return self.run_coroutine(self.FeatureDict_coroutine, access_paths)
    
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
        feature_df = {}
        features_dict = self.FeaturesDict(access_paths)
        for exp_id, map_dict in features_dict.items():
            for feature_index, feature_dict in map_dict.items():
                feature_df["{}:Feature[{}]".format(exp_id,feature_dict['openms_feature_id'])] = feature_dict
        feature_df = pd.DataFrame(feature_df).transpose().sort_values(by=['EXP_ID','Feature_Index'])
        feature_df.index.name = 'feature_uid'
        return feature_df