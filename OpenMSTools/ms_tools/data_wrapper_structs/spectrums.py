from __future__ import annotations
from pydantic import BaseModel, ConfigDict
import pyopenms as oms
import rtree
import pandas as pd
import numpy as np
import dask
import dask.bag as db
import re
from re import Pattern
import bisect
from typing import Optional, Union, List, Tuple, Dict, Literal, ClassVar

class SpectrumMap(BaseModel):
    
    model_config = ConfigDict({"arbitrary_types_allowed": True})
    
    scan_id_matcher: ClassVar[Pattern] = re.compile(r'scan=(\d+)')
    
    exp_name: str
    ms1_df: Optional[pd.DataFrame] = None
    ms2_df: Optional[pd.DataFrame] = None
    rtree_index: Optional[rtree.index.Index] = None
    exp_meta: Optional[pd.Series] = None
    
    @staticmethod
    def get_exp_meta(exp: oms.MSExperiment) -> pd.Series:
        spec: oms.MSSpectrum = exp[0]
        meta_info_string = spec.getMetaValue("filter string")
        meta_info_list = meta_info_string.split(" ")
        ms_type, ion_mode, ion_source = meta_info_list[0], meta_info_list[1], meta_info_list[3]
        return pd.Series(
            {
                "ms_type": ms_type,
                "ion_mode": ion_mode,
                "ion_source": ion_source,
            }
        )
        
    @staticmethod
    def get_scan_index(spec: oms.MSSpectrum) -> int:
        scan_id_match = SpectrumMap.scan_id_matcher.search(spec.getNativeID())
        if scan_id_match:
            return int(scan_id_match.group(1))
        else:
            raise ValueError(f"Cannot extract scan index from spectrum native ID: {spec.getNativeID()}")
    
    @staticmethod
    def ms2spec2dfdict(spec: oms.MSSpectrum) -> Dict[
        Literal[
            "spec_id",
            "rt",
            "precursor_mz",
            "base_peak_mz",
            "base_peak_intensity",
            "mz_array",
            "intensity_array",
        ],
        Union[int, float, np.ndarray]
    ]:
        spec_id = SpectrumMap.get_scan_index(spec)
        rt = spec.getRT()
        precursor_mz = spec.getPrecursors()[0].getMZ()
        base_peak_mz = spec.getMetaValue("base peak m/z")
        base_peak_intensity = spec.getMetaValue("base peak intensity")
        mz_array, intensity_array = spec.get_peaks()
        return {
            "spec_id": spec_id,
            "rt": rt,
            "precursor_mz": precursor_mz,
            "base_peak_mz": base_peak_mz,
            "base_peak_intensity": base_peak_intensity,
            "mz_array": mz_array,
            "intensity_array": intensity_array,
        }
        
    @staticmethod
    def ms1spec2dfdict(spec: oms.MSSpectrum) -> Dict[
        Literal[
            "spec_id",
            "rt",
            "mz_array",
            "intensity_array",
        ],
        Union[int, float, np.ndarray]
    ]:
        spec_id = SpectrumMap.get_scan_index(spec)
        rt = spec.getRT()
        mz_array, intensity_array = spec.get_peaks()
        return {
            "spec_id": spec_id,
            "rt": rt,
            "mz_array": mz_array,
            "intensity_array": intensity_array,
        }
        
    def insert_ms1_id_to_ms2(self) -> None:
        '''
        如果MS2谱图没有对应的MS1谱图ID，则插入-1
        '''
        if self.ms1_df is None or self.ms2_df is None:
            raise ValueError("MS1 and MS2 dataframes must be loaded before inserting MS1 IDs to MS2 dataframe")
        self.ms2_df['ms1_id'] = self.ms2_df.index.map(lambda x: bisect.bisect_left(self.ms1_df.index.values,x) - 1).tolist()
        self.ms2_df['ms1_id'] = self.ms2_df['ms1_id'].map(lambda x: self.ms1_df.index.values[x] if x >= 0 else -1)
        
    def convert_scan_to_spec_id(self) -> None:
        if isinstance(self.ms1_df.index.values[0], np.int64):
            self.ms1_df.index = self.ms1_df.index.map(lambda x: f"{self.exp_name}::ms1::{x}")
        if isinstance(self.ms2_df.index.values[0], np.int64):
            self.ms2_df.index = self.ms2_df.index.map(lambda x: f"{self.exp_name}::ms2::{x}")
        if isinstance(self.ms2_df['ms1_id'].iloc[0], np.int64):
            self.ms2_df['ms1_id'] = self.ms2_df['ms1_id'].map(lambda x: f"{self.exp_name}::ms1::{x}" if x >= 0 else "")
            
    def modify_ms2_rt(self) -> None:
        self.ms2_df['rt'] = self.ms2_df.index.map(lambda x: self.ms1_df.loc[self.ms2_df.loc[x,"ms1_id"],"rt"] if self.ms2_df.loc[x,"ms1_id"] != "" else self.ms2_df.loc[x,"rt"]).tolist()
        
    def init_rtree_index(self) -> None:
        if self.ms2_df is None:
            raise ValueError("MS2 dataframe must be loaded before initializing R-tree index")
        self.rtree_index = rtree.index.Index()
        for i,(spec_id,rt,precursor_mz) in enumerate(zip(self.ms2_df.index,self.ms2_df['rt'],self.ms2_df['precursor_mz'])):
            self.rtree_index.insert(
                id=i,
                coordinates=(precursor_mz, rt, precursor_mz, rt),
                obj=spec_id
            )
            
    def search_ms2_by_range(
        self, 
        coordinates: Tuple[
            float, # min_mz
            float, # min_rt
            float, # max_mz
            float, # max_rt
        ]
    ) -> List[str]:
        if self.rtree_index is None:
            raise ValueError("R-tree index must be initialized before searching MS2 by range")
        return list(self.rtree_index.intersection(coordinates, objects="raw"))
    
    @classmethod
    def from_oms(
        cls, 
        exp: oms.MSExperiment,
        exp_name: str,
    ) -> SpectrumMap:
        spec_bag = db.from_sequence(exp)
        ms1_bag = spec_bag.filter(lambda x: x.getMSLevel() == 1)
        ms2_bag = spec_bag.filter(lambda x: x.getMSLevel() == 2)
        ms1_bag = ms1_bag.map(cls.ms1spec2dfdict)
        ms2_bag = ms2_bag.map(cls.ms2spec2dfdict)
        ms1,ms2 = dask.compute(ms1_bag,ms2_bag,scheduler='threads')
        ms1_df = pd.DataFrame(ms1).set_index('spec_id',inplace=False)
        ms2_df = pd.DataFrame(ms2).set_index('spec_id',inplace=False)
        exp_meta = cls.get_exp_meta(exp)
        spectrum_map = cls(
            exp_name=exp_name,
            ms1_df=ms1_df,
            ms2_df=ms2_df,
            exp_meta=exp_meta,
        )
        spectrum_map.insert_ms1_id_to_ms2()
        spectrum_map.convert_scan_to_spec_id()
        spectrum_map.modify_ms2_rt()
        spectrum_map.init_rtree_index()
        return spectrum_map
        