from __future__ import annotations

import copy
import json
import re
from re import Pattern
from typing import ClassVar, Literal

import dask
import dask.bag as db
import numpy as np
import pandas as pd
import polars as pl
import pyopenms as oms
import rtree
from pydantic import Field

from .ABCs import BaseMap


class SpectrumMap(BaseMap):

    scan_id_matcher: ClassVar[Pattern] = re.compile(r'scan=(\d+)')

    ms1_index: pd.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="csv",
        description="MS1谱图的索引"
    )
    ms1_df: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="MS1谱图的DataFrame，基于polars"
    )
    ms2_index: pd.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="csv",
        description="MS2谱图的索引"
    )
    ms2_df: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="MS2谱图的DataFrame，基于polars"
    )
    ms2_rtree_index: rtree.index.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="rtree",
        build_func='build_ms2_rtree_index',
        init_func='init_ms2_rtree_index',
        description="基于R-Tree的MS2索引，可以基于mz和rt范围快速索引MS2数据。\
                    可以通过`init_ms2_rtree_index`方法从ms2_df中初始化。",
    )

    @staticmethod
    def get_exp_meta(exp: oms.MSExperiment) -> dict[str, str]:
        spec: oms.MSSpectrum = exp[0]
        meta_info_string = spec.getMetaValue("filter string")
        meta_info_list = meta_info_string.split(" ")
        ms_type = meta_info_list[0]
        ion_mode = meta_info_list[1]
        ion_source = meta_info_list[3]
        return {
            "ms_type": ms_type,
            "ion_mode": ion_mode,
            "ion_source": ion_source,
        }

    @staticmethod
    def get_scan_index(spec: oms.MSSpectrum) -> int:
        scan_id_match = SpectrumMap.scan_id_matcher.search(spec.getNativeID())
        if scan_id_match:
            return int(scan_id_match.group(1))
        else:
            raise ValueError(
                f"Cannot extract scan index from \
                spectrum native ID: {spec.getNativeID()}"
            )

    @staticmethod
    def ms2spec2dfdict(spec: oms.MSSpectrum) -> dict[
        Literal[
            "spec_id",
            "rt",
            "precursor_mz",
            "base_peak_mz",
            "base_peak_intensity",
            "mz_array",
            "intensity_array",
        ],
        int | float | np.ndarray
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
            "mz_array": mz_array.astype(np.float32),
            "intensity_array": intensity_array.astype(np.float32),
        }

    @staticmethod
    def ms1spec2dfdict(spec: oms.MSSpectrum) -> dict[
        Literal[
            "spec_id",
            "rt",
            "mz_array",
            "intensity_array",
        ],
        int | float | np.ndarray
    ]:
        spec_id = SpectrumMap.get_scan_index(spec)
        rt = spec.getRT()
        mz_array, intensity_array = spec.get_peaks()
        return {
            "spec_id": spec_id,
            "rt": rt,
            "mz_array": mz_array.astype(np.float32),
            "intensity_array": intensity_array.astype(np.float32),
        }

    def insert_ms1_id_to_ms2(self) -> None:
        '''
        如果MS2谱图没有对应的MS1谱图ID，则插入null
        '''
        if self.ms1_df is None or self.ms2_df is None:
            raise ValueError(
                "MS1 and MS2 dataframes must be loaded \
                    before inserting MS1 IDs to MS2 dataframe"
            )
        ms1_df_mapping = self.ms1_df.with_columns(
            pl.col('spec_id').alias('ms1_id')
        ).select(['spec_id','ms1_id'])
        self.ms2_df = self.ms2_df.join_asof(
            ms1_df_mapping,
            left_on='spec_id',
            right_on='spec_id',
            strategy='backward'
        )

    def convert_scan_to_spec_id(self) -> None:
        if self.ms1_df.schema['spec_id'] in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
        ):
            self.ms1_df = self.ms1_df.with_columns(
                (f"{self.exp_name}::ms1::" + self.ms1_df['spec_id'].cast(str)).alias('spec_id')
            )
            self.ms1_index = pd.Index(self.ms1_df['spec_id'].to_list())
        if self.ms2_df.schema['spec_id'] in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
        ):
            self.ms2_df = self.ms2_df.with_columns(
                (f"{self.exp_name}::ms2::" + self.ms2_df['spec_id'].cast(str)).alias('spec_id')
            )
            self.ms2_index = pd.Index(self.ms2_df['spec_id'].to_list())
        if self.ms2_df.schema['ms1_id'] in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
        ):
            self.ms2_df = self.ms2_df.with_columns(
                (f"{self.exp_name}::ms1::" + self.ms2_df['ms1_id'].cast(str)).alias('ms1_id')
            )

    def modify_ms2_rt(self) -> None:
        ms1_rt_df = self.ms1_df.select(['spec_id', 'rt']).rename({'rt': 'ms1_rt','spec_id':'ms1_id'})
        joined_df = self.ms2_df.join(ms1_rt_df, on='ms1_id', how='left')
        self.ms2_df = joined_df.with_columns(
            pl.when(pl.col('ms1_id').is_not_null())
            .then(pl.col('ms1_rt'))
            .otherwise(pl.col('rt'))
            .alias('rt')
        ).drop('ms1_rt')

    def build_ms2_rtree_index(self, path : str | None = None) -> rtree.index.Index:
        if self.ms2_df is None:
            raise ValueError(
                "MS2 dataframe must be loaded before initializing R-tree index"
            )
        ms2_rtree_index = rtree.index.Index(path)
        for i,(rt,precursor_mz) in enumerate(
            zip(
                self.ms2_df['rt'],
                self.ms2_df['precursor_mz'],
            )
        ):
            ms2_rtree_index.insert(
                id=i,
                coordinates=(precursor_mz, rt, precursor_mz, rt),
                obj=i
            )
        return ms2_rtree_index

    def init_ms2_rtree_index(self):
        self.ms2_rtree_index = self.build_ms2_rtree_index()

    def search_ms2_by_range(
        self,
        coordinates: tuple[
            float, # min_mz
            float, # min_rt
            float, # max_mz
            float, # max_rt
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        if self.ms2_rtree_index is None:
            self.init_ms2_rtree_index()
        index = list(self.ms2_rtree_index.intersection(coordinates, objects="raw"))
        if return_type == "id":
            return self.ms2_index[index].tolist()
        elif return_type == "df":
            return self.ms2_df[index]
        else:
            return index

    @classmethod
    def from_oms(
        cls,
        exp: oms.MSExperiment,
        exp_name: str,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> SpectrumMap:
        spec_bag = db.from_sequence(exp,npartitions=num_workers)
        ms1_bag = spec_bag.filter(lambda x: x.getMSLevel() == 1)
        ms2_bag = spec_bag.filter(lambda x: x.getMSLevel() == 2)
        ms1_bag = ms1_bag.map(cls.ms1spec2dfdict)
        ms2_bag = ms2_bag.map(cls.ms2spec2dfdict)
        ms1,ms2 = dask.compute(ms1_bag,ms2_bag,scheduler=worker_type,num_workers=num_workers)
        ms1_df = pl.DataFrame(ms1)
        ms1_df = ms1_df.with_columns(
            (pl.col('rt') / 60.0).cast(pl.Float32),
        )
        ms1_index = pd.Index(ms1_df['spec_id'])
        ms2_df = pl.DataFrame(ms2)
        ms2_df = ms2_df.with_columns(
            (pl.col('rt') / 60.0).cast(pl.Float32),
            pl.col('precursor_mz').cast(pl.Float32),
            pl.col('base_peak_mz').cast(pl.Float32),
            pl.col('base_peak_intensity').cast(pl.Float32),
        )
        ms2_index = pd.Index(ms2_df['spec_id'])
        metadata = cls.get_exp_meta(exp)
        spectrum_map = cls(
            exp_name=exp_name,
            metadata=metadata,
            ms1_index=ms1_index,
            ms1_df=ms1_df,
            ms2_index=ms2_index,
            ms2_df=ms2_df,
        )
        spectrum_map.insert_ms1_id_to_ms2()
        spectrum_map.convert_scan_to_spec_id()
        spectrum_map.modify_ms2_rt()
        return spectrum_map

    def save(self, save_dir_path: str):

        self_to_save = copy.copy(self)
        self_to_save.ms1_df = self.ms1_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )
        self_to_save.ms2_df = self.ms2_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )

        super(SpectrumMap, self_to_save).save(save_dir_path)

    @classmethod
    def load(cls, save_dir_path: str):

        data_dict = cls._base_load(save_dir_path)

        if 'ms1_df' in data_dict:
            if isinstance(data_dict['ms1_df'], pl.DataFrame):
                data_dict['ms1_df'] = data_dict['ms1_df'].with_columns(
                    pl.col("mz_array").map_elements(
                        lambda x: np.array(json.loads(x)),
                        return_dtype=pl.Object,
                    ),
                    pl.col("intensity_array").map_elements(
                        lambda x: np.array(json.loads(x)),
                        return_dtype=pl.Object,
                    ),
                )
        if 'ms2_df' in data_dict:
            if isinstance(data_dict['ms2_df'], pl.DataFrame):
                data_dict['ms2_df'] = data_dict['ms2_df'].with_columns(
                    pl.col("mz_array").map_elements(
                        lambda x: np.array(json.loads(x)),
                        return_dtype=pl.Object,
                    ),
                    pl.col("intensity_array").map_elements(
                        lambda x: np.array(json.loads(x)),
                        return_dtype=pl.Object,
                    ),
                )

        return cls(**data_dict)
