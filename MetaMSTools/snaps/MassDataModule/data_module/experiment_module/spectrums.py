from __future__ import annotations

import copy
import re
from re import Pattern
from typing import ClassVar, Literal

import dask
import dask.bag as db
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyopenms as oms
from pydantic import Field

from .ABCs import BaseMap


class SpectrumMap(BaseMap):

    scan_id_matcher: ClassVar[Pattern] = re.compile(r'scan=(\d+)')

    ms1_index: gpd.GeoDataFrame | None = Field(
        default=None,
        data_type="index",
        save_mode="parquet",
        description="MS1谱图的空间索引表，基于geopandas"
    )
    ms1_df: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="MS1谱图的DataFrame，基于polars"
    )
    ms2_index: gpd.GeoDataFrame | None = Field(
        default=None,
        data_type="index",
        save_mode="parquet",
        description="MS2谱图的空间索引表，基于geopandas"
    )
    ms2_df: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="MS2谱图的DataFrame，基于polars"
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
            "mz_array": mz_array.tolist(),
            "intensity_array": intensity_array.tolist(),
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
            "mz_array": mz_array.tolist(),
            "intensity_array": intensity_array.tolist(),
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
            self.ms1_index.index = pd.Index(self.ms1_df['spec_id'].to_list())
        if self.ms2_df.schema['spec_id'] in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
        ):
            self.ms2_df = self.ms2_df.with_columns(
                (f"{self.exp_name}::ms2::" + self.ms2_df['spec_id'].cast(str)).alias('spec_id')
            )
            self.ms2_index.index = pd.Index(self.ms2_df['spec_id'].to_list())
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

    def search_ms2_by_range(
        self,
        coordinates: tuple[
            float, # min_rt
            float, # min_mz
            float, # max_rt
            float, # max_mz
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        iloc = list(self.ms2_index.sindex.intersection(coordinates))
        if return_type == "id":
            return self.ms2_index.index[iloc].tolist()
        elif return_type == "df":
            return self.ms2_df[iloc]
        else:
            return iloc

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
        ms1_df = pl.DataFrame(ms1,schema={
            "spec_id":pl.Int32,
            "rt":pl.Float32,
            "mz_array":pl.List(pl.Float32),
            "intensity_array":pl.List(pl.Float32),
        })
        ms1_df = ms1_df.with_columns(
            (pl.col('rt') / 60.0),
        )
        ms1_index = gpd.GeoDataFrame(
            {"iloc":range(len(ms1_df))},
            index=ms1_df['spec_id'],
            geometry=gpd.points_from_xy(
                x=ms1_df['rt'],
                y=[0] * len(ms1_df),
            )
        )
        ms2_df = pl.DataFrame(ms2,schema={
            "spec_id":pl.Int32,
            "rt":pl.Float32,
            "precursor_mz":pl.Float32,
            "base_peak_mz":pl.Float32,
            "base_peak_intensity":pl.Float32,
            "mz_array":pl.List(pl.Float32),
            "intensity_array":pl.List(pl.Float32),
        })
        ms2_df = ms2_df.with_columns(
            (pl.col('rt') / 60.0),
        )
        ms2_index = gpd.GeoDataFrame(
            {"iloc":range(len(ms2_df))},
            index=ms2_df['spec_id'],
            geometry=gpd.points_from_xy(
                x=ms2_df['rt'],
                y=ms2_df['precursor_mz'],
            )
        )
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
            ("[" + pl.col("mz_array").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("mz_array"),
            ("[" + pl.col("intensity_array").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("intensity_array"),
        )
        self_to_save.ms2_df = self.ms2_df.with_columns(
            ("[" + pl.col("mz_array").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("mz_array"),
            ("[" + pl.col("intensity_array").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("intensity_array"),
        )

        super(SpectrumMap, self_to_save).save(save_dir_path)

    @classmethod
    def load(cls, save_dir_path: str):

        data_dict = cls._base_load(save_dir_path)

        if 'ms1_df' in data_dict:
            if isinstance(data_dict['ms1_df'], pl.DataFrame):
                data_dict['ms1_df'] = data_dict['ms1_df'].with_columns(
                    pl.col("mz_array")
                        .str.strip_chars_start("[")
                        .str.strip_chars_end("]")
                        .str.split(",")
                        .cast(pl.List(pl.Float32)),
                    pl.col("intensity_array")
                        .str.strip_chars_start("[")
                        .str.strip_chars_end("]")
                        .str.split(",")
                        .cast(pl.List(pl.Float32)),
                )
        if 'ms2_df' in data_dict:
            if isinstance(data_dict['ms2_df'], pl.DataFrame):
                data_dict['ms2_df'] = data_dict['ms2_df'].with_columns(
                    pl.col("mz_array")
                        .str.strip_chars_start("[")
                        .str.strip_chars_end("]")
                        .str.split(",")
                        .cast(pl.List(pl.Float32)),
                    pl.col("intensity_array")
                        .str.strip_chars_start("[")
                        .str.strip_chars_end("]")
                        .str.split(",")
                        .cast(pl.List(pl.Float32)),
                )

        return cls(**data_dict)
