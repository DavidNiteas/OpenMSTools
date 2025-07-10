from __future__ import annotations

import copy
import json
import os
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
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine


class SpectrumMap(BaseModel):


    model_config = ConfigDict({"arbitrary_types_allowed": True})

    scan_id_matcher: ClassVar[Pattern] = re.compile(r'scan=(\d+)')

    exp_name: str
    ms1_index: pd.Index | None = None
    ms1_df: pl.DataFrame | None = None
    ms2_index: pd.Index | None = None
    ms2_df: pl.DataFrame | None = None
    ms2_rtree_index: rtree.index.Index | None = None
    exp_meta: pd.Series | None = None

    @staticmethod
    def get_exp_meta(exp: oms.MSExperiment) -> pd.Series:
        spec: oms.MSSpectrum = exp[0]
        meta_info_string = spec.getMetaValue("filter string")
        meta_info_list = meta_info_string.split(" ")
        ms_type = meta_info_list[0]
        ion_mode = meta_info_list[1]
        ion_source = meta_info_list[3]
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
            "mz_array": mz_array,
            "intensity_array": intensity_array,
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
            "mz_array": mz_array,
            "intensity_array": intensity_array,
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

    def init_ms2_rtree_index(self) -> None:
        if self.ms2_df is None:
            raise ValueError(
                "MS2 dataframe must be loaded before initializing R-tree index"
            )
        self.ms2_rtree_index = rtree.index.Index()
        for i,(rt,precursor_mz) in enumerate(
            zip(
                self.ms2_df['rt'],
                self.ms2_df['precursor_mz'],
            )
        ):
            self.ms2_rtree_index.insert(
                id=i,
                coordinates=(precursor_mz, rt, precursor_mz, rt),
                obj=i
            )

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
        exp_meta = cls.get_exp_meta(exp)
        spectrum_map = cls(
            exp_name=exp_name,
            ms1_index=ms1_index,
            ms1_df=ms1_df,
            ms2_index=ms2_index,
            ms2_df=ms2_df,
            exp_meta=exp_meta,
        )
        spectrum_map.insert_ms1_id_to_ms2()
        spectrum_map.convert_scan_to_spec_id()
        spectrum_map.modify_ms2_rt()
        return spectrum_map

    def __getstate__(self):
        state = copy.deepcopy(super().__getstate__())
        state['__dict__']['ms2_df'] = state['__dict__']['ms2_df'].to_pandas()
        state['__dict__']['ms1_df'] = state['__dict__']['ms1_df'].to_pandas()
        return state

    def __setstate__(self, state):
        state['__dict__']['ms2_df'] = pl.from_pandas(state['__dict__']['ms2_df'])
        state['__dict__']['ms1_df'] = pl.from_pandas(state['__dict__']['ms1_df'])
        need_init_ms2_rtree_index = False
        ms2_rtree_index:rtree.index.Index | None = state['__dict__']['ms2_rtree_index']
        if isinstance(ms2_rtree_index, rtree.index.Index):
            if ms2_rtree_index.get_size() == 0:
                state['__dict__']['ms2_rtree_index'] = None
                need_init_ms2_rtree_index = True
        super().__setstate__(state)
        if need_init_ms2_rtree_index:
            self.init_ms2_rtree_index()

    def save(self, save_dir_path: str):

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        index_dir_path = os.path.join(save_dir_path, "index")
        if not os.path.exists(index_dir_path):
            os.makedirs(index_dir_path)

        metadata_path = os.path.join(save_dir_path, "metadata.json")
        metadata = {"exp_name": self.exp_name, **self.exp_meta.to_dict()}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        ms1_index_path = os.path.join(index_dir_path, "ms1_index.csv")
        ms2_index_path = os.path.join(index_dir_path, "ms2_index.csv")
        pd.Series(self.ms1_index).to_csv(ms1_index_path, header=False)
        pd.Series(self.ms2_index).to_csv(ms2_index_path, header=False)

        ms2_rtree_index_path = os.path.join(index_dir_path, "ms2_rtree_index")
        saved_ms2_rtree_index = rtree.index.Index(ms2_rtree_index_path)
        for i,(rt,precursor_mz) in enumerate(
            zip(
                self.ms2_df['rt'],
                self.ms2_df['precursor_mz'],
            )
        ):
            saved_ms2_rtree_index.insert(
                id=i,
                coordinates=(precursor_mz, rt, precursor_mz, rt),
                obj=i
            )

        db_path = os.path.join(save_dir_path, "data.db")
        engine = create_engine(f"sqlite:///{db_path}")

        ms1_df_to_save = self.ms1_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )
        ms1_df_to_save.write_database(table_name="ms1_data", connection=engine.connect(), if_table_exists="replace")
        ms2_df_to_save = self.ms2_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )
        ms2_df_to_save.write_database(table_name="ms2_data", connection=engine.connect(), if_table_exists="replace")

        engine.dispose()

    @classmethod
    def load(cls, save_dir_path: str) -> SpectrumMap:

        index_dir_path = os.path.join(save_dir_path, "index")

        metadata_path = os.path.join(save_dir_path, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        exp_name = metadata.pop('exp_name')
        exp_meta = pd.Series(metadata)

        ms1_index_path = os.path.join(index_dir_path, "ms1_index.csv")
        ms2_index_path = os.path.join(index_dir_path, "ms2_index.csv")
        ms1_index = pd.Index(pd.read_csv(ms1_index_path, header=None, index_col=0).iloc[:,0])
        ms2_index = pd.Index(pd.read_csv(ms2_index_path, header=None, index_col=0).iloc[:,0])

        ms2_rtree_index_path = os.path.join(index_dir_path, "ms2_rtree_index")
        ms2_rtree_index = rtree.index.Index(ms2_rtree_index_path)

        db_path = os.path.join(save_dir_path, "data.db")
        engine = create_engine(f"sqlite:///{db_path}")

        ms1_df = pl.read_database(query="SELECT * FROM ms1_data", connection=engine.connect())
        ms1_df = ms1_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: np.array(json.loads(x)),
                return_dtype=pl.Object,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: np.array(json.loads(x)),
                return_dtype=pl.Object,
            ),
        )
        ms2_df = pl.read_database(query="SELECT * FROM ms2_data", connection=engine.connect())
        ms2_df = ms2_df.with_columns(
            pl.col("mz_array").map_elements(
                lambda x: np.array(json.loads(x)),
                return_dtype=pl.Object,
            ),
            pl.col("intensity_array").map_elements(
                lambda x: np.array(json.loads(x)),
                return_dtype=pl.Object,
            ),
        )

        engine.dispose()

        return cls(
            exp_name=exp_name,
            ms1_index=ms1_index,
            ms1_df=ms1_df,
            ms2_index=ms2_index,
            ms2_df=ms2_df,
            ms2_rtree_index=ms2_rtree_index,
            exp_meta=exp_meta,
        )
