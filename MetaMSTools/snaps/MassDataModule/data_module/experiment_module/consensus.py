from __future__ import annotations

import copy
from typing import Literal

import dask.bag as db
import geopandas as gpd
import polars as pl
import pyopenms as oms
from pydantic import Field

from ..module_abc.map_ABCs import BaseMap


class ConsensusMap(BaseMap):

    queue_name: str = Field(
        ...,
        data_type="metadata",
        save_mode="json",
        description="队列名称"
    )
    consensus_index: gpd.GeoDataFrame | None = Field(
        None,
        data_type="index",
        save_mode="parquet",
        description="共识图索引，基于geopandas"
    )
    consensus_info: pl.DataFrame | None = Field(
        None,
        data_type="data",
        save_mode="sqlite",
        description="共识图数据表，基于polars"
    )
    exp_name: str = Field(
        "default",
        data_type="metadata",
        save_mode="json",
        description="实验名称，对于共识图而言，这一参数应与队列名称一致"
    )

    def model_post_init(self, context):
        if self.exp_name == "default":
            self.exp_name = self.queue_name

    @classmethod
    def from_oms(
        cls,
        queue_name: str,
        consensus_map: oms.ConsensusMap,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> ConsensusMap:
        raw_consensus_info = consensus_map.get_df()
        exp_names = raw_consensus_info.columns[5:][::-1]
        consensus_bag = db.from_sequence(consensus_map, npartitions=num_workers)
        feature_id_bag = consensus_bag.map(
            lambda x: \
                [
                    f.getUniqueId() if f.getUniqueId() is str \
                    else f"{exp_names[f.getMapIndex()]}::{f.getUniqueId()}" \
                    for f in x.getFeatureList()
                ]
        )
        consensus_info = consensus_map.get_df().iloc[:,2:].reset_index(drop=True)
        consensus_info['feature_ids'] = feature_id_bag.compute(scheduler=worker_type, num_workers=num_workers)
        consensus_info.index.name = "consensus_id"
        consensus_info = pl.from_pandas(consensus_info,include_index=True)
        consensus_info = consensus_info.with_columns(
            pl.col(consensus_info.columns).exclude(['feature_ids','consensus_id']).cast(pl.Float32),
            (f"{queue_name}::" + pl.col('consensus_id').cast(pl.String)).alias('consensus_id')
        )
        consensus_index = gpd.GeoDataFrame(
            {"iloc": range(len(consensus_info))},
            geometry=gpd.points_from_xy(
                consensus_info["RT"],
                consensus_info["mz"],
                consensus_info['quality']
            ),
            index=consensus_info['consensus_id']
        )

        return cls(
            consensus_info=consensus_info,
            consensus_index=consensus_index,
            queue_name=queue_name,
        )

    def search_consensus_by_range(
        self,
        coordinates: tuple[
            float, # min_rt
            float, # min_mz
            float, # max_rt
            float, # max_mz
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        iloc = list(self.consensus_index.sindex.intersection(coordinates))
        if return_type == "id":
            return self.consensus_index.iloc[iloc].index.tolist()
        elif return_type == "df":
            return self.consensus_info[iloc]
        else:
            return iloc

    def get_oms_feature_map(self) -> oms.FeatureMap:
        feature_map = oms.FeatureMap()
        for i in range(len(self.consensus_info)):
            feature = oms.Feature()
            feature.setUniqueId(i)
            feature.setMZ(self.consensus_info[i, "mz"])
            feature.setRT(self.consensus_info[i, "RT"])
            feature_map.push_back(feature)
        return feature_map

    def save(self, save_dir_path: str):

        self_to_save = copy.copy(self)

        self_to_save.consensus_info = self.consensus_info.with_columns(
            ("[" + pl.col("feature_ids").list.join(",") + "]")
            .alias("feature_ids"),
        )

        super(ConsensusMap, self_to_save).save(save_dir_path)

    @classmethod
    def load(cls, save_dir_path: str):

        data_dict = cls._base_load(save_dir_path)

        consensus_info: pl.DataFrame | None = data_dict.pop("consensus_info")

        if consensus_info is not None:
            data_dict['consensus_info'] = consensus_info.with_columns(
                pl.col("feature_ids")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(","),
                pl.col(consensus_info.columns)
                    .exclude(['feature_ids','consensus_id'])
                    .cast(pl.Float32),
            )

        data_dict['queue_name'] = data_dict['metadata'].pop('queue_name')

        return cls(**data_dict)
