from typing import Literal

import dask
import dask.bag as db
import pandas as pd
import polars as pl

from ..experiment_module import FeatureMap, SpectrumMap


def infer_sub_hull_from_feature(
    df_feat: pl.DataFrame,
    df_hull: pl.DataFrame,
    *,
    feature_id_col: str = "feature_id",
    num_hull_col: str = "hull_num",
    hull_id_col: str = "hull_id",
) -> list[pl.DataFrame]:
    df_ids = (
        df_feat
        .with_columns(
            idx=pl.int_ranges(0, pl.col(num_hull_col))
        )
        .explode("idx")
        .with_columns(
            hull_id=pl.format("{}::{}", pl.col(feature_id_col), pl.col("idx"))
        )
    )
    merged = df_ids.join(
        df_hull,
        left_on="hull_id",
        right_on=hull_id_col,
        how="left",
    )
    sub_map = (
        merged.group_by(feature_id_col)
        .agg(pl.all())                # 每组的全部列 -> 列表
        .partition_by(feature_id_col, as_dict=False)
    )
    id_order = df_feat[feature_id_col].to_list()
    id_to_df = {df[feature_id_col][0]: df.drop(feature_id_col) for df in sub_map}
    return [id_to_df.get(fid, pl.DataFrame()).select(
                ["hull_id", "RTstart", "RTend", "MZstart", "MZend",
                "rt_points", "mz_points", "intens_points"]
            ).explode(
                ["hull_id", "RTstart", "RTend", "MZstart", "MZend",
                "rt_points", "mz_points", "intens_points"]) \
            for fid in id_order]

def link_ms2_to_feature(
    feature_hulls: pd.DataFrame | pl.DataFrame,
    spectrum_map: SpectrumMap
) -> list[str]:
    spectrum_id_list = []
    for mz_start,rt_start,mz_end,rt_end in zip(
        feature_hulls['MZstart'],
        feature_hulls['RTstart'],
        feature_hulls['MZend'],
        feature_hulls['RTend'],
    ):
        spectrum_id_list += spectrum_map.search_ms2_by_range(
            (rt_start,mz_start,rt_end,mz_end),"id"
        )
    return spectrum_id_list

def link_ms2_and_feature_map(
    feature_map: FeatureMap,
    spectrum_map: SpectrumMap,
    key_id: Literal["feature","spectrum"] = "feature",
    worker_type: Literal["threads","processes","synchronous"] = "threads",
    num_workers: int | None = None,
) -> pl.DataFrame:
    hull_info_queue = infer_sub_hull_from_feature(feature_map.feature_info, feature_map.hull_info)
    hull_info_bag = db.from_sequence(hull_info_queue, npartitions=num_workers)
    spectrum_id_bag = hull_info_bag.map(
        lambda x: link_ms2_to_feature(x,spectrum_map)
    )
    spectrum_id_list = dask.compute(
        spectrum_id_bag, scheduler=worker_type, num_workers=num_workers
    )[0]
    mapping_df = pl.DataFrame(
        data = {
            "feature_id": feature_map.feature_info['feature_id'],
            "spectrum_id": spectrum_id_list,
        },
        schema=pl.Schema({
            "feature_id": pl.String,
            "spectrum_id": pl.List(pl.String),
        })
    )
    if key_id == "spectrum":
        mapping_df = mapping_df.explode("spectrum_id").filter(
            pl.col("spectrum_id").is_not_null()
        ).select(["spectrum_id", "feature_id"])
    return mapping_df
