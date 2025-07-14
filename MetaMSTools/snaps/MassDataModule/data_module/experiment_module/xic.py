from __future__ import annotations

from typing import ClassVar, Literal

import geopandas as gpd
import polars as pl
import pyopenms as oms
from pydantic import Field

from ..module_abc.map_ABCs import BaseMap


class XICMap(BaseMap):

    table_schema: ClassVar[dict[str,dict]] = {
        "ion_df": {
            "mz": pl.Float32,
            "rt": pl.Float32,
            "i": pl.Float32,
        }
    }

    ion_index: gpd.GeoDataFrame | None = Field(
        default=None,
        data_type="index",
        save_mode="parquet",
        description="离子流的空间索引，基于geopandas",
    )
    ion_df: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="离子流的空间数据，基于polars",
    )

    @classmethod
    def from_oms(
        cls,
        exp: oms.MSExperiment,
        exp_name: str,
    ) -> XICMap:
        ion_df = pl.from_pandas(
            exp.get_massql_df()[0][["mz","rt","i"]],
            schema_overrides = {
                "mz": pl.Float32,
                "rt": pl.Float32,
                "i": pl.Float32,
            }
        )
        ion_index = gpd.GeoDataFrame(
            {"iloc": range(len(ion_df))},
            geometry=gpd.points_from_xy(
                ion_df['rt'],
                ion_df['mz'],
                ion_df['i'],
            )
        )
        return cls(
            exp_name=exp_name,
            ion_index=ion_index,
            ion_df=ion_df,
        )

    def search_ion_by_range(
        self,
        coordinates: tuple[
            float, # min_rt
            float, # min_mz
            float, # max_rt
            float, # max_mz
        ],
        return_type: Literal['id', 'indices', 'df'] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        iloc = list(self.ion_index.sindex.intersection(coordinates))
        if return_type == "id":
            return self.ion_index[iloc].index.tolist()
        elif return_type == "df":
            return self.ion_df[iloc]
        else:
            return iloc
