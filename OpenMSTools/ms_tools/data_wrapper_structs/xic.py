from __future__ import annotations

import pandas as pd
import pyopenms as oms
import rtree
from pydantic import BaseModel, ConfigDict


class XICMap(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    ion_index: rtree.index.Index
    ion_df: pd.DataFrame

    @classmethod
    def from_oms(
        cls,
        exp: oms.MSExperiment,
    ) -> XICMap:
        ion_df = exp.get_massql_df()[0][["mz","rt","i"]]
        ion_df['rt'] = ion_df['rt'] * 60
        ion_index = rtree.index.Index()
        for i,(ion_id,ion_mz,ion_rt) in enumerate(
            zip(
                ion_df.index,
                ion_df['mz'],
                ion_df['rt'],
            )
        ):
            ion_index.insert(
                id=i,
                coordinates=(ion_mz, ion_rt),
                obj=ion_id,
            )
        return cls(ion_index=ion_index, ion_df=ion_df)

    @classmethod
    def from_ms1(
        cls,
        ms1: pd.DataFrame,
    ) -> XICMap:
        ion_df = {
            "mz": [],
            "rt": [],
            "i": [],
        }
        ion_index = rtree.index.Index()
        ion_id = 0
        for _,row in ms1.iterrows():
            rt = row['rt']
            for mz,intensity in zip(row['mz_array'],row['intensity_array']):
                ion_df['mz'].append(mz)
                ion_df['rt'].append(rt)
                ion_df['i'].append(intensity)
                ion_index.insert(
                    id=ion_id,
                    coordinates=(mz, rt),
                    obj=ion_id,
                )
                ion_id += 1
        ion_df = pd.DataFrame(ion_df)
        return cls(ion_index=ion_index, ion_df=ion_df)

    def search_ion_by_range(
        self,
        coordinates: tuple[
            float, # min_mz
            float, # min_rt
            float, # max_mz
            float, # max_rt
        ]
    ) -> list[int]:
        return list(self.ion_index.intersection(coordinates, objects="raw"))
