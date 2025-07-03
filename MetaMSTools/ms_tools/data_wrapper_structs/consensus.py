from __future__ import annotations

import dask.bag as db
import pandas as pd
import pyopenms as oms
from pydantic import BaseModel, ConfigDict


class ConsensusMap(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    consensus_df: pd.DataFrame
    consensus_feature_mapping: pd.Series
    feature_consensus_mapping: pd.Series

    @classmethod
    def from_oms(cls, consensus_map: oms.ConsensusMap) -> ConsensusMap:
        raw_consensus_df = consensus_map.get_df()
        exp_names = raw_consensus_df.columns[5:][::-1]
        consensus_bag = db.from_sequence(consensus_map)
        feature_id_bag = consensus_bag.map(
            lambda x: \
                [
                    f.getUniqueId() if f.getUniqueId() is str \
                    else f"{exp_names[f.getMapIndex()]}::{f.getUniqueId()}" \
                    for f in x.getFeatureList()
                ]
        )
        consensus_df = consensus_map.get_df().iloc[:,2:].reset_index(drop=True)
        consensus_df.index.name = "consensus_id"
        consensus_feature_mapping = pd.Series(
            feature_id_bag.compute(scheduler="threads"),
            index=consensus_df.index
        )
        consensus_feature_mapping.name = "feature_ids"
        consensus_feature_mapping.index.name = "consensus_id"
        feature_consensus_mapping = {}
        for cid,fids in consensus_feature_mapping.items():
            for fid in fids:
                feature_consensus_mapping[fid] = cid
        feature_consensus_mapping = pd.Series(feature_consensus_mapping)
        feature_consensus_mapping.name = "consensus_id"
        feature_consensus_mapping.index.name = "feature_id"
        return cls(
            consensus_df=consensus_df,
            consensus_feature_mapping=consensus_feature_mapping,
            feature_consensus_mapping=feature_consensus_mapping
        )

    def as_oms_feature_map(self) -> oms.FeatureMap:
        feature_map = oms.FeatureMap()
        for i,row in self.consensus_df.iterrows():
            feature = oms.Feature()
            feature.setUniqueId(i)
            feature.setMZ(row["mz"])
            feature.setRT(row["RT"])
            feature.setIntensity(row.iloc[3:].max())
            feature_map.push_back(feature)
        return feature_map
