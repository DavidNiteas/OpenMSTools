from __future__ import annotations

import polars as pl
from pydantic import Field
from typing_extensions import Self

from ..experiment_module import ConsensusMap, FeatureMap, MetaMSExperimentDataQueue, SpectrumMap
from ..module_abc import BaseLinker
from .link_func import link_ms2_and_feature_map


class SampleLevelLinker(BaseLinker):

    exp_name: str = Field(
        ...,
        description="实验名称"
    )
    ms2_ms1_mapping: pl.DataFrame | None = Field(
        default=None,
        description="MS2-MS1映射表"
    )
    ms2_feature_mapping: pl.DataFrame | None = Field(
        default=None,
        description="MS2-特征映射表"
    )
    hull_feature_mapping: pl.DataFrame | None = Field(
        default=None,
        description="离子流凸包-特征映射表",
    )

    def link_ms1_ms2(
        self,
        spectrum_map: SpectrumMap,
    ) -> None:
        ms2_ms1_mapping = spectrum_map.ms2_df.select(['spec_id','ms1_id'])
        ms2_ms1_mapping.columns = ['ms2_id','ms1_id']
        self.ms2_ms1_mapping = ms2_ms1_mapping

    def link_ms2_feature(
        self,
        spectrum_map: SpectrumMap,
        feature_map: FeatureMap,
    ) -> None:
        self.ms2_feature_mapping = link_ms2_and_feature_map(
            spectrum_map,feature_map,
            key_id="spectrum",
        )

    def link_hull_feature(
        self,
        feature_map: FeatureMap,
    ) -> None:
        hull_feature_mapping = (
            feature_map.feature_info
            .with_columns(
                pl.int_ranges(0, pl.col('hull_num')).alias('hull_id')
            )
            .select(['hull_id', 'feature_id'])
            .explode('hull_id')
            .with_columns(
                (pl.col('feature_id') + "::" + pl.col('hull_id').cast(str)).alias('hull_id')
            )
        )
        self.hull_feature_mapping = hull_feature_mapping

    def link_exp_data(
        self,
        spectrum_map: SpectrumMap | None = None,
        feature_map: FeatureMap | None = None,
    ):
        if spectrum_map is not None:
            self.link_ms1_ms2(spectrum_map)
            if feature_map is not None:
                self.link_ms2_feature(spectrum_map, feature_map)
        if feature_map is not None:
            self.link_hull_feature(feature_map)

    @classmethod
    def link_exp_data_from_cls(
        cls,
        exp_name: str,
        spectrum_map: SpectrumMap | None = None,
        feature_map: FeatureMap | None = None,
    ) -> Self:
        linker = cls(exp_name=exp_name)
        linker.link_exp_data(spectrum_map, feature_map)
        return linker

class QueueLevelLinker(BaseLinker):

    queue_name: str = Field(
        ...,
        description="队列名称"
    )
    exp_names: pl.Series = Field(
        default_factory=pl.Series,
        description="实验名称列表"
    )
    sample_level_linkers: list[SampleLevelLinker] = Field(
        default_factory=list,
        description="样本级联合器列表"
    )
    feature_consensus_mapping: pl.DataFrame | None = Field(
        default=None,
        description="特征-共识特征映射表"
    )

    def link_feature_consensus(
        self,
        consensus_map: ConsensusMap,
    ) -> None:
        feature_consensus_mapping = consensus_map.consensus_info.select(['feature_id', 'consensus_id'])
        feature_consensus_mapping = feature_consensus_mapping.explode('feature_id')
        self.feature_consensus_mapping = feature_consensus_mapping

    def link_exp_data(
        self,
        queue_data: MetaMSExperimentDataQueue,
    ):
        self.queue_name = queue_data.queue_name
        self.exp_names = queue_data.exp_names
        for exp_name, spectrum_map, feature_map in zip(
            self.exp_names,
            queue_data.spectrum_maps,
            queue_data.feature_maps,
        ):
            sample_level_linker = SampleLevelLinker(exp_name=exp_name)
            sample_level_linker.link(spectrum_map, feature_map)
            self.sample_level_linkers.append(sample_level_linker)
        if queue_data.consensus_map is not None:
            self.link_feature_consensus(queue_data.consensus_map)

    @classmethod
    def link_exp_data_from_cls(
        cls,
        queue_data: MetaMSExperimentDataQueue,
    ) -> Self:
        linker = cls(queue_name=queue_data.queue_name)
        linker.link_exp_data(queue_data)
