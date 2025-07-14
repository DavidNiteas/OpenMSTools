from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .consensus import ConsensusMap
from .features import FeatureMap
from .spectrums import SpectrumMap
from .xic import XICMap


class MetaMSDataWrapper(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    file_paths: list[str] | None = None
    exp_names: list[str] | None = None
    spectrum_maps: list[SpectrumMap] | None = None
    xic_maps: list[XICMap] | None = None
    feature_maps: list[FeatureMap] | None = None
    ms2_feature_mapping: list[pd.Series] | None = None
    consensus_map: ConsensusMap | None = None

    @staticmethod
    def _get_size(obj: list | None) -> int:
        if obj is None:
            return 0
        return len(obj)

    def size(self) -> int:
        return max(
            self._get_size(self.file_paths),
            self._get_size(self.exp_names),
            self._get_size(self.spectrum_maps),
            self._get_size(self.xic_maps),
            self._get_size(self.feature_maps),
            self._get_size(self.ms2_feature_mapping),
        )

    def _split_obj(self, obj: list | None, n: int) -> str | SpectrumMap | XICMap | FeatureMap | None:
        if self._get_size(obj) != self.size():
            return None
        return obj[n]

    @staticmethod
    def _merge_step(obj_1: list | None, obj_2: list | None) -> list | None:
        if obj_1 is None and obj_2 is None:
            return None
        elif obj_1 is None and obj_2 is not None:
            return obj_2
        elif obj_1 is not None and obj_2 is None:
            return obj_1 + [obj_2]
        else:
            return obj_1 + obj_2

    @classmethod
    def merge(cls, data_wrappers: list[MetaMSDataWrapper]) -> MetaMSDataWrapper:
        merged_obj = cls()
        for data_wrapper in data_wrappers:
            merged_obj.file_paths = cls._merge_step(merged_obj.file_paths, data_wrapper.file_paths)
            merged_obj.exp_names = cls._merge_step(merged_obj.exp_names, data_wrapper.exp_names)
            merged_obj.spectrum_maps = cls._merge_step(merged_obj.spectrum_maps, data_wrapper.spectrum_maps)
            merged_obj.xic_maps = cls._merge_step(merged_obj.xic_maps, data_wrapper.xic_maps)
            merged_obj.feature_maps = cls._merge_step(merged_obj.feature_maps, data_wrapper.feature_maps)
            merged_obj.ms2_feature_mapping = cls._merge_step(
                merged_obj.ms2_feature_mapping, data_wrapper.ms2_feature_mapping
            )
            if data_wrapper.consensus_map is not None and merged_obj.consensus_map is None:
                merged_obj.consensus_map = data_wrapper.consensus_map
        return merged_obj

    def split(self) -> list[MetaMSDataWrapper]:
        data_wrappers = []
        for i in range(self.size()):
            data_wrapper = MetaMSDataWrapper(
                file_paths=self._split_obj(self.file_paths, i),
                exp_names=self._split_obj(self.exp_names, i),
                spectrum_maps=self._split_obj(self.spectrum_maps, i),
                xic_maps=self._split_obj(self.xic_maps, i),
                feature_maps=self._split_obj(self.feature_maps, i),
                consensus_map=self.consensus_map,
            )
            data_wrappers.append(data_wrapper)
        return data_wrappers
