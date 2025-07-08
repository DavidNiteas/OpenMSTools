from __future__ import annotations

import dask
import dask.bag as db
import pyopenms as oms
from pydantic import BaseModel, ConfigDict

from .io import load_exp_file


class OpenMSDataWrapper(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    file_paths: list[str] | None = None
    exp_names: list[str] | None = None
    exps: list[oms.MSExperiment] | None = None
    mass_traces: list[list[oms.MassTrace]] | None = None
    chromatogram_peaks: list[list[oms.ChromatogramPeak]] | None = None
    features: list[oms.FeatureMap] | None = None
    ref_feature_for_align: oms.FeatureMap | None = None
    trafos: list[oms.TransformationDescription] | None = None
    consensus_map: oms.ConsensusMap | None = None

    def init_exps(self):

        if self.file_paths is not None:
            file_bag = db.from_sequence(self.file_paths)
            file_bag = file_bag.map(load_exp_file)
            exp_name_bag = file_bag.pluck(0)
            exp_bag = file_bag.pluck(1)
            self.exp_names, self.exps = dask.compute(exp_name_bag, exp_bag, scheduler='threads')

    def infer_ref_feature_for_align(self):

        max_feature_num = 0
        max_feature_map = None
        for feature_map in self.features:
            if feature_map.size() > max_feature_num:
                max_feature_num = feature_map.size()
                max_feature_map = feature_map
        self.ref_feature_for_align = max_feature_map

    @staticmethod
    def _get_size(obj: list | None) -> int:
        if obj is None:
            return 0
        return len(obj)

    def size(self) -> int:
        return max(
            self._get_size(self.file_paths),
            self._get_size(self.exp_names),
            self._get_size(self.exps),
            self._get_size(self.mass_traces),
            self._get_size(self.chromatogram_peaks),
            self._get_size(self.features),
            self._get_size(self.trafos),
        )

    def _split_obj(
        self, obj: list | None, n: int
    ) -> str |\
            oms.MSExperiment |\
            list[oms.MassTrace] |\
            list[oms.ChromatogramPeak] |\
            oms.FeatureMap |\
            oms.TransformationDescription:
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
    def merge(cls, data_wrappers: list[OpenMSDataWrapper]) -> OpenMSDataWrapper:
        merged_obj = cls()
        for data_wrapper in data_wrappers:
            cls._merge_step(merged_obj.file_paths, data_wrapper.file_paths)
            cls._merge_step(merged_obj.exp_names, data_wrapper.exp_names)
            cls._merge_step(merged_obj.exps, data_wrapper.exps)
            cls._merge_step(merged_obj.mass_traces, data_wrapper.mass_traces)
            cls._merge_step(merged_obj.chromatogram_peaks, data_wrapper.chromatogram_peaks)
            cls._merge_step(merged_obj.features, data_wrapper.features)
            cls._merge_step(merged_obj.trafos, data_wrapper.trafos)
            if data_wrapper.ref_feature_for_align is not None and merged_obj.ref_feature_for_align is None:
                merged_obj.ref_feature_for_align = data_wrapper.ref_feature_for_align
            if data_wrapper.consensus_map is not None and merged_obj.consensus_map is None:
                merged_obj.consensus_map = data_wrapper.consensus_map
        return merged_obj

    def split(self) -> list[OpenMSDataWrapper]:
        data_wrappers = []
        for i in range(self.size()):
            data_wrapper = OpenMSDataWrapper(
                file_paths=self._split_obj(self.file_paths, i),
                exp_names=self._split_obj(self.exp_names, i),
                exps=self._split_obj(self.exps, i),
                mass_traces=self._split_obj(self.mass_traces, i),
                chromatogram_peaks=self._split_obj(self.chromatogram_peaks, i),
                features=self._split_obj(self.features, i),
                ref_feature_for_align=self.ref_feature_for_align,
                trafos=self._split_obj(self.trafos, i),
                consensus_map=self.consensus_map,
            )
            data_wrappers.append(data_wrapper)
        return data_wrappers
