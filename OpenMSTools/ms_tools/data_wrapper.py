from __future__ import annotations

from pathlib import Path

import dask
import dask.bag as db
import pyopenms as oms
from pydantic import BaseModel, ConfigDict


def load_exp_file(file_path: str | Path) -> tuple[str, oms.MSExperiment]:

    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    exp_name = file_path.name
    file_type = file_path.suffix.lower()
    exp = oms.MSExperiment()
    if file_type == ".mzml":
        oms.MzMLFile().load(str(file_path), exp)
    elif file_type == ".mzxml":
        oms.MzXMLFile().load(str(file_path), exp)
    else:
        raise ValueError(
            f"Unsupported file type: {file_type} for file {file_path}, supported types are .mzML and .mzXML"
        )
    return exp_name, exp

class ConsensusMap(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    @classmethod
    def from_oms(cls, consensus_map: oms.ConsensusMap) -> ConsensusMap:
        pass

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
