from pydantic import BaseModel, ConfigDict
import pyopenms as oms
from pathlib import Path
import dask
import dask.bag as db
from typing import Optional, Union, List, Tuple

def load_exp_file(file_path: Union[str,Path]) -> Tuple[str, oms.MSExperiment]:
    
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
        raise ValueError(f"Unsupported file type: {file_type} for file {file_path}, supported types are .mzML and .mzXML")
    return exp_name, exp

class OpenMSDataWrapper(BaseModel):
    
    model_config = ConfigDict({"arbitrary_types_allowed": True})
    
    file_paths: Optional[List[str]] = None
    exp_names: Optional[List[str]] = None
    exps: Optional[List[oms.MSExperiment]] = None
    mass_traces: Optional[List[List[oms.MassTrace]]] = None
    chromatogram_peaks: Optional[List[List[oms.ChromatogramPeak]]] = None
    features: Optional[List[oms.FeatureMap]] = None
    ref_feature_for_align: Optional[oms.FeatureMap] = None
    trafos: Optional[List[oms.TransformationDescription]] = None
    consensus_map: Optional[oms.ConsensusMap] = None
    
    def init_exps(self):
        
        if self.file_paths is not None:
            file_bag = db.from_sequence(self.file_paths)
            file_bag = file_bag.map(load_exp_file)
            exp_name_bag = file_bag.pluck(0)
            exp_bag = file_bag.pluck(1)
            self.exp_names, self.exps = dask.compute(exp_name_bag, exp_bag, scheduler='threads')