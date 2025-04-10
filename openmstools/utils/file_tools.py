from .base_tools import oms,calculate_mzfile_hash
from .async_tools import use_coroutine,run_coroutine
from typing import Tuple,Dict,Literal,Hashable

def load_mzml_file(file_path: str) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)
    return exp

def load_mzxml_file(file_path: str) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.MzXMLFile().load(file_path, exp)
    return exp

def load_exp_ms_file(file_path: str) -> oms.MSExperiment:
    if file_path.lower().endswith(".mzml"):
        return load_mzml_file(file_path)
    elif file_path.lower().endswith(".mzxml"):
        return load_mzxml_file(file_path)
    else:
        raise ValueError("Unsupported file format")

@use_coroutine
def load_exp_ms_file_coroutine(file_path: str):
    return load_exp_ms_file(file_path)

def load_exp_ms_files(
    file_path_list: list,
    key_type: Literal[
        "index",
        "file_path",
        "file_name",
        "md5", "sha1", "sha256"
    ] = "file_path",
    use_progress: bool = True,
) -> Tuple[
    Dict[Hashable, oms.MSExperiment], 
    Dict[Hashable, str],
]:
    inps = {file_path:((file_path,),{}) for file_path in file_path_list}
    exps:Dict[str,oms.MSExperiment] = run_coroutine(load_exp_ms_file_coroutine, inps, use_progress=use_progress, description="Loading MSExperiment files")
    result_dict = {}
    path_map = {}
    if key_type == "file_path":
        result_dict = exps
        path_map = dict(zip(file_path_list, file_path_list))
    elif key_type == "file_name":
        for file_path, exp in exps.items():
            file_name = file_path.split("/")[-1]
            result_dict[file_name] = exp
            path_map[file_name] = file_path
    elif key_type == "index":
        for i, file_path in enumerate(exps.keys()):
            result_dict[i] = exps[file_path]
            path_map[i] = file_path
    else:
        for file_path, exp in exps.items():
            hash_value = calculate_mzfile_hash(file_path, key_type)
            result_dict[hash_value] = exp
            path_map[hash_value] = file_path
    return result_dict, path_map