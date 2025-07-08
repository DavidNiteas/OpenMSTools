from pathlib import Path

import pyopenms as oms


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
