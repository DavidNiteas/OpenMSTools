import logging
from pathlib import Path

import pyopenms as oms


def load_exp_file(
    file_path: str | Path,
    re_try: int = 3,
) -> tuple[str, oms.MSExperiment]:

    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    exp_name = file_path.name
    file_type = file_path.suffix.lower()
    if file_type == ".mzml":
        file_loader = oms.MzMLFile()
    elif file_type == ".mzxml":
        file_loader = oms.MzXMLFile()
    else:
        raise ValueError(
            f"Unsupported file type: {file_type} for file {file_path}, supported types are .mzML and .mzXML"
        )
    try_num = 0
    while True:
        exp = oms.MSExperiment()
        if try_num >= re_try:
            raise ValueError(f"Failed to load {file_path} after {re_try} tries")
        try:
            file_loader.load(str(file_path), exp)
            break
        except Exception as e:
            try_num += 1
            logging.warning(f"Error while loading {file_path}, trying again ({try_num}/{re_try})")
            logging.warning(e)
    return exp_name, exp
