import os

import pandas as pd
from MetaMSTools.txn.experiment_analysis import ExperimentAnalysis, ExperimentAnalysisConfig


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def main():

    script_path = os.path.abspath(__file__)
    root_path = os.path.dirname(script_path)
    os.chdir(root_path)
    exp_file_paths = get_all_file_paths(
        "data/large_files/900_human_metabolites_db/mzml"
    )
    exp_file_paths = [path for path in exp_file_paths if "STD" in path]

    config = ExperimentAnalysisConfig(
        use_rt_aligner=False,
        use_feature_linker=False,
        worker_type="processes",
        num_workers=os.cpu_count() / 2
    )
    ea = ExperimentAnalysis(config = config)
    wrapper = ea(exp_file_paths)
    return wrapper

if __name__ == "__main__":
    wrapper = main()
    feature_df = pd.concat([feature.feature_info for feature in wrapper.feature_maps],axis=0)
    feature_df.to_excel("data/large_files/900_human_metabolites_db/features.xlsx")
