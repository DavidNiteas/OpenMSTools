import os
import time

from MetaMSTools.txn.experiment_analysis import ExperimentAnalysis, ExperimentAnalysisConfig
from MetaMSTools.txn.reference_analysis import ReferenceFeatureFinderConfig


def main():

    config = ExperimentAnalysisConfig(
        # use_rt_aligner=False,
        # use_feature_linker=False,
        reference_analysis_config = ReferenceFeatureFinderConfig(
            mode = 'max_size_feature'
        ),
        # worker_type="threads",
        worker_type="processes",
        num_workers=2,
        num_threads_per_worker=4,
        num_high_load_worker=None,
    )
    ea = ExperimentAnalysis(config = config)
    cwd = os.getcwd()
    # exp_file_paths = [
    #     os.path.join(cwd, "tests","data","raw_files","Metabolomics_1.mzML"),
    #     os.path.join(cwd, "tests","data","raw_files","Metabolomics_2.mzML"),
    # ]
    # ref_file_paths = [
    #     os.path.join(cwd, "tests","data","raw_files","QC1.mzML"),
    #     os.path.join(cwd, "tests","data","raw_files","QC2.mzML"),
    # ]
    exp_file_paths = [
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","sample","MX03-030-1ul-Y02.mzML"),
        # os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","sample","MX03-030-1ul-Y03.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","sample","MX03-030-1ul-Y138.mzML"),  # noqa: E501
    ]
    ref_file_paths = [
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC1.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC2.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC3.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC4.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC5.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC6.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC7.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC8.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC9.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC10.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC11.mzML"),
        os.path.join(cwd, "tests","data","large_files","MX2503-030-DR-Taxol","mzml","qc","QC12.mzML"),
    ]
    wrapper,linker = ea(
        queue_name="debug_queue",
        exp_file_paths=exp_file_paths,
        ref_file_paths=ref_file_paths,
        save_dir_path=os.path.join(cwd, "tests","cache","debug_queue"),
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间
    print(f"Execution time: {execution_time:.2f} seconds")  # 打印执行时间
