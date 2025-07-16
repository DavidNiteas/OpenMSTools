import os
import time

from MetaMSTools.txn.experiment_analysis import ExperimentAnalysis, ExperimentAnalysisConfig


def main():

    config = ExperimentAnalysisConfig(
        # use_rt_aligner=False,
        # use_feature_linker=False,
        # worker_type="threads",
        worker_type="processes",
        num_workers=2,
        num_threads_per_worker=4,
        # error_mode="raise_in_worker",
    )
    ea = ExperimentAnalysis(config = config)
    cwd = os.getcwd()
    wrapper,linker = ea(
        queue_name="test_queue",
        exp_file_paths=[
            os.path.join(cwd, "tests","data","raw_files","Metabolomics_1.mzML"),
            os.path.join(cwd, "tests","data","raw_files","Metabolomics_2.mzML"),
        ],
        ref_file_paths=[
            os.path.join(cwd, "tests","data","raw_files","QC1.mzML"),
            os.path.join(cwd, "tests","data","raw_files","QC2.mzML"),
        ],
        save_dir_path=os.path.join(cwd, "tests","cache","test_queue"),
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间
    print(f"Execution time: {execution_time:.2f} seconds")  # 打印执行时间
