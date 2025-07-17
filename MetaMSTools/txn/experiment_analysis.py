import multiprocessing
import os
import threading
from contextlib import nullcontext as No_Semaphore
from functools import partial
from multiprocessing.synchronize import Semaphore as P_Semaphore
from threading import Semaphore as T_Semaphore
from typing import Literal

import dask.bag as db
import polars as pl
import pyopenms as oms
from pydantic import Field

from ..ms_tools import (
    AdductDetector,
    AdductDetectorConfig,
    Centrizer,
    CentrizerConfig,
    FeatureFinder,
    FeatureFinderConfig,
    FeatureLinker,
    FeatureLinkerConfig,
    MSTool,
    MSToolConfig,
    RTAligner,
    RTAlignerConfig,
    TICSmoother,
    TICSmootherConfig,
)
from ..snaps.MassDataModule.data_module.linkers import QueueLevelLinker, SampleLevelLinker
from ..snaps.MassDataModule.data_module.maps import ConsensusMap, FeatureMap, SpectrumMap, XICMap
from ..snaps.MassDataModule.data_module.wrappers import MetaMSDataWrapper, MetaMSExperimentDataQueue, OpenMSDataWrapper
from .reference_analysis import ReferenceFeatureFinder, ReferenceFeatureFinderConfig


class ExperimentAnalysisConfig(MSToolConfig):

    use_tic_smoother: bool = Field(
        default=True,
        description="是否使用TIC平滑"
    )
    tic_smoother_config: TICSmootherConfig = Field(
        default=TICSmootherConfig(),
        description="TIC平滑的配置"
    )
    use_centrizer: bool = Field(
        default=False,
        description="是否使用质心化校正"
    )
    centrizer_config: CentrizerConfig = Field(
        default=CentrizerConfig(),
        description="质心化校正的配置"
    )
    feature_finder_config: FeatureFinderConfig = Field(
        default=FeatureFinderConfig(),
        description="特征提取的配置"
    )
    use_rt_aligner: bool = Field(
        default=True,
        description="是否使用RT对齐"
    )
    rt_aligner_config: RTAlignerConfig = Field(
        default=RTAlignerConfig(),
        description="RT对齐的配置"
    )
    reference_analysis_config: ReferenceFeatureFinderConfig = Field(
        default=ReferenceFeatureFinderConfig(),
        description="参考文件的分析配置，用于RT对齐"
    )
    use_adduct_detector: bool = Field(
        default=True,
        description="是否使用加合物检测"
    )
    adduct_detector_config: AdductDetectorConfig = Field(
        default=AdductDetectorConfig(),
        description="加合物检测的配置"
    )
    use_feature_linker: bool = Field(
        default=True,
        description="是否使用特征链接（如果只有一个输入文件，则永远不会进行特征链接）"
    )
    feature_linker_config: FeatureLinkerConfig = Field(
        default=FeatureLinkerConfig(),
        description="特征链接的配置（如果只有一个输入文件，则不会进行特征链接）"
    )
    worker_type: Literal["processes", "threads", "synchronous"] = Field(
        default="processes",
        description="使用的工作类型，\
            `processes`表示使用多进程，\
            `threads`表示使用多线程 \
            `synchronous`表示使用单线程同步调度，在调式时非常有用"
    )
    num_workers: int | None = Field(
        default=None,
        description="并行计算的线程数/进程数"
    )
    num_threads_per_worker: int = Field(
        default=4,
        description="每个worker的线程数"
    )
    num_high_load_worker: int | None = Field(
        default=None,
        description="高负载工作worker的最大数量",
    )

class ExperimentAnalysisWorkerError(Exception):

    def __init__(self, message, file_path, worker_error=None):
        super().__init__(message)
        self.file_path = file_path
        self.worker_error = worker_error

    def __str__(self):
        error_message = f"ExperimentAnalysisWorkerError: {self.args[0]}"
        if self.worker_error:
            error_message += rf"\Worker error: {self.worker_error}"
        error_message += f"\nFile path: {self.file_path}"
        return error_message

class ExperimentAnalysisError(Exception):

    def __init__(self, worker_errors):
        super().__init__("Experiment analysis encountered one or more worker errors.")
        self.worker_errors = worker_errors

    def __str__(self):
        error_message = "ExperimentAnalysisError: Experiment analysis encountered one or more worker errors.\n"
        for error in self.worker_errors:
            error_message += f" - {error}\n"
        return error_message.strip()

    @classmethod
    def check(cls, worker_outputs: list[tuple[MetaMSDataWrapper, SampleLevelLinker] | ExperimentAnalysisWorkerError]):
        worker_errors = [
            worker_output \
                for worker_output in worker_outputs \
                if isinstance(worker_output, ExperimentAnalysisWorkerError)
        ]
        if len(worker_errors) > 0:
            raise cls(worker_errors)

class ExperimentAnalysis(MSTool):

    config_type = ExperimentAnalysisConfig
    config: ExperimentAnalysisConfig

    @staticmethod
    def _single_file_pipeline(
        runtime_config: ExperimentAnalysisConfig,
        exp_file_path: str,
        ref: oms.MSExperiment | oms.FeatureMap | bytes | None = None,
        save_dir_path: str | None = None,
        semaphore: T_Semaphore | P_Semaphore | No_Semaphore = No_Semaphore(),
    ) -> tuple[MetaMSDataWrapper, SampleLevelLinker] | ExperimentAnalysisWorkerError:

        # higly loaded worker
        with semaphore:
            open_ms_wrapper = OpenMSDataWrapper(file_paths=[exp_file_path])
            open_ms_wrapper.init_exps(
                worker_type="synchronous",
                num_workers=1,
            )
            if runtime_config.use_tic_smoother:
                open_ms_wrapper = TICSmoother(config=runtime_config.tic_smoother_config)(
                    open_ms_wrapper,
                    worker_type="synchronous",
                    num_workers=1,
                )
            if runtime_config.use_centrizer:
                open_ms_wrapper = Centrizer(config=runtime_config.centrizer_config)(
                    open_ms_wrapper,
                    worker_type="synchronous",
                    num_workers=1,
                )
            open_ms_wrapper = FeatureFinder(config=runtime_config.feature_finder_config)(
                open_ms_wrapper,
                worker_type="synchronous",
                num_workers=1,
            )
            if runtime_config.use_rt_aligner and ref is not None:
                if isinstance(ref, (oms.MSExperiment, oms.FeatureMap)):
                    open_ms_wrapper.ref_for_align = ref
                else:
                    open_ms_wrapper.ref_for_align = oms.MSExperiment()
                    oms.MzMLFile().loadBuffer(ref, open_ms_wrapper.ref_for_align)
                open_ms_wrapper = RTAligner(config=runtime_config.rt_aligner_config)(
                    open_ms_wrapper,
                    worker_type="synchronous",
                    num_workers=1,
                )
            if runtime_config.use_adduct_detector:
                open_ms_wrapper = AdductDetector(config=runtime_config.adduct_detector_config)(
                    open_ms_wrapper,
                    worker_type="synchronous",
                    num_workers=1,
                )

        # lowly loaded worker
        meta_ms_wrapper = MetaMSDataWrapper(
            file_paths=open_ms_wrapper.file_paths,
            exp_names=pl.Series([open_ms_wrapper.exp_names[0]]),
            spectrum_maps=[SpectrumMap.from_oms(
                open_ms_wrapper.exps[0], open_ms_wrapper.exp_names[0],
                worker_type="threads" \
                    if runtime_config.worker_type != "synchronous" \
                    else "synchronous",
                num_workers=runtime_config.num_threads_per_worker,
            )],
            xic_maps=[XICMap.from_oms(
                open_ms_wrapper.exps[0],open_ms_wrapper.exp_names[0],
            )],
            feature_maps=[FeatureMap.from_oms(
                open_ms_wrapper.exp_names[0],
                open_ms_wrapper.features[0],
                open_ms_wrapper.chromatogram_peaks[0],
                worker_type="threads" \
                    if runtime_config.worker_type != "synchronous" \
                    else "synchronous",
                num_workers=runtime_config.num_threads_per_worker,
            )],
        )
        sample_level_linker = SampleLevelLinker.link_exp_data_from_cls(
            exp_name=meta_ms_wrapper.exp_names[0],
            spectrum_map=meta_ms_wrapper.spectrum_maps[0],
            feature_map=meta_ms_wrapper.feature_maps[0],
        )
        if save_dir_path is not None:
            meta_ms_wrapper.save(
                save_dir_path,
                worker_type="threads" \
                    if runtime_config.worker_type != "synchronous" \
                    else "synchronous",
                num_workers=3,
            )
            sample_level_linker.save(
                os.path.join(save_dir_path, "exp_datas", sample_level_linker.exp_name, "linker.sqlite")
            )
        return meta_ms_wrapper,sample_level_linker

    def __call__(
        self,
        queue_name: str,
        exp_file_paths: list[str],
        ref_file_paths: list[str] | None = None,
        save_dir_path: str | None = None,
        *args,
        **kwargs,
    ) -> tuple[MetaMSExperimentDataQueue, QueueLevelLinker]:

        runtime_config = self.config.get_runtime_config(**kwargs)

        if len(exp_file_paths) > 1 and runtime_config.use_feature_linker:
            if ref_file_paths is None:
                ref_file_paths = exp_file_paths
            ref_exp = ReferenceFeatureFinder(config=runtime_config.reference_analysis_config)(
                ref_file_paths,
                result_type="experiment",
                worker_type=runtime_config.worker_type \
                    if runtime_config.worker_type != "processes" \
                    else "threads",
                num_workers=runtime_config.num_workers,
            )
        else:
            ref_exp = None
        if runtime_config.worker_type == "processes" and ref_exp is not None:
            ref_exp_buffer = oms.String()
            oms.MzMLFile().storeBuffer(ref_exp_buffer, ref_exp)
            ref_exp = ref_exp_buffer.c_str()
        if runtime_config.num_high_load_worker is None:
            semaphore = No_Semaphore()
        elif runtime_config.worker_type == "processes":
            manager = multiprocessing.Manager()
            semaphore = manager.Semaphore(runtime_config.num_high_load_worker)
        else:
            semaphore = threading.Semaphore(runtime_config.num_high_load_worker)
        single_file_pipeline = partial(
            ExperimentAnalysis._single_file_pipeline,
            runtime_config,
            ref=ref_exp,
            save_dir_path=save_dir_path,
            semaphore=semaphore,
        )
        exp_bag = db.from_sequence(exp_file_paths,npartitions=runtime_config.num_workers)
        results_bag = exp_bag.map(single_file_pipeline)
        results_list = results_bag.compute(
            scheduler=runtime_config.worker_type,
            num_workers=runtime_config.num_workers,
        )
        meta_ms_wrapper_list = []
        sample_level_linker_list = []
        for wrapper,linker in results_list:
            meta_ms_wrapper_list.append(wrapper)
            sample_level_linker_list.append(linker)
        meta_ms_wrapper = MetaMSDataWrapper.merge(meta_ms_wrapper_list)
        meta_ms_wrapper.queue_name = queue_name
        if runtime_config.use_feature_linker and len(exp_file_paths) > 1:
            feature_list = [feature_map.get_oms_feature_map() for feature_map in meta_ms_wrapper.feature_maps]
            open_ms_wrapper = OpenMSDataWrapper(
                file_paths=meta_ms_wrapper.file_paths,
                exp_names=meta_ms_wrapper.exp_names,
                features=feature_list,
            )
            open_ms_wrapper = FeatureLinker(runtime_config.feature_linker_config)(open_ms_wrapper)
            meta_ms_wrapper.consensus_map = ConsensusMap.from_oms(
                meta_ms_wrapper.queue_name,
                open_ms_wrapper.consensus_map,
                worker_type=runtime_config.worker_type \
                    if runtime_config.worker_type != "processes" \
                    else "threads",
                num_workers=runtime_config.num_workers,
            )
        queue_level_linker = QueueLevelLinker(
            queue_name=meta_ms_wrapper.queue_name,
            exp_names=meta_ms_wrapper.exp_names,
            sample_level_linkers=sample_level_linker_list,
        )
        if meta_ms_wrapper.consensus_map is not None:
            queue_level_linker.link_feature_consensus(meta_ms_wrapper.consensus_map)
        if save_dir_path is not None:
            queue_level_linker.save(os.path.join(save_dir_path, "linker.sqlite"))
            meta_ms_wrapper.save_metadata(save_dir_path)
        meta_ms_wrapper = MetaMSExperimentDataQueue(**meta_ms_wrapper.model_dump())
        return meta_ms_wrapper, queue_level_linker
