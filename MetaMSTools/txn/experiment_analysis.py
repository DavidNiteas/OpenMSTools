import concurrent.futures
import logging
import multiprocessing
import os
import threading
from contextlib import nullcontext
from typing import Literal

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
from ..snaps.MassDataModule.data_module import ConsensusMap, FeatureMap, SpectrumMap, XICMap, link_ms2_and_feature_map
from ..snaps.MassDataModule.data_module.data_wrapper import MetaMSDataWrapper, OpenMSDataWrapper
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
    num_workers: int = Field(
        default=os.cpu_count() if os.cpu_count() <= 60 else 60,
        description="并行计算的线程数/进程数"
    )
    high_load_worker_rate: float = Field(
        default=0.5,
        description="高负载工作的比例,用于负载均衡控制",
        ge=0.0, le=1.0,
    )
    error_mode: Literal["raise_in_worker", "raise_after_worker", "raise_at_pool"] = Field(
        default="raise_after_worker",
        description="当工作线程/进程发生错误时，选择在何时抛出异常?\
            `raise_in_worker`表示在工作线程/进程发生错误时抛出异常，\
            `raise_after_worker`表示在工作线程/进程结束时抛出异常，\
            `raise_at_pool`表示在工作池中抛出异常"
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
    def check(cls, worker_outputs: list[MetaMSDataWrapper | ExperimentAnalysisWorkerError]):
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

    def _single_file_pipeline(
        self,
        exp_file_path: str,
        ref: ConsensusMap | FeatureMap | None = None,
        semaphore: threading.Semaphore | nullcontext = nullcontext(),
    ) -> MetaMSDataWrapper | ExperimentAnalysisWorkerError:
        try:
            with semaphore:
                # 基于C++的高负载函数，使用信号量来实现均衡负载
                open_ms_wrapper = OpenMSDataWrapper(file_paths=[exp_file_path])
                open_ms_wrapper.init_exps()
                if self.config.use_tic_smoother:
                    open_ms_wrapper = TICSmoother(config=self.config.tic_smoother_config)(open_ms_wrapper)
                if self.config.use_centrizer:
                    open_ms_wrapper = Centrizer(config=self.config.centrizer_config)(open_ms_wrapper)
                open_ms_wrapper = FeatureFinder(config=self.config.feature_finder_config)(open_ms_wrapper)
                if self.config.use_rt_aligner and ref is not None:
                    ref_map = ref.get_oms_feature_map()
                    open_ms_wrapper.ref_feature_for_align = ref_map
                    open_ms_wrapper = RTAligner(config=self.config.rt_aligner_config)(open_ms_wrapper)
                if self.config.use_adduct_detector:
                    open_ms_wrapper = AdductDetector(config=self.config.adduct_detector_config)(open_ms_wrapper)
            # 基于Python的低负载函数
            meta_ms_wrapper = MetaMSDataWrapper(
                file_paths=open_ms_wrapper.file_paths,
                exp_names=open_ms_wrapper.exp_names,
                spectrum_maps=[SpectrumMap.from_oms(open_ms_wrapper.exps[0], open_ms_wrapper.exp_names[0])],
                xic_maps=[XICMap.from_oms(open_ms_wrapper.exps[0])],
                feature_maps=[FeatureMap.from_oms(
                    open_ms_wrapper.features[0],
                    open_ms_wrapper.chromatogram_peaks[0],
                    open_ms_wrapper.exp_names[0]
                )],
            )
            meta_ms_wrapper.ms2_feature_mapping = [link_ms2_and_feature_map(
                feature_map = meta_ms_wrapper.feature_maps[0],
                spectrum_map = meta_ms_wrapper.spectrum_maps[0],
                key_id = "feature",
                worker_mode = "threads" if self.config.worker_type == "synchronous" else "synchronous",
                num_workers = 1,
            )]
            return meta_ms_wrapper
        except Exception as e:
            if self.config.error_mode == "raise_in_worker":
                raise e
            return ExperimentAnalysisWorkerError(f"Worker处理文件时出错: {exp_file_path}", exp_file_path, str(e))

    def _single_file_pipeline_pool(
        self,
        exp_file_paths: list[str],
        ref: ConsensusMap | FeatureMap | None = None,
    ) -> list[MetaMSDataWrapper | ExperimentAnalysisWorkerError]:
        refs = [ref] * len(exp_file_paths)
        high_load_worker_num = int(self.config.num_workers * self.config.high_load_worker_rate)
        high_load_worker_num += 1 if high_load_worker_num == 0 else 0
        if self.config.worker_type == "processes":
            manager = multiprocessing.Manager().__enter__()
            semaphore = manager.Semaphore(high_load_worker_num)
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.config.num_workers)
        elif self.config.worker_type == "threads":
            semaphore = threading.Semaphore(high_load_worker_num)
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            semaphore = nullcontext()
            pool = None
        results = []
        if pool is not None:
            with pool as executor:
                futures = [executor.submit(
                    self._single_file_pipeline,
                    exp_file_path,
                    ref,
                    semaphore,
                ) for exp_file_path, ref in zip(exp_file_paths, refs)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if isinstance(result, ExperimentAnalysisWorkerError):
                        if self.config.error_mode == "raise_after_worker":
                            raise result
                        else:
                            logging.error(result)
                    results.append(result)
        else:
            for exp_file_path, ref in zip(exp_file_paths, refs):
                result = self._single_file_pipeline(exp_file_path, ref, semaphore)
                if isinstance(result, ExperimentAnalysisWorkerError):
                    if self.config.error_mode == "raise_after_worker":
                        raise result
                    else:
                        logging.error(result)
                results.append(result)
        if self.config.worker_type == "processes":
            manager.__exit__(None, None, None)
        if self.config.error_mode == "raise_at_pool":
            ExperimentAnalysisError.check(results)
        return results

    def __call__(
        self,
        exp_file_paths: list[str],
        ref_file_paths: list[str] | None = None,
    ) -> MetaMSDataWrapper:
        if len(exp_file_paths) > 1 and self.config.use_feature_linker:
            if ref_file_paths is None:
                ref_file_paths = exp_file_paths
            ref = ReferenceFeatureFinder(config=self.config.reference_analysis_config)(ref_file_paths)
        else:
            ref = None
        meta_ms_wrapper_list = self._single_file_pipeline_pool(exp_file_paths, ref)
        meta_ms_wrapper = MetaMSDataWrapper.merge(meta_ms_wrapper_list)
        if self.config.use_feature_linker and len(exp_file_paths) > 1:
            feature_list = [feature_map.get_oms_feature_map() for feature_map in meta_ms_wrapper.feature_maps]
            open_ms_wrapper = OpenMSDataWrapper(
                file_paths=meta_ms_wrapper.file_paths,
                exp_names=meta_ms_wrapper.exp_names,
                features=feature_list,
            )
            open_ms_wrapper = FeatureLinker(self.config.feature_linker_config)(open_ms_wrapper)
            meta_ms_wrapper.consensus_map = ConsensusMap.from_oms(open_ms_wrapper.consensus_map)
        return meta_ms_wrapper
