from __future__ import annotations

from typing import ClassVar, Literal

import dask
import dask.bag as db
import pyopenms as oms
from pydantic import Field

from ..module_abc import BaseWrapper
from .io import load_exp_file


class OpenMSDataWrapper(BaseWrapper):

    '''
    OpenMSDataWrapper是对一组OpenMS实验数据的封装。

    其中包含：
    - 实验文件路径列表
    - 实验名称列表
    - OpenMS的MSExperiment对象列表
    - 质量踪迹列表
    - xic的色谱峰列表
    - 特征图列表
    - 用于对齐的参考特征图
    - 用于对齐的转换描述列表
    - 整个队列所有特征的共识图
    - 质谱数据队列的名称

    以上参数均为可选参数，可以根据需要选择性使用。
    '''

    _column_attributes: ClassVar[list[str]] = [
        "file_paths",
        "exp_names",
        "exps",
        "mass_traces",
        "chromatogram_peaks",
        "features",
        "ref_feature_for_align",
        "trafos",
    ]

    file_paths: list[str] | None = Field(
        None,
        description="实验文件的路径列表"
    )
    exp_names: list[str] | None = Field(
        None,
        description="实验名称列表"
    )
    exps: list[oms.MSExperiment] | None = Field(
        None,
        description="OpenMS的MSExperiment对象列表"
    )
    mass_traces: list[list[oms.MassTrace]] | None = Field(
        None,
        description="每个MSExperiment对象中提取到的质量踪迹列表"
    )
    chromatogram_peaks: list[list[oms.ChromatogramPeak]] | None = Field(
        None,
        description="每个MSExperiment对象中提取到的xic的色谱峰列表"
    )
    features: list[oms.FeatureMap] | None = Field(
        None,
        description="每个实验文件中找到的特征图（OpenMS的FeatureMap对象）列表"
    )
    ref_for_align: oms.FeatureMap | oms.MSExperiment | None = Field(
        None,
        description="用于RT对齐的参考样本"
    )
    trafos: list[oms.TransformationDescription] | None = Field(
        None,
        description="用于对齐的转换描述（OpenMS的TransformationDescription对象）列表"
    )
    consensus_map: oms.ConsensusMap | None = Field(
        None,
        description="整个队列所有特征的共识图（OpenMS的ConsensusMap对象）"
    )
    queue_name: str = Field(
        None,
        description="质谱数据队列的名称"
    )

    def init_exps(
        self,
        worker_type: Literal["threads", "synchronous"] = "threads",
        num_workers: int | None = None,
    ):

        if self.file_paths is not None:
            file_bag = db.from_sequence(self.file_paths, npartitions=num_workers)
            file_bag = file_bag.map(load_exp_file)
            exp_name_bag = file_bag.pluck(0)
            exp_bag = file_bag.pluck(1)
            self.exp_names, self.exps = dask.compute(
                exp_name_bag, exp_bag,
                scheduler=worker_type,
                num_workers=num_workers,
            )

class OpenMSExperimentDataQueue(OpenMSDataWrapper):

    '''
    OpenMSExperimentDataQueue是对一组质谱实验数据队列的封装。
    其中包含：
    - 质谱数据队列的名称 （必选参数）
    - 实验文件路径列表
    - 实验名称列表
    - OpenMS的MSExperiment对象列表
    - 质量踪迹列表
    - xic的色谱峰列表
    - 特征图列表
    - 用于对齐的参考特征图
    - 用于对齐的转换描述列表
    - 整个队列所有特征的共识图
    - 质谱数据队列的质谱范围
    '''

    queue_name: str = Field(
        ...,
        description="质谱数据队列的名称"
    )
