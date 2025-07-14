from __future__ import annotations

from typing import ClassVar

import polars as pl
from pydantic import Field

from ..module_abc import BaseWrapper
from .consensus import ConsensusMap
from .features import FeatureMap
from .spectrums import SpectrumMap
from .xic import XICMap


class MetaMSDataWrapper(BaseWrapper):

    _column_attributes: ClassVar[list[str]] = [
        "file_paths",
        "exp_names",
        "spectrum_maps",
        "xic_maps",
        "feature_maps",
    ]

    queue_name: str | None = Field(
        None,
        description="队列名称"
    )
    file_paths: list[str] | None = Field(
        None,
        description="原始文件路径列表"
    )
    exp_names: pl.Series | None = Field(
        None,
        description="实验名称列表"
    )
    spectrum_maps: list[SpectrumMap] | None = Field(
        None,
        description="谱图数据列表"
    )
    xic_maps: list[XICMap] | None = Field(
        None,
        description="XIC数据列表"
    )
    feature_maps: list[FeatureMap] | None = Field(
        None,
        description="特征图数据列表"
    )
    consensus_map: ConsensusMap | None = Field(
        None,
        description="共识特征图数据"
    )

class MetaMSExperimentDataQueue(MetaMSDataWrapper):

    queue_name: str = Field(
        ...,
        description="队列名称"
    )
    file_paths: list[str] = Field(
        ...,
        description="原始文件路径列表"
    )
    exp_names: pl.Series = Field(
        ...,
        description="实验名称列表"
    )
    spectrum_maps: list[SpectrumMap] = Field(
        ...,
        description="谱图数据列表"
    )
    xic_maps: list[XICMap] = Field(
        ...,
        description="XIC数据列表"
    )
    feature_maps: list[FeatureMap] = Field(
        ...,
        description="特征图数据列表"
    )
