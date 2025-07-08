from __future__ import annotations

from typing import Protocol, TypeVar

from .experiment_module import MetaMSDataWrapper
from .openms_module import OpenMSDataWrapper


class DataWrapper(Protocol):

    @classmethod
    def merge(cls, data_wrappers: list[DataWrapper]) -> DataWrapper:
        ...

    def split(self) -> list[DataWrapper]:
        ...

data_wrapper = TypeVar("data_wrapper",bound=DataWrapper)

def merge_data_wrapper(data_wrappers: list[data_wrapper]) -> data_wrapper:
    return data_wrappers[0].merge(data_wrappers[1:])

def split_data_wrapper(data_wrapper: data_wrapper) -> list[data_wrapper]:
    return data_wrapper.split()

__all__ = [
    "DataWrapper",
    "OpenMSDataWrapper",
    "MetaMSDataWrapper",
    "merge_data_wrapper",
    "split_data_wrapper",
]
