from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

data_wrapper_type = TypeVar("data_wrapper_type", bound="BaseWrapper")
wrapper_item_obj_type = TypeVar("_wrapper_item_obj_type")

class BaseWrapper(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})
    _column_attributes: ClassVar[list[str]] = []

    @staticmethod
    def _get_size(obj: list | None) -> int:
        if obj is None:
            return 0
        return len(obj)

    def size(self) -> int:
        return max(self._get_size(getattr(self, attr)) for attr in self._column_attributes)

    def _split_obj(
        self,
        obj: list[wrapper_item_obj_type] | None,
        n: int
    ) -> wrapper_item_obj_type | None:
        if self._get_size(obj) != self.size():
            return None
        return obj[n]

    def split(self) -> list[Self]:
        data_wrappers = []
        for i in range(self.size()):
            data_dict = {}
            for attr_name in type(self).model_fields:
                if attr_name in self._column_attributes:
                    data_dict[attr_name] = self._split_obj(getattr(self, attr_name), i)
                else:
                    data_dict[attr_name] = getattr(self, attr_name)
            data_wrapper = type(self)(**data_dict)
            data_wrappers.append(data_wrapper)
        return data_wrappers

    @staticmethod
    def _merge_step(obj_1: list | None, obj_2: list | None) -> list | None:
        if obj_1 is None and obj_2 is None:
            return None
        elif obj_1 is None and obj_2 is not None:
            return obj_2
        elif obj_1 is not None and obj_2 is None:
            return obj_1 + [obj_2]
        else:
            return obj_1 + obj_2

    @classmethod
    def merge(cls, data_wrappers: list[data_wrapper_type]) -> data_wrapper_type:
        merged_data_dict = {
            attr_name: None \
                for attr_name in cls.model_fields
        }
        for data_wrapper in data_wrappers:
            for attr_name in merged_data_dict:
                if attr_name in cls._column_attributes:
                    merged_data_dict[attr_name] = cls._merge_step(
                        merged_data_dict[attr_name],
                        getattr(data_wrapper, attr_name)
                    )
                elif merged_data_dict[attr_name] is None:
                    merged_data_dict[attr_name] = getattr(data_wrapper, attr_name)
        return cls(**merged_data_dict)
