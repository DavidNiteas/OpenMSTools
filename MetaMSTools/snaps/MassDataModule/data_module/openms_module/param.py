from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..config_module import TomlConfig


class OpenMSParam(ABC):

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def keys(self) -> list[bytes]:
        pass

    @abstractmethod
    def values(self) -> list[Any]:
        pass

    @abstractmethod
    def __getitem__(self, key: bytes) -> Any:
        pass

    @abstractmethod
    def descriptions(self) -> list[str]:
        pass

    @abstractmethod
    def getDescription(self,key:bytes) -> str:
        pass

class OpenMSMethodParam(ABC, TomlConfig):

    @abstractmethod
    def dump2openms(self) -> dict[str, Any]:
        pass

class OpenMSMethodParamWrapper(OpenMSMethodParam):

    wrapper_name: ClassVar[str]

    def dump2openms(self) -> dict[str, Any]:
        param_dict = {}
        for field_name in self.model_fields:
            field_data = getattr(self, field_name)
            if isinstance(field_data, OpenMSMethodParam):
                for openms_key, openms_value in field_data.dump2openms().items():
                    param_dict[f"{self.wrapper_name}:{openms_key}"] = openms_value
            else:
                param_dict[f"{self.wrapper_name}:{field_name}"] = field_data
        return param_dict
