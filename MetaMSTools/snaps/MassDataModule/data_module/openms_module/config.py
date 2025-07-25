from abc import ABC
from typing import ClassVar

import pandas as pd
import pyopenms as oms
from pydantic.fields import FieldInfo

from ..config_module import TomlConfig
from .method import OpenMSMethod
from .param import OpenMSMethodParam, OpenMSParam


class OpenMSMethodConfig(ABC, TomlConfig):

    openms_method: ClassVar[type[OpenMSMethod]]

    @staticmethod
    def _is_openms_method_param(field_info: FieldInfo) -> bool:
        if field_info.json_schema_extra is not None:
            is_openms_method_param = field_info.json_schema_extra.get("is_openms_method_param", True)
        else:
            is_openms_method_param = True
        return is_openms_method_param

    @property
    def param(self) -> OpenMSParam:
        param = oms.Param()
        for key,f in type(self).model_fields.items():
            if self._is_openms_method_param(f):
                value = getattr(self, key)
                if isinstance(value, OpenMSMethodParam):
                    openms_dict = value.dump2openms()
                    for openms_key, openms_value in openms_dict.items():
                        param.setValue(openms_key, openms_value)
                else:
                    param.setValue(key, value)
        return param

    @property
    def param_info(self) -> pd.DataFrame:
        method = self.openms_method()
        default_param = method.getDefaults()
        default_config = self.__class__()
        infos = {"value":{},"default_value":{},"openms_default_value":{},"description":{}}
        names = default_param.keys()
        for name in names:
            infos["openms_default_value"][name.decode()] = default_param[name]
            infos["description"][name.decode()] = default_param.getDescription(name)
        for field_name,f in type(self).model_fields.items():
            if self._is_openms_method_param(f):
                field_data = getattr(self, field_name)
                if isinstance(field_data, OpenMSMethodParam):
                    for name, value in field_data.dump2openms().items():
                        infos["value"][name] = value
                else:
                    infos["value"][field_name] = field_data
        for field_name,f in type(default_config).model_fields.items():
            if self._is_openms_method_param(f):
                field_data = getattr(default_config, field_name)
                if isinstance(field_data, OpenMSMethodParam):
                    for name, value in field_data.dump2openms().items():
                        infos["default_value"][name] = value
                else:
                    infos["default_value"][field_name] = field_data
        infos = pd.DataFrame(infos)
        infos.index.name = "param_name"
        return infos
