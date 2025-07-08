from abc import ABC, abstractmethod

from ..snaps.MassDataModule.data_module.configs import ConvertMethodConfig, OpenMSMethodConfig, TomlConfig
from ..snaps.MassDataModule.data_module.data_wrapper import OpenMSDataWrapper


class MSToolConfig(TomlConfig):

    pass

class MSTool(ABC):

    config_type:type[MSToolConfig] |\
                type[ConvertMethodConfig] |\
                type[OpenMSMethodConfig] |\
                type[TomlConfig]

    def __init__(
        self,
        config: MSToolConfig |\
                ConvertMethodConfig |\
                OpenMSMethodConfig |\
                TomlConfig |\
                None = None,
    ):
        if not isinstance(config, self.config_type):
            self.config = self.config_type()
        else:
            self.config = config

    @abstractmethod
    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:
        pass
