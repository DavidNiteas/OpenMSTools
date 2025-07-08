from abc import ABC, abstractmethod

from .param import OpenMSParam


class OpenMSMethod(ABC):

    @abstractmethod
    def getParameters(self) -> OpenMSParam:
        pass

    @abstractmethod
    def getDefaults(self) -> OpenMSParam:
        pass
