from collections.abc import Hashable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import trio
from rich.console import Console
from rich.progress import GetTimeCallable, Progress, ProgressColumn, TaskID

from .base_tools import get_kv_pairs


class ProgressManager(Progress):

    def __init__(
        self,
        *columns: Union[str, ProgressColumn],
        console: Optional[Console] = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30.0,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: Optional[GetTimeCallable] = None,
        disable: bool = False,
        expand: bool = False,
    ):
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )
        self.task_name_id_map = {}

    @property
    def Name2ID(self) -> Dict[Hashable,TaskID]:
        return self.task_name_id_map

    def add_task(
        self,
        task_name: Hashable,
        description: Optional[str] = None,
        start: bool = True,
        total: Optional[float] = None,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> None:
        if description is None:
            description = f'{str(task_name)}:'
        task_id = super().add_task(
            description,
            start=start,
            total=total,
            completed=completed,
            visible=visible,
            **fields,
        )
        self.Name2ID[task_name] = task_id

    def update(
        self,
        task_name: Hashable,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if task_name not in self.Name2ID:
            self.add_task(task_name)
        task_id = self.Name2ID[task_name]
        super().update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

    def update_total(
        self,
        task_name: Hashable,
        advance: float = 1,
        completed: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if task_name in self.Name2ID:
            if completed is None:
                if self._tasks[self.Name2ID[task_name]].total is None:
                    self.update(task_name, total=advance, description=description)
                else:
                    self.update(task_name, total=self._tasks[self.Name2ID[task_name]].total+advance, description=description)
            else:
                self.update(task_name, total=completed, description=description)

def use_coroutine(func):

    async def coroutine(
        key: Union[Hashable, int],
        data_dict: Dict[Hashable,Dict[int,Any]],
        args: Tuple[Any],
        kwargs: Dict[str,Any],
        progress: Union[ProgressManager,None] = None,
    ):
        data_dict[key] = func(*args, **kwargs)

        if progress is not None:
            progress.update(func.__name__, advance=1)

    coroutine.__inner_func_name__ = func.__name__

    return coroutine

async def start_coroutine(
    func: Callable,
    func_inps: Union[
        Dict[Hashable, Tuple[tuple,dict]],
        List[Tuple[tuple,dict]],
    ],
    data_dict: Dict[Hashable,Any],
    progress: Union[ProgressManager,None] = None,
):
    async with trio.open_nursery() as nursery:
        for key, (args, kwargs) in get_kv_pairs(func_inps):
            nursery.start_soon(func, key, data_dict, args, kwargs, progress)

def run_coroutine(
    func: Callable,
    func_inps: Union[
        Dict[Hashable, Tuple[tuple,dict]],
        List[Tuple[tuple,dict]],
    ],
    data_dict: Optional[Dict[Hashable,Dict[int,Any]]] = None,
    coroutine: Optional[
        Callable[
            [
                Callable,
                Union[
                    Dict[Hashable, Tuple[tuple,dict]],
                    List[Tuple[tuple,dict]],
                ],
                Dict[Hashable,Any],
            ],None
        ]
    ] = None,
    use_progress: bool = True,
    progress: Union[Progress,None] = None,
    description: str = None,
) -> Dict:
    need_close = False
    if data_dict is None:
        data_dict = {}
    if coroutine is None:
        coroutine = start_coroutine
    if use_progress:
        if progress is None:
            progress = ProgressManager()
            progress.add_task(task_name=func.__inner_func_name__,total=len(func_inps),description=description)
            progress.start()
            need_close = True
    else:
        progress = None
    trio.run(coroutine, func, func_inps, data_dict, progress)
    if need_close:
        progress.stop()
    return data_dict
