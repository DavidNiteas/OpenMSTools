from .base_tools import oms,oms_exps,get_kv_pairs
from .async_tools import trio
from typing import Dict, Union

async def copy_ms_experiment_step(ms_exp: oms.MSExperiment, key:str, result_dict:dict, use_blank:bool=False):
    if use_blank:
        result_dict[key] = oms.MSExperiment()
    else:
        result_dict[key] = oms.MSExperiment(ms_exp)
    
async def copy_ms_experiments_step(ms_exps: Dict[str, oms.MSExperiment], result_dict:dict, use_blank:bool=False):
    async with trio.open_nursery() as nursery:
        for key, ms_exp in ms_exps.items():
            nursery.start_soon(copy_ms_experiment_step, ms_exp, key, result_dict, use_blank)

def copy_ms_experiments(ms_exps: oms_exps, use_blank:bool=False) -> oms_exps:
    if isinstance(ms_exps, oms.MSExperiment):
        return oms.MSExperiment(ms_exps)
    result_dict = {}
    if isinstance(ms_exps, dict):
        trio.run(copy_ms_experiments_step, ms_exps, result_dict, use_blank)
        return result_dict
    else:
        trio.run(copy_ms_experiment_step, dict(zip(range(len(ms_exps)), ms_exps)), result_dict, use_blank)
        return list(result_dict.values())
    
def merge_exps(ms_exps: oms_exps) -> oms.MSExperiment:
    merged_exp = oms.MSExperiment()
    for _,exp in get_kv_pairs(ms_exps):
        for spec in exp.getSpectra():
            merged_exp.addSpectrum(spec)
    return merged_exp