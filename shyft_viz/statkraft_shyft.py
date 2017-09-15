import os
import sys
from shyft import api

#sys.path.insert(0,os.path.join(os.getenv('SHYFTDATA', '.'), '..', 'shyft_config', 'orchestration'))

from statkraft_shyft_config import ConfigGenerator
from statkraft_shyft_simulator import Simulator
from statkraft_shyft_evaluator import Evaluator

from .view_funcs import ViewFuncs

viewers = ViewFuncs()
utc = api.Calendar()

class Regions(object):
    def __init__(self):
        region_names = list(ConfigGenerator.MASTER_CFG.keys())
        [setattr(self, rg_name.replace('-', '_'), Region(rg_name, ConfigGenerator())) for rg_name in
         region_names]

class Region(object):
    def __init__(self, rg_name, cfg_gen):
        self.region_name = rg_name
        self.cfg_gen = cfg_gen

    def get_shop_module_names(self):
        shop_cfg = self.cfg_gen.MASTER_CFG.get(self.region_name)
        return [m['module_name']for m in shop_cfg if m['module_group'] == 'SHOP']

    def get_evaluator(self, model_stack='ptgsk', dxy='2000', targets_filter=None, adjust_discharge_state=True):
        region_model_id = '{}#{}#{}m'.format(self.region_name, model_stack, dxy)
        return Evaluator(self.cfg_gen, region_model_id, targets_filter=targets_filter, adjust_discharge_state=adjust_discharge_state)

    def evaluate_continuous_sim_and_view(self, hist_run_period, model_stack='ptgsk', dxy='2000', targets_filter=None,
                                         adjust_discharge_state=True, input_src='AROME', intervals=None,init_state=None,
                                         reg_param=None,catch_params={}):
        evaluator = self.get_evaluator(model_stack=model_stack, dxy=dxy, targets_filter=targets_filter,
                                       adjust_discharge_state=adjust_discharge_state)
        res = evaluator.evaluate_historical_sim(hist_run_period, input_src=input_src, intervals=intervals,
                                                init_state=init_state, reg_param=reg_param, catch_params=catch_params)
        print(res)
        return evaluator, viewers.view_continuous_sim(evaluator)

    def evaluate_forecast_sim_and_view(self, run_time, fc_eval_period, model_stack='ptgsk', dxy='2000', targets_filter=None,
                                       adjust_discharge_state=True, historical_run_start_t=None, hist_input_src='AROME',
                                       init_state=None,reg_param=None, update_state=True, intervals=None,
                                       fc_run_type='AROME', fc_delay=0):
        evaluator = self.get_evaluator(model_stack=model_stack, dxy=dxy, targets_filter=targets_filter,
                                       adjust_discharge_state=adjust_discharge_state)
        res = evaluator.evaluate_forecast_sim(run_time, fc_eval_period, historical_run_start_t=historical_run_start_t, hist_input_src=hist_input_src,
                                              init_state=init_state,reg_param=reg_param, update_state=update_state, intervals=intervals,
                                              fc_run_type=fc_run_type, fc_delay=fc_delay)
        print(res)
        return evaluator, viewers.view_hindcast_sim(evaluator)

regions = Regions()
