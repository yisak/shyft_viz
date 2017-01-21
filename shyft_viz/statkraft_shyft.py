import os
import sys
from dateutil.parser import *
from shyft import api

sys.path.insert(0,os.path.join(os.getenv('SHYFTDATA', '.'), '..', 'shyft_config', 'orchestration'))

from statkraft_shyft_config import ConfigGenerator
from statkraft_shyft_simulator import Simulator

from .data_extractors.shyft_multi_regmod_data import DataExtractor
from .data_extractors.smg_discharge_data import SMGDataExtractor
from .data_viewer import Viewer

utc = api.Calendar()


class Container(object):
    def __init__(self):
        pass

class Models(object):
    def __init__(self):
        self.region = Container()
        cfg_gen = ConfigGenerator()
        region_names = list(cfg_gen.MASTER_CFG.keys())
        [setattr(self.region, rg_name.replace('-', '_'), Region(rg_name, cfg_gen)) for rg_name in
         region_names]

class Region(object):
    def __init__(self, rg_name, cfg_gen):
        self.rg_name = rg_name
        t_now = api.utctime_now()  # For usage with current date-time
        #t_now = utc.time(2016, 9, 3)  # For usage with any specified date-time
        self.t = utc.trim(t_now, api.Calendar.HOUR)
        self.cfg_gen = cfg_gen
        shop_cfg = cfg_gen.MASTER_CFG.get(rg_name)
        self.region_model_id = '{}#ptgsk#1000m'.format(rg_name)
        self.simulator = None
        self.shop_module_names, self.subcat_ids_in_shop_module = zip(*[(m['module_name'],[c['subcat_id'] for c in m['subcats_in_module']]) for m in shop_cfg if m['module_group']=='SHOP'])

    def _get_config(self):
        pass

    def _get_simulator(self):
        pass

    #def _get_viewer(self):
    def view(self, t=None, plots={}, fetch_ref_data=False):
        if t is not None:
            t_datetime = parse(t)
            t_num = utc.time(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second)
            self.t = utc.trim(t_num, api.Calendar.HOUR)
        if self.simulator is None:
            self.simulator = Simulator(self.cfg_gen, self.region_model_id)
            self.simulator.run_system(self.t, save_end_state=False, save_result_timeseries=False)
        else:
            if t is not None:
                self.simulator = Simulator(self.cfg_gen, self.region_model_id)
                self.simulator.run_system(self.t, save_end_state=False, save_result_timeseries=False)
        sim = self.simulator
        rm_update = sim.region_model_update
        rm_dct = {'arome00_ec00': [rm_update, sim.region_model_arome00, sim.region_model_arome00_ec00],
                  'arome06_ec00': [rm_update, sim.region_model_arome06, sim.region_model_arome06_ec00],
                  'arome12_ec00': [rm_update, sim.region_model_arome12, sim.region_model_arome12_ec00],
                  'arome18_ec00': [rm_update, sim.region_model_arome18, sim.region_model_arome18_ec00],
                  'arome00_ec12': [rm_update, sim.region_model_arome00, sim.region_model_arome00_ec12],
                  'arome06_ec12': [rm_update, sim.region_model_arome06, sim.region_model_arome06_ec12],
                  'arome12_ec12': [rm_update, sim.region_model_arome12, sim.region_model_arome12_ec12],
                  'arome18_ec12': [rm_update, sim.region_model_arome18, sim.region_model_arome18_ec12]
                  }
        #rm_nm_arome00 = [nm for nm in rm_dct if 'arome00' in nm]
        module_data_ext = {k: DataExtractor(v, catch_select=self.subcat_ids_in_shop_module,
                                            catch_names=self.shop_module_names, agg=True) for k, v in rm_dct.items()}
        custom_plots = {'arome00-ec12_PTQ': {'arome00_ec12': ['temp', 'q_avg', 'prec']},
                        'arome00,18-ec12,00_Q': {k: ['q_avg'] for k in ['arome00_ec12','arome18_ec12','arome00_ec00']}}
        custom_plots.update(plots)
        if fetch_ref_data:
            smg_data_ext = {'Qobs_SMG': SMGDataExtractor(list(module_data_ext.values())[0], sim.reference_ts_repo, sim.reference_ts_spec)}
            module_data_ext.update(smg_data_ext)
            [p.update({'Qobs_SMG': ['q_avg']}) for nm, p in custom_plots.items() if 'Q' in nm]
        return Viewer(module_data_ext,
                      custom_plots,
                      time_marker=self.t, data_ext_pt=None, background=None, default_var='q_avg',
                      default_ds='arome00_ec12')

models=Models()