import os
import sys
from dateutil.parser import *
from shyft import api

#sys.path.insert(0,os.path.join(os.getenv('SHYFTDATA', '.'), '..', 'shyft_config', 'orchestration'))

from statkraft_shyft_config import ConfigGenerator
from statkraft_shyft_simulator import Simulator

from .data_extractors.shyft_multi_regmod_data import DataExtractor
from .data_extractors.shyft_regmod_data import SubcatDataExtractor
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
        self.cfg_gen = cfg_gen
        self.region_model_id = '{}#ptgsk#2000m'.format(rg_name)
        self.simulator = None

    @property
    def shop_module_info(self):
        shop_cfg = self.cfg_gen.MASTER_CFG.get(self.region_model_id.split('#')[0])
        return [{'name':m['module_name'], 'subcat_ids':[c['subcat_id'] for c in m['subcats_in_module']]} for m in shop_cfg if
              m['module_group'] == 'SHOP']

    def view(self, t=None, plots={}, fetch_ref_data=True, arome_delay=api.deltahours(3), ec_delay=api.deltahours(8),
             mode='forecast'):
        if mode not in ['historical', 'forecast']:
            print("kwarg 'mode' should be either 'forecast' or 'historical'!")
            return
        if t is not None:
            t_datetime = parse(t)
            t_start = utc.time(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second)
        else:
            t_start = api.utctime_now()
        if self.simulator is None:
            self.simulator = Simulator(self.cfg_gen, self.region_model_id)
        else:
            if self.simulator.region_model_id != self.region_model_id:
                self.simulator = Simulator(self.cfg_gen, self.region_model_id)
        self.simulator.run_system(t_start, save_end_state=False, save_result_timeseries=False,
                                  update_state=True, arome_fc_delay=arome_delay, ec_fc_delay=ec_delay)
        sim = self.simulator
        rm_update = sim.region_model_update
        shop_module_names, subcat_ids_in_shop_module = zip(*[(m['name'],m['subcat_ids']) for m in self.shop_module_info])
        geom = SubcatDataExtractor(rm_update, catch_select=subcat_ids_in_shop_module,
                                   catch_names=shop_module_names).geom
        if mode == 'forecast':
            rm_dct = {'arome00_ec00': [rm_update, sim.region_model_arome00, sim.region_model_arome00_ec00],
                      'arome06_ec00': [rm_update, sim.region_model_arome06, sim.region_model_arome06_ec00],
                      'arome12_ec00': [rm_update, sim.region_model_arome12, sim.region_model_arome12_ec00],
                      'arome18_ec00': [rm_update, sim.region_model_arome18, sim.region_model_arome18_ec00],
                      'arome00_ec12': [rm_update, sim.region_model_arome00, sim.region_model_arome00_ec12],
                      'arome06_ec12': [rm_update, sim.region_model_arome06, sim.region_model_arome06_ec12],
                      'arome12_ec12': [rm_update, sim.region_model_arome12, sim.region_model_arome12_ec12],
                      'arome18_ec12': [rm_update, sim.region_model_arome18, sim.region_model_arome18_ec12]
                      }
            custom_plots = {'arome00-ec12_PTQ': {'arome00_ec12': ['temp', 'q_avg', 'prec']},
                            'arome00,18-ec12,00_Q': {k: ['q_avg'] for k in
                                                     ['arome00_ec12', 'arome18_ec12', 'arome00_ec00']}}

            module_data_ext = {k: DataExtractor(v, catch_select=subcat_ids_in_shop_module,
                                                catch_names=shop_module_names, geom=geom, agg=True) for k, v in rm_dct.items()}
            default_ds = 'arome00_ec12'
            time_marker = t_start
        if mode == 'historical':
            rm_preupdate = sim.region_model_preupdate
            module_data_ext = {'arome_concat': SubcatDataExtractor(rm_preupdate, catch_select=subcat_ids_in_shop_module,
                                                      catch_names=shop_module_names)}
            custom_plots = {'arome_concat_PTQ': {'arome_concat': ['temp', 'q_avg', 'prec']},
                            'QsimVSQobs': {k: ['q_avg'] for k in ['arome_concat']}}
            default_ds = 'arome_concat'
            time_marker = None

        custom_plots.update(plots)
        if fetch_ref_data:
            smg_data_ext = {'Qobs_SMG': SMGDataExtractor(list(module_data_ext.values())[0], sim.reference_ts_repo, sim.reference_ts_spec)}
            module_data_ext.update(smg_data_ext)
            [p.update({'Qobs_SMG': ['q_avg']}) for nm, p in custom_plots.items() if 'Q' in nm]
        return Viewer(module_data_ext,
                      custom_plots,
                      time_marker=time_marker, data_ext_pt=None, background=None, default_var='q_avg',
                      default_ds=default_ds)

models=Models()