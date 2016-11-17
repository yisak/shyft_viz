import os
import yaml
from dateutil.parser import *
from shyft import api
from shyft.orchestration.configuration.yaml_configs import YAMLForecastConfig
from shyft.orchestration.simulators.config_simulator import ConfigForecaster

from .data_extractors.shyft_multi_regmod_data import DataExtractor
from .data_extractors.smg_discharge_data import SMGDataExtractor
from .data_viewer import Viewer

utc = api.Calendar()
config_dir = os.path.join(os.getenv('SHYFTDATA', '.'), '..', 'shyft_config', 'yaml')


class Container(object):
    def __init__(self):
        pass

class Models(object):
    def __init__(self):
        self.region = Container()
        #self.shop_module = Container()
        sim_cfg_file = os.path.join(config_dir, "simulation_config_realtime-run.yaml")
        shop_cfg_file = os.path.join(config_dir, "shop_config.yaml")
        with open(shop_cfg_file, encoding='utf8') as cfg:
            shop_cfg = yaml.load(cfg)
        region_names = list(shop_cfg.keys())
        [setattr(self.region, rg_name.replace('-','_'), Region(rg_name, sim_cfg_file, shop_cfg)) for rg_name in region_names]

class Region(object):
    def __init__(self, rg_name, sim_cfg_file, shop_cfg):
        self.rg_name = rg_name
        t_now = api.utctime_now()  # For usage with current date-time
        #t_now = utc.time(2016, 9, 3)  # For usage with any specified date-time
        self.t = utc.trim(t_now, api.Calendar.DAY)

        self.config_file = sim_cfg_file # os.path.join(config_dir, "simulation_config_realtime-run.yaml")
        #self.cfg = YAMLForecastConfig(self.config_file, rg_name, ['AROME'], forecast_time=self.t)
        shop_cfg = shop_cfg[rg_name]
        self.rg_name = rg_name
        self.simulator = None
        self.shop_module_names, self.subcat_ids_in_shop_module = zip(*[(m['module_name'],[c['subcat_id'] for c in m['subcats_in_module']]) for m in shop_cfg if m['module_group']=='SHOP'])


    def _get_config(self):
        pass

    def _get_simulator(self):
        pass

    #def _get_viewer(self):
    def view(self, t=None):
        if t is not None:
            t_datetime = parse(t)
            t_num = utc.time(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second)
            self.t = utc.trim(t_num, api.Calendar.DAY)
        if self.simulator is None:
            self.cfg = YAMLForecastConfig(self.config_file, self.rg_name, ['AROME'], forecast_time=self.t)
            self.simulator = ConfigForecaster(self.cfg)
            self.simulator.run(save_end_state=False, save_result_timeseries=False)
        else:
            if t is not None:
                self.cfg = YAMLForecastConfig(self.config_file, self.rg_name, ['AROME'], forecast_time=self.t)
                self.simulator = ConfigForecaster(self.cfg)
                self.simulator.run(save_end_state=False, save_result_timeseries=False)
        cell_data_ext = DataExtractor(self.simulator.historical_sim.region_model,
                                      self.simulator.forecast_sim['AROME'].region_model)
        module_data_ext = DataExtractor(self.simulator.historical_sim.region_model,
                                        self.simulator.forecast_sim['AROME'].region_model,
                                        catch_select=self.subcat_ids_in_shop_module,
                                        catch_names=self.shop_module_names, agg=True)
        smg_data_ext = SMGDataExtractor(module_data_ext, self.cfg.sim_config.get_reference_repo())
        return Viewer({'sim_Cell': cell_data_ext, 'sim_ShopModule': module_data_ext, 'obs_ShopModule': smg_data_ext},
                      {'PTQ': {'sim_ShopModule': ['temp', 'q_avg', 'prec'], 'obs_ShopModule': ['q_avg']}},
                      time_marker=self.t, data_ext_pt=None, background=None, default_var='q_avg', default_ds='sim_ShopModule')

models=Models()