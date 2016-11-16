import sys
sys.path.insert(0,'D:/users/ysa/shyft_fork')
sys.path.insert(0,'D:/users/ysa/shyft_main/shyft')
import os

import numpy as np
from dateutil.parser import *
from shyft import api
from shyft.orchestration.configuration.yaml_configs import YAMLForecastConfig
from shyft.orchestration.simulators.config_simulator import ConfigForecaster

from shyft_viz.data_extractors.shyft_multi_regmod_data import DataExtractor
from shyft_viz.data_extractors.smg_discharge_data import SMGDataExtractor
from shyft_viz.data_viewer import Viewer

utc = api.Calendar()
config_dir = "D:/users/ysa/shyft_config/yaml"
shop_table = np.load("D:/users/ysa/shyft_config/yaml/shop_table.npy")

region_names = ['Nore',
                'Mår',
                'Tokke',
                'Ulla-Førre',
                'Folgefonn',
                'Tyssedal',
                'Sima',
                'Bjølvo',
                'Vik',
                'Leirdøla',
                'Jostedal',
                'Høyanger',
                'Grytten',
                'Aura',
                'Trollheim',
                'Svorka',
                'Nea-Nidelv',
                'Røssåga',
                'Rana',
                'Svartisen',
                'Kobbelv',
                'Skjomen',
                'Barduelva',
                'Alta',
                'Adamselv'
                ]

class Container(object):
    def __init__(self):
        pass

class Models(object):
    def __init__(self):
        self.region = Container()
        #self.shop_module = Container()
        [setattr(self.region, rg_name.replace('-','_'), Region(rg_name)) for rg_name in region_names]

class Region(object):
    def __init__(self, rg_name):
        self.rg_name = rg_name
        t_now = api.utctime_now()  # For usage with current date-time
        #t_now = utc.time(2016, 9, 3)  # For usage with any specified date-time
        self.t = utc.trim(t_now, api.Calendar.DAY)

        self.config_file = os.path.join(config_dir, "simulation_config_realtime-run.yaml")
        #self.cfg = YAMLForecastConfig(self.config_file, rg_name, ['AROME'], forecast_time=self.t)
        self.rg_name = rg_name
        self.simulator = None
        self.shop_module_names = np.unique(shop_table['shop_delfelt'][shop_table['shop_flomvassdrag'] == self.rg_name])
        self.subcat_ids_in_shop_module = [shop_table['subcat_id'][shop_table['shop_delfelt'] == nm] for nm in self.shop_module_names]

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
                      time_marker=self.t, data_ext_pt=None, background=self.rg_name, default_var='q_avg', default_ds='sim_ShopModule')
        #return SMGDataExtractor(module_data_ext, self.cfg.sim_config.get_reference_repo())


models=Models()