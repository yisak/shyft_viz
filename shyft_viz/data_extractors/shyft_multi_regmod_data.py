from shyft import api

import numpy as np

from .shyft_regmod_data import CellDataExtractor, SubcatDataExtractor


class DataExtractor(object):
    def __init__(self, rm_lst, agg = False, catch_select=None, clip=False, catch_names=None, geom=None):
        if agg:
            self.cell_data_ext = [SubcatDataExtractor(rm, catch_select=catch_select, clip=clip, catch_names=catch_names, geom=geom)
                                  for rm in rm_lst]
        else:
            self.cell_data_ext = [CellDataExtractor(rm, catch_select=catch_select, clip=clip, catch_names=catch_names, geom=geom)
                                  for rm in rm_lst]
        self.ts_fetching_lst = self.cell_data_ext[0].ts_fetching_lst
        self.map_fetching_lst = self.cell_data_ext[0].map_fetching_lst
        self.n = [data_ext.t_ax.n for data_ext in self.cell_data_ext]
        self.n_cum = np.cumsum(self.n)
        self.cum = np.cumsum([0]+self.n)
        self.geom = self.cell_data_ext[0].geom
        self.catch_names = self.cell_data_ext[0].catch_names
        self.var_units = self.cell_data_ext[0].var_units
        self.cal = api.Calendar()
        self.t_ax = api.Timeaxis(self.cell_data_ext[0].t_ax.start, self.cell_data_ext[0].t_ax.delta_t, sum(self.n))

    def time_num_2_str(self, ti):
        return self.cal.to_string(self.t_ax.time(ti))

    def get_map(self, var_name, cat_id_lst, t):
        data_ext_idx = np.searchsorted(self.n_cum, t, side='right')
        t_idx = t-self.cum[data_ext_idx]
        return self.cell_data_ext[data_ext_idx].get_map(var_name, cat_id_lst, int(t_idx))

    def get_ts(self, var_name, cell_idx):
        return np.concatenate([data_ext.get_ts(var_name, cell_idx) for data_ext in self.cell_data_ext])

    def get_geo_data(self, var_name, cat_id_lst):
        return self.cell_data_ext[0].get_geo_data(var_name, cat_id_lst)