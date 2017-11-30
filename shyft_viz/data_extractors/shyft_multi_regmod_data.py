from shyft import api

import numpy as np

from .shyft_regmod_data import CellDataExtractor, SubcatDataExtractor


class DataExtractor(object):
    def __init__(self, rm_lst, periods, agg=False, catch_select=None, clip=False, catch_names=None, geom=None):
        if agg:
            self.cell_data_ext = [SubcatDataExtractor(rm, period=period, catch_select=catch_select, clip=clip, catch_names=catch_names, geom=geom)
                                  for rm, period in zip(rm_lst, periods)]
        else:
            self.cell_data_ext = [CellDataExtractor(rm, period=period, catch_select=catch_select, clip=clip, catch_names=catch_names, geom=geom)
                                  for rm, period in zip(rm_lst, periods)]
        self.ts_fetching_lst = self.cell_data_ext[0].ts_fetching_lst
        self.map_fetching_lst = self.cell_data_ext[0].map_fetching_lst
        self.n = [len(data_ext.t_ax) for data_ext in self.cell_data_ext]
        self.n_cum = np.cumsum(self.n)
        self.cum = np.cumsum([0]+self.n)
        self.geom = self.cell_data_ext[0].geom
        self.catch_names = self.cell_data_ext[0].catch_names
        self.var_units = self.cell_data_ext[0].var_units
        self.temporal_vars = self.cell_data_ext[0].temporal_vars
        self.static_vars = self.cell_data_ext[0].static_vars
        self.cal = api.Calendar()
        self.t_ax_shyft = api.TimeAxisFixedDeltaT(self.cell_data_ext[0].t_ax_shyft.start, self.cell_data_ext[0].t_ax_shyft.delta_t, sum(self.n))
        self.t_ax = np.array([self.t_ax_shyft.time(i) for i in range(self.t_ax_shyft.size())])

    def time_num_2_str(self, t_num):
        return self.cal.to_string(self.t_ax_shyft.time(self._time_num_2_idx(t_num)))

    def _time_num_2_idx(self, t_num):
        t = int(t_num)
        if t < self.t_ax_shyft.start:
            return 0
        elif t >= self.t_ax_shyft.total_period().end:
            return self.t_ax_shyft.n-1
        return self.t_ax_shyft.index_of(t)

    def get_map(self, var_name, cat_id_lst, t_num):
        t = self._time_num_2_idx(t_num)
        data_ext_idx = np.searchsorted(self.n_cum, t, side='right')
        # t_idx = t-self.cum[data_ext_idx]
        # return self.cell_data_ext[data_ext_idx].get_map(var_name, cat_id_lst, int(t_idx))
        return self.cell_data_ext[data_ext_idx].get_map(var_name, cat_id_lst, t_num)

    def get_ts(self, var_name, cell_idx):
        ts_lst = [data_ext.get_ts(var_name, cell_idx) for data_ext in self.cell_data_ext]
        return np.concatenate([ts[0] for ts in ts_lst]), np.concatenate([ts[1] for ts in ts_lst])

    def get_geo_data(self, var_name, cat_id_lst):
        return self.cell_data_ext[0].get_geo_data(var_name, cat_id_lst)