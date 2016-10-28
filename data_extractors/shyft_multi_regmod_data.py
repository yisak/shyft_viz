from shyft import api

import numpy as np

from .shyft_regmod_data import CellDataExtractor, SubcatDataExtractor


class DataExtractor(object):
    def __init__(self, rm1, rm2, agg = False, catch_select=None, clip=False, catch_names=None):
        if agg:
            self.cell_data_ext1 = SubcatDataExtractor(rm1, catch_select=None, clip=False, catch_names=None)
            self.cell_data_ext2 = SubcatDataExtractor(rm2, catch_select=None, clip=False, catch_names=None)
        else:
            self.cell_data_ext1 = CellDataExtractor(rm1, catch_select=None, clip=False, catch_names=None)
            self.cell_data_ext2 = CellDataExtractor(rm2, catch_select=None, clip=False, catch_names=None)
        self.ts_fetching_lst = self.cell_data_ext1.ts_fetching_lst
        self.map_fetching_lst = self.cell_data_ext1.map_fetching_lst
        self.n1 = self.cell_data_ext1.t_ax.n
        self.n2 = self.cell_data_ext2.t_ax.n
        self.geom = self.cell_data_ext1.geom
        self.catch_names = self.cell_data_ext1.catch_names
        self.var_units = self.cell_data_ext1.var_units
        self.cal = api.Calendar()
        self.t_ax = api.Timeaxis(self.cell_data_ext1.t_ax.start, self.cell_data_ext1.t_ax.delta_t,
                                 self.cell_data_ext1.t_ax.n+self.cell_data_ext2.t_ax.n)

    def time_num_2_str(self, ti):
        return self.cal.to_string(self.t_ax.time(ti))

    def get_map(self, var_name, cat_id_lst, t):
        if t<self.n1:
            return self.cell_data_ext1.get_map(var_name, cat_id_lst, t)
        else:
            return self.cell_data_ext2.get_map(var_name, cat_id_lst, t-self.n1)
        #return self._ts_map_ext_methods[var_name](cat_id_lst, t).to_numpy()


    def get_ts(self, var_name, cell_idx):
        return np.concatenate([self.cell_data_ext1.get_ts(var_name, cell_idx),
                               self.cell_data_ext2.get_ts(var_name, cell_idx)])
        #return self._ts_ext_methods[var_name](self.cells[int(cell_idx)]).v.to_numpy()


    def get_geo_data(self, var_name, cat_id_lst):
        return self.cell_data_ext1.get_geo_data(var_name, cat_id_lst)
        #return self.cell_geo_data[var_name][np.in1d(self.cid, cat_id_lst)]