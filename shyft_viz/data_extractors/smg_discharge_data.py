import numpy as np


class Geom(object):
    def __init__(self, shyft_geom, idx):
        self.patches = [shyft_geom.patches[i] for i in idx]
        self.polys = [shyft_geom.polys[i] for i in idx]
        self.bbox = shyft_geom.bbox


class SMGDataExtractor(object):
    def __init__(self, shyft_data_ext, ref_repo, ref_ts):

        self.var_units = {'q_avg': 'm3_per_sec'} # shyft_data_ext.var_units
        self.t_ax = shyft_data_ext.t_ax
        self.tsp = ref_repo.read([ts_info['uid'] for ts_info in ref_ts], shyft_data_ext.t_ax.total_period())
        self.catch_names, self.ts_uid = zip(*[(ts_info['module_name'], ts_info['uid']) for ts_info in ref_ts])
        shyft_catch_names = shyft_data_ext.catch_names
        idx = [shyft_catch_names.index(c) for c in self.catch_names]
        self.geom = Geom(shyft_data_ext.geom, idx)
        self.map_fetching_lst = list(range(len(self.catch_names)))
        self.ts_fetching_lst = list(range(len(self.catch_names)))
        self.data = {'q_avg':np.array([self.tsp[uid].v.to_numpy() for uid in self.ts_uid])}

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.data[var_name][:, t]

    def get_ts(self, var_name, cat_id):
        return self.data[var_name][cat_id]

    def get_geo_data(self, var_name, cat_id_lst_grp):
        pass
