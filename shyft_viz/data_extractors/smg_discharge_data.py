import numpy as np


class Geom(object):
    def __init__(self, shyft_geom, idx):
        self.patches = [shyft_geom.patches[i] for i in idx]
        self.polys = [shyft_geom.polys[i] for i in idx]
        self.bbox = shyft_geom.bbox


class SMGDataExtractor(object):
    def __init__(self, shyft_data_ext, ref_repo):

        self.var_units = shyft_data_ext.var_units
        self.t_ax = shyft_data_ext.t_ax
        self.tsp = ref_repo[0]['repository'].read([ts_info['uid'] for ts_info in ref_repo[0]['1D_timeseries']],
                                                  shyft_data_ext.t_ax.total_period())
        self.catch_names, self.ts_uid = zip(*[(ts_info['module_name'],ts_info['uid']) for ts_info in ref_repo[0]['1D_timeseries']])
        shyft_catch_names = shyft_data_ext.catch_names
        idx = [shyft_catch_names.index(c) for c in self.catch_names]
        self.geom = Geom(shyft_data_ext.geom, idx)
        self.map_fetching_lst = [0]
        self.ts_fetching_lst = [0]
        self.data = {'q_avg':np.array([self.tsp[uid].v.to_numpy() for uid in self.ts_uid])}

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.data[var_name][:, t]

    def get_ts(self, var_name, cat_id):
        return self.data[var_name][cat_id]

    def get_geo_data(self, var_name, cat_id_lst_grp):
        pass

        # self.map_fetching_lst = {k: v.map_fetching_lst for k, v in data_ext.items()}
        # self.ts_fetching_lst = {k: v.ts_fetching_lst for k, v in data_ext.items()}
        # self.patches = {k: v.geom.patches for k, v in data_ext.items()}
        # self.polys = {k: v.geom.polys for k, v in data_ext.items()}
        # self.nb_catch = {k: len(v.geom.polys) for k, v in data_ext.items()}
        # self.catch_nms = {k: v.catch_names for k, v in data_ext.items()}
        # self.var_units = {k: v.var_units for k, v in data_ext.items()}
        # # -Just picking the value for one of the datasets for now-
        # bbox = {k: v.geom.bbox for k, v in data_ext.items()}
        # self.bbox = list(bbox.values())[0]
        # t_ax = {k: v.t_ax for k, v in data_ext.items()}
        # self.t_ax = list(t_ax.values())[0]
        # max_ti = {k: v.t_ax.n-1 for k, v in data_ext.items()}
        # self.max_ti = list(max_ti.values())[0]
        # times = {k: [datetime.utcfromtimestamp(v.t_ax.time(i)) for i in range(v.t_ax.size())] for k, v in data_ext.items()}
        # self.times = list(times.values())[0]
