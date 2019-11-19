import numpy as np
from shyft import api

class Geom(object):
    def __init__(self, shyft_geom, idx):
        self.patches = [shyft_geom.patches[i] for i in idx]
        self.polys = [shyft_geom.polys[i] for i in idx]
        self.bbox = shyft_geom.bbox


class TsRepoDataExtractor(object):
    def __init__(self, shyft_data_ext, ref_repo, ref_ts, period):

        self.var_units = {'q_avg': 'm3_per_sec'} # shyft_data_ext.var_units
        #self.t_ax = shyft_data_ext.t_ax

        # self.tsp = ref_repo.read([ts_info['uid'] for ts_info in ref_ts], shyft_data_ext.t_ax_shyft.total_period())
        self.tsp = ref_repo.read([ts_info['uid'] for ts_info in ref_ts], period)
        self.catch_names, self.ts_uid = zip(*[(ts_info['module_name'], ts_info['uid']) for ts_info in ref_ts])
        shyft_catch_names = shyft_data_ext.catch_names
        idx = [shyft_catch_names.index(c) for c in self.catch_names]
        self.geom = Geom(shyft_data_ext.geom, idx)
        self.map_fetching_lst = list(range(len(self.catch_names)))
        self.ts_fetching_lst = list(range(len(self.catch_names)))
        ts_vct = api.TsVector()
        [ts_vct.push_back(self.tsp[uid]) for uid in self.ts_uid]
        #self.data = {'q_avg':np.array([self.tsp[uid].v.to_numpy() for uid in self.ts_uid])}
        self.data = {'q_avg': ts_vct}
        self.t_ax = self._flatten_tsvct_t_2_numpy(ts_vct)
        self.cal = api.Calendar()
        self.t_ax_shyft = api.TimeAxis(api.UtcTimeVector(self.t_ax.tolist()))

        self.static_vars = []  # TODO: make this a property

    @property
    def temporal_vars(self):
        return list(self.data.keys())

    def get_closest_time(self, t_num):
        return self.t_ax_shyft.time(self.t_ax_shyft.index_of(int(t_num), 0))

    def time_num_2_str(self, t_num):
        return self.cal.to_string(self.get_closest_time(t_num))

    def _flatten_tsvct_t_2_numpy(self, ts_vct):
        return np.sort(np.unique(np.concatenate([ts.time_axis.time_points for ts in ts_vct])))

    # def get_map(self, var_name, cat_id_lst_grp, t):
    #     return self.data[var_name][:, t]

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.data[var_name].values_at_time(int(t))

    # def get_ts(self, var_name, cat_id):
    #     return self.data[var_name][cat_id]

    def get_ts(self, var_name, cat_id):
        ts = self.data[var_name][cat_id]
        #return self.get_tarr_from_ts(ts), ts.v.to_numpy()
        return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()

    def get_geo_data(self, var_name, cat_id_lst_grp):
        pass
