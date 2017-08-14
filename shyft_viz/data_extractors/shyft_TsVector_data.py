import numpy as np
#from datetime import datetime
#from pytz import utc
from shyft import api

class TsVectorDataExtractor(object):
    def __init__(self, ts_vct, catch_names, geom, preprocess=False):
        #ts_vct can be a list of Ts or TsVector
        if not isinstance(ts_vct, api.TsVector):
            shyft_ts_vct = api.TsVector()
            [shyft_ts_vct.append(ts) for ts in ts_vct]
            ts_vct = shyft_ts_vct
        # ---Attributes expected by Viewer---
        self.var_units = {'q_avg': 'm3_per_sec'}
        self.t_ax = self._flatten_tsvct_t_2_numpy(ts_vct)
        self.catch_names = catch_names
        self.map_fetching_lst = list(range(len(self.catch_names)))
        self.ts_fetching_lst = list(range(len(self.catch_names)))
        self.geom = geom
        # ---***---

        self.cal = api.Calendar()
        self.t_ax_shyft = api.TimeAxis(api.UtcTimeVector(self.t_ax.tolist()))

        if preprocess:
            ts_t, ts_v = zip(*[(m.time_axis.time_points, m.v.to_numpy()) for m in ts_vct])
            self.data = {'q_avg': {'t': ts_t, 'v': ts_v}}
            self.get_ts = self._get_ts_from_preprocessed
        else:
            self.data = {'q_avg': ts_vct}
            self.get_ts = self._get_ts_from_tsvct

    def _get_tarr_from_ts(self, ts):
        # Using time_axis.time_points
        # return ts.time_axis.time_points
        # Extracting by loop
        return [ts.time(i) for i in range(ts.size())]

    def get_closest_time(self, t_num):
        return self.t_ax_shyft.time(self.t_ax_shyft.index_of(t_num))

    def time_num_2_str(self, t_num):
        return self.cal.to_string(self.get_closest_time(t_num))

    def _flatten_tsvct_t_2_numpy(self, ts_vct):
        return np.sort(np.unique(np.concatenate([ts.time_axis.time_points for ts in ts_vct])))

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.data[var_name].values_at_time(int(t))

    def _get_ts_from_tsvct(self, var_name, cat_id):
        ts = self.data[var_name][cat_id]
        #return self.get_tarr_from_ts(ts), ts.v.to_numpy()
        return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()

    def _get_ts_from_preprocessed(self, var_name, cat_id):
        return self.data[var_name]['t'][cat_id], self.data[var_name]['v'][cat_id]

    def get_geo_data(self, var_name, cat_id_lst_grp):
        pass
