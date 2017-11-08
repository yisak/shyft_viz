import numpy as np
from datetime import datetime
from pytz import utc

class HindcastDataExtractor(object):
    def __init__(self, dict_w_ts_vct_lst, catch_names, geom):
        var_nm_map = {'precipitation': 'prec', 'temperature': 'temp', 'wind_speed': 'ws', 'relative_humidity': 'rh',
                           'radiation': 'rad', 'q_avg': 'q_avg'}
        dict_w_ts_vct_lst = {var_nm_map[k]: v for k, v in dict_w_ts_vct_lst.items()}
        # ---Attributes expected by Viewer---
        self._var_units = {'q_avg': 'm3_per_sec', 'prec': 'mm_per_hr', 'temp': 'degree_celcius'}
        #self.t_ax = self._flatten_tsvct_t_2_numpy(ts_vct_lst[0])
        self.t_ax = self._flatten_tsvct_t_2_numpy(dict_w_ts_vct_lst[list(dict_w_ts_vct_lst)[0]][0])
        self.catch_names = catch_names
        self.map_fetching_lst = list(range(len(self.catch_names)))
        self.ts_fetching_lst = list(range(len(self.catch_names)))
        self.geom = geom
        # ---***---

        self.t_ax_datetime = [datetime.utcfromtimestamp(t).replace(tzinfo=utc) for t in self.t_ax]
        #self.data = {'q_avg':np.array([self.tsp[uid].v.to_numpy() for uid in self.ts_uid])}
        #self.data = {'q_avg': np.array([self._flatten_tsvct_v_2_numpy(ts_vct) for ts_vct in ts_vct_lst])}
        self.data = {k: np.array([self._flatten_tsvct_v_2_numpy(ts_vct) for ts_vct in ts_vct_lst]) for k, ts_vct_lst in dict_w_ts_vct_lst.items()}

        self.static_vars = []  # TODO: make this a property

    @property
    def var_units(self):
        return {k: self._var_units[k] for k in self.data}

    @property
    def temporal_vars(self):
        return list(self.data.keys())

    def get_closest_time(self, t_num):
        return self.t_ax[self._time_num_2_idx(t_num)]

    def time_num_2_str(self, t_num):
        return datetime.utcfromtimestamp(self.get_closest_time(t_num)).strftime('%Y-%m-%d %H:%M:%S')

    def _time_num_2_idx(self, t_num):
        return np.searchsorted(self.t_ax, t_num)

    def _flatten_tsvct_t_2_numpy(self, vct):
        if vct[0].total_period().overlaps(vct[1].total_period()) and vct[0].size() != 1:
            arr_t = np.empty((vct.size(), vct[0].size() + 1))
            for i in range(vct.size()):
                #arr_t[i, :-1] = [vct[i].time(j) for j in range(vct[i].size())]
                arr_t[i, :-1] = vct[i].time_axis.time_points[0:vct[i].size()]
            arr_t[:, -1] = arr_t[:, -2]
            return arr_t.flatten()
        else:
            return np.concatenate([ts.time_axis.time_points[0:ts.size()] for ts in vct])


    def _flatten_tsvct_v_2_numpy(self, vct):
        # Plotting forecasts as one line
        if vct[0].total_period().overlaps(vct[1].total_period()) and vct[0].size() != 1:
            arr_v = np.empty((vct.size(), vct[0].size() + 1))
            arr_v.fill(np.nan)
            for i in range(vct.size()):
                arr_v[i, :-1] = vct[i].values.to_numpy()
            return arr_v.flatten()
        else:
            return np.concatenate([ts.v.to_numpy() for ts in vct])

    def _flatten_tsvct_2_numpy(self, vct):
        # Plotting forecasts as one line
        arr_v = np.empty((vct.size(), vct[0].size() + 1))
        arr_v.fill(np.nan)
        arr_t = np.empty(arr_v.shape)
        for i in range(vct.size()):
            arr_v[i, :-1] = vct[i].values.to_numpy()
            #arr_t[i, :-1] = [vct[i].time(j) for j in range(vct[i].size())]
            arr_t[i, :-1] = vct[i].time_axis.time_points[0:vct[i].size()]
        arr_t[:, -1] = arr_t[:, -2]
        #T = [datetime.utcfromtimestamp(t) for t in arr_t.flatten()]
        return arr_t.flatten(), arr_v.flatten()

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.data[var_name][:, self._time_num_2_idx(t)]

    def get_ts(self, var_name, cat_id):
        #return self.t_ax_datetime, self.data[var_name][cat_id]
        return self.t_ax, self.data[var_name][cat_id]

    def get_geo_data(self, var_name, cat_id_lst_grp):
        pass
