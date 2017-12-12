import numpy as np
#from datetime import datetime
#from pytz import utc
from shyft import api
from ..geom_preps.forecast_grid_geom import GridViewerPrep

class GeoTsVectorDataExtractorError(Exception):
    pass

class GeoTsVectorDataExtractor(object):
    def __init__(self, ts_vct_dict, as_pt_dataset=True, geom=None, model_cs=None, fc_cs=None, dxy=None, pt_names=None):
        self.cal = api.Calendar()
        self.std_units = {'temp': 'degree_celsius', 'prec': 'mm_per_hr', 'rad': 'W_per_m2', 'ws': 'm_per_sec',
                          'rh':'fraction [0-1]'}
        #ts_vct_dict can be of type api.ARegionEnvironment
        if isinstance(ts_vct_dict, api.ARegionEnvironment):
            self.src_types = {'precipitation': 'prec', 'temperature': 'temp', 'wind_speed': 'ws', 'rel_hum': 'rh',
                              'radiation': 'rad'}
            self.ts_vct_dict = {src_type: getattr(ts_vct_dict, nm) for nm, src_type in self.src_types.items()}
        else:
            self.src_types = {'precipitation': 'prec', 'temperature': 'temp', 'wind_speed': 'ws', 'relative_humidity': 'rh',
                              'radiation': 'rad'}
            self.ts_vct_dict = {self.src_types[nm]: ts_vct_dict[nm] for nm, src_vct in ts_vct_dict.items()}

        xyz = {nm: np.array([[src.mid_point().x, src.mid_point().y, src.mid_point().z] for src in vct]) for nm, vct
                    in self.ts_vct_dict.items()}

        if len(xyz) > 1:
            if not self._all_pts_match_across_src_type(xyz):
                raise GeoTsVectorDataExtractorError('The points do not match across the different source types.')

        self.xyz = list(xyz.values())[0]
        # ---Attributes expected by Viewer---
        self.var_units = {nm: self.std_units[nm] for nm in self.ts_vct_dict}
        if as_pt_dataset:
            self.nb_pts = len(self.xyz)
            self.names = [str(i) for i in range(self.nb_pts)] if pt_names is None else pt_names
            self.coord = self.xyz[:, 0:2]
        else:
            ref_ts = list(self.ts_vct_dict.values())[0][0].ts
            self.t_ax_shyft = ref_ts.time_axis
            self.t_ax = np.array([self.t_ax_shyft.time(i) for i in range(self.t_ax_shyft.size())])
            self.catch_names = [str(i) for i in range(len(self.xyz))]
            self.geom = geom
            if geom is None:
                self.geom = GridViewerPrep(self.xyz[:,0], self.xyz[:,1], model_cs=model_cs, fc_cs=fc_cs, dxy=dxy)
            self.ts_fetching_lst = self.geom.ts_fetching_lst
            self.map_fetching_lst = self.geom.map_fetching_lst
        # ---***---

        self.static_vars = []  # TODO: make this a property

    @property
    def temporal_vars(self):
        return list(self.ts_vct_dict.keys())

    def _all_pts_match_across_src_type(self, xyz):
        L = list(xyz.values())
        return np.all(np.array([len(arr) for arr in L]) == len(L[0])) and (np.diff(np.vstack(L).reshape(len(L),-1),axis=0)==0).all()

    def time_num_2_str(self, t_num):
        return self.cal.to_string(self.get_closest_time(t_num))

    def _time_num_2_idx(self, t_num):
        return self.t_ax_shyft.index_of(int(t_num), 0)

    def get_closest_time(self, t_num):
        return self.t_ax_shyft.time(self._time_num_2_idx(t_num))

    def get_ts(self, var_name, pt_idx):
        ts = self.ts_vct_dict[var_name][int(pt_idx)].ts
        return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()

    def get_map(self, var_name, cat_id_lst_grp, t):
        return self.ts_vct_dict[var_name].values_at_time(int(t))
