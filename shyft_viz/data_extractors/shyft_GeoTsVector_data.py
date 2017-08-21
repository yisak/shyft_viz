import numpy as np
#from datetime import datetime
#from pytz import utc
from shyft import api

class GeoTsVectorDataExtractorError(Exception):
    pass

class GeoTsVectorDataExtractor(object):
    def __init__(self, ts_vct_dict, preprocess=False):
        self.std_units = {'temp': 'degree_celsius', 'prec': 'mm_per_hr', 'rad': 'W_per_m2', 'ws': 'm_per_sec',
                          'rh':'fraction [0-1]'}
        #ts_vct_dict can be of type api.ARegionEnvironment
        if isinstance(ts_vct_dict, api.ARegionEnvironment):
            self.src_types = {'prec': 'precipitation', 'temp': 'temperature', 'ws': 'wind_speed', 'rh': 'rel_hum',
                              'rad': 'radiation'}
            self.ts_vct_dict = {nm: getattr(ts_vct_dict, src_type) for nm, src_type in self.src_types.items()}
        else:
            self.ts_vct_dict = ts_vct_dict

        xyz = {nm: np.array([[src.mid_point().x, src.mid_point().y, src.mid_point().z] for src in vct]) for nm, vct
                    in self.ts_vct_dict.items()}

        if len(xyz) > 1:
            if not self._all_pts_match_across_src_type(xyz):
                raise GeoTsVectorDataExtractorError('The points do not match across the different source types.')

        self.xyz = list(xyz.values())[0]
        # ---Attributes expected by Viewer---
        self.var_units = {nm: self.std_units[nm] for nm in self.ts_vct_dict}
        self.nb_pts = len(self.xyz)
        self.names = [str(i) for i in range(self.nb_pts)]
        self.coord = self.xyz[:, 0:2]
        # ---***---

        self.temporal_vars = ['prec', 'temp', 'rad', 'ws', 'rh']  # TODO: make this a property
        self.static_vars = []  # TODO: make this a property

    def _all_pts_match_across_src_type(self, xyz):
        L = list(xyz.values())
        return np.all(np.array([len(arr) for arr in L]) == len(L[0])) and (np.diff(np.vstack(L).reshape(len(L),-1),axis=0)==0).all()

    def get_ts(self, var_name, pt_idx):
        ts = self.ts_vct_dict[var_name][int(pt_idx)].ts
        return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()
