import numpy as np

class SourceDataExtractor(object):
    def __init__(self, rg_env):
        self.rg_env = rg_env
        self.src_types = {'prec': 'precipitation', 'temp': 'temperature', 'ws': 'wind_speed', 'rh': 'rel_hum',
                          'rad': 'radiation'}
        self.src_vct = {nm: getattr(self.rg_env, src_type) for nm, src_type in self.src_types.items()}

        self.xyz = {nm: np.array([[src.mid_point().x, src.mid_point().y, src.mid_point().z] for src in vct]) for nm, vct in self.src_vct.items()}
        xyz_arr = np.vstack(tuple(self.xyz.values()))
        xy_arr = xyz_arr[:, 0:2]
        self.xy_unique = np.array(list(set(tuple(p) for p in xy_arr)))

        self.idx = {nm: [np.hypot(x - self.xy_unique[:, 0], y - self.xy_unique[:, 1]).argmin() for x, y in v[:, 0:2]] for nm, v in
                    self.xyz.items()}