from shyft import api

import numpy as np

from ..geom_preps.shyft_cell_geom import CellViewerPrep, SubcatViewerPrep


class ArealDataExtractor(object):
    def __init__(self, rm):
        self.rm = rm
        stack_names = {'PTHSKModel':'pthsk','PTGSKModel':'ptgsk','PTSSKModel':'ptssk','PTHSKOptModel':'pthsk','PTGSKOptModel':'ptgsk','PTSSKOptModel':'ptssk'}
        self.stack_nm = stack_names[rm.__class__.__name__]
        self.cells = rm.cells
        self.cal = api.Calendar()
        self.t_ax = self.rm.time_axis
        self.inputs = {'prec': 'precipitation', 'temp': 'temperature', 'ws': 'wind_speed', 'rh': 'rel_hum',
                       'rad': 'radiation'}
        self.basic_outputs = {'q_avg': 'discharge'}
        self.outputs = {'ptgsk':
                            {'priestley_taylor_response': {'pet': 'output'},
                            'actual_evaptranspiration_response': {'aet': 'output'},
                            'gamma_snow_response': {'sout': 'outflow', 'swe': 'swe', 'sca': 'sca','glacier_melt': 'glacier_melt'},
                            'kirchner_state': {'q_inst': 'discharge'},
                            'gamma_snow_state': {'acc_melt': 'acc_melt', 'albedo': 'albedo', 'alpha': 'alpha',
                                                'iso_pot_energy': 'iso_pot_energy', 'lwc': 'lwc',
                                                'surface_heat': 'surface_heat', 'temp_swe': 'temp_swe',
                                                'sdc_melt_mean': 'sdc_melt_mean'}},
                        'ptssk':
                            {'priestley_taylor_response': {'pet': 'output'},
                             'actual_evaptranspiration_response': {'aet': 'output'},
                             'skaugen_snow_response': {'sout': 'outflow', 'glacier_melt': 'glacier_melt', 'total_stored_water': 'total_stored_water'},
                             'kirchner_state': {'q_inst': 'discharge'},
                             'skaugen_snow_state': {'alpha': 'alpha', 'lwc': 'lwc', 'nu': 'nu',
                                                   'residual': 'residual', 'sca': 'sca', 'swe': 'swe'}},
                        'pthsk':
                            {'priestley_taylor_response': {'pet': 'output'},
                             'actual_evaptranspiration_response': {'aet': 'output'},
                             'hbv_snow_response': {'sout': 'outflow', 'glacier_melt': 'glacier_melt'},
                             'kirchner_state': {'q_inst': 'discharge'},
                             'hbv_snow_state': {'sca': 'sca', 'swe': 'swe'}}
                        }
        u = ['degree_celsius', 'mm', 'm3_per_sec', 'mm_per_hr', 'W_per_m2', 'm_per_sec', 'fraction [0-1]', 'm']
        self.var_units = {'prec': u[3], 'temp': u[0], 'swe': u[1], 'q_avg': u[2], 'rad': u[4], 'ws': u[5], 'rh': u[6],
                          'elev': u[7]}
        self.geo_attr = ['z', 'gf', 'lf', 'rf', 'ff']
        geo_attr_all = ['x', 'y', 'z', 'area', 'c_id', 'rsf', 'gf', 'lf', 'rf', 'ff', 'uf']
        self.cell_geo_data = np.rec.fromrecords(self.cells.geo_cell_data_vector(self.cells).to_numpy().reshape(
            len(self.cells), len(geo_attr_all)),names=','.join(geo_attr_all))
        self.cid = self.cell_geo_data.c_id.astype(int)

        # Map
        self._ts_map_ext_methods = {}
        # Input
        self._ts_map_ext_methods.update({k: getattr(self.rm.statistics, v) for k, v in self.inputs.items()})
        # Output
        self._ts_map_ext_methods.update({k: getattr(self.rm.statistics, v) for k, v in self.basic_outputs.items()})

        self._ts_map_ext_methods.update({k: getattr(getattr(self.rm, output_grp), v) for output_grp in self.outputs[self.stack_nm]
                                         for k, v in self.outputs[self.stack_nm][output_grp].items()})

    def time_num_2_str(self, ti):
        return self.cal.to_string(self.t_ax.time(ti))


class CellDataExtractor(ArealDataExtractor):
    def __init__(self, rm, catch_select=None, clip=False, catch_names=None):
        super().__init__(rm)

        self.responses = {'ptgsk':
                              {'q_avg': 'avg_discharge', 'pet': 'pe_output', 'aet': 'ae_output', 'sout': 'snow_outflow',
                                'swe': 'snow_swe', 'sca': 'snow_sca'},
                          'ptssk':
                              {'q_avg': 'avg_discharge', 'pet': 'pe_output', 'aet': 'ae_output', 'sout': 'snow_outflow', 'glacier_melt':'glacier_melt','snow_total_stored_water':'snow_total_stored_water'},
                          'pthsk':
                              {'q_avg': 'avg_discharge', 'pet': 'pe_output', 'aet': 'ae_output', 'sout': 'snow_outflow'}
                          }
        self.states = {'ptgsk':
                           {'q_inst': 'kirchner_discharge', 'acc_melt': 'gs_acc_melt', 'albedo': 'gs_albedo',
                            'alpha': 'gs_alpha', 'iso_pot_energy': 'gs_iso_pot_energy', 'lwc': 'gs_lwc',
                            'sdc_melt_mean': 'gs_sdc_melt_mean', 'surface_heat': 'gs_surface_heat',
                            'temp_swe': 'gs_temp_swe'},
                       'ptssk':
                           {'q_inst': 'kirchner_discharge', 'swe': 'snow_swe', 'sca':'snow_sca','alpha':'snow_alpha','lwc':'snow_lwc','nu':'snow_nu','residual':'snow_residual'},
                       'pthsk':
                           {'q_inst': 'kirchner_discharge', 'swe': 'snow_swe', 'sca':'snow_sca'}
                       }
        self.geom = CellViewerPrep(self.cid, self.rm.gis_info, self.rm.catchment_id_map,
                                   self.rm.bounding_region.geometry, catch_select=catch_select, clip=clip)
        self.ts_fetching_lst = self.geom.ts_fetching_lst
        self.map_fetching_lst = self.geom.map_fetching_lst
        if catch_names is None:
            self.catch_names = ['c_'+str(i) for i in range(len(self.cid))]
        else:
            self.catch_names = catch_names

        # Timeseries
        self._ts_ext_methods = {}
        # Input
        self._ts_ext_methods.update({k: lambda cell, v=v: getattr(cell.env_ts, v) for k, v in self.inputs.items()})
        # Response
        self._ts_ext_methods.update({k: lambda cell, v=v: getattr(cell.rc, v) for k, v in self.responses[self.stack_nm].items()})
        # State
        self._ts_ext_methods.update({k: lambda cell, v=v: getattr(cell.sc, v) for k, v in self.states[self.stack_nm].items()})

    def get_map(self, var_name, cat_id_lst, t):
        return self._ts_map_ext_methods[var_name](cat_id_lst, t).to_numpy()

    def get_ts(self, var_name, cell_idx):
        if (self.stack_nm  in ['ptssk','pthsk'] and var_name in ['swe']):
            return self._ts_ext_methods[var_name](self.cells[int(cell_idx)]).v.to_numpy()[0:-1]
        else:
            return self._ts_ext_methods[var_name](self.cells[int(cell_idx)]).v.to_numpy()

    def get_geo_data(self, var_name, cat_id_lst):
        return self.cell_geo_data[var_name][np.in1d(self.cid,cat_id_lst)]


class SubcatDataExtractor(ArealDataExtractor):
    def __init__(self, rm, catch_select=None, clip=False, catch_names=None):
        super().__init__(rm)

        self.geom = SubcatViewerPrep(self.cid, self.rm.gis_info, self.rm.catchment_id_map,
                                     self.rm.bounding_region.geometry, catch_grp=catch_select, clip=clip)
        self.ts_fetching_lst = self.geom.ts_fetching_lst
        self.map_fetching_lst = self.geom.map_fetching_lst
        if catch_names is None:
            self.catch_names = ['c_'+str(i) for i in range(len(self.map_fetching_lst))]
        else:
            self.catch_names = catch_names

        # Map
        self._val_ext_methods = {}
        # Input
        self._val_ext_methods.update({k: getattr(self.rm.statistics, v+'_value') for k, v in self.inputs.items()})
        # Output
        self._val_ext_methods.update({k: getattr(self.rm.statistics, v+'_value') for k, v in self.basic_outputs.items()})

        self._val_ext_methods.update({k: getattr(getattr(self.rm, output_grp), v+'_value') for output_grp in self.outputs[self.stack_nm]
                                         for k, v in self.outputs[self.stack_nm][output_grp].items()})

    def get_map(self, var_name, cat_id_lst_grp, t):
        return np.array([self._val_ext_methods[var_name](cat_id_lst, t) for cat_id_lst in cat_id_lst_grp])

    def get_ts(self, var_name, cat_id_lst):
        if (self.stack_nm in ['ptssk', 'pthsk'] and var_name in ['swe']):
            return self._ts_map_ext_methods[var_name](cat_id_lst).v.to_numpy()[0:-1]
        else:
            return self._ts_map_ext_methods[var_name](cat_id_lst).v.to_numpy()

    def get_geo_data(self, var_name, cat_id_lst_grp):
        idx = [np.in1d(self.cid,cat_id_lst) for cat_id_lst in cat_id_lst_grp]
        return np.array([np.average(self.cell_geo_data[var_name][idx[i]], weights=self.cell_geo_data['area'][idx[i]])
                for i, cat_id_lst in enumerate(cat_id_lst_grp)])