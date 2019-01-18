from shyft import api

import numpy as np

from ..geom_preps.shyft_cell_geom import CellViewerPrep, SubcatViewerPrep


class ArealDataExtractor(object):
    def __init__(self, rm, period=None):
        self.rm = rm
        stack_names = {'PTHSKModel':'pthsk','PTGSKModel':'ptgsk','PTSSKModel':'ptssk','PTHSKOptModel':'pthsk','PTGSKOptModel':'ptgsk','PTSSKOptModel':'ptssk'}
        self.stack_nm = stack_names[rm.__class__.__name__]
        self.cells = rm.cells
        self.cal = api.Calendar()
        self.t_ax_shyft_full = self.rm.time_axis
        self.t_ax_full = np.array([self.t_ax_shyft_full.time(i) for i in range(self.t_ax_shyft_full.size())])
        if period is not None:
            self.start_idx = self._time_num_2_idx(period.start)
            self.nb_pts = self._time_num_2_idx(period.end)-self.start_idx+1
            self.t_ax_shyft = api.TimeAxisFixedDeltaT(self.t_ax_shyft_full.time(self.start_idx),self.t_ax_shyft_full.delta_t,self.nb_pts)
            self.t_ax  = np.array([self.t_ax_shyft.time(i) for i in range(self.t_ax_shyft.size())])
        else:
            self.start_idx = 0
            self.nb_pts = self.t_ax_shyft_full.size()
            self.t_ax_shyft = self.t_ax_shyft_full
            self.t_ax = self.t_ax_full
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
                          'z': u[7], 'sout': u[2], 'gf': u[6] , 'lf': u[6], 'rf': u[6], 'ff': u[6], 'uf': u[6]}
        # self.geo_attr = ['z', 'gf', 'lf', 'rf', 'ff', 'uf']
        self.temporal_vars = ['prec', 'temp', 'swe', 'q_avg', 'rad', 'ws', 'rh', 'sout']  # TODO: make this a property
        self.static_vars = ['z', 'gf', 'lf', 'rf', 'ff', 'uf']  # TODO: make this a property
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
                                         for k, v in self.outputs[self.stack_nm][output_grp].items() if hasattr(self.rm, output_grp)})

    def time_num_2_str(self, t_num):
        return self.cal.to_string(self.get_closest_time(t_num))

    def _time_num_2_idx(self, t_num):
        if t_num < self.t_ax_shyft_full.time(0):
            return 0
        elif t_num > self.t_ax_shyft_full.time(self.t_ax_shyft_full.size()-1):
            return self.t_ax_shyft_full.size()-1
        else:
            return self.t_ax_shyft_full.index_of(int(t_num))

    def get_closest_time(self, t_num):
        return self.t_ax_shyft_full.time(self._time_num_2_idx(t_num))


class CellDataExtractor(ArealDataExtractor):
    def __init__(self, rm, period=None, catch_select=None, clip=False, catch_names=None, geom=None):
        super().__init__(rm, period=period)

        self.responses = {'ptgsk':
                              {'q_avg': 'avg_discharge', 'pet': 'pe_output', 'aet': 'ae_output', 'sout': 'snow_outflow',
                                'swe': 'snow_swe', 'sca': 'snow_sca'},
                          'ptssk':
                              {'q_avg': 'avg_discharge', 'pet': 'pe_output', 'aet': 'ae_output', 'sout': 'snow_outflow',
                               'glacier_melt':'glacier_melt','snow_total_stored_water':'snow_total_stored_water'},
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
        self.geom = geom
        if geom is None:
            self.geom = CellViewerPrep(self.cid, self.rm.gis_info, self.rm.catchment_id_map,
                                       self.rm.bounding_region.geometry, catch_select=catch_select, clip=clip)
        self.ts_fetching_lst = self.geom.ts_fetching_lst
        self.map_fetching_lst = self.geom.map_fetching_lst
        if catch_names is None:
            #self.catch_names = ['c_'+str(i) for i in range(len(self.cid))]
            self.catch_names = ['c_' + str(i) for i in self.ts_fetching_lst]  # to get cell indices matching order in region_modell cells
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
        return self._ts_map_ext_methods[var_name](cat_id_lst, self._time_num_2_idx(t)).to_numpy()

    def get_ts(self, var_name, cell_idx):
        # if (self.stack_nm  in ['ptssk','pthsk'] and var_name in ['swe']):
        #     return self._ts_ext_methods[var_name](self.cells[int(cell_idx)]).v.to_numpy()[0:-1]
        # else:
        #     return self._ts_ext_methods[var_name](self.cells[int(cell_idx)]).v.to_numpy()
        ts = self._ts_ext_methods[var_name](self.cells[int(cell_idx)])
        #return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()
        return ts.TimeSeries.time_axis.time_points[self.start_idx:self.start_idx+self.nb_pts], ts.v.to_numpy()[self.start_idx:self.start_idx+self.nb_pts]

    def get_geo_data(self, var_name, cat_id_lst):
        return self.cell_geo_data[var_name][np.in1d(self.cid,cat_id_lst)]


class SubcatDataExtractor(ArealDataExtractor):
    def __init__(self, rm, period=None, catch_select=None, clip=False, catch_names=None, geom=None):
        super().__init__(rm, period=period)
        self.geom = geom
        if geom is None:
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
                                         for k, v in self.outputs[self.stack_nm][output_grp].items() if hasattr(self.rm, output_grp)})

    def get_map(self, var_name, cat_id_lst_grp, t):
        return np.array([self._val_ext_methods[var_name](cat_id_lst, self._time_num_2_idx(t)) for cat_id_lst in cat_id_lst_grp])

    def get_ts(self, var_name, cat_id_lst):
        # if (self.stack_nm in ['ptssk', 'pthsk'] and var_name in ['swe']):
        #     return self._ts_map_ext_methods[var_name](cat_id_lst).v.to_numpy()[0:-1]
        # else:
        #     return self._ts_map_ext_methods[var_name](cat_id_lst).v.to_numpy()
        ts = self._ts_map_ext_methods[var_name](cat_id_lst)
        #return ts.time_axis.time_points[0:ts.size()], ts.v.to_numpy()
        return ts.time_axis.time_points[self.start_idx:self.start_idx+self.nb_pts], ts.v.to_numpy()[self.start_idx:self.start_idx+self.nb_pts]

    def get_geo_data(self, var_name, cat_id_lst_grp):
        idx = [np.in1d(self.cid,cat_id_lst) for cat_id_lst in cat_id_lst_grp]
        return np.array([np.average(self.cell_geo_data[var_name][idx[i]], weights=self.cell_geo_data['area'][idx[i]])
                for i, cat_id_lst in enumerate(cat_id_lst_grp)])