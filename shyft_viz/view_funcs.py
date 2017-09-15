import os
import pickle
from datetime import datetime
from shyft import api

from shapely.ops import cascaded_union

from .geom_preps.shyft_cell_geom import SubcatViewerPrep
from .data_extractors.hindcast_data import HindcastDataExtractor
from .data_extractors.shyft_TsVector_data import TsVectorDataExtractor
#from .data_extractors.shyft_regmod_data import SubcatDataExtractor
from .data_viewer import Viewer
from .geom_preps.utils import mpoly_2_pathpatch


class ViewFuncs(object):
    def __init__(self):
        GIS_DATA_DIR = os.path.join(os.environ['SHYFTDATA'], 'repository', 'gis_repository_cache')
        with open(os.path.join(GIS_DATA_DIR, "All_subcat_polygons.pkl"), "rb") as pkl_fil:
            self.subcat_polys = pickle.load(pkl_fil)
        #poly = cascaded_union(shapes).buffer(dxy * pad)

    def view_hindcast_sim(self, evaluator):
        region_name = evaluator.region_model_id.split('#')[0]
        shapes = [poly for poly, region in zip(self.subcat_polys['polygon'], self.subcat_polys['region_name']) if region == region_name]
        subcat_patches = [mpoly_2_pathpatch(shp) for shp in shapes]
        subcat_props = dict(facecolor='none', edgecolor='blue')
        foreground_patches = [{'patches': subcat_patches, 'props': subcat_props}]
        rm, eval_res = evaluator.region_model_preupdate, evaluator.result_forecast_eval

        shop_module_names, subcat_ids_in_shop_module = zip(*[(m['module_name'], m['catchment_indexes'].to_numpy().tolist())
                                                             for m in eval_res])

        cell_cid_full = [cell.geo.catchment_id() for cell in rm.cells]
        cell_shapes_full, catchment_id_map, bbox = rm.gis_info, rm.catchment_id_map, rm.bounding_region.geometry
        geom = SubcatViewerPrep(cell_cid_full, cell_shapes_full, catchment_id_map, bbox,
                                catch_grp=subcat_ids_in_shop_module, clip=False)
        # Can also get geom from rm data extractor
        # geom = ext(rm_update, catch_select=subcat_ids_in_shop_module, catch_names=shop_module_names).geom

        ts_vct_lst = [res['sim_discharge'] for res in eval_res]
        ext = {'Qsim_raw': HindcastDataExtractor(ts_vct_lst, shop_module_names, geom)}
        ts_lst = [res['obs_discharge'] for res in eval_res]
        ext.update({'Qobs_raw': TsVectorDataExtractor(ts_lst, shop_module_names, geom)})

        if 'interval_stats' in eval_res[0]:
            intervals = [interval_res['interval_spec'] for interval_res in eval_res[0]['interval_stats']]
            for i in range(len(intervals)):
                start, delta_t, n = intervals[i]
                interval_str = ','.join([str(t // api.deltahours(1)) for t in [start, delta_t, n * delta_t]])
                ts_vct_lst = [res['interval_stats'][i]['sim_discharge'] for res in eval_res]
                ext.update({'Qsim_'+interval_str: HindcastDataExtractor(ts_vct_lst, shop_module_names, geom)})
                ts_vct_lst = [res['interval_stats'][i]['obs_discharge'] for res in eval_res]
                ext.update({'Qobs_'+interval_str: HindcastDataExtractor(ts_vct_lst, shop_module_names, geom)})
        #period = api.UtcPeriod(eval_res[0]['sim_discharge'][0].time(0), eval_res[0]['sim_discharge'][-1].time(0))
        #ext.update({'Qsim': SubcatDataExtractor(rm, period=period, catch_select=subcat_ids_in_shop_module,
        #                    catch_names=shop_module_names)})

        return Viewer(ext,['q_avg'], [],
                      {'QsimVSQobs': {'Qsim_raw': ['q_avg'], 'Qobs_raw': ['q_avg']}},
                      time_marker=None, data_ext_pt=None, background_img=None, foreground_patches=foreground_patches,
                      default_var='q_avg',
                      default_ds='Qsim_raw')

    def view_continuous_sim(self, evaluator):
        region_name = evaluator.region_model_id.split('#')[0]
        shapes = [poly for poly, region in zip(self.subcat_polys['polygon'], self.subcat_polys['region_name']) if region == region_name]
        subcat_patches = [mpoly_2_pathpatch(shp) for shp in shapes]
        subcat_props = dict(facecolor='none', edgecolor='blue')
        foreground_patches = [{'patches': subcat_patches, 'props': subcat_props}]
        rm, eval_res = evaluator.region_model_preupdate, evaluator.result_historical_eval

        shop_module_names, subcat_ids_in_shop_module = zip(*[(m['module_name'], m['catchment_indexes'].to_numpy().tolist())
                                                             for m in eval_res])

        cell_cid_full = [cell.geo.catchment_id() for cell in rm.cells]
        cell_shapes_full, catchment_id_map, bbox = rm.gis_info, rm.catchment_id_map, rm.bounding_region.geometry
        geom = SubcatViewerPrep(cell_cid_full, cell_shapes_full, catchment_id_map, bbox,
                                catch_grp=subcat_ids_in_shop_module, clip=False)
        # Can also get geom from rm data extractor
        # geom = ext(rm_update, catch_select=subcat_ids_in_shop_module, catch_names=shop_module_names).geom

        ts_lst = [res['sim_discharge'] for res in eval_res]
        ext = {'Qsim_raw': TsVectorDataExtractor(ts_lst, shop_module_names, geom)}
        ts_lst = [res['obs_discharge'] for res in eval_res]
        ext.update({'Qobs_raw': TsVectorDataExtractor(ts_lst, shop_module_names, geom)})

        if 'interval_stats' in eval_res[0]:
            intervals = [interval_res['interval_spec'] for interval_res in eval_res[0]['interval_stats']]
            for i in range(len(intervals)):
                start, delta_t, n = intervals[i]
                interval_str = ','.join([datetime.utcfromtimestamp(start).strftime('%Y%m%d'), str(delta_t//api.deltahours(1)), str((n * delta_t)//api.deltahours(1))])
                ts_lst = [res['interval_stats'][i]['sim_discharge'] for res in eval_res]
                ext.update({'Qsim_'+interval_str: TsVectorDataExtractor(ts_lst, shop_module_names, geom)})
                ts_lst = [res['interval_stats'][i]['obs_discharge'] for res in eval_res]
                ext.update({'Qobs_'+interval_str: TsVectorDataExtractor(ts_lst, shop_module_names, geom)})

        return Viewer(ext,['q_avg'], [],
                      {'QsimVSQobs': {'Qsim_raw': ['q_avg'], 'Qobs_raw': ['q_avg']}},
                      time_marker=None, data_ext_pt=None, background_img=None, foreground_patches=foreground_patches,
                      default_var='q_avg',
                      default_ds='Qsim_raw')