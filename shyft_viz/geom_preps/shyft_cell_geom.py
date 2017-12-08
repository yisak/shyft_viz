import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union
from .utils import mpoly_2_pathpatch

class ViewerPrep(object):
    def __init__(self, cell_cid_full, cell_shapes_full, catchment_id_map, bbox):
        #self.rm = rm
        self.catchment_id_map = catchment_id_map # self.rm.catchment_id_map
        self.polys = None
        self.patches = None
        self.bbox = bbox # self.rm.bounding_region.geometry
        self.cell_shapes_full = cell_shapes_full # self.rm.gis_info
        #self.cell_data_ext = CellDataExtractor(self.rm)
        self.cell_cid_full = cell_cid_full # self.cell_data_ext.cid

    @staticmethod
    def shp_2_polypatch(polygons):
        if not isinstance(polygons, list):
            polygons = [polygons]
        return [mpoly_2_pathpatch(p) for p in polygons]

    def plot_region(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # ax.add_collection(PatchCollection([PolygonPatch(p,facecolor=None,edgecolor="black") for p in self.catch_polys], match_original=True))
        ax.add_collection(PatchCollection(self.patches, facecolor='none', edgecolor='blue'))
        ax.set_xlim(self.bbox[0], self.bbox[2])
        ax.set_ylim(self.bbox[1], self.bbox[3])
        plt.show()

    def clip_extent(self):
        #union_poly = cascaded_union(self.polys)
        #self.bbox = union_poly.bounds
        self.bbox = MultiPolygon(polygons=self.polys).bounds


class CellViewerPrep(ViewerPrep):
    def __init__(self, cell_cid_full, cell_shapes_full, catchment_id_map, bbox, catch_select=None, clip=False):
        super().__init__(cell_cid_full, cell_shapes_full, catchment_id_map, bbox)

        if catch_select is None:
            self.catch_ids_select = np.array(self.catchment_id_map).tolist()
            self.cell_idx_select = np.in1d(self.cell_cid_full, self.catchment_id_map).nonzero()[0].tolist()
        else:
            self.catch_ids_select = catch_select
            self.cell_idx_select = np.in1d(self.cell_cid_full, catch_select).nonzero()[0].tolist()

        #self.cell_idx_select = np.in1d(self.cell_cid_full, self.catch_ids_select).nonzero()[0]
        #self.cell_shapes_select = self.cell_shapes_full[self.cell_idx_select]

        self.ts_fetching_lst = self.cell_idx_select
        self.map_fetching_lst = self.catch_ids_select

        self.polys = self.cell_shapes_full[self.cell_idx_select].tolist()
        self.patches = self.shp_2_polypatch(self.polys)

        if clip:
            self.clip_extent()

class SubcatViewerPrep(ViewerPrep):
    def __init__(self, cell_cid_full, cell_shapes_full, catchment_id_map, bbox, catch_grp=None, clip=False):
        super().__init__(cell_cid_full, cell_shapes_full, catchment_id_map, bbox)

        if catch_grp is None:
            self.catch_ids_select = np.array(self.catchment_id_map).tolist()
            self.catch_grp_select = [[cid] for cid in self.catchment_id_map]
        else:
            self.catch_ids_select = [item for sublist in catch_grp for item in sublist]
            self.catch_grp_select = [list(seq) for seq in catch_grp]

        self.ts_fetching_lst = self.map_fetching_lst = self.catch_grp_select

        self.polys = [cascaded_union(self.cell_shapes_full[np.in1d(self.cell_cid_full,cid_lst)].tolist())
                      for cid_lst in self.catch_grp_select]
        self.patches = self.shp_2_polypatch(self.polys)

        if clip:
            self.clip_extent()

