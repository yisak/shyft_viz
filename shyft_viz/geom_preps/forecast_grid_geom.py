import numpy as np
from shapely.geometry import MultiPolygon,box
from shapely.ops import transform
import pyproj
from functools import partial
from .utils import mpoly_2_pathpatch

def transform_poly(poly, src_cs, dest_cs):
    # Transform using pyproj
    project = partial(
        pyproj.transform,
        src_cs,  # source coordinate system
        dest_cs)  # destination coordinate system
    return transform(project, poly)

class GridViewerPrep(object):
    def __init__(self, x_model, y_model, model_cs, fc_cs, dxy):
        orig_proj = pyproj.Proj(fc_cs)
        model_proj = pyproj.Proj(model_cs)
        x, y = pyproj.transform(model_proj, orig_proj, x_model, y_model)
        x, y = np.around(np.array([x, y]), decimals=1)

        dxy2 = 0.5 * dxy
        # box(xmin,ymin,xmax,ymax)
        multipoly = MultiPolygon([box(xc - dxy2, yc - dxy2, xc + dxy2, yc + dxy2) for xc, yc in zip(x, y)])
        polys = transform_poly(multipoly, orig_proj, model_proj)
        pathpatches = [mpoly_2_pathpatch(p) for p in polys]

        self.bbox = polys.bounds
        self.ts_fetching_lst = list(range(len(polys)))
        self.map_fetching_lst = list(range(len(polys)))

        self.polys = polys
        self.patches = pathpatches