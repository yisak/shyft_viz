import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from shapely.geometry import MultiPolygon, Polygon


def mpoly_2_pathpatch(mpoly):
    #print(type(mpoly))
    assert isinstance(mpoly, (MultiPolygon, Polygon))

    codes = []
    all_x = []
    all_y = []

    if(isinstance(mpoly, Polygon)):
        mpoly = [mpoly]

    for poly in mpoly:
        x = np.array(poly.exterior)[:,0].tolist()
        y = np.array(poly.exterior)[:,1].tolist()
        # skip boundary between individual rings
        codes += [mpath.Path.MOVETO] + (len(x)-1)*[mpath.Path.LINETO]
        all_x += x
        all_y += y

    carib_path = mpath.Path(np.column_stack((all_x,all_y)), codes)
    carib_patch = mpatches.PathPatch(carib_path)#,lw=0.5,fc='blue', alpha=0.3)#facecolor='none'

    return carib_patch