# get basic stuff in to play
import numpy as np
from datetime import datetime
from statkraft.ssa.timeseries import Calendar,Period,UtcTime
# get snowman-service up and running
from statkraft.ssa.environment import SNOWMAN_PROD
from statkraft.ssa.snowmanrepository import SnowmanService, Coordinate

# have a look at the measurements from the web app here: http://webapp/snowman2/
# to find interesting spots etc.


# construct a simple function to create a coordinate with epsg (requirement for inputs)
def coordinate(x, y, epsg):
    c = Coordinate()
    c.X = x
    c.Y = y
    c.Epsg = epsg  # 32633 # need to specify on input
    return c

class SnowmanDataExtractor(object):
    def __init__(self, bbox, start, end):
        self.units = {'swe': 'mm'}

        #oslo = Calendar.Local  # so that we use local calendar,Europe/Oslo, but still utc-time for dealing with time
        utc = Calendar.Utc
        ss = SnowmanService(SNOWMAN_PROD)  # get our snowman service -> ss
        #with SnowmanService(SNOWMAN_PROD) as ss:
        # then get all snowman-locations, should be fast (or fail fast)
        self.snow_locations = ss.get_locations()  # returns a list of all locations, describing position, name etc.
        # create a period that we use for query out measurements
        #print(oslo.to_utctime(2010, 1, 1))
        #p = Period(oslo.to_utctime(2010, 1, 1), oslo.to_utctime(2017, 1, 1))
        p = Period(utc.to_utctime(start.year,start.month,start.day), utc.to_utctime(end.year,end.month,end.day))

        xmin, ymin, dx, dy = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        lower_left = coordinate(xmin, ymin, 32633)

        self.bb_measurements = ss.get_measurements_by_bb(p, lower_left, dx, dy)
        self.m_radar = [m for m in self.bb_measurements if 'Radar' in m.Type]
        self.nb_pts = len(self.m_radar)
        self.times = [[datetime.utcfromtimestamp(m.RegistrationDate.ToUnixTime())] for m in self.m_radar]
        self.SWE = [[m.SWE] for m in self.m_radar]
        self.names = [m.Location.Name for m in self.m_radar]
        #x = [s.m_radar[0].Profile.Points[i].Coordinate.X for i in range(len(s.m_radar[0].Profile.Points))]
        self.coord = np.array([[m.Location.Coordinate.X, m.Location.Coordinate.Y] for m in self.m_radar])


    def get_ts(self, var_name, indx):
        return self.times[indx],self.SWE[indx]


if __name__=='__main__':
    s = SnowmanDataExtractor()
    #print(s.snow_locations[0].name)