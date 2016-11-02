import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button, Cursor
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from datetime import datetime
from itertools import cycle


def plot_background(ax, f_path):
    from osgeo import gdal
    gdal.UseExceptions()

    # ds = gdal.Open('G:/Work/ENKI/NVE_Atlas/Vinjevatn_background_3.tiff')
    ds = gdal.Open(f_path)
    # band = ds.GetRasterBand(1)
    # elevation = band.ReadAsArray()
    elevation = ds.ReadAsArray()  # all bands

    nrows, ncols = elevation[0].shape

    # I'm making the assumption that the image isn't rotated/skewed/etc.
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()

    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows

    # plt.imshow(elevation, cmap='gist_earth', extent=[x0, x1, y1, y0])
    ax.imshow(np.rollaxis(elevation, 0, 3), cmap='gist_earth', extent=[x0, x1, y1, y0])  # all bands


class Viewer(object):
    def __init__(self, data_ext, data_ext_pt, time_marker=None, background=False):

        self.data_ext_pt = data_ext_pt # {k: v for k, v in data_ext_pt.items()}

        self.data_ext = {k: v for k, v in data_ext.items()}
        self.map_fetching_lst = {k: v.map_fetching_lst for k, v in data_ext.items()}
        self.ts_fetching_lst = {k: v.ts_fetching_lst for k, v in data_ext.items()}
        self.patches = {k: v.geom.patches for k, v in data_ext.items()}
        self.polys = {k: v.geom.polys for k, v in data_ext.items()}
        self.nb_catch = {k: len(v.geom.polys) for k, v in data_ext.items()}
        self.catch_nms = {k: v.catch_names for k, v in data_ext.items()}
        self.var_units = {k: v.var_units for k, v in data_ext.items()}
        # -Just picking the value for one of the datasets for now-
        bbox = {k: v.geom.bbox for k, v in data_ext.items()}
        self.bbox = list(bbox.values())[0]
        t_ax = {k: v.t_ax for k, v in data_ext.items()}
        self.t_ax = list(t_ax.values())[0]
        max_ti = {k: v.t_ax.n-1 for k, v in data_ext.items()}
        self.max_ti = list(max_ti.values())[0]
        times = {k: [datetime.utcfromtimestamp(v.t_ax.time(i)) for i in range(v.t_ax.size())] for k, v in data_ext.items()}
        self.times = list(times.values())[0]
        # --------------------------------------------------------

        #self.alreadyplottedCatchIndx = {k: np.zeros((len(v.geom.polys)), dtype=np.int) for k, v in data_ext.items()}
        #self.alreadyplottedDistVar = {k: [] for k in data_ext}

        self.ds_names_pt = ['SNOWMAN']
        self.ds_actve_pt = ['SNOWMAN']
        self.pt_vars = ['swe']
        self.pt_var = 'swe'

        self.ds_names = list(data_ext.keys())
        self.ds_active = self.ds_names[0]

        self.map = {k: None for k in data_ext}
        self.cbar = {k: None for k in data_ext}

        self.geo_data = ['z']
        self.dist_vars = ['temp', 'swe', 'q_avg', 'rad', 'prec', 'z']
        self.data_lim = {'temp': [-20., 40.], 'swe': [0., 500], 'q_avg': [0., 500], 'rad': [0., 1000.], 'prec': [0., 50.],
                         'z': [0., 3000.]}
        self.dist_var = self.dist_vars[0]
        self.ti = 0
        self.data = None

        self.time_marker = datetime.utcfromtimestamp(time_marker) if time_marker is not None else time_marker

        #self.tsplot = TsPlot(datetime.utcfromtimestamp(time_marker) if time_marker is not None else time_marker)
        self.tsplots = []

        #self.plt_mode = {'Plot_Source': False, 'Multi_Series': False, 'Re-plot': False, 'Custom_Plot': True}
        self.plt_mode = {'Plot_Source': False, 'Custom_Plot': True}
        self.custom_plt_vars = {'PTQ': ['temp', 'q_avg', 'prec']}
        self.custom_plt_types = list(self.custom_plt_vars.keys())
        self.custom_plt_active = self.custom_plt_types[0]

        self.data_lim_current = {nm: [0, 1] for nm in self.dist_vars}

        self.fig = plt.figure(figsize=(15, 6))#, facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.1,0.7,0.2]) #, height_ratios=[2,1])

        gs_var_select = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 0])
        gs_plot = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], height_ratios=[0.1,0.8,0.1])
        gs_options = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2])
        gs_lim_slider = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_plot[2, 0], width_ratios=[0.7,0.3])
        gs_navigate = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_plot[0, 0], width_ratios=[0.1,0.1,0.1,0.1,0.1,0.5])

        self.ax_plt = self.fig.add_subplot(gs_plot[1, 0])
        ax_dataset_slect = self.fig.add_subplot(gs_var_select[0, 0])
        ax_map_var_slect = self.fig.add_subplot(gs_var_select[1, 0])
        #ax_geo_data_slect = self.fig.add_subplot(gs_var_select[1, 0])
        ax_pt_var_slect = self.fig.add_subplot(gs_var_select[2, 0])
        #ax_options_1 = self.fig.add_subplot(gs[0,2])
        ax_options_1 = self.fig.add_subplot(gs_options[0,0])
        ax_oper_plots = self.fig.add_subplot(gs_options[1, 0])
        ax_min_slider = self.fig.add_subplot(gs_lim_slider[0, 0])
        ax_max_slider = self.fig.add_subplot(gs_lim_slider[1, 0])
        ax_reset_button = self.fig.add_subplot(gs_lim_slider[:, 1])
        ax_time_slider = self.fig.add_subplot(gs_navigate[0, 5])
        ax_navigate = {nm: self.fig.add_subplot(gs_navigate[0, i]) for i, nm in enumerate(['Prev', 'Play', 'Pause', 'Next', 'Update'])}

        if background:
            plot_background(self.ax_plt, 'G:/Work/shyft/background_img/NeaNidelva_background_3.tiff')

        self.add_plot()
        self.add_pt_plot()
        self.set_labels()
        #self.cbar[self.ds_active].set_visible(True)


        self.option_btn = self.add_check_button(ax_options_1, 'Options', list(self.plt_mode.keys()), list(self.plt_mode.values()), self.OnPltModeBtnClk)
        self.custom_plt_btn = self.add_radio_button(ax_oper_plots, 'Custom Plots', self.custom_plt_types,
                                                    self.OnCustomPltBtnClk)
        self.custom_plt_btn.set_active(self.custom_plt_types.index(self.custom_plt_active))
        self.add_data_lim_sliders(ax_min_slider, ax_max_slider)
        self.add_time_slider(ax_time_slider)
        self.add_media_button(ax_navigate)
        self.reset_lim_btn = self.add_reset_button(ax_reset_button, 'Reset', self.update_cbar_by_data_lim)
        self.pt_var_sel_btn = self.add_radio_button(ax_pt_var_slect, 'Pt_Vars', self.pt_vars, None)
        self.dist_var_sel_btn = self.add_radio_button(ax_map_var_slect, 'Dist_Vars', self.dist_vars, self.OnDistVarBtnClk)
        self.dist_var_sel_btn.set_active(self.dist_vars.index(self.dist_var)) # not available on older version of matplotlib

        self.dataset_sel_btn = self.add_radio_button(ax_dataset_slect, 'Datasets', self.ds_names,
                                                      self.OnDatasetSelect)
        self.dataset_sel_btn.set_active(self.ds_names.index(self.ds_active))

        gs.tight_layout(self.fig)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()


    def add_plot(self):
        self.ax_plt.set_xlim(self.bbox[0], self.bbox[2])
        self.ax_plt.set_ylim(self.bbox[1], self.bbox[3])
        self.ax_plt.set_aspect('equal')
        self.ax_plt.format_coord = self.format_coord
        divider = make_axes_locatable(self.ax_plt)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        for ds in self.ds_names:
            self.map[ds] = self.ax_plt.add_collection(PatchCollection(self.patches[ds], alpha=0.9))
            self.data = self.data_ext[ds].get_map(self.dist_var, self.map_fetching_lst[ds], self.ti)
            self.map[ds].set_array(self.data)
            self.cbar[ds] = self.fig.colorbar(self.map[ds], cax=cax, orientation='vertical')
            self.map[ds].set_visible(False)

    def add_pt_plot(self):
        self.ax_plt.plot(self.data_ext_pt.coord[:, 0], self.data_ext_pt.coord[:, 1], marker='o', ls='None', ms=8, mfc='none',
                        mec='r', mew=2, picker=5)

    def update_cbar(self, event):
        self.map[self.ds_active].set_clim([self.slidermin.val, self.slidermax.val])
        self.fig.canvas.draw()

    def OnDatasetSelect(self, label):
        self.map[self.ds_active].set_visible(False)
        print(label)
        self.map[label].set_visible(True)
        self.ds_active = label
        self.data = self.map[self.ds_active].get_array()
        self.fig.canvas.draw()

    def OnDistVarBtnClk(self, label):
        self.ax_plt.set_title(self.ax_plt.get_title().replace(self.dist_var, label), fontsize=12)
        self.dist_var = label
        print(self.dist_var)
        if self.dist_var in self.geo_data:
            self.data = self.data_ext[self.ds_active].get_geo_data(self.dist_var, self.map_fetching_lst[self.ds_active])
            [self.media_buttons[nm].disconnect(cid) for nm, cid in zip(self.media_btn_nms, self.media_btn_cids)]
        else:
            self.data = self.data_ext[self.ds_active].get_map(self.dist_var, self.map_fetching_lst[self.ds_active], self.ti)
            self.media_btn_cids = [getattr(self.media_buttons[nm], 'on_clicked')(func) for nm, func in
                                   zip(self.media_btn_nms, self.media_btn_funcs)]
        self.map[self.ds_active].set_array(self.data)
        #self.map.set_clim([data.min(), data.max()])
        lo, hi = self.data_lim_current[self.dist_var]
        self.map[self.ds_active].set_clim([self.scale_data_lim(v) for v in [lo, hi]])
        self.slidermin.set_val(lo)
        self.slidermax.set_val(hi)
        #self.map.update_scalarmappable()

        self.fig.canvas.draw()

    def OnCustomPltBtnClk(self, label):
        self.custom_plt_type = label

    def OnPltModeBtnClk(self, label):
        self.plt_mode[label] = not self.plt_mode[label]
        print(label,self.plt_mode[label])

    def update_cbar_by_slider_lim(self):
        self.map[self.ds_active].set_clim([self.scale_data_lim(v) for v in [self.slidermin.val, self.slidermax.val]])
        self.fig.canvas.draw()

    def update_cbar_by_data_lim(self, event):
        #data = self.map.get_array()
        self.map[self.ds_active].set_clim([self.data.min(), self.data.max()])
        self.fig.canvas.draw()

    def add_radio_button(self, but_ax, title, labels, func):
        axcolor = 'lightgoldenrodyellow'
        but_ax.set_title(title)
        radio_but = RadioButtons(but_ax, labels)
        radio_but.on_clicked(func)
        return radio_but

    def add_check_button(self, but_ax1, title, labels, vals, func):
        axcolor = 'lightgoldenrodyellow'
        but_ax1.set_title(title)
        check = CheckButtons(but_ax1, labels, vals)
        check.on_clicked(func)
        return check

    def add_data_lim_sliders(self, axmin, axmax):
        axcolor = 'lightgoldenrodyellow'
        self.slidermin = Slider(axmin, 'Min', 0., 1., valinit=self.data_lim_current[self.dist_var][0])
        self.slidermax = Slider(axmax, 'Max', 0., 1., valinit=self.data_lim_current[self.dist_var][1])
        self.slidermin.valtext.set_text(self.scale_data_lim(self.slidermin.val))
        self.slidermax.valtext.set_text(self.scale_data_lim(self.slidermax.val))
        self.slidermin.on_changed(self.update_min_lim)
        self.slidermax.on_changed(self.update_max_lim)

    def update_min_lim(self, val):
        self.data_lim_current[self.dist_var][0] = val
        self.slidermin.valtext.set_text(self.scale_data_lim(val))
        self.update_cbar_by_slider_lim()

    def update_max_lim(self, val):
        self.data_lim_current[self.dist_var][1] = val
        self.slidermax.valtext.set_text(self.scale_data_lim(val))
        self.update_cbar_by_slider_lim()

    def scale_data_lim(self, val):
        return self.data_lim[self.dist_var][0] + val * (self.data_lim[self.dist_var][1] - self.data_lim[self.dist_var][0])

    def add_time_slider(self, ax_slider):
        axcolor = 'lightgoldenrodyellow'
        self.time_slider = Slider(ax_slider, 'Time', self.t_ax.start,
                                  self.t_ax.start + self.t_ax.delta_t * (self.t_ax.size() - 1),
                                  valinit=self.t_ax.start)
        self.time_slider.valtext.set_text(
            self.data_ext[self.ds_active].time_num_2_str(self.t_ax.index_of(self.time_slider.val)))
        self.time_slider.on_changed(self.update_time)

    def update_time(self,val):
        t_indx = self.t_ax.index_of(int(self.time_slider.val))
        #self.ti = t_indx
        #self.update_plot() # will hnag if slider dragged too fast
        self.time_slider.valtext.set_text(self.data_ext[self.ds_active].time_num_2_str(t_indx))

    def add_media_button(self, ax_navigate):
        axcolor = 'lightgoldenrodyellow'
        self.media_btn_nms = ['Prev', 'Play', 'Pause', 'Next', 'Update']
        self.media_btn_funcs = [self.OnPrev, self.OnPlay, self.OnPause, self.OnNext, self.OnUpdate]
        self.media_buttons = {nm: Button(ax_navigate[nm], nm, color=axcolor, hovercolor='0.975') for nm in self.media_btn_nms}
        self.media_btn_cids = [getattr(self.media_buttons[nm], 'on_clicked')(func) for nm, func in zip(self.media_btn_nms, self.media_btn_funcs)]

    def add_reset_button(self, ax_button, label, func):
        axcolor = 'lightgoldenrodyellow'
        reset_button = Button(ax_button, label, color=axcolor, hovercolor='0.975')
        reset_button.on_clicked(func)
        return reset_button

    def which_catch(self, x, y):
        p = Point(x, y)
        indx = None
        for i in range(self.nb_catch[self.ds_active]):
            if self.polys[self.ds_active][i].contains(p):
                indx = i
        return indx

    def format_coord(self, x, y):
        indx = self.which_catch(x, y)
        if (indx is None):
            info = 'x=%1.4f, y=%1.4f, val=None' % (x, y)
        else:
            z = self.data[indx]
            info = 'x=%1.4f, y=%1.4f, val=%1.4f, name=%s' % (x, y, z, self.catch_nms[self.ds_active][indx])
        return info

    def OnNext(self, *args):
        self.ti += 1
        if (self.ti > self.max_ti): self.ti = 0
        # if(self.ti<0):self.ti=self.max_ti
        self.update_plot()

    def OnPrev(self, *args):
        self.ti -= 1
        # if(self.ti>self.max_ti):self.ti=0
        if (self.ti < 0): self.ti = self.max_ti
        self.update_plot()

    def OnPlay(self, event):
        # self.ani = animation.FuncAnimation(self.fig, self.OnNext,blit=False, interval=10,repeat=False)
        # self.fig.canvas.manager.window.after(100, self.OnNext)
        self.timer.add_callback(self.OnNext)  # , selfax)
        self.timer.start()

    def OnPause(self, event):
        self.timer.remove_callback(self.OnNext)

    def OnUpdate(self, event):
        t_indx = self.t_ax.index_of(int(self.time_slider.val))
        if self.ti != t_indx:
            self.ti = t_indx
            self.update_plot()

    def update_plot(self):
        self.data = self.data_ext[self.ds_active].get_map(self.dist_var, self.map_fetching_lst[self.ds_active], self.ti)
        self.map[self.ds_active].set_array(self.data)
        self.ax_plt.title.set_text(
            '%s - %s' % (self.dist_var, self.data_ext[self.ds_active].time_num_2_str(self.ti)))
        self.fig.canvas.draw()

    def set_labels(self):
        self.fig.canvas.set_window_title('Shyft-viz')
        self.ax_plt.set_title('%s - %s' % (self.dist_var, self.data_ext[self.ds_active].time_num_2_str(self.ti)), fontsize=12)

    def on_click(self, event):
        if event.inaxes is not self.ax_plt: return True
        tb = self.fig.canvas.manager.toolbar
        if not self.plt_mode['Plot_Source'] and tb.mode == '':
            x = event.xdata
            y = event.ydata
            catchind = self.which_catch(x, y)
            print(catchind)
            if catchind is None: return True
            tsplots_active = [p for p in self.tsplots if p.fig is not None and p.plt_mode['Plot_over']]
            if len(tsplots_active)==0 or self.plt_mode['Custom_Plot']:
                tsplot = TsPlot(self.time_marker)
                tsplot.alreadyplottedPtIndx = np.zeros(self.data_ext_pt.nb_pts, dtype=np.int)
                tsplot.alreadyplottedPtVar = []

            #if self.tsplot.fig is None:
                #self.alreadyplottedCatchIndx[self.ds_active][:] = 0
                #self.alreadyplottedDistVar[self.ds_active] = []
                tsplot.alreadyplottedCatchIndx = {k: np.zeros(self.nb_catch[k], dtype=np.int) for k in self.ds_names}
                tsplot.alreadyplottedDistVar = {k: [] for k in self.ds_names}
                #var_indx = np.nonzero(self.var_select)
                if self.plt_mode['Custom_Plot']:
                    dist_vars = ['temp', 'q_avg', 'prec']
                else:
                    dist_vars = [self.dist_var]

                props = [{} for _ in dist_vars]
                if 'prec' in dist_vars:
                    props[dist_vars.index('prec')].update({'drawstyle':'steps'})
                #ts_v = self.data_ext.get_ts(self.dist_var, self.ts_fetching_lst[catchind])
                tsplot.init_plot(self.times,
                                      [self.data_ext[self.ds_active].get_ts(dist_var, self.ts_fetching_lst[self.ds_active][catchind]) for dist_var in dist_vars],
                                      [dist_var + '_' + self.catch_nms[self.ds_active][catchind] for dist_var in dist_vars],
                                      [self.var_units[self.ds_active][dist_var] for dist_var in dist_vars], props)
                tsplot.alreadyplottedCatchIndx[self.ds_active][catchind] = 1
                tsplot.alreadyplottedDistVar[self.ds_active].append(self.dist_var)
                self.tsplots.append(tsplot)
            else:
                for tsplot in tsplots_active:
                    print(self.ds_active)
                    #if self.alreadyplottedCatchIndx[self.ds_active][catchind] and self.dist_var in self.alreadyplottedDistVar[self.ds_active] and not self.plt_mode['Re-plot']: return True
                    if tsplot.alreadyplottedCatchIndx[self.ds_active][catchind] and self.dist_var in \
                            tsplot.alreadyplottedDistVar[self.ds_active] and not tsplot.plt_mode['Re-plot']: return True
                    # if not self.plt_mode['Multi_Series']:
                    #     self.tsplot.clear_plot()
                    #     self.alreadyplottedCatchIndx[self.ds_active][:] = 0
                    #     self.alreadyplottedDistVar[self.ds_active] = []
                    #var_indx = np.nonzero(self.var_select)
                    ts_v = self.data_ext[self.ds_active].get_ts(self.dist_var, self.ts_fetching_lst[self.ds_active][catchind])
                    print(self.dist_var)
                    tsplot.add_plot(self.times, [ts_v],
                                         [self.dist_var + '_' + self.catch_nms[self.ds_active][catchind]],
                                         [self.var_units[self.ds_active][self.dist_var]], [{}])
                    tsplot.alreadyplottedCatchIndx[self.ds_active][catchind] = 1
                    tsplot.alreadyplottedDistVar[self.ds_active].append(self.dist_var)
            #self.alreadyplottedCatchIndx[self.ds_active][catchind] = 1
            #self.alreadyplottedDistVar[self.ds_active].append(self.dist_var)


    def on_pick(self, event):
        tb = self.fig.canvas.manager.toolbar
        if self.plt_mode['Plot_Source'] and tb.mode == '':

            # if event.artist!=self.overview: return True

            N = len(event.ind)
            if not N: return True

            # the click locations
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata

            distances = np.hypot(x - self.data_ext_pt.coord[event.ind, 0], y - self.data_ext_pt.coord[event.ind, 1])
            indmin = distances.argmin()
            catchind = event.ind[indmin]

            self.lastind = catchind
            tsplots_active = [p for p in self.tsplots if p.fig is not None and p.plt_mode['Plot_over']]
            if len(tsplots_active)==0: # or self.plt_mode['Custom_Plot']:
                tsplot = TsPlot(self.time_marker)

                tsplot.alreadyplottedPtIndx = np.zeros(self.data_ext_pt.nb_pts, dtype=np.int)
                tsplot.alreadyplottedPtVar = []
                ts_t, ts_v = self.data_ext_pt.get_ts(self.pt_var, catchind)
                tsplot.init_plot(ts_t, [ts_v],
                                         [self.pt_var + '_' + self.data_ext_pt.names[catchind]],
                                         [self.data_ext_pt.units[self.pt_var]],[{'ls':'None'}])
                tsplot.alreadyplottedPtIndx[catchind] = 1
                tsplot.alreadyplottedPtVar.append(self.pt_var)
                self.tsplots.append(tsplot)
            else:
                for tsplot in tsplots_active:
                    if tsplot.alreadyplottedPtIndx[catchind] and self.pt_var in \
                            tsplot.alreadyplottedPtVar and not tsplot.plt_mode['Re-plot']: return True

                    ts_t, ts_v = self.data_ext_pt.get_ts(self.pt_var, catchind)
                    print(self.pt_var)
                    tsplot.add_plot(ts_t, [ts_v],
                                         [self.pt_var + '_' + self.data_ext_pt.names[catchind]],
                                         [self.data_ext_pt.units[self.pt_var]], [{'ls':'None'}])
                    tsplot.alreadyplottedPtIndx[catchind] = 1
                    tsplot.alreadyplottedPtVar.append(self.pt_var)


class TsPlot(object):
    def __init__(self, time_marker):  # ,timesteps,values):
        self.time_marker = time_marker
        self.fig = None
        self.ax = None
        self.plotted_unit = None
        self.axes = None
        self.reset_plot()
        self.colors = ('Green', 'Red', 'Blue', 'Cyan') # colors = cycle(matplotlib.rcParams['axes.color_cycle']) # plt.rcParams['axes.color_cycle']
        #colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        #self.line_styles = [cycle(['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>',
        #                         '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']) for _ in range(4)]
        self.line_styles = [cycle(['-', '--', '-.', ':']) for _ in range(4)]
        # Filter out filled markers and marker settings that do nothing.
        # We use iteritems from six to make sure that we get an iterator
        # in both python 2 and 3
        unfilled_markers = [m for m, func in Line2D.markers.items()
                            if func != 'nothing' and m not in Line2D.filled_markers]
        markers = {'*': 'star', '2': 'tri_up', 2: 'tickup', 'o': 'circle', 4: 'caretleft', 5: 'caretright',
                   '_': 'hline', '.': 'point', 'd': 'thin_diamond', '4': 'tri_right', '': 'nothing', 'None': 'nothing',
                   3: 'tickdown', ' ': 'nothing', 7: 'caretdown', 'x': 'x', 0: 'tickleft', '+': 'plus',
                   '<': 'triangle_left', '|': 'vline', '8': 'octagon', 1: 'tickright', 6: 'caretup', 's': 'square',
                   'p': 'pentagon', ',': 'pixel', '^': 'triangle_up', 'D': 'diamond', None: 'nothing', 'H': 'hexagon2',
                   '3': 'tri_left', '>': 'triangle_right', 'h': 'hexagon1', 'v': 'triangle_down', '1': 'tri_down'}
        self.markers = [cycle(['o', '*', 's', 'v', 'x', 'p', '+']) for _ in range(4)]
        self.temp = [0.85, 0.75, 0.6]
        self.i = 1

        self.plt_mode = {'Plot_over': False, 'Re-plot': False}
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[0.1, 0.9])  # , height_ratios=[2,1])


    def add_check_button(self, but_ax1, title, labels, vals, func):
        axcolor = 'lightgoldenrodyellow'
        but_ax1.set_title(title)
        check = CheckButtons(but_ax1, labels, vals)
        check.on_clicked(func)
        return check

    def OnPltModeBtnClk(self, label):
        self.plt_mode[label] = not self.plt_mode[label]
        print(label,self.plt_mode[label])

    def reset_plot(self):
        self.plotted_unit = []
        self.axes = []
        self.lines = []
        #self.fig.autofmt_xdate()
        #self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)


    def init_plot(self, t, v, labels, units, prop):
        #self.fig, self.ax = plt.subplots()
        self.fig = plt.figure()#, (15, 6))  # , facecolor='white')
        self.ax = self.fig.add_subplot(self.gs[0, 1])
        ax_options_1 = self.fig.add_subplot(self.gs[0, 0])
        self.option_btn = self.add_check_button(ax_options_1, 'Options', list(self.plt_mode.keys()),
                                                list(self.plt_mode.values()), self.OnPltModeBtnClk)
        #self.ax = host_subplot(111)

        #self.fig.subplots_adjust(hspace=0.1, bottom=0.2)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        #plt.setp([a.get_xticklabels() for a in self.fig.axes[:-1]], visible=False)
        #self.subplot_autofmt_xdate(self.fig.axes[-1])
        # self.subplot_autofmt_xdate(self.ax) # for multiple subplots
        self.fig.autofmt_xdate() # for one subplot
        self.add_plot(t, v, labels, units, prop)
        #self.multi = MultiCursor(self.fig.canvas, self.ax, color='r', lw=1)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)
        self.gs.tight_layout(self.fig)

    def plot(self,ax, t, v, kwargs):
        return ax.plot(t, v, **kwargs)[0]

    def add_plot(self, t, v, labels, units, prop):
        for k in range(len(v)):
            if (not np.all(np.isnan(v[k]))):
                if (len(self.plotted_unit) == 0):
                    self.axes.append(self.ax)
                    #self.lines.append(self.axes[0].plot(t, v[k], ls=next(self.line_styles[0]), color=self.colors[0], label=labels[k], marker=marker, ms=8, markevery=None)[0])
                    l_prop = dict(ls='-', color=self.colors[0], label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)
                    l_prop.update(prop[k])
                    self.lines.append(self.plot(self.axes[0],t, v[k], l_prop))
                    #self.lines.append(self.axes[0].plot(t, v[k], ls=ls, color=self.colors[0], label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None, drawstyle='steps')[0])

                    self.plotted_unit.append(units[k])
                    self.axes[0].set_ylabel(units[k],color=self.colors[0])
                    self.axes[0].tick_params(axis='y', colors=self.colors[0])
                    if self.time_marker is not None:
                        self.axes[0].axvline(x=self.time_marker)
                else:
                    if (units[k] in self.plotted_unit):
                        idx=self.plotted_unit.index(units[k])
                        color = self.colors[idx]
                        #self.lines.append(self.axes[idx].plot(t, v[k], ls=next(self.line_styles[idx]), color=color, label=labels[k], marker=marker, ms=8, markevery=None)[0])
                        l_prop = dict(ls='-', color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)
                        l_prop.update(prop[k])
                        self.lines.append(self.plot(self.axes[idx], t, v[k], l_prop))
                        #self.lines.append(self.axes[idx].plot(t, v[k], ls=ls, color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)[0])

                        self.axes[idx].tick_params(axis='y', colors=color)
                    else:
                        self.plotted_unit.append(units[k])
                        idx = self.plotted_unit.index(units[k])
                        color = self.colors[idx]
                        self.axes.append(self.axes[0].twinx())
                        #self.lines.append(self.axes[-1].plot(t, v[k], ls=next(self.line_styles[idx]), color=color, label=labels[k], marker=marker, ms=8, markevery=None)[0])
                        l_prop = dict(ls='-', color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)
                        l_prop.update(prop[k])
                        self.lines.append(self.plot(self.axes[-1], t, v[k], l_prop))
                        #self.lines.append(self.axes[-1].plot(t, v[k], ls=ls, color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)[0])

                        print(self.lines)

                        self.axes[-1].set_ylabel(units[k])
                        self.axes[-1].format_coord = self.make_format(self.axes[-1], self.axes[0])
                        self.axes[-1].tick_params(axis='y', colors=color)

                    if len(self.axes)>2:
                        # Make some space on the right side for the extra y-axis.
                        #self.fig.subplots_adjust(right=0.75)
                        temp = self.temp[len(self.axes)-2]
                        right_additive = (0.98 - temp) / float(len(self.axes)-2)
                        self.fig.subplots_adjust(right=temp)
                        # Move the last y-axis spine over to the right by 20% of the width of the axes
                        #self.axes[-1].spines['right'].set_position(('axes', 1.2))
                        self.axes[-1].spines['right'].set_position(('axes', 1. + right_additive * self.i))
                        self.i += 1
                        # To make the border of the right-most axis visible, we need to turn the frame
                        # on. This hides the other plots, however, so we need to turn its fill off.
                        self.axes[-1].set_frame_on(True)
                        self.axes[-1].patch.set_visible(False)

                # self.axes[i][0].legend()

                #self.subplot_autofmt_xdate(self.ax)


                self.show_legend(0)
                self.fig.canvas.draw()

    def make_format(self, current, other):
        # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.transData.transform((x, y))
            inv = other.transData.inverted()
            # convert back to data coords with respect to ax
            ax_coord = inv.transform(display_coord)
            coords = [ax_coord, (x, y)]
            return ('Left: {:<40}    Right: {:<}'
                    .format(
                *['({}, {:.3f})'.format(mdates.num2date(xi).strftime('%Y-%m-%d %H:%M'), yi) for xi, yi in coords]))

        return format_coord

    def handle_close(self, evt):
        self.fig = None
        self.reset_plot()

    def show_legend(self, ax_i):
        labs = [l.get_label() for l in self.lines]
        self.axes[ax_i].legend(self.lines, labs, loc=0)

    def clear_plot(self):

        for ax in self.axes:
            if not isinstance(ax, list):
                ax.cla()
            else:
                j = 0
                for ax_twin in ax:
                    ax_twin.cla()
                    if (j > 0):
                        plt.setp(ax_twin.get_yticklabels(), visible=False)
                        # plt.setp(ax_twin.get_yticks(), visible=False)
                        ax_twin.set_yticks([])
                    j += 1
        # plt.setp([a.get_xticklabels() for a in self.fig.axes[:-1]], visible=False)
        self.reset_plot()

    #        for ax in self.axes:
    #            for ax_twin in ax:
    #                ax_twin.lines=[]
    #            if(len(ax)>0): ax[0].legend_ = None
    # self.fig.canvas.draw()

    def subplot_autofmt_xdate(self, ax):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)


