import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import dates as mdate
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button, Cursor
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from datetime import datetime
from pytz import utc
from itertools import cycle
plt.style.use('ggplot')

def utctimestamp_2_datetime(utc_lst):
    return [datetime.utcfromtimestamp(t_num).replace(tzinfo=utc) for t_num in utc_lst]


def plot_background(ax, f_path):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(f_path)
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


class AnnoteFinder(object):
    def __init__(self, xdata, ydata, annotes, ax):
        self.x = xdata
        self.y = ydata
        self.annotes = annotes
        self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def __call__(self, event):
        tb = self.ax.figure.canvas.manager.toolbar
        if tb.mode == '': # event.inaxes:
            if not len(event.ind): return True
            ind = event.ind
            distances = np.hypot(event.mouseevent.xdata - np.take(self.x, ind), event.mouseevent.ydata - np.take(self.y, ind))
            indmin = distances.argmin()
            ptind = event.ind[indmin]
            self.drawAnnote(event.mouseevent.inaxes, self.x[ptind], self.y[ptind], self.annotes[ptind])
            for l in self.links:
                l.drawSpecificAnnote(self.annotes[ptind])

    def drawAnnote(self, ax, x, y, annote):
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " - %s" % (annote),)
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in zip(self.xdata, self.ydata, self.annotes) if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)


class Viewer(object):
    def __init__(self, data_ext, temporal_vars, static_vars, custom_plt, time_marker=None, background_img=None, foreground_patches=None,
                 data_ext_pt=None, default_var=None, default_ds=None, default_pt_var=None, default_pt_ds=None):
        # Set-up Dist dataset
        #self.data_ext = {k: v for k, v in data_ext.items()}
        self.data_ext = {k: v for k, v in data_ext.items() if any(i in temporal_vars for i in v.temporal_vars) or any(j in static_vars for j in v.static_vars)}
        self.map_fetching_lst = {k: v.map_fetching_lst for k, v in self.data_ext.items()}
        self.ts_fetching_lst = {k: v.ts_fetching_lst for k, v in self.data_ext.items()}
        self.patches = {k: v.geom.patches for k, v in self.data_ext.items()}
        self.polys = {k: v.geom.polys for k, v in self.data_ext.items()}
        self.nb_catch = {k: len(v.geom.polys) for k, v in self.data_ext.items()}
        self.catch_nms = {k: v.catch_names for k, v in self.data_ext.items()}
        self.var_units = {k: v.var_units for k, v in self.data_ext.items()}
        self.dist_vars = list(set([var for v in self.data_ext.values() for var in v.temporal_vars if var in temporal_vars]))
        self.dist_vars_pr_ds = {k: [var for var in v.temporal_vars if var in temporal_vars] for k, v in
                             self.data_ext.items()}
        self.geo_data = list(set([var for v in self.data_ext.values() for var in v.static_vars if var in static_vars]))
        # -Just picking the value for one of the datasets for now-
        bbox = {k: v.geom.bbox for k, v in self.data_ext.items()}
        self.bbox = list(bbox.values())[0]
        t_ax = {k: v.t_ax for k, v in self.data_ext.items()}
        if len(t_ax) > 1:
            self.t_ax = np.sort(np.unique(np.concatenate([t_arr for t_arr in t_ax.values()])))
        else:
            self.t_ax = list(t_ax.values())[0]
        self.t_min, self.t_max = self.t_ax[0], self.t_ax[-1]
        self.max_ti = len(self.t_ax)-1

        self.ds_names = list(self.data_ext.keys())
        self.ds_active = default_ds
        if default_ds is None:
            self.ds_active = self.ds_names[0]

        self.map = {k: None for k in self.data_ext}
        self.cbar = {k: None for k in self.data_ext}

        self.data_lim = {'temp': [-20., 40.], 'swe': [0., 500], 'q_avg': [0., 500], 'rad': [0., 1000.],
                         'prec': [0., 50.],
                         'z': [0., 3000.]}
        self.dist_var = default_var
        if default_var is None:
            self.dist_var = self.dist_vars[0]
        self.ti = 0
        self.data = None

        self.record = {}

        self.ts_name_separator = '#'

        # Set-up Point dataset
        if data_ext_pt is not None:
            self.data_ext_pt = {k: v for k, v in data_ext_pt.items() if any(i in temporal_vars for i in v.temporal_vars)}
            self.var_units_pt = {k: v.var_units for k, v in self.data_ext_pt.items()}
            self.nb_pts = {k: v.nb_pts for k, v in self.data_ext_pt.items()}
            self.pt_nms = {k: v.names for k, v in self.data_ext_pt.items()}
            self.pt_coord = {k: v.coord for k, v in self.data_ext_pt.items()}

            #all_pt_vars = list(set().union(*[v.temporal_vars for v in self.data_ext_pt.values()]))
            #self.pt_vars = [var for var in temporal_vars if var in all_pt_vars]
            self.pt_vars_pr_ds = {k: [var for var in v.temporal_vars if var in temporal_vars] for k, v in self.data_ext_pt.items()}
            self.pt_vars = list(set([var for v in self.data_ext_pt.values() for var in v.temporal_vars if var in temporal_vars]))
            self.ds_names_pt = list(self.data_ext_pt.keys())

            self.ds_active_pt = default_pt_ds
            if default_pt_ds is None:
                self.ds_active_pt = self.ds_names_pt[0]
            self.pt_var = default_pt_var
            if default_pt_var is None:
                self.pt_var = self.pt_vars[0]
            self.map_pt = {k: None for k in self.data_ext_pt}

        self.time_marker = datetime.utcfromtimestamp(time_marker) if time_marker is not None else time_marker

        self.tsplots = []


        plt_mode_label = ['Plot_dist_dataset', 'Add_dist_ds_to_rec', 'Del_dist_ds_from_rec', 'Custom_Plot']
        self.custom_plt = custom_plt
        self.custom_plt_types = list(self.custom_plt.keys())
        self.custom_plt_active = self.custom_plt_types[0] if len(custom_plt) else None

        self.data_lim_current = {nm: [0, 1] for nm in self.dist_vars}

        self.fig = plt.figure(figsize=(15, 6))#, facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.15,0.7,0.15]) #, height_ratios=[2,1])

        if data_ext_pt is None:
            gs_var_select = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0])
        else:
            gs_var_select = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 0])
        gs_plot = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], height_ratios=[0.1,0.8,0.1])
        gs_options = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 2], height_ratios=[0.5,0.1,0.1,0.3])
        gs_lim_slider = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_plot[2, 0], width_ratios=[0.025,0.775,0.2])
        gs_navigate = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs_plot[0, 0], width_ratios=[0.1,0.1,0.1,0.1,0.1,0.02,0.3,0.18])

        self.ax_plt = self.fig.add_subplot(gs_plot[1, 0])
        ax_dataset_slect = self.fig.add_subplot(gs_var_select[0, 0])
        ax_map_var_slect = self.fig.add_subplot(gs_var_select[1, 0])

        if data_ext_pt is not None:
            plt_mode_label.extend(['Plot_pt_dataset', 'Add_pt_ds_to_rec', 'Del_pt_ds_from_rec'])
            ax_pt_dataset_slect = self.fig.add_subplot(gs_var_select[2, 0])
            ax_pt_var_slect = self.fig.add_subplot(gs_var_select[3, 0])
            self.add_pt_plot()
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        ax_options_1 = self.fig.add_subplot(gs_options[0,0])
        ax_oper_plots = self.fig.add_subplot(gs_options[3, 0])
        ax_plot_rec_btn = self.fig.add_subplot(gs_options[1, 0])
        ax_clear_rec_btn = self.fig.add_subplot(gs_options[2, 0])
        ax_min_slider = self.fig.add_subplot(gs_lim_slider[0, 1])
        ax_max_slider = self.fig.add_subplot(gs_lim_slider[1, 1])
        ax_reset_button = self.fig.add_subplot(gs_lim_slider[:, 2])
        ax_time_slider = self.fig.add_subplot(gs_navigate[0, 5+1])
        ax_navigate = {nm: self.fig.add_subplot(gs_navigate[0, i]) for i, nm in enumerate(['Prev', 'Play', 'Pause', 'Next', 'Update'])}

        if background_img is not None:
            plot_background(self.ax_plt, background_img)

        self.add_plot()
        self.set_labels()

        if foreground_patches is not None:
            [self.ax_plt.add_collection(PatchCollection(p['patches'], **p['props'])) for p in foreground_patches]

        plt_mode_default = plt_mode_label[0]
        self.plt_mode_active = plt_mode_default

        self.plt_mode_sel_btn = self.add_radio_button(ax_options_1, 'Timeseries Plt Options', plt_mode_label, self.OnPltModeBtnClk)
        self.plt_mode_sel_btn.set_active(plt_mode_label.index(plt_mode_default))

        self.custom_plt_btn = self.add_radio_button(ax_oper_plots, 'Custom Plots', self.custom_plt_types,
                                                    self.OnCustomPltBtnClk)
        if self.custom_plt_active:
            self.custom_plt_btn.set_active(self.custom_plt_types.index(self.custom_plt_active))
        self.add_data_lim_sliders(ax_min_slider, ax_max_slider)
        self.add_time_slider(ax_time_slider)
        self.add_media_button(ax_navigate)
        self.reset_lim_btn = self.add_button(ax_reset_button, 'Reset', self.update_cbar_by_data_lim)
        self.clear_rec_btn = self.add_button(ax_clear_rec_btn, 'Clear Record', self.OnClearRecord)
        self.plot_rec_btn = self.add_button(ax_plot_rec_btn, 'Plot Record', self.OnPlotDataInRecord)

        if data_ext_pt is not None:
            self.pt_var_sel_btn = self.add_radio_button(ax_pt_var_slect, 'Pt_Vars', self.pt_vars, self.OnPtVarBtnClk)
            self.pt_var_sel_btn.set_active(self.pt_vars.index(self.pt_var))
            self.pt_dataset_sel_btn = self.add_radio_button(ax_pt_dataset_slect, 'Pt_Datasets', self.ds_names_pt,
                                                            self.OnPtDatasetSelect)
            self.pt_dataset_sel_btn.set_active(self.ds_names_pt.index(self.ds_active_pt))
        self.dist_var_sel_btn = self.add_radio_button(ax_map_var_slect, 'Dist_Vars', self.dist_vars, self.OnDistVarBtnClk)
        self.dist_var_sel_btn.set_active(self.dist_vars.index(self.dist_var)) # not available on older version of matplotlib

        self.dataset_sel_btn = self.add_radio_button(ax_dataset_slect, 'Dist_Datasets', self.ds_names,
                                                      self.OnDatasetSelect)
        self.dataset_sel_btn.set_active(self.ds_names.index(self.ds_active))
        gs.tight_layout(self.fig)
        self.timer = self.fig.canvas.new_timer(interval=50)
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
            self.data = self.data_ext[ds].get_map(self.dist_var, self.map_fetching_lst[ds], self.t_ax[self.ti])
            self.map[ds].set_array(self.data)
            self.cbar[ds] = self.fig.colorbar(self.map[ds], cax=cax, orientation='vertical')
            self.map[ds].set_visible(False)

    def add_pt_plot(self):
        # self.map_pt = self.ax_plt.plot(self.pt_coord[self.ds_active_pt][:, 0], self.pt_coord[self.ds_active_pt][:, 1], marker='o', ls='None', ms=8, mfc='none',
        #                 mec='r', mew=2, picker=5)
        for ds in self.ds_names_pt:
            self.map_pt[ds], = self.ax_plt.plot(self.pt_coord[ds][:, 0], self.pt_coord[ds][:, 1],
                                           marker='o', ls='None', ms=8, mfc='none',
                                           mec='r', mew=2, picker=5)
            self.map_pt[ds].set_visible(False)

    def update_cbar(self, event):
        self.map[self.ds_active].set_clim([self.slidermin.val, self.slidermax.val])
        self.fig.canvas.draw()

    def OnDatasetSelect(self, label):
        if self.dist_var_sel_btn.value_selected not in self.var_units[label]:
            print("Dist.Ds.Sel.: Dist. Dataset '{}' does not contain Variable '{}'".format(label, self.dist_var_sel_btn.value_selected))
            dist_var_auto_sel = self.dist_vars_pr_ds[label][0]
            self.dist_var_sel_btn.set_active(self.dist_vars.index(dist_var_auto_sel))
            print("Dist.Ds.Sel.: Auto-selected Variable '{}'".format(dist_var_auto_sel))
        self.map[self.ds_active].set_visible(False)
        self.map[label].set_visible(True)
        self.ds_active = label
        self.fig.canvas.draw()

    def OnDistVarBtnClk(self, label):
        if label not in self.var_units[self.ds_active]:
            print("Dist.Var.Sel.: Dist. Dataset '{}' does not contain Variable '{}'".format(self.ds_active, label))
            dist_ds_auto_sel = self.ds_names[[label in self.dist_vars_pr_ds[nm] for nm in self.ds_names].index(True)]
            self.dataset_sel_btn.set_active(self.ds_names.index(dist_ds_auto_sel))
            print("Dist.Var.Sel.:Auto-selected Dist. Dataset '{}'".format(dist_ds_auto_sel))

        self.ax_plt.set_title(self.ax_plt.get_title().replace(self.dist_var, label), fontsize=12)
        self.dist_var = label
        if self.dist_var in self.geo_data:
            self.data = self.data_ext[self.ds_active].get_geo_data(self.dist_var, self.map_fetching_lst[self.ds_active])
        else:
            self.data = self.data_ext[self.ds_active].get_map(self.dist_var, self.map_fetching_lst[self.ds_active],
                                                              self.t_ax[self.ti])
        self.map[self.ds_active].set_array(self.data)
        self.update_cbar_by_data_lim()

    def OnPtDatasetSelect(self, label):
        if self.pt_var_sel_btn.value_selected not in self.var_units_pt[label]:
            print("Pt.Ds.Sel.: Pt. Dataset '{}' does not contain Variable '{}'".format(label, self.pt_var_sel_btn.value_selected))
            pt_var_auto_sel = self.pt_vars_pr_ds[label][0]
            self.pt_var_sel_btn.set_active(self.pt_vars.index(pt_var_auto_sel))
            print("Pt.Ds.Sel.: Auto-selected Variable '{}'".format(pt_var_auto_sel))
        self.map_pt[self.ds_active_pt].set_visible(False)
        self.map_pt[label].set_visible(True)
        self.ds_active_pt = label
        self.fig.canvas.draw()

    def OnPtVarBtnClk(self, label):
        if label not in self.var_units_pt[self.ds_active_pt]:
            print("Pt.Var.Sel.: Pt. Dataset '{}' does not contain Variable '{}'".format(self.ds_active_pt, label))
            pt_ds_auto_sel = self.ds_names_pt[[label in self.pt_vars_pr_ds[nm] for nm in self.ds_names_pt].index(True)]
            print('Pt.Var.Sel.: pt_ds_auto_sel',pt_ds_auto_sel)
            print("Pt.Var.Sel.: Auto-selected Pt. Dataset '{}'".format(pt_ds_auto_sel))
            self.pt_dataset_sel_btn.set_active(self.ds_names_pt.index(pt_ds_auto_sel))
        self.pt_var = label


    def OnCustomPltBtnClk(self, label):
        self.custom_plt_active = label

    def OnPltModeBtnClk(self, label):
        self.plt_mode_active = label

    def update_cbar_by_slider_lim(self):
        self.map[self.ds_active].set_clim([self.scale_data_lim(v) for v in [self.slidermin.val, self.slidermax.val]])
        self.fig.canvas.draw()

    def update_cbar_by_data_lim(self, event=None):
        self.map[self.ds_active].set_clim([self.data.min(), self.data.max()])
        self.fig.canvas.draw()

    def add_radio_button(self, but_ax, title, labels, func):
        but_ax.set_title(title)
        radio_but = RadioButtons(but_ax, labels)
        radio_but.on_clicked(func)
        return radio_but

    def add_check_button(self, but_ax1, title, labels, vals, func):
        but_ax1.set_title(title)
        check = CheckButtons(but_ax1, labels, vals)
        check.on_clicked(func)
        return check

    def add_data_lim_sliders(self, axmin, axmax):
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
        self.time_slider = Slider(ax_slider, 'Time', self.t_ax[0], self.t_ax[-1], valinit=self.t_ax[0])
        self.time_slider.valtext.set_text(datetime.utcfromtimestamp(int(self.time_slider.val)).strftime('%Y-%m-%d %H:%M:%S'))
        self.time_slider.on_changed(self.update_time)

    def update_time(self,val):
        t_num = int(self.time_slider.val)
        self.time_slider.valtext.set_text(datetime.utcfromtimestamp(t_num).strftime('%Y-%m-%d %H:%M:%S'))

    def add_media_button(self, ax_navigate):
        axcolor = 'lightgoldenrodyellow'
        self.media_btn_nms = ['Prev', 'Play', 'Pause', 'Next', 'Update']
        self.media_btn_funcs = [self.OnPrev, self.OnPlay, self.OnPause, self.OnNext, self.OnUpdate]
        self.media_buttons = {nm: Button(ax_navigate[nm], nm, color=axcolor, hovercolor='0.975') for nm in self.media_btn_nms}
        self.media_btn_cids = [getattr(self.media_buttons[nm], 'on_clicked')(func) for nm, func in zip(self.media_btn_nms, self.media_btn_funcs)]

    def add_button(self, ax_button, label, func):
        axcolor = 'lightgoldenrodyellow'
        button = Button(ax_button, label, color=axcolor, hovercolor='0.975')
        button.on_clicked(func)
        return button

    def which_catch(self, x, y, ds_active):
        p = Point(x, y)
        indx = None
        for i in range(self.nb_catch[ds_active]):
            if self.polys[ds_active][i].contains(p):
                indx = i
        return indx

    def format_coord(self, x, y):
        indx = self.which_catch(x, y, self.ds_active)
        if (indx is None):
            info = 'x=%1.4f, y=%1.4f, val=None' % (x, y)
        else:
            z = self.data[indx]
            info = 'x=%1.4f, y=%1.4f, val=%1.4f, name=%s' % (x, y, z, self.catch_nms[self.ds_active][indx])
        return info

    def OnNext(self, *args):
        if not self._is_var_static(self.dist_var):
            self.ti += 1
            if (self.ti > self.max_ti): self.ti = 0
            # if(self.ti<0):self.ti=self.max_ti
            self.update_plot()

    def OnPrev(self, *args):
        if not self._is_var_static(self.dist_var):
            self.ti -= 1
            # if(self.ti>self.max_ti):self.ti=0
            if (self.ti < 0): self.ti = self.max_ti
            self.update_plot()

    def OnPlay(self, event):
        if not self._is_var_static(self.dist_var):
            # self.ani = animation.FuncAnimation(self.fig, self.OnNext,blit=False, interval=10,repeat=False)
            # self.fig.canvas.manager.window.after(100, self.OnNext)
            self.timer.add_callback(self.OnNext)  # , selfax)
            self.timer.start()

    def OnPause(self, event):
        self.timer.remove_callback(self.OnNext)

    def OnUpdate(self, event):
        if not self._is_var_static(self.dist_var):
            #t_indx = self.t_ax.index_of(int(self.time_slider.val))
            t_indx = np.searchsorted(self.t_ax, int(self.time_slider.val))
            if self.ti != t_indx:
                self.ti = t_indx
                self.update_plot()

    def _is_var_static(self,var_name):
        if var_name in self.geo_data:
            print("Variable '{}' has no time dimension!".format(var_name))
            return True
        else:
            return False

    def update_plot(self):
        self.data = self.data_ext[self.ds_active].get_map(self.dist_var, self.map_fetching_lst[self.ds_active], self.t_ax[self.ti])
        self.map[self.ds_active].set_array(self.data)
        self.ax_plt.title.set_text(
            '%s - %s' % (self.dist_var, self.data_ext[self.ds_active].time_num_2_str(self.t_ax[self.ti])))
        self.fig.canvas.draw()

    def set_labels(self):
        self.fig.canvas.set_window_title('Shyft-viz')
        self.ax_plt.set_title('%s - %s' % (self.dist_var, self.data_ext[self.ds_active].time_num_2_str(self.t_ax[self.ti])),
                              fontsize=12)

    def print_record_content(self):
        print('Data currently in record:\n{}'.format(list(self.record.keys())))

    def is_data_in_record(self, data_name):
        chk = data_name in self.record
        if chk:
            print("Data '{}' already in record!".format(data_name))
            self.print_record_content()
        return chk

    def OnClearRecord(self,event):
        self.record = {}
        print('Cleared the record!')

    def add_data_to_record(self, data):
        self.record.update(data)
        print("Data '{}' added to record!".format(list(data.keys())[0]))
        self.print_record_content()

    def remove_data_from_record(self, data_name):
        if data_name in self.record:
            self.record.pop(data_name)
            print("Data '{}' removed from record!".format(data_name))
        else:
            print("Data '{}' not in record!".format(data_name))
        self.print_record_content()

    def OnPlotDataInRecord(self, event):
        if len(self.record) == 0:
            print('No data stored in record!')
        else:
            #unique_names = list(self.record.keys())
            # ts_t, ts_v, units, props = zip(*[[self.record[nm]['t'], self.record[nm]['v'], self.record[nm]['unit'],
            #                                   self.record[nm]['props']] for nm in unique_names])
            tsplot = TsPlot(self.time_marker)
            #tsplot.unique_ts_names = unique_names
            #tsplot.init_plot(list(ts_t), ts_v, unique_names, units, props)
            tsplot.init_plot(self.record)
            #tsplot.unique_ts_names.extend(unique_names)
            self.tsplots.append(tsplot)

    def on_click(self, event):
        if event.inaxes is not self.ax_plt: return True
        if self.plt_mode_active == 'Custom_Plot' and self.ds_active not in self.custom_plt[self.custom_plt_active].keys():
            return True
        tb = self.fig.canvas.manager.toolbar
        if tb.mode != '':
            print('You clicked on something, but toolbar is in mode pan/zoom.')
            return True
        #if not self.plt_mode_active == 'Plot_Source' and tb.mode == '':
        if self.plt_mode_active in ['Plot_dist_dataset', 'Add_dist_ds_to_rec', 'Del_dist_ds_from_rec', 'Custom_Plot']:
            x = event.xdata
            y = event.ydata
            catchind = {self.ds_active: self.which_catch(x, y, self.ds_active)}
            if catchind[self.ds_active] is None: return True
            unique_names = [self.ts_name_separator.join([self.ds_active, self.dist_var, self.catch_nms[self.ds_active][catchind[self.ds_active]]])]
            if self.plt_mode_active == 'Add_dist_ds_to_rec':
                if not self.is_data_in_record(unique_names[0]):
                    if not self._is_var_static(self.dist_var):
                        valid_ds = [self.ds_active]
                        dist_vars = {self.ds_active: [self.dist_var]}
                        ts_t, ts_v = zip(
                            *[self.data_ext[ds_active].get_ts(dist_var,
                                                              self.ts_fetching_lst[ds_active][catchind[ds_active]])
                              for ds_active in valid_ds for dist_var in dist_vars[ds_active]])
                        units = [self.var_units[ds_active][dist_var] for ds_active in valid_ds for dist_var in dist_vars[ds_active]]
                        props = [{} for _ in unique_names]
                        found_prec = ['prec' in nm for nm in unique_names]
                        if any(found_prec):
                            props[found_prec.index(True)].update({'drawstyle': 'steps'})
                        data = {nm: {'t': t, 'v': v, 'unit': u, 'props': p} for nm, t, v, u, p in zip(
                            unique_names, list(ts_t), ts_v, units, props)}
                        self.add_data_to_record(data)
                return True

            if self.plt_mode_active == 'Del_dist_ds_from_rec':
                self.remove_data_from_record(unique_names[0])
                return True

            tsplots_active = [p for p in self.tsplots if p.fig is not None and p.plt_mode['Plot_over']]
            if len(tsplots_active)==0 or self.plt_mode_active == 'Custom_Plot':
                tsplot = TsPlot(self.time_marker)
                #tsplot.unique_ts_names = []
                if self.plt_mode_active == 'Custom_Plot':
                    catchind_ = {k: self.which_catch(x, y, k) for k in self.custom_plt[self.custom_plt_active] if k!= self.ds_active}
                    catchind.update({k: v for k, v in catchind_.items() if v is not None})
                    valid_ds = list(catchind.keys())
                    dist_vars = self.custom_plt[self.custom_plt_active]
                    unique_names = [self.ts_name_separator.join([ds_active, dist_var, self.catch_nms[ds_active][catchind[ds_active]]])
                                    for ds_active in valid_ds for dist_var in dist_vars [ds_active]]
                else:
                    if not self._is_var_static(self.dist_var):
                        valid_ds = [self.ds_active]
                        dist_vars = {self.ds_active: [self.dist_var]}
                    else:
                        return True

                props = [{} for _ in unique_names]
                found_prec = ['prec' in nm for nm in unique_names]
                if any(found_prec):
                    props[found_prec.index(True)].update({'drawstyle':'steps'})
                ts_t, ts_v = zip(
                    *[self.data_ext[ds_active].get_ts(dist_var, self.ts_fetching_lst[ds_active][catchind[ds_active]])
                                       for ds_active in valid_ds for dist_var in dist_vars[ds_active]])
                units = [self.var_units[ds_active][dist_var] for ds_active in valid_ds for dist_var in dist_vars[ds_active]]
                data_to_plot = {nm: {'t': t, 'v': v, 'unit': u, 'props': p} for nm, t, v, u, p in zip(
                            unique_names, list(ts_t), ts_v, units, props)}
                # tsplot.init_plot(list(ts_t), ts_v, unique_names,
                #                  [self.var_units[ds_active][dist_var] for ds_active in valid_ds for dist_var in dist_vars[ds_active]], props)
                tsplot.init_plot(data_to_plot)
                #tsplot.unique_ts_names.extend(unique_names)
                self.tsplots.append(tsplot)
            else:
                for tsplot in tsplots_active:
                    print(self.ds_active)

                    #if unique_names[0] in tsplot.unique_ts_names and not tsplot.plt_mode['Re-plot']: return True
                    if unique_names[0] in tsplot.data and not tsplot.plt_mode['Re-plot']: return True

                    if tsplot.plt_mode['Re-plot']:
                        unique_names[0] = unique_names[0]+'-1'

                    ts_t, ts_v = self.data_ext[self.ds_active].get_ts(self.dist_var, self.ts_fetching_lst[self.ds_active][catchind[self.ds_active]])
                    units = self.var_units[self.ds_active][self.dist_var]
                    props = {'drawstyle': 'steps'} if 'prec' in unique_names[0] else {}
                    data_to_plot = {unique_names[0]:{'t': ts_t, 'v': ts_v, 'unit': units, 'props': props}}

                    # tsplot.add_plot([ts_t], [ts_v], unique_names, [self.var_units[self.ds_active][self.dist_var]],
                    #                 [{'drawstyle':'steps'}] if 'prec' in unique_names[0] else [{}])
                    tsplot.add_plot(data_to_plot)
                    #tsplot.unique_ts_names.extend(unique_names)

    def on_pick(self, event):
        tb = self.fig.canvas.manager.toolbar
        if tb.mode != '':
            print('You clicked on something, but toolbar is in mode pan/zoom.')
            return True
        if self.plt_mode_active in ['Plot_pt_dataset', 'Add_pt_ds_to_rec', 'Del_pt_ds_from_rec']:

            # if event.artist!=self.overview: return True

            N = len(event.ind)
            if not N: return True

            # the click locations
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata

            distances = np.hypot(x - self.pt_coord[self.ds_active_pt][event.ind, 0], y - self.pt_coord[self.ds_active_pt][event.ind, 1])
            indmin = distances.argmin()
            catchind = event.ind[indmin]

            self.lastind = catchind
            unique_names = [self.ts_name_separator.join([self.ds_active_pt, self.pt_var, self.pt_nms[self.ds_active_pt][catchind]])]

            if self.plt_mode_active == 'Add_pt_ds_to_rec':
                if not self.is_data_in_record(unique_names[0]):
                    ts_t, ts_v = self.data_ext_pt[self.ds_active_pt].get_ts(self.pt_var, catchind)
                    unit = self.var_units_pt[self.ds_active_pt][self.pt_var]
                    props = {'drawstyle': 'steps'} if 'prec' in unique_names[0] else {}
                    data = {unique_names[0]: {'t': ts_t, 'v': ts_v, 'unit': unit, 'props': props}}
                    self.add_data_to_record(data)
                return True

            if self.plt_mode_active == 'Del_pt_ds_from_rec':
                self.remove_data_from_record(unique_names[0])
                return True

            tsplots_active = [p for p in self.tsplots if p.fig is not None and p.plt_mode['Plot_over']]
            if len(tsplots_active)==0: # or self.plt_mode['Custom_Plot']:
                tsplot = TsPlot(self.time_marker)
                #tsplot.unique_ts_names = []
                props = {'drawstyle': 'steps'} if 'prec' in unique_names[0] else {}
                ts_t, ts_v = self.data_ext_pt[self.ds_active_pt].get_ts(self.pt_var, catchind)
                units = self.var_units_pt[self.ds_active_pt][self.pt_var]
                data_to_plot = {unique_names[0]: {'t': ts_t, 'v': ts_v, 'unit': units, 'props': props}}
                # tsplot.init_plot([ts_t], [ts_v], unique_names, [self.var_units_pt[self.ds_active_pt][self.pt_var]],
                #                  props)# ,[{'ls':'None'}])
                tsplot.init_plot(data_to_plot)
                #tsplot.unique_ts_names.extend(unique_names)
                self.tsplots.append(tsplot)
            else:
                for tsplot in tsplots_active:
                    #if unique_names[0] in tsplot.unique_ts_names and not tsplot.plt_mode['Re-plot']: return True
                    if unique_names[0] in tsplot.data and not tsplot.plt_mode['Re-plot']: return True

                    ts_t, ts_v = self.data_ext_pt[self.ds_active_pt].get_ts(self.pt_var, catchind)
                    units = self.var_units_pt[self.ds_active_pt][self.pt_var]
                    props = {'drawstyle': 'steps'} if 'prec' in unique_names[0] else {}
                    data_to_plot = {unique_names[0]: {'t': ts_t, 'v': ts_v, 'unit': units, 'props': props}}
                    # tsplot.add_plot([ts_t], [ts_v], unique_names, [self.var_units_pt[self.ds_active_pt][self.pt_var]],
                    #                 [{'drawstyle':'steps'}] if 'prec' in unique_names[0] else [{}])# ,[{'ls':'None'}])
                    tsplot.add_plot(data_to_plot)
                    #tsplot.unique_ts_names.extend(unique_names)


class TsPlot(object):
    def __init__(self, time_marker):  # ,timesteps,values):
        self.time_marker = time_marker
        self.fig = None
        self.ax = None
        self.plotted_unit = None
        self.axes = None
        self.reset_plot()
        self.colors = cycle(['b', 'g', 'r', 'k', 'm', 'y', 'c'])
        self.line_styles = [cycle(['-', '--', '-.', ':']) for _ in range(4)]
        self.markers = [cycle([None, 'o', '*', 's', 'v', 'x', 'p', '+']) for _ in range(4)]
        self.temp = [0.85, 0.75, 0.6]
        self.i = 1
        self.plt_mode = {'Plot_over': False, 'Re-plot': False}
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[0.1, 0.9])  # , height_ratios=[2,1])
        self.gs_plot_opt = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[0, 0])
        self.data = {}

    def add_click_button(self, ax_button, label, func):
        axcolor = 'lightgoldenrodyellow'
        button = Button(ax_button, label, color=axcolor, hovercolor='0.975')
        button.on_clicked(func)
        return button

    def add_check_button(self, but_ax1, title, labels, vals, func):
        axcolor = 'lightgoldenrodyellow'
        but_ax1.set_title(title)
        check = CheckButtons(but_ax1, labels, vals)
        check.on_clicked(func)
        return check

    def OnPltModeBtnClk(self, label):
        self.plt_mode[label] = not self.plt_mode[label]
        #print(label,self.plt_mode[label])

    def reset_plot(self):
        self.plotted_unit = []
        self.axes = []
        self.lines = []
        #self.fig.autofmt_xdate()
        #self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)

    #def init_plot(self, t, v, labels, units, prop):
    def init_plot(self, data_dict):
        self.fig = plt.figure(figsize=(15, 6))#, (15, 6))  # , facecolor='white')
        self.ax = self.fig.add_subplot(self.gs[0, 1])
        ax_options_1 = self.fig.add_subplot(self.gs_plot_opt[0, 0]) # self.gs[0, 0])
        self.option_btn = self.add_check_button(ax_options_1, 'Options', list(self.plt_mode.keys()),
                                                list(self.plt_mode.values()), self.OnPltModeBtnClk)
        ax_options_2 = self.fig.add_subplot(self.gs_plot_opt[1, 0])
        self.comp_btn = self.add_click_button(ax_options_2, 'Compare', self.OnCompare)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.fig.autofmt_xdate()  # for one subplot
        self.ax.xaxis_date()
        self.add_plot(data_dict)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)

    def OnCompare(self, event):
        fetch_data_from_plot = False
        if fetch_data_from_plot:
            lines = self.ax.get_lines()
            labels = [line.get_label() for line in lines]
            is_obs = ['Obs' in label for label in labels]
            if any(is_obs):
                obs_idx = is_obs.index(True)
                sim_idx = is_obs.index(False)
            else:
                obs_idx = 0
                sim_idx = 1
            # line.get_xdata() - returns x in datetime
            # line.get_data() - returns x & y where x is in datetime
            # line.get_xydata() - returns [[x0,y0], [x1,y1], ...] where x is num
            obs_nm, sim_nm = labels[obs_idx], labels[sim_idx]
            x_obs, y_obs = lines[obs_idx].get_data()
            x_sim, y_sim = lines[sim_idx].get_data()

            x_obs = mdate.num2epoch(mdate.date2num(x_obs))
            x_sim = mdate.num2epoch(mdate.date2num(x_sim))
        else:
            err_msg = self._data_not_ok_for_comparison(self.data)
            if err_msg:
                print(err_msg)
                return True
            labels = list(self.data.keys())
            is_obs = ['Obs' in label for label in labels]
            if any(is_obs):
                obs_idx = is_obs.index(True)
                sim_idx = is_obs.index(False)
            else:
                obs_idx = 0
                sim_idx = 1
                print("Could not identify which series is the reference using substring 'obs'."
                      "{} will be assumed as the reference series.".format(labels[obs_idx]))
            obs_nm, sim_nm = labels[obs_idx], labels[sim_idx]
            x_obs, y_obs = self.data[obs_nm]['t'], self.data[obs_nm]['v']
            x_sim, y_sim = self.data[sim_nm]['t'], self.data[sim_nm]['v']

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        print(y_min, y_max)
        x_min, x_max = mdate.num2epoch([x_min, x_max])
        ScatterPlot(x_sim, y_sim, sim_nm, x_obs, y_obs, obs_nm, x_min, x_max, y_min, y_max, self.data[obs_nm]['unit'])

    def _data_not_ok_for_comparison(self, data):
        err = []
        if len(data) != 2:
            err.append('The number of timeseries to compare should be 2.')
        if len(set([v['unit'] for v in data.values()])) != 1:
            err.append('The timeseries to compare should have the same units.')
        return '; '.join(err)

    def plot(self,ax, t, v, kwargs):
        return ax.plot(t, v, **kwargs)[0]

    def add_plot(self, data_dict):
        labels = list(data_dict.keys())
        t_, v, units, prop = zip(*[[data_dict[nm]['t'], data_dict[nm]['v'],data_dict[nm]['unit'], data_dict[nm]['props']] for nm in labels])
        t = list(t_)
        for k in range(len(v)):
            if (not np.all(np.isnan(v[k]))):
                if not isinstance(t[k][0], datetime):
                    t[k] = utctimestamp_2_datetime(t[k])
                if (len(self.plotted_unit) == 0):
                    self.axes.append(self.ax)
                    color = next(self.colors)
                    #self.lines.append(self.axes[0].plot(t, v[k], ls=next(self.line_styles[0]), color=self.colors[0], label=labels[k], marker=marker, ms=8, markevery=None)[0])
                    l_prop = dict(ls='-', color=color, label=labels[k], marker=next(self.markers[0]), ms=4, mec=color, markevery=7)
                    l_prop.update(prop[k])
                    self.lines.append(self.plot(self.axes[0],t[k], v[k], l_prop))
                    #self.lines.append(self.axes[0].plot(t, v[k], ls=ls, color=self.colors[0], label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None, drawstyle='steps')[0])

                    self.plotted_unit.append(units[k])
                    self.axes[0].set_ylabel(units[k],color=color)
                    self.axes[0].tick_params(axis='y', colors=color)
                    if self.time_marker is not None:
                        self.axes[0].axvline(x=self.time_marker, lw=2, color='k', ls='-')
                else:
                    if (units[k] in self.plotted_unit):
                        idx=self.plotted_unit.index(units[k])
                        #color = self.colors[idx]
                        color = next(self.colors)
                        #color = next(self.colors_)
                        #marker = next(self.markers[idx])
                        marker = None
                        #self.lines.append(self.axes[idx].plot(t, v[k], ls=next(self.line_styles[idx]), color=color, label=labels[k], marker=marker, ms=8, markevery=None)[0])
                        l_prop = dict(ls='-', color=color, label=labels[k], marker=marker, ms=4, mec=color, markevery=7)
                        l_prop.update(prop[k])
                        self.lines.append(self.plot(self.axes[idx], t[k], v[k], l_prop))
                        #self.lines.append(self.axes[idx].plot(t, v[k], ls=ls, color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)[0])

                        self.axes[idx].tick_params(axis='y', colors=color)
                    else:
                        self.plotted_unit.append(units[k])
                        idx = self.plotted_unit.index(units[k])
                        #color = self.colors[idx]
                        color = next(self.colors)
                        self.axes.append(self.axes[0].twinx())
                        #self.lines.append(self.axes[-1].plot(t, v[k], ls=next(self.line_styles[idx]), color=color, label=labels[k], marker=marker, ms=8, markevery=None)[0])
                        l_prop = dict(ls='-', color=color, label=labels[k], marker=next(self.markers[idx]), ms=4, mec=color, markevery=7)
                        l_prop.update(prop[k])
                        self.lines.append(self.plot(self.axes[-1], t[k], v[k], l_prop))
                        #self.lines.append(self.axes[-1].plot(t, v[k], ls=ls, color=color, label=labels[k], marker=next(self.markers[0]), ms=8, markevery=None)[0])

                        #print(self.lines)

                        self.axes[-1].set_ylabel(units[k])
                        self.axes[-1].format_coord = self.make_format(self.axes[-1], self.axes[0])
                        self.axes[-1].tick_params(axis='y', colors=color)
                        self.axes[-1].set_ylabel(units[k], color=color)

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
                self.data.update({labels[k]: data_dict[labels[k]]})  # to take only data which has been sucessfully plotted

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
        #self.axes[ax_i].legend(bbox_to_anchor = (0, 1), loc = 'upper left', ncol = 1)
        #self.axes[ax_i].legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
        #                        ncol = 2, mode = "expand", borderaxespad = 0.)

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


class ScatterPlot(object):
    def __init__(self, x_sim, y_sim, label_sim, x_obs, y_obs, label_obs, x_min, x_max, y_min, y_max, unit):
        in_zoom_idx = np.nonzero(((x_sim>=x_min) & (x_sim<=x_max) & (np.isfinite(y_sim))))[0]
        x_sim_sel = x_sim[in_zoom_idx]
        y_sim_sel = y_sim[in_zoom_idx]

        #xsorted = np.argsort(obs_x)
        #ypos = np.searchsorted(obs_x[xsorted], sim_x)
        #indices = xsorted[ypos]

        pos_obs = np.searchsorted(x_obs, x_sim_sel)  # assumes that every element of x_sim_sel is in x_obs

        y_obs_sel = y_obs[pos_obs]

        # Consider using rec arr
        # data = np.rec.fromarrays((x_sim_sel, y_sim_sel, y_obs_sel), names='time, sim, obs')

        y_lim_mask = ((y_obs_sel>=y_min) & (y_obs_sel<=y_max))
        y_sim_sel = y_sim_sel[y_lim_mask]
        y_obs_sel = y_obs_sel[y_lim_mask]
        x_sim_sel = x_sim_sel[y_lim_mask]

        if any([np.isnan(y_sim_sel).any(), np.isnan(y_obs_sel).any()]):
            valid_v_mask = ((np.isfinite(y_obs_sel)) & (np.isfinite(y_sim_sel)))
            y_sim_sel = y_sim_sel[valid_v_mask]
            y_obs_sel = y_obs_sel[valid_v_mask]
            x_sim_sel = x_sim_sel[valid_v_mask]

        gs = gridspec.GridSpec(1, 2, width_ratios=[0.7, 0.3])

        self.fig = plt.figure() # figsize=(15, 6))  # , (15, 6))  # , facecolor='white')
        self.ax_plot = self.fig.add_subplot(gs[0,0]) #211)
        self.ax_plot.scatter(y_obs_sel, y_sim_sel, picker=True)

        lims = [
            np.min([self.ax_plot.get_xlim(), self.ax_plot.get_ylim()]),  # min of both axes
            np.max([self.ax_plot.get_xlim(), self.ax_plot.get_ylim()]),  # max of both axes
        ]
        self.ax_plot.plot(lims, lims, 'k-', alpha=0.75, zorder=0) # identity line
        self.ax_plot.set_aspect('equal')
        self.ax_plot.set_xlim(lims)
        self.ax_plot.set_ylim(lims)
        self.ax_plot.set_xlabel(label_obs)
        self.ax_plot.set_ylabel(label_sim)

        af = AnnoteFinder(y_obs_sel, y_sim_sel,
                          [datetime.utcfromtimestamp(d).strftime('%Y-%m-%d %H:%M') for d in x_sim_sel], self.ax_plot)
        self.fig.canvas.mpl_connect('pick_event', af)

        self.ax_table = self.fig.add_subplot(gs[0,1]) # 212)
        #self.ax_table.axis('tight')
        self.ax_table.axis('off')

        fmt = lambda f: '{:.3f}'.format(f)

        stats = [['Nash', '-', fmt(self.calc_nash_np(y_sim_sel, y_obs_sel))],
                 ['MAPE', '%', fmt(100.*self.calc_vol_err_avg_np(y_sim_sel, y_obs_sel))],
                 ['Mean_sim', unit, fmt(y_sim_sel.mean())],
                 ['Mean_obs', unit, fmt(y_obs_sel.mean())],
                 ['Min_sim', unit, fmt(y_sim_sel.min())],
                 ['Min_obs', unit, fmt(y_obs_sel.min())],
                 ['Max_sim', unit, fmt(y_sim_sel.max())],
                 ['Max_obs', unit, fmt(y_obs_sel.max())],
                 ['Std_sim', unit, fmt(np.std(y_sim_sel))],
                 ['Std_obs', unit, fmt(np.std(y_obs_sel))]]

        self.table_stat = self.ax_table.table(cellText=stats, colLabels=['Stat name', 'unit', 'value'], loc='center')
        self.table_stat.auto_set_font_size(False)
        self.table_stat.set_fontsize(10)
        gs.tight_layout(self.fig)

    @staticmethod
    def calc_vol_err_avg_np(sim, obs, relative=True):
        if relative:
            return (np.abs(obs - sim) / obs).mean()
        else:
            return np.abs(obs - sim).mean()

    @staticmethod
    def calc_nash_np(sim, obs):
        SSres = np.square(obs - sim).sum()
        SStot = np.square(obs - obs.mean()).sum()
        return 1 - SSres / SStot


class StaticViewer(object):
    """ A class to view static variables for a Region """
    def __init__(self, polygons, polygon_data, foreground_patches=None, points=None):

        self.polygons = polygons
        self.polygon_data = polygon_data
        self.points = points
        self.foreground_patches = foreground_patches

        self.left = None
        self.main_plot = None
        self.colorbar = None

        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])

        self.main_plot = plt.subplot(gs[1])
        self.main_plot.axis('equal')

        # Selector for polygon data
        h = 0.5
        cell_data_box = plt.axes([0.05, 1-h, 0.15, h], aspect='equal')
        cell_data_box.set_title("Cell data")
        cell_data_box.set_facecolor('lightgoldenrodyellow')
        self.data_select_button = RadioButtons(cell_data_box, [key for key in polygon_data.keys()])
        self.data_select_button.on_clicked(self.selected_dataset_changed)

        if len(points) > 0:
            # Selector for points
            point_data_box = plt.axes([0.05,1-h-h, 0.15, h], aspect='equal')
            point_data_box.set_title("Points")
            point_data_box.set_facecolor('lightgoldenrodyellow')
            point_labels = [key for key in points.keys()]
            self.point_select_status = {key: False for key in points.keys()}
            self.point_select_button = CheckButtons(point_data_box, point_labels, (False,)*len(point_labels))
            self.point_select_button.on_clicked(self.selected_points_changed)

        # Add elevation dataset (assume this is always present for now)
        self.selected_dataset = None
        self.selected_dataset_changed('Elevation')

        # Perform autoscale once
        self.main_plot.autoscale(True)
        self.main_plot.autoscale(False)

        self.fig.canvas.draw()

        plt.show()

    def update_plot(self):
        # Save current view
        xlim = self.main_plot.axes.get_xlim()
        ylim = self.main_plot.axes.get_ylim()

        # Clear old plot and add new
        self.main_plot.cla()
        pc = PatchCollection(self.polygons)
        pc.set_array(np.array(self.polygon_data[self.selected_dataset]))
        self.main_plot.add_collection(pc)

        # Restore view
        self.main_plot.axes.set_xlim(xlim)
        self.main_plot.axes.set_ylim(ylim)

        # (Re)create colorbar
        if self.colorbar:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(pc, ax=self.main_plot)

        # Add foreground patches
        if self.foreground_patches is not None:
            [self.main_plot.add_collection(PatchCollection(p['patches'], **p['props']))
             for p in self.foreground_patches]

        # Add points
        for key in self.points:
            if self.point_select_status[key]:
                vals = self.points[key][0]
                opts = self.points[key][1]
                self.add_points(vals, **opts)

        self.fig.canvas.draw()

    def selected_dataset_changed(self, label):
        self.selected_dataset = label
        self.update_plot()

    def selected_points_changed(self, label):
        self.point_select_status[label] = not self.point_select_status[label]
        self.update_plot()

    def add_points(self, points, **kwargs):
        for idx in range(len(points)):
            x = points[idx][0]
            y = points[idx][1]
            self.main_plot.plot(x, y, **kwargs)
            if len(points[idx]) > 2:
                self.main_plot.annotate(points[idx][2], xy=(x,y), textcoords='data')
