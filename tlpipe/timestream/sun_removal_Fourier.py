import numpy as np
import h5py
import ephem
import aipy as a
import timestream_task
# from tlpipe.container.timestream import Timestream
from tlpipe.container.raw_timestream import RawTimestream
from scipy.interpolate import interp1d, Rbf
from numpy import linalg as LA
import scipy.signal as signal
import numpy.ma as ma
import logging


# Set the module logger.
logger = logging.getLogger(__name__)


class SunRemoval(timestream_task.TimestreamTask):

    params_init = {
        'filter_order': 2.0,  # 'none', linear', 'nearest' or 'rbf'
        'cutoff_freq_1' : 0.05,
        'cutoff_freq_2' : 0.0001,
        'span' : 600, # minutes
    }

    prefix = 'sr_'

    def process(self, rt):

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        filter_order = self.params['filter_order']
        cutoff_freq_1 = self.params['cutoff_freq_1']
        cutoff_freq_2 = self.params['cutoff_freq_2']
        span = self.params['span']

        logger.info("Task %s: Fourier filtering" % self.__class__.__name__)
        print('filter_order: {}'.format(filter_order))
        print('cutoff_freq_1: {}'.format(cutoff_freq_1))
        print('cutoff_freq_2: {}'.format(cutoff_freq_2))

        nt = rt.local_vis.shape[0] # number of time points of this process

        if nt > 0:

            srclist, cutoff, catalogs = a.scripting.parse_srcs('Sun', 'misc')
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            s = cat.values()[0] # the Sun

            # get transit time of calibrator
            # array
            aa = rt.array
            local_juldate = rt['jul_date'].local_data
            aa.set_jultime(local_juldate[0]) # the first obs time point of this process

            mask_inds = []

            # previous transit
            prev_transit = aa.previous_transit(s)
            prev_transit_start = a.phs.ephem2juldate(prev_transit - 0.5 * span * ephem.minute) # Julian date
            prev_transit_end = a.phs.ephem2juldate(prev_transit + 0.5 * span * ephem.minute) # Julian date
            prev_transit_start_ind = np.searchsorted(local_juldate, prev_transit_start, side='left')
            prev_transit_end_ind = np.searchsorted(local_juldate, prev_transit_end, side='right')
            if prev_transit_end_ind > 0:
                mask_inds.append((prev_transit_start_ind, prev_transit_end_ind))

            # next transit
            next_transit = aa.next_transit(s)
            next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
            next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
            next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
            next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
            if next_transit_start_ind < nt:
                mask_inds.append((next_transit_start_ind, next_transit_end_ind))

            # then all next transit if data is long enough
            while (next_transit_end_ind < nt):
                aa.set_jultime(next_transit_end)
                next_transit = aa.next_transit(s)
                next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
                next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
                next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
                next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
                if next_transit_start_ind < nt:
                    mask_inds.append((next_transit_start_ind, next_transit_end_ind))

        rt.create_dataset('sun_removal',data=np.array(mask_inds))
        rt['sun_removal'].attrs['dimname'] = 'begin, end sun_removal'
        rt['sun_removal'].attrs['unit'] = 'index in file'

        logger.info("[(Sunrise,Sunset)]: %s" %str(mask_inds))

        auto_inds = np.where(rt.bl[:, 0]==rt.bl[:, 1])[0].tolist() # inds for auto-correlations

        for fi in xrange(rt.local_vis.shape[1]):
            for bl in xrange(rt.local_vis.shape[2]):
                if bl in auto_inds:
                    continue
                for si, ei in mask_inds:
                    vis_1bl_real = np.copy(np.real(rt.local_vis[:, fi, bl]))
                    vis_1bl_imag = np.copy(np.imag(rt.local_vis[:, fi, bl]))
                    for ti in xrange(len(vis_1bl_real)):
                        if rt.local_vis_mask[ti, fi, bl] == True:
                            vis_1bl_real[ti] = np.nan
                            vis_1bl_imag[ti] = np.nan
                    nans_r, r = nan_helper(vis_1bl_real)
                    nans_i, i = nan_helper(vis_1bl_imag)

                    if (np.sum(nans_r)==rt.local_vis.shape[0]) or (np.sum(nans_i)==rt.local_vis.shape[0]):
                        continue

                    vis_1bl_real[nans_r]= np.interp(r(nans_r), r(~nans_r), vis_1bl_real[~nans_r])
                    vis_1bl_imag[nans_i]= np.interp(i(nans_i), i(~nans_i), vis_1bl_imag[~nans_i])

                    B, A = signal.butter(filter_order, cutoff_freq_1, output='ba')
                    vr_f = signal.filtfilt(B,A, vis_1bl_real)
                    vr_i = signal.filtfilt(B,A, vis_1bl_imag)
                    residual_r = vis_1bl_real - vr_f
                    residual_i = vis_1bl_imag - vr_i
                    B, A = signal.butter(filter_order, cutoff_freq_2, output='ba')
                    vr_f2 = signal.filtfilt(B,A, vis_1bl_real)
                    vr_i2 = signal.filtfilt(B,A, vis_1bl_imag)
                    residual_r = residual_r + vr_f2
                    residual_i = residual_i + vr_i2

                    for ti in xrange(si,ei):
                        if rt.local_vis_mask[ti, fi, bl] == False:
                            rt.local_vis[ti, fi, bl] = residual_r[ti] + 1j*residual_i[ti]
                    
                    del vis_1bl_real, vis_1bl_imag, nans_r, r, nans_i, i, B, A, vr_f, vr_i, residual_r, residual_i, vr_f2, vr_i2

        logger.info("Task %s: Done Fourier filtering." % self.__class__.__name__)

        return super(SunRemoval, self).process(rt)


