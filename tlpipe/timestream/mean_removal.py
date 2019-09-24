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


class MeanRemoval(timestream_task.TimestreamTask):

    params_init = {
        'span' : 600, # minutes
        'remove_mean' : False
    }

    prefix = 'mr_'

    def process(self, rt):

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        remove_mean = self.params['remove_mean']
        span = self.params['span']

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

        rt.create_dataset('mean_removal',data=np.array(mask_inds))
        rt['mean_removal'].attrs['span (seconds)'] = float(60*span)
        rt['mean_removal'].attrs['dimname'] = 'begin, end sun_removal'
        rt['mean_removal'].attrs['unit'] = 'index in file'

        logger.info("[(Sunrise,Sunset)]: %s" %str(mask_inds))

        if remove_mean:
            logger.info("Task %s: Removing the mean..." % self.__class__.__name__)
            for si, ei in mask_inds:
                #print('si: {}, ei: {}'.format(si,ei))
                for fi in xrange(rt.local_vis.shape[1]):
                    vis = ma.array(rt.local_vis[:,fi,:], mask=rt.local_vis_mask[:,fi,:])
                    me1 = ma.mean(ma.array(rt.local_vis[:si,fi,:], mask=rt.local_vis_mask[:si,fi,:]),axis=0)
                    me2 = ma.mean(ma.array(rt.local_vis[ei:,fi,:], mask=rt.local_vis_mask[ei:,fi,:]),axis=0)
                    med = ma.mean([me1, me2],axis=0)
                    del me1, me2
                    vis = vis - med
                    for ti in xrange(vis.shape[0]):
                        if np.all(rt.local_vis_mask[ti, fi, :]):
                            continue
                        else:
                            rt.local_vis[ti,fi,:] = vis[ti,:]
                    del med, vis

        return super(MeanRemoval, self).process(rt)


