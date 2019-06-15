import numpy as np
import h5py
import timestream_task
# from tlpipe.container.timestream import Timestream
from tlpipe.container.raw_timestream import RawTimestream
from scipy.interpolate import interp1d, Rbf


class Interpolation(timestream_task.TimestreamTask):

    params_init = {
        'interp': 'linear',  # 'none', linear', 'nearest' or 'rbf'
    }

    prefix = 'ip_'

    def process(self, rt):

        assert isinstance(
            rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        interp = self.params['interp']

        if interp != 'none':
            for fi in xrange(rt.local_vis.shape[1]):
                for bi in xrange(rt.local_vis.shape[2]):
                    # interpolate for local_vis
                    true_inds = np.where(rt.local_vis_mask[:, fi, bi])[0] # masked inds
                    if len(true_inds) > 0:
                        false_inds = np.where(~rt.local_vis_mask[:, fi, bi])[0] # un-masked inds
                        if len(false_inds) > 0.1 * rt.local_vis.shape[0]:
        # nearest interpolate for local_vis
                            if interp in ('linear', 'nearest'):
                                itp_real = interp1d(false_inds, rt.local_vis[false_inds, fi, bi].real, kind=interp, fill_value='extrapolate', assume_sorted=True)
                                itp_imag = interp1d(false_inds, rt.local_vis[false_inds, fi, bi].imag, kind=interp, fill_value='extrapolate', assume_sorted=True)
                            elif interp == 'rbf':
                                itp_real = Rbf(false_inds, rt.local_vis[false_inds, fi, bi].real, smooth=10)
                                itp_imag = Rbf(false_inds, rt.local_vis[false_inds, fi, bi].imag, smooth=10)
                            else:
                                raise ValueError('Unknown interpolation method: %s' % interp)
                            rt.local_vis[true_inds, fi, bi] = itp_real(true_inds) + 1.0J * itp_imag(true_inds) # the interpolated vis
                        else:
                            rt.local_vis[:, fi, bi] = 0 # TODO: may need to take special care

        return super(Interpolation, self).process(rt)
