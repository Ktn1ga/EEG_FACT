import scipy.signal as signal
import numpy as np
import copy
import torch

class filterBank:
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30  # stopband attenuation
        aPass = 3  # passband attenuation
        nFreq = fs / 2  # Nyquist frequency

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass = bandFiltCutF[1] / nFreq
            fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass = bandFiltCutF[0] / nFreq
            fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass = (np.array(bandFiltCutF) / nFreq).tolist()
            fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):
        data = copy.deepcopy(data1)

        d = data
        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:, :, i] = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                                               self.axis, self.filtType)

        # remove any redundant 3rd dimension
        if len(self.filtBank) <= 1:
            out = np.squeeze(out, axis=2)

        # data['data'] = torch.from_numpy(out).float()
        return out