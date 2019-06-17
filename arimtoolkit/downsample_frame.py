import numpy as np
import arim


def downsample_frame(frame, k):
    """
    Return a Frame obtained by grouping the probe elements by 'k'
    
    New scanlines are obtained by averaging original scanlines.
    
    Emulates a Frame with a larger pitch. 1D array only.
    
    If 'k' is not a divisor of the number of elements, do not use
    remaining elements.
    
    """
    probe = frame.probe
    new_numelements = probe.numelements // k
    numelements_to_use = k * new_numelements

    new_locations = np.zeros((new_numelements, 3))

    for i in range(new_numelements):
        new_locations[i] = probe.locations[i * k : (i + 1) * k].mean(axis=0)

    new_probe = arim.Probe(
        new_locations, probe.frequency, bandwidth=probe.bandwidth, pcs=probe.pcs.copy()
    )

    # Prepare downsampled FMC:
    new_scanlines = []
    new_tx = []
    new_rx = []
    elements_idx = np.arange(numelements_to_use)
    for i in range(new_numelements):
        for j in range(new_numelements):
            retained_scanlines_idx = np.logical_and(
                np.isin(frame.tx, elements_idx[i * k : (i + 1) * k]),
                np.isin(frame.rx, elements_idx[j * k : (j + 1) * k]),
            )
            if np.sum(retained_scanlines_idx) != k * k:
                # Missing signals, ignore
                pass
            else:
                new_tx.append(i)
                new_rx.append(j)
                new_scanlines.append(
                    frame.scanlines[retained_scanlines_idx].mean(axis=0)
                )
    new_tx = np.array(new_tx)
    new_rx = np.array(new_rx)
    new_scanlines = np.array(new_scanlines)

    new_frame = arim.Frame(
        new_scanlines, frame.time, new_tx, new_rx, new_probe, frame.examination_object
    )
    return new_frame
