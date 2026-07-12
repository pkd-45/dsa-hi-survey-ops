# Two-minute interview brief

The DSA job is not only an imaging problem. The advert links operational choices - zoom-band placement, Doppler setting, RFI handling, continuum subtraction, compression, and source selection - to a real-time radio-camera survey.

I therefore built a small, transparent sandbox around that decision chain.

First, a catalogue-aware optimiser propagates each galaxy's redshift uncertainty, expected W50, barycentric motion, guard channels, target priority, and RFI overlap into a frequency interval. It then places a fixed-width nearby-H I product to maximise fully contained, weighted targets.

Second, a synthetic rotating H I cube is passed through continuum, ripple, RFI, noise, subtraction, channel averaging, quantisation, and two source masks. Instead of validating with a generic image loss, I measure integrated H I flux, centroid, W50, position angle, completeness, and purity.

The point is not that this is production DSA software. It is evidence that I understand how to turn an instrument configuration into measurable science-preservation tests, and how to structure the software so that better RFI, continuum, compression, and source-finding methods can be compared fairly.


## v0.2.0 real-cube answer

**Question:** Does the synthetic QA survive realistic cube backgrounds?

**Answer:** The repository now includes a FITS-cube injection adapter. It selects
finite background patches from an existing cube, calibrates each forward-modelled
disk to the requested matched-filter S/N, and measures completeness plus
paired-background-corrected flux, centroid, and W50 recovery. It compares strict
and grown masks and repeats the measurement after channel averaging and float16
quantisation. The release tests the full code path through a synthetic FITS
round-trip and a structured background; I do not present a named survey result
until the public cube, checksum, command, and output are archived. That boundary
is intentional validation discipline, not a limitation hidden by a headline
number.
