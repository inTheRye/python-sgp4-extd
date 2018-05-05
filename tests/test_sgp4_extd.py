"""Test suite for SGP4 extd."""

from unittest import TestCase, main

import os

from sgp4.earth_gravity import wgs72
from sgp4_extd.sgp4_extd import SpaceObjects
from sgp4 import io
import numpy as np

thisdir = os.path.dirname(__file__)
error = 2e-7


class TestSpaceObject(TestCase):

    def test_tle_verify(self):
        # Check whether a test run produces the output in tcppver.out

        whichconst = 'wgs72'
        error_list = []
        actual = generate_test_output(whichconst, error_list)

        # Iterate across "tcppver.out", making sure that we ourselves
        # produce a line that looks very much like the corresponding
        # line in that file.

        tcppath = os.path.join(thisdir, 'tcppver_short_LEO.out')
        with open(tcppath) as tcpfile:
            for i, expected_line in enumerate(tcpfile, start=1):

                try:
                    actual_line = next(actual)
                except StopIteration:
                    raise ValueError(
                        'WARNING: our output ended early, on line %d' % (i,))

                # Compare the lines. We allow a small error due
                # to rounding differences

                if 'xx' in actual_line:
                    similar = (actual_line == expected_line)
                else:
                    afields = actual_line
                    efields = expected_line.split()
                    actual7 = [float(a) for a in afields[:7]]
                    expected7 = [float(e) for e in efields[:7]]
                    similar = (
                        len(actual7) == len(expected7)
                        and
                        all(
                            -error < (a - e) < error
                            for a, e in zip(actual7, expected7)
                             )
                        )

                if not similar:
                    raise ValueError(
                        'Line %d of output does not match:\n'
                        '\n'
                        'Expect: %s'
                        '\n'
                        'Actual: %s'
                        % (i, expected7, actual7))


def generate_test_output(whichconst, error_list):
    """Generate lines like those in the test file tcppver.out.

    This iterates through the satellites in "SGP4-VER.TLE", which are
    each supplemented with a time start/stop/step over which we are
    supposed to print results.

    """
    whichconst = wgs72
    tlepath = os.path.join(thisdir, 'SGP4-VER-LEO.TLE')
    with open(tlepath) as tlefile:
        tlelines = iter(tlefile.readlines())

    for line1 in tlelines:

        if not line1.startswith('1'):
            continue

        line2 = next(tlelines)
        satrec = io.twoline2rv(line1, line2, whichconst)
        tle_list = [line1, line2]

        yield '%ld xx\n' % (satrec.satnum,)

        for line in generate_satellite_output(tle_list, line2, whichconst, satrec.satnum):
            yield line


def generate_satellite_output(tle_list, line2, whichconst, satnum):
    """Print a data line for each time in line2's start/stop/step field.
    """
    tstart, tend, tstep = (float(field) for field in line2[69:].split())
    print("Test for catalog No. {}".format(satnum))

    so = SpaceObjects(tle_list, whichconst, tstart, tend, tstep, julian_date=True)
    df = so.propagate()
    a = df.loc[satnum].T
    for t, x, y, z, vx, vy, vz in np.hstack((np.array([a.index.values]).T, a.values)):
        yield t, f"{x:.8f}", f"{y:.8f}", f"{z:.8f}", f"{vx:.9f}", f"{vy:.9f}", f"{vz:.9f}"


if __name__ == '__main__':
    main()
