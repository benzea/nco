#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pipelined_cordic
from myhdl import *

import numpy as np
from matplotlib import pyplot


INTMAX = 2**17
INTMIN = -INTMAX
PHASE_MAX = 2**18

MIN_SAMPLES = 2**11
MAX_SAMPLES = 2**13
SKIP = 1 #max(1, SAMPLES / 8192)
FREQ_EXTRA = 2**16
FREQ = 0.311
PHASE_STEP = int(FREQ*PHASE_MAX*FREQ_EXTRA) / float(FREQ_EXTRA)
#PHASE_STEP = 42949673 / float(FREQ_EXTRA)
#PHASE_STEP = 1
FREQ = PHASE_STEP / float(PHASE_MAX)

# Search for an optimal number of samples (in the allowed range).
# We need to ensure that we hit the frequency of the sine with the FFT.
best_error = 1
for samples in range(MIN_SAMPLES, MAX_SAMPLES+1):
    freq_resolution = 1.0 / samples

    error = (FREQ % freq_resolution)

    if best_error > error:
        best_error = error
        SAMPLES = samples

#SAMPLES=2**14

def _test_vectorcalc():

    clock = Signal(bool(0))
    i     = Signal(intbv(0,min=INTMIN,max=INTMAX))
    q     = Signal(intbv(0,min=INTMIN,max=INTMAX))
    phase = Signal(intbv(0,min=0,max=PHASE_MAX))

    LUT_DEPTH=5
    CORDIC_STAGES = 10
    DEPTH = CORDIC_STAGES + 2

    vcalc = pipelined_cordic.CordicPipeline(clock, phase, i, q, CORDIC_STAGES=CORDIC_STAGES, LUT_DEPTH=LUT_DEPTH)
 
    global outvals, phases
    outvals = []
    phases = []
  
    @always(delay(1))
    def clockgen():
         clock.next = not clock

    @instance
    def check():
        yield clock.negedge
        result_pos = -DEPTH-1
        phase.next = 100
        for sample in range(0,SAMPLES,1):
            yield clock.negedge

            angle = int(round(sample * PHASE_STEP) % PHASE_MAX)
            phase.next = angle
            # Select whether to compare to optimal value, or input value
            #phases.append((round(sample * FREQ) % PHASE_MAX) * 2*math.pi / PHASE_MAX)
            phases.append(((sample * PHASE_STEP) % PHASE_MAX) * 2*math.pi / PHASE_MAX)

            yield clock.posedge
            result_pos += 1

            if result_pos >= 0:
                outvals.append((int(i), int(q)))

        yield clock.negedge
        phase.next = 100

        while result_pos < len(phases) - 1:
            yield clock.posedge
            result_pos += 1

            outvals.append((int(i), int(q)))

        raise StopSimulation                 

    return instances()
 
def test_cordic(n=None):
    global outvals, phases

    Simulation(_test_vectorcalc()).run(n)

    phases = np.array(phases)
    outvals = np.array(outvals)

    ideal_i = np.cos(phases) * (INTMAX-1)
    ideal_q = np.sin(phases) * (INTMAX-1)

    pyplot.figure(figsize=(15, 10))

    pyplot.subplot(411, title="result")
    #pyplot.plot(phases, outvals[:,0], '.')
    #pyplot.plot(phases, outvals[:,1], '.')

    intime = outvals[:min(outvals.shape[0], 250)]

    pyplot.plot(np.arange(0, intime.shape[0]), intime[:,0])
    pyplot.plot(np.arange(0, intime.shape[0]), intime[:,1])

    pyplot.subplot(412, title="amplitude error")
    pyplot.plot(phases, np.sqrt(np.power(outvals[:,0], 2) + np.power(outvals[:,1], 2)) - (INTMAX-1), '.')
    pyplot.axhline(math.sqrt(0.5**2+0.5**2))
    pyplot.axhline(-math.sqrt(0.5**2+0.5**2))

    print 'Average amplitude error:', np.average(np.sqrt(np.power(outvals[:,0], 2) + np.power(outvals[:,1], 2)) - (INTMAX-1))

    pyplot.subplot(413, title="phase error of CORDIC")
    pyplot.plot(phases, ((np.arctan2(outvals[:,1], outvals[:,0]) - phases + np.pi) % (2*np.pi) - np.pi), '.')
    pyplot.axhline(np.arctan2(math.sqrt(0.5**2+0.5**2), INTMAX-1), ls=':', label="Rect")
    pyplot.axhline(-np.arctan2(math.sqrt(0.5**2+0.5**2), INTMAX-1), ls=':')

    print 'Average phase error:', np.average((np.arctan2(outvals[:,1], outvals[:,0]) - phases + np.pi) % (2*np.pi) - np.pi)

    pyplot.axhline(2*np.pi / PHASE_MAX / 2, ls='--', label="Phase")
    pyplot.axhline(-2*np.pi / PHASE_MAX / 2, ls='--')

    pyplot.legend()

    pyplot.subplot(414, title="FFT (%i Points)" % SAMPLES)
    pyplot.grid(True)
    pyplot.xlim(-0.5, 0.5)

    freqs = np.fft.fftfreq(SAMPLES, 1)
    fft = np.maximum(np.abs(np.fft.fft(outvals[:,0] + 1j*outvals[:,1])), 1)/(INTMAX-1)/SAMPLES
    fft = fft[::SKIP]
    pyplot.plot(np.fft.fftshift(freqs), 20*np.log10(np.fft.fftshift(fft))[::SKIP])


    pyplot.savefig("error.pdf", dpi=300)
    #pyplot.show(block=True)

test_cordic()

