#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013, Benjamin Berg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
sys.path.append('..')
from myhdl import *

def NCOCordic(clock, reset, phi_step, out_cos, out_sin, PHASE_PRECISION=None, DITHER=None, LUT_DEPTH=None, CORDIC_STAGES=None):

    if PHASE_PRECISION is None:
        # Infer a sensible precision based on out_cos bit depth (ie. +1)
        PHASE_PRECISION = len(out_cos) + 1

    # Calculate a name for the NCO
    NCOCordic.func_name = "NCOCordic_%i_%i_%i_%i" % (PHASE_PRECISION, len(out_cos), LUT_DEPTH if LUT_DEPTH else 0, CORDIC_STAGES if CORDIC_STAGES else 0)

    instances = []

    if DITHER == 0:
        DITHER = None

    if DITHER is not None:
        dither_val = Signal(intbv(0, -2**(DITHER-1), 2**(DITHER-1)))

        import prng.rule30 as rule30
        rand = rule30.Rule30(clock, reset, dither_val)

        instances.append(rand)

    # The accumulator has the same size as the step input
    phi_acc = Signal(intbv(0)[len(phi_step):])
    nco_phase = Signal(intbv(0)[PHASE_PRECISION:])

    @always_seq(clock.posedge, reset=reset)
    def process():
        # Move the phase forward
        phi_acc.next = (phi_acc + phi_step) % (2**len(phi_acc))
    instances.append(process)

    # Do a dither step if requested
    if DITHER:
        @always_seq(clock.posedge, reset=reset)
        def dither():
            nco_phase.next = (phi_acc[:len(phi_acc)-PHASE_PRECISION] + dither_val) % (2**PHASE_PRECISION)
            
        instances.append(dither)
    else:
        @always_comb
        def passtrough():
            nco_phase.next = phi_acc[:len(phi_acc)-PHASE_PRECISION]

        instances.append(passtrough)

    # Instantiate the main Cordic pipeline
    from cordic import pipelined_cordic
    cordic = pipelined_cordic.CordicPipeline(clock, nco_phase, out_cos, out_sin, LUT_DEPTH=LUT_DEPTH, CORDIC_STAGES=CORDIC_STAGES, NON_SATURATING=True)

    instances.append(cordic)

    return tuple(instances)

if __name__ == "__main__":
    clock = Signal(bool())
    reset = ResetSignal(bool(True), True, True)
    phi_step = Signal(intbv(0)[32:])
    BITS = 16
    cos_out = Signal(intbv(0, -2**(BITS-1), 2**(BITS-1)))
    sin_out = Signal(intbv(0, -2**(BITS-1), 2**(BITS-1)))

    try:
        from toVHDL_kh import toVHDL_kh as toVHDL
    except:
        print "Not keeping hierarchy as toVHDL_kh could not be imported!"

    toVHDL(NCOCordic, clock, reset, phi_step, cos_out, sin_out, LUT_DEPTH=10, DITHER=3)

