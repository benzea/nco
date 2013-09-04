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


import  math
from myhdl import *


def CordicPipeline(clock, phase, output_cos, output_sin, CORDIC_STAGES=None, LUT_DEPTH=5, NON_SATURATING=True, RENAME=True):
    """Implement a Cordic polar to rectangular transformation. The
    length of the vector is one unit. The bit depth and number of pipelineing
    stages is (or will be) configurable.
    Optimized for Xlininx® Kintex 7 and similar devices.

    The pipeline is build as following:
     * Initial Stage (configurable depth):
       This stage does two things:
        - Map the rotation into the first quadrant
        - Lookup an initial value for the first quadrant based on a LUT.

       The rotation into the first quadrant is simply means removing the upper
       two bits of the phase, and storing it into a pipeline of flipflops for
       the last stage.

       The LUT requires a bit more work. The idea is to use a LUT that can be
       directly programmed into the FPGAs logic elements. For this reason the
       depth is configurable. The default value of 5 is optimized for the
       Xilinx® 7 Series of FPGAs, which can create a 5 input LUT with 2 outputs.

       The LUT is generated statically in the script, for this the knowledge
       about further stages is required, as the LUT already contains the
       required scaling factor.

     * After the initial stage a first CORDIC pipeline is placed. This pipeline
       uses a table for the needed atan() and exponent values (we can skip
       some exponents because of the LUT).
       The this stage is only used as long as atan(sigma_i) != sigma_i.

     * The second CORDIC pipeline follows. It uses sigma_i = atan(sigma_0) >> i.
       where sigma_0 is a constant. This pipeline is identical to the previous
       one except for the way that it approximates the value of atan(sigma).

     * The last stage is the output stage. It has the following tasks:
        - Rotate the result vector back into the correct quadrant.
       (- Ensure that no value overflows; this is only done if NON_SATURATING is
          turned off.)

    Options:
     * output/input widths:
       The widths of the output and input are discovered automatically. The
       core bases the internal precisions on the width in bits of these values!
     * CORDIC_STAGES:
       The overall number of pipeline steps for the CORDIC algorithm. The
       overall latency of the core is CORDIC_STAGES + 2.
       If set to None a sane default value is selected. This default value is
       based on the phase resolution of the input phase. With this setting the
       maximum phase error is about 1 LSB of the input phase.
       Default: PHASE_BITS - LUT_DEPTH - 2 (quadrant) + 1
     * LUT_DEPTH:
       The number of bits of the phase that should be used for the lookup table.
       You should set this to the number of inputs that your device can handle
       well for lookup tables. Setting it to a lower value means wasting
       resources and setting it to a higher value is a waste of resources.
       Default: 5 (also used when set to None)
     * NON_SATURATING:
       The output is a fractional number so its range can be [-1..1). This core
       scales the output so that the range becomes (-1..1). This means that no
       error will occur for a zero phase as 1.0 is not reached.
       If set to False, then the error is caught and the output is set to the
       largest possible value.
       WARNING: Must currently be True!


    Input signals:
     * phase: The input phase, width of this signal is used to decide on the
              length of the pipeline!

    Output signals:
     * output_cos, output_sin: fixed point number in the range of (-1..1).
       See also the NON_SATURATING option.
       This output amplitude of the core tends to be slightly smaller than
       requested.

    NOTES:
     * If one does the multiplication at the output, then there is no need
       to increase the precision for the rect pipeline maybe.
    """

    def generate_lut(depth, maximum, output_bits, Kscale):
        """Generate a lookup table for the first quadrant"""
        lut = []

        for angle in range(0, 2**depth):
            # divide by four as we only want the first quadrant
            a = (angle + 0.5) / float(2**depth) * 2*math.pi / 4.0

            s = math.sin(a)
            c = math.cos(a)

            # Add offset for the implicit floor() in the output stage.
            s += 2**(-output_bits)
            c += 2**(-output_bits)

            s = s * maximum * Kscale
            c = c * maximum * Kscale

            s, c = int(s), int(c)

            lut.append((c, s))

        lut = zip(*lut)

        return tuple(lut)

    def generate_cordic_atan(max_phase, precision, max_depth):

        Kscale = 1.0

        def _convert_phase(_phase):
            return int(round(_phase / (2*math.pi) * 2**precision))

        # Find the angle that corresponds to the phase change
        exponent = 1
        _phase = 100
        while _phase > max_phase:
            exponent -= 1
            _phase = math.atan(math.pow(2, exponent))

        phases = []
        exponents = []

        while _convert_phase(_phase) != _convert_phase(math.pow(2, exponent)):
            exponents.append(abs(exponent))
            phases.append(_convert_phase(_phase))

            Kscale = Kscale / math.sqrt(1+math.pow(2, 2*exponent))

            # Do not want more of these ...
            if len(phases) == max_depth:
                break

            exponent -= 1
            _phase = math.atan(math.pow(2, exponent))

        for i in range(0, max_depth - len(phases)):
            Kscale = Kscale / math.sqrt(1+math.pow(2, 2*(exponent - i)))

        if len(phases) == 0:
            phases = None
            exponents = None
        else:
            phases = tuple(phases)
            exponents = tuple(exponents)

        return phases, exponents, abs(exponent), _convert_phase(_phase), Kscale


    ################################################
    ################################################

    # Sane? Requires min/max  to be set!
    assert output_cos.min == output_sin.min and output_cos.max == output_sin.max
    assert output_cos.min == -output_cos.max
    assert len(output_cos) == len(output_sin)

    #####
    # CONSTANTS

    QUAD_DEPTH = 2
    PHASE_PRECISION = len(phase)

    if LUT_DEPTH is None:
        LUT_DEPTH=5

    if CORDIC_STAGES is None:
        CORDIC_STAGES = PHASE_PRECISION - 2 - LUT_DEPTH + 1

    if RENAME:
        # Only has an effect if the hierarchy is extracted using toVHDL_kh.py currently
        CordicPipeline.func_name = "CordicPipeline_%i_%i_%i_%i" % (PHASE_PRECISION, len(output_cos), LUT_DEPTH, CORDIC_STAGES)

    #: Extra precision for the phase register.
    if CORDIC_STAGES:
        # Every stage does one addition/subtraction, ensure the rounding error
        # never becomes relevant.
        # Why is the -2 fine?
        PHASE_NEEDED_PRECISION = CORDIC_STAGES + LUT_DEPTH + int(math.ceil(math.log(CORDIC_STAGES)/math.log(2))) + QUAD_DEPTH - 2
    else:
        PHASE_NEEDED_PRECISION = PHASE_PRECISION

    PHASE_EXTRA_PRECISION = max(0, PHASE_NEEDED_PRECISION - PHASE_PRECISION)

    # All the add/subtracts should not spill into the data bits; so add some bits
    OUT_EXTRA_PRECISION = int(math.ceil(math.log(CORDIC_STAGES) / math.log(2))) + 1

    PIPELINE_PRECISION = len(output_cos) + OUT_EXTRA_PRECISION
    OUT_MIN = -2**(PIPELINE_PRECISION - 1)
    OUT_MAX = 2**(PIPELINE_PRECISION - 1)
    CORDIC_PHASE_PRECISION = PHASE_PRECISION - QUAD_DEPTH - LUT_DEPTH + PHASE_EXTRA_PRECISION

    # Unsigned calculations!
    assert phase.min is None or phase.min == 0

    PIPELINE_DEPTH = CORDIC_STAGES + 1 # cordic stages + input stage

    assert PHASE_PRECISION >= LUT_DEPTH + QUAD_DEPTH

    ####################
    # ROMs
    ####################

    cordic_atan_rom, cordic_exp_rom, cordic_no_atan_first_shift, cordic_no_atan_first_phase, Kscale = generate_cordic_atan(math.pi*2**(-LUT_DEPTH-QUAD_DEPTH), PHASE_PRECISION + PHASE_EXTRA_PRECISION, CORDIC_STAGES)

    # The additional offset for rounding purposes is done inside generate_lut.
    is_cos_rom, is_sin_rom = generate_lut(LUT_DEPTH, OUT_MAX/2 - NON_SATURATING*int(2**(OUT_EXTRA_PRECISION-1)), len(output_cos), Kscale)

    if cordic_atan_rom is None:
        cordic_atan_stages = 0
    else:
        cordic_atan_stages = len(cordic_atan_rom)

    ########
    # Internal Signals
    ########

    # The phase in the pipeline is signed!
    pipeline_phase = [Signal(intbv(0, -2**(CORDIC_PHASE_PRECISION), 2**(CORDIC_PHASE_PRECISION))) for i in range(PIPELINE_DEPTH)]
    pipeline_quadrant = [Signal(intbv(0, 0, 3+1)) for i in range(PIPELINE_DEPTH)]
    pipeline_cos = [Signal(intbv(0, OUT_MIN, OUT_MAX)) for i in range(PIPELINE_DEPTH)]
    pipeline_sin = [Signal(intbv(0, OUT_MIN, OUT_MAX)) for i in range(PIPELINE_DEPTH)]

    ###########
    # Implementation
    ###########

    @always(clock.posedge)
    def initial_stage():
        pipeline_cos[0].next = is_cos_rom[int(phase[PHASE_PRECISION - QUAD_DEPTH:PHASE_PRECISION - QUAD_DEPTH - LUT_DEPTH])]
        pipeline_sin[0].next = is_sin_rom[int(phase[PHASE_PRECISION - QUAD_DEPTH:PHASE_PRECISION - QUAD_DEPTH - LUT_DEPTH])]

        pipeline_quadrant[0].next = phase[PHASE_PRECISION:PHASE_PRECISION-2]

        if PHASE_PRECISION-LUT_DEPTH-QUAD_DEPTH > 0:
            remainder = phase[PHASE_PRECISION-LUT_DEPTH-QUAD_DEPTH:]
            pipeline_phase[0].next = (remainder << PHASE_EXTRA_PRECISION) - 2**(CORDIC_PHASE_PRECISION - 1)
        else:
            pipeline_phase[0].next = 0

    @always(clock.posedge)
    def cordic_atan_pipe():
        for i in range(cordic_atan_stages):
            exp = cordic_exp_rom[i]
            atan = cordic_atan_rom[i]

            if pipeline_phase[i] < 0:
                pipeline_phase[i+1].next = pipeline_phase[i] + atan

                # XXX: MyHDL Bug! I need the brackets around the shift!
                pipeline_cos[i+1].next = pipeline_cos[i] + (pipeline_sin[i] >> exp)
                pipeline_sin[i+1].next = pipeline_sin[i] - (pipeline_cos[i] >> exp)
            else:
                pipeline_phase[i+1].next = pipeline_phase[i] - atan

                pipeline_cos[i+1].next = pipeline_cos[i] - (pipeline_sin[i] >> exp)
                pipeline_sin[i+1].next = pipeline_sin[i] + (pipeline_cos[i] >> exp)

            pipeline_quadrant[i+1].next = pipeline_quadrant[i]

    @always(clock.posedge)
    def cordic_atan_fast_pipe():
        for i in range(cordic_atan_stages, CORDIC_STAGES):
            extra_shift = (i - cordic_atan_stages)
            phase_base = intbv(cordic_no_atan_first_phase, pipeline_phase[0].min, pipeline_phase[0].max)

            if pipeline_phase[i] < 0:
                pipeline_phase[i+1].next = pipeline_phase[i] + (phase_base >> extra_shift)

                # XXX: MyHDL Bug? I need the brackets around the shift!
                pipeline_cos[i+1].next = pipeline_cos[i] + (pipeline_sin[i] >> (cordic_no_atan_first_shift + extra_shift))
                pipeline_sin[i+1].next = pipeline_sin[i] - (pipeline_cos[i] >> (cordic_no_atan_first_shift + extra_shift))
            else:
                pipeline_phase[i+1].next = pipeline_phase[i] - (phase_base >> extra_shift)

                pipeline_cos[i+1].next = pipeline_cos[i] - (pipeline_sin[i] >> (cordic_no_atan_first_shift + extra_shift))
                pipeline_sin[i+1].next = pipeline_sin[i] + (pipeline_cos[i] >> (cordic_no_atan_first_shift + extra_shift))

            pipeline_quadrant[i+1].next = pipeline_quadrant[i]

    cordic_pipe = []
    if cordic_atan_stages > 0:
        cordic_pipe.append(cordic_atan_pipe)
    if CORDIC_STAGES - cordic_atan_stages > 0:
        cordic_pipe.append(cordic_atan_fast_pipe)

    OUTPUT_MAX = output_cos.max

    # Commented code, because the dead branch does not work in VHDL
    assert NON_SATURATING

    @always(clock.posedge)
    def output_stage():
        tmp_cos = pipeline_cos[CORDIC_STAGES] >> (OUT_EXTRA_PRECISION - 1)
        tmp_sin = pipeline_sin[CORDIC_STAGES] >> (OUT_EXTRA_PRECISION - 1)

#        if NON_SATURATING == False:
#            if tmp_cos >= OUTPUT_MAX:
#                tmp_cos = OUTPUT_MAX - 1
#            else:
#                tmp_cos = tmp_cos
#
#            if tmp_sin >= OUTPUT_MAX:
#                tmp_sin = OUTPUT_MAX - 1
#            else:
#                tmp_sin = tmp_sin

        if pipeline_quadrant[CORDIC_STAGES] == 0:
            output_cos.next = tmp_cos
            output_sin.next = tmp_sin
        elif pipeline_quadrant[CORDIC_STAGES] == 1:
            output_cos.next = -tmp_sin
            output_sin.next = tmp_cos
        elif pipeline_quadrant[CORDIC_STAGES] == 2:
            output_cos.next = -tmp_cos
            output_sin.next = -tmp_sin
        elif pipeline_quadrant[CORDIC_STAGES] == 3:
            output_cos.next = tmp_sin
            output_sin.next = -tmp_cos

    return [initial_stage] + cordic_pipe + [output_stage]


if __name__ == "__main__":
    clock = Signal(bool())
    phase = Signal(intbv(0, 0, 2**16))
    bits = 15
    output_cos = Signal(intbv(0, -2**(bits-1), 2**(bits-1)))
    output_sin = Signal(intbv(0, -2**(bits-1), 2**(bits-1)))

    CORDIC_STAGES = 10
    LUT_DEPTH = 5

    #toVerilog(CordicPipeline, clock, phase, output_cos, output_sin, CORDIC_STAGES)
    toVHDL(CordicPipeline, clock, phase, output_cos, output_sin, CORDIC_STAGES, LUT_DEPTH)

