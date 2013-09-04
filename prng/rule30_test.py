#!/usr/bin/env python
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


import myhdl
import rule30
import rule30_reference


def test_rule30():
    # We verify that both automata output the same 10 first states
    # Note that the bit order is reversed in the two, but that does not really
    # matter
    INITIAL_STATE=[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
    UPDATE_RULE=[0,1,1,1,1,0,0,0]

    reference = rule30_reference.CellularAutomata(UPDATE_RULE, INITIAL_STATE[::-1])

    clock = myhdl.Signal(bool(False))
    reset = myhdl.ResetSignal(True, True, True)
    state = myhdl.Signal(myhdl.intbv(0)[len(INITIAL_STATE):])

    dut = rule30.Rule30(clock, reset, state, INITIAL_STATE, UPDATE_RULE)

    @myhdl.always(myhdl.delay(10 / 2))
    def advance_clock():
        clock.next = not clock

    @myhdl.instance
    def check_equality():
        yield clock.posedge
        reset.next = False

        yield clock.posedge

        # Simply check the initial value and the first couple of transitions
        for i in range(10):
            ref_state = reference.state[::-1]

            for s, ref in zip(state, ref_state):
                assert s == ref

            yield clock.posedge
            reference.update_state()

        raise myhdl.StopSimulation()

    # Use all instances for simulation
    return myhdl.instances()

s = myhdl.Simulation(test_rule30())

s.run(None)

print "Everything seems good!"

