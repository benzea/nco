#! /usr/bin/env python

from myhdl import *

def Rule30(clock, reset, rout, INITIAL_STATE=[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], UPDATE_RULE=[0,1,1,1,1,0,0,0]):

    INITIAL_STATE = tuple(INITIAL_STATE)
    UPDATE_RULE = tuple(UPDATE_RULE)

    state_width = len(INITIAL_STATE)
    initial_state = 0
    for i, s in enumerate(INITIAL_STATE):
        initial_state += s << i
    output_width = len(rout)

    # XXX: This should be non-numeric!
    state = Signal(intbv(initial_state, min=0)[state_width:])

    @always_seq(clock.posedge, reset=reset)
    def process():
        rot_r = concat(state[state_width-1:], state[state_width-1])
        rot_l = concat(state[0], state[state_width:1])

        for i in range(state_width):
            addr = concat(rot_l[i], state[i], rot_r[i])

            # Bug: UPDATE_RULE[int(concat(a2, a1, a0))] does not work (with VHDL)

            state.next[i] = UPDATE_RULE[addr]

    if rout.min < 0:
        @always_comb
        def output():
            rout.next = state[output_width:].signed()
    else:
        @always_comb
        def output():
            rout.next = state[output_width:]

    return [process, output]

if __name__ == "__main__":
    clock = Signal(bool())
    reset = ResetSignal(bool(True), True, True)
    rout = Signal(intbv(0)[3:])

    #toVerilog(Rule30, reset, clock, rout)
    toVHDL(Rule30, clock, reset, rout)

