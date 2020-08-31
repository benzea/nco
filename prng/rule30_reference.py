# -*- coding: utf-8 -*-

#=======================================================================
#
# Copyright notice for the CellularAutomata class.
#
# The upstream version is from http://opencores.org/project,ca_prng
#
#########
#
# Fast and simple ca_prng conformant cellular automata model in
# Python. This model is actually implented as a general 1D CA class
# and the rule and size of the CA array is provided as parameters.
#
#
# Author: Joachim Strömbergson
# Copyright (c) 2008, Kryptologik
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Kryptologik ''AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL Kryptologik BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#=======================================================================

#-------------------------------------------------------------------
# class CellularAutomata()
#
# This class implements a 1D cellular automata. The class expects
# to be initalized with an array of arbitrary length with initial
# cell state values (0 or 1) as well as an array with update rules.
#
# The update rule is expected to contain eight values (0 or 1)
# that define the update value for a cell given by the current
# state of the cell and two nearest neighbours. Note that nearest
# neighbour is calculated with wrap around, that is the cell
# array is treated as a ring.
#-------------------------------------------------------------------
class CellularAutomata():
    def __init__(self, rule, init_state):
        self.rule = rule
        self.state = init_state


    def print_state(self):
        """Print the current state of the cellular automata."""
        print(self.state)


    def update_state(self):
        """Update the cells in the cellular automata array."""

        # Create a new CA array to store the updated state.
        new_state = [x for x in range(len(self.state))]

        # For each cell we extract three consequtive bits from the
        # current state and use wrap around at the edges.
        for curr_bit in range(len(self.state)):
            if curr_bit == 0:
                bit_left = self.state[-1]
                bit_mid = self.state[0]
                bit_right = self.state[1]
            elif curr_bit == (len(self.state) - 1):
                bit_left = self.state[(curr_bit - 1)]
                bit_mid = self.state[curr_bit]
                bit_right = self.state[0]
            else:
                bit_left = self.state[(curr_bit - 1)]
                bit_mid = self.state[curr_bit]
                bit_right = self.state[(curr_bit + 1)]

            # Use the extraxted bits to calculate an index for
            # the update rule array and update the cell.
            rule_index = 4 * bit_left + 2 * bit_mid + bit_right
            new_state[curr_bit] = self.rule[rule_index]

        # Replace the old state array with the new array.
        self.state = new_state

