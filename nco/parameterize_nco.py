#! /usr/bin/env python
# -*- coding: utf-8 -*-

from gi.repository import Gtk, GLib, GObject

GLib.threads_init()

import threading
import math
import copy

import nco
import myhdl

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar

NavigationToolbar.draw_idle = lambda toolbar, *args: toolbar.canvas.draw_idle()

def format_num(num):
    ranges = [
        (-24, "y"),
        (-21, "z"),
        (-18, "a"),
        (-15, "f"),
        (-12, "p"),
        (-9, "n"),
        (-6, u"µ"),
        (-3, "m"),
        (0, ""),
        (3, "k"),
        (6, 'M'),
        (9, 'G'),
        (12, 'T'),
        (15, 'P'),
        (18, 'E'),
        (21, 'Z'),
        (24, 'Y'),
    ]

    # Try to find first digit that is below 100
    for exp, sym in ranges[::-1]:
        if abs(num) > 10**(exp - 1):
            break

    return "%.3f %s" % (num * 10**(-exp), sym)

class Parameters(object):
    params = {'lut_depth', 'phase_acc_bits', 'phase_bits', 'out_bits',
        'cordic_stages', 'phase_dither', 'samples', 'frequency',
        'tone'}

    float_params = {'frequency', 'tone'}

    def __init__(self):
        self._adjustments = None

        for param in self.params:
            setattr(self, param, 1)

    def connect_ui(self, builder):
        self._adjustments = dict()

        for param in self.params:
            self._adjustments[param] = builder.get_object(param)
            if param in self.float_params:
                setattr(self, param, self._adjustments[param].get_value())
            else:
                setattr(self, param, int(self._adjustments[param].get_value()))

            # Force the spinner to update, why is this needed?
            self._adjustments[param].emit('value-changed')

            self._adjustments[param].connect('value-changed', self._adjustment_changed_cb)

        self.labels = dict()
        self.labels['latency'] = builder.get_object('latency_label')
        self.labels['phase_frequency'] = builder.get_object('phase_frequency')
        self.labels['frequency_resolution'] = builder.get_object('frequency_resolution')

        self.update_labels()

    def _adjustment_changed_cb(self, *args):
        assert self._adjustments is not None

        for param in self.params:
            if param in self.float_params:
                setattr(self, param, self._adjustments[param].get_value())
            else:
                setattr(self, param, int(self._adjustments[param].get_value()))

        self.ensure_valid()

        self.update_labels()
        self.update_adjustments()

    def ensure_valid(self):
        # Sanity checks
        if self.tone > self.frequency / 2:
            self.tone = self.frequency / 2

        if self.phase_bits > self.phase_acc_bits:
            self.phase_bits = self.phase_acc_bits

        if self.lut_depth + 2 > self.phase_bits:
            self.phase_bits = self.lut_depth + 2

        # Very large dithers are simply insane ...
        if self.phase_dither > 2 + self.phase_bits:
            self.phase_dither = self.phase_bits - 2

    def update_labels(self):
        self.labels['latency'].set_text('%i' % self.pipeline_length)

        self.labels['phase_frequency'].set_text(u'𝚫 %sHz (%i)' % (format_num(1000*1000*(self.actual_tone - self.tone)), self.phase_step))

        self.labels['frequency_resolution'].set_text(u'%sHz (%sHz)' % (format_num(1e6 * self.frequency / (2**self.phase_acc_bits)), format_num(1e6 * self.frequency / (2**self.phase_bits))))

    def update_adjustments(self):
        assert self._adjustments is not None

        for param in self.params:
            value = getattr(self, param)
            cur_val = self._adjustments[param].get_value()
            if value != cur_val:
                self._adjustments[param].set_value(value)

    @property
    def pipeline_length(self):
        dither_depth = 1 if self.phase_dither else 0

        # Magic ... :-/
        return 2 + dither_depth + self.cordic_stages

    @property
    def phase_step(self):
        return int(round(2**self.phase_acc_bits * self.tone / self.frequency))

    @property
    def actual_tone(self):
        return self.phase_step / float(2**self.phase_acc_bits) * self.frequency

    @property
    def fft_samples(self):
        """The number of samples (below the maximum sample count) which is
        best approximates the output frequency.
        This is required so that the FFT does not contain too much aliasing."""

        # The normalized frequency
        freq = self.actual_tone / self.frequency

        best_error = 1
        result = 0
        for samples in range(self.samples / 2, self.samples + 1):
            freq_resolution = 1.0 / samples

            error = (freq % freq_resolution)

            if best_error > error:
                best_error = error
                result = samples

        return result

    def __copy__(self):
        new = Parameters()

        # Simply copy all params (but no other properties)
        for param in self.params:
            setattr(new, param, getattr(self, param))

        return new

    __deepcopy__ = __copy__

class Simulate(GObject.Object):

    __gsignals__ = {
        'simulation_progress' : (GObject.SIGNAL_RUN_FIRST, GObject.TYPE_NONE, ([float, ])),
        'simulation_done' : (GObject.SIGNAL_RUN_FIRST, GObject.TYPE_NONE, ([object, object]))
    }

    def __init__(self):
        GObject.Object.__init__(self)

        self.optimal_samples = 4096
        self.params = Parameters()
        self._thread = None

    def set_parameters(self, params):
        # Cancel any running simulation; just in case
        self.cancel()

        # Store a copy of the parameters
        self.params = copy.copy(params)

    def start(self):
        # Ensure no simulation is running right now
        self.cancel()

        self._thread = threading.Thread(target=self.simulation_thread, name="Cordic NCO Simulation")
        self._thread_abort = False
        self._thread.start()

    def cancel(self):
        if self._thread is None:
            return

        if not self._thread.is_alive():
            self._thread = None
            return

        # Signal thread to abort
        self._thread_abort = True

        # Wait for thread to quit
        self._thread.join()

        self._thread = None

    def simulation_thread(self):
        params = copy.copy(self.params)

        clock = myhdl.Signal(bool())
        reset = myhdl.ResetSignal(bool(True), True, True)
        phase_step = myhdl.Signal(myhdl.intbv(0)[self.params.phase_acc_bits:])
        BITS = self.params.out_bits
        cos_out = myhdl.Signal(myhdl.intbv(0, -2**(BITS-1), 2**(BITS-1)))
        sin_out = myhdl.Signal(myhdl.intbv(0, -2**(BITS-1), 2**(BITS-1)))

        # Constants
        samples = self.params.fft_samples

        dut = nco.NCOCordic(clock, reset, phase_step, cos_out, sin_out,
            PHASE_PRECISION=self.params.phase_bits, LUT_DEPTH=self.params.lut_depth,
            DITHER=self.params.phase_dither, CORDIC_STAGES=self.params.cordic_stages)

        output = []

        @myhdl.always(myhdl.delay(1))
        def clockgen():
            clock.next = not clock

        @myhdl.instance
        def simulate():
            # A couple of cycles to reset the circuit
            reset.next = True
            phase_step.next = 0
            yield clock.negedge
            yield clock.negedge

            reset.next = False
            phase_step.next = self.params.phase_step

            # Flush the pipeline once
            for i in range(0, self.params.pipeline_length):
                yield clock.negedge

            # Now the first correct sample is at the output
            for sample in range(0, samples):
                time = sample / self.params.frequency
                theoretical_phase = (sample / self.params.frequency * self.params.actual_tone % 1.0) * 2*np.pi

                # Save the result
                output.append((time, theoretical_phase, int(cos_out.val), int(sin_out.val)))

                self.emit('simulation_progress', sample / float(samples))

                if self._thread_abort:
                    raise myhdl.StopSimulation("Simulation Aborted")

                yield clock.negedge

            raise myhdl.StopSimulation

        myhdl.Simulation(dut, clockgen, simulate).run(None)

        simulation_result = np.array(output, dtype=np.float64)

        self.emit('simulation_done', simulation_result, params)

class Plots(object):
    def __init__(self, notebook):
        self._notebook = notebook

        self.data = None
        self.params = None

        ##########
        # Time Domain
        ##########
        self.time_domain = Figure()
        self.time_domain_ax = self.time_domain.add_subplot('111')
        #self.time_domain.tight_layout()

        grid = Gtk.Grid()
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        label = Gtk.Label()
        label.set_markup("<b>Time Domain</b>")
        canvas = FigureCanvas(self.time_domain)
        canvas.set_size_request(200, 200)
        canvas.set_hexpand(True)
        canvas.set_vexpand(True)

        toolbar = NavigationToolbar(canvas, notebook.get_toplevel())

        grid.attach(canvas, 0, 0, 1, 1)
        grid.attach(toolbar, 0, 1, 1, 1)

        notebook.append_page(grid, label)

        ##########
        # FFT
        ##########
        self.fft = Figure()
        self.fft_ax = self.fft.add_subplot('111')

        grid = Gtk.Grid()
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        label = Gtk.Label()
        label.set_markup("<b>FFT</b>")
        canvas = FigureCanvas(self.fft)
        canvas.set_size_request(200, 200)
        canvas.set_hexpand(True)
        canvas.set_vexpand(True)

        toolbar = NavigationToolbar(canvas, notebook.get_toplevel())

        grid.attach(canvas, 0, 0, 1, 1)
        grid.attach(toolbar, 0, 1, 1, 1)

        notebook.append_page(grid, label)


        ##########
        # Phase Error
        ##########
        self.phase_error = Figure()
        self.phase_error_ax = self.phase_error.add_subplot('111')

        grid = Gtk.Grid()
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        label = Gtk.Label()
        label.set_markup("<b>Phase Error</b>")
        canvas = FigureCanvas(self.phase_error)
        canvas.set_size_request(200, 200)
        canvas.set_hexpand(True)
        canvas.set_vexpand(True)

        toolbar = NavigationToolbar(canvas, notebook.get_toplevel())

        grid.attach(canvas, 0, 0, 1, 1)
        grid.attach(toolbar, 0, 1, 1, 1)

        notebook.append_page(grid, label)

        notebook.show_all()

    def set_data(self, data, params):
        # Copy data
        self.data = np.array(data)
        self.params = copy.copy(params)

        ########
        # Time Domain
        ########
        self.time_domain_ax.clear()
        self.time_domain_ax.plot(self.data[:,0], self.data[:,2])
        self.time_domain_ax.plot(self.data[:,0], self.data[:,3])

        ########
        # FFT
        ########
        samples = self.data.shape[0]
        amplitude = 2**(self.params.out_bits - 1) - 1

        freqs = np.fft.fftfreq(samples, 1 / self.params.frequency)
        fft = np.maximum(np.abs(np.fft.fft(self.data[:,2] + 1j*self.data[:,3])), 1)/amplitude/samples

        self.fft_ax.clear()
        self.fft_ax.plot(np.fft.fftshift(freqs), 20*np.log10(np.fft.fftshift(fft)))

        self.fft_ax.set_xlim(-self.params.frequency / 2, self.params.frequency / 2)
        self.fft_ax.set_ylim(20*np.log10(1.0/amplitude/samples), 0)

        ########
        # Phase Error
        ########
        phase_precision = 1.0 / 2**(self.params.phase_bits)
        self.phase_error_ax.clear()
        self.phase_error_ax.plot(self.data[:,1], ((np.arctan2(self.data[:,3], self.data[:,2]) - self.data[:,1] + np.pi) % (2*np.pi) - np.pi), 'x')
        self.phase_error_ax.axhline(np.arctan2(math.sqrt(0.5**2+0.5**2), amplitude), ls=':', label="Rect")
        self.phase_error_ax.axhline(-np.arctan2(math.sqrt(0.5**2+0.5**2), amplitude), ls=':')
        self.phase_error_ax.axhline(2*np.pi * phase_precision / 2, ls='--', label="Phase")
        self.phase_error_ax.axhline(-2*np.pi * phase_precision / 2, ls='--')
        self.phase_error_ax.set_xlim(0, 2*np.pi)

        ########
        # Labels/Grids/etc.
        ########
        self.phase_error_ax.set_xlabel('Phase')
        self.phase_error_ax.set_ylabel('Error')
        self.phase_error_ax.grid(True)

        self.time_domain_ax.set_xlabel('Time [ms]')
        self.time_domain_ax.set_ylabel('Amplitude')
        self.time_domain_ax.grid(True)

        self.fft_ax.set_xlabel("Frequency [MHz]")
        self.fft_ax.set_ylabel("FFT [dB]")
        self.fft_ax.grid(True)

        # XXX: Workaround, it doesn't repaint properly
        self._notebook.queue_draw()

class Parameterize(object):
    
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file('./parameterize_nco.glade')

        self.builder.connect_signals(self)

        self.window = self.builder.get_object('main')
        self.statusbar = self.builder.get_object('statusbar')
        self._simulation_ctx = self.statusbar.get_context_id('simulation')

        self.params = Parameters()
        self.params.connect_ui(self.builder)

        self.plots = Plots(self.builder.get_object('plots'))

        self.simulate = Simulate()

        self.simulate.connect('simulation_progress', self.simulation_thread_progress)
        self.simulate.connect('simulation_done', self.simulation_thread_done)

        self.simulation_update_throttle = False


    def simulation_progress(self, progress):
        self.statusbar.remove_all(self._simulation_ctx)
        self.statusbar.push(self._simulation_ctx, 'Simulation %i %% complete' % (int(progress * 100)))
        self.simulation_update_throttle = False

    def simulation_done(self, output, params):
        self.statusbar.remove_all(self._simulation_ctx)
        self.statusbar.push(self._simulation_ctx, 'Simulation done')

        self.plots.set_data(output, params)

    def simulation_thread_progress(self, sim, progress):
        if self.simulation_update_throttle:
            return

        # This is called from another thread
        GLib.idle_add(self.simulation_progress, progress)
        self.simulation_update_throttle = True

    def simulation_thread_done(self, sim, output, params):
        # This is called from another thread
        GLib.idle_add(self.simulation_done, output, params)

    def run(self):
        # Show the window, and run the mainloop
        self.window.show()
        Gtk.main()

    def start_simulation(self, *args):
        self.simulate.set_parameters(self.params)

        self.statusbar.remove_all(self._simulation_ctx)
        self.statusbar.push(self._simulation_ctx, 'Starting Simulation')
        # Uh, fresh simulation ...
        self.simulation_update_throttle = False

        self.simulate.start()

    def quit(self, *args):
        self.simulate.cancel()

        Gtk.main_quit()

if __name__ == "__main__":
    gui = Parameterize()
    gui.run()

