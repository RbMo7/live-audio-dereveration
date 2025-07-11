import customtkinter as ctk
import numpy as np
import sounddevice as sd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import queue
from collections import deque
from scipy.signal import fftconvolve, freqz

# --- Main Application ---
class LiveAudioEnhancerApp(ctk.CTk):
    """
    A real-time audio processing application to remove reverb from a live microphone feed.
    It works by:
    1.  Calibrating to the room's echo by recording a short, sharp sound (a clap).
    2.  Calculating an 'inverse filter' that cancels out this echo.
    3.  Applying this filter to the live audio stream using an efficient convolution method.
    """
    def __init__(self):
        super().__init__()
        self.title("Live Audio De-reverberation Tool")
        self.geometry("1000x800")

        # --- Audio & App State ---
        self.fs = 44100
        # A larger block size is more efficient for convolution but increases latency.
        self.blocksize = 2048
        self.stream = None
        self.plot_queue_in = queue.Queue()
        self.plot_queue_out = queue.Queue()

        # Fixed-size buffers for plotting (deque is efficient for this)
        self.plot_buffer_size = 4096
        self.plot_data_in = deque(np.zeros(self.plot_buffer_size), maxlen=self.plot_buffer_size)
        self.plot_data_out = deque(np.zeros(self.plot_buffer_size), maxlen=self.plot_buffer_size)

        self.room_impulse_response = None
        self.inverse_filter = None
        # This buffer is crucial for the continuous convolution (Overlap-Save method)
        self.overlap_buffer = None

        self.deverb_mix = ctk.DoubleVar(value=0.7)

        # --- GUI Widgets ---
        self.main_label = ctk.CTkLabel(self, text="Live Voice De-reverberation", font=ctk.CTkFont(size=20, weight="bold"))
        self.main_label.pack(pady=10)

        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=10, padx=10, fill="x")
        control_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.calibrate_button = ctk.CTkButton(control_frame, text="1. Clap to Calibrate", command=self.calibrate_room)
        self.calibrate_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.start_button = ctk.CTkButton(control_frame, text="2. Start Processing", command=self.start_stream, state="disabled")
        self.start_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.stop_button = ctk.CTkButton(control_frame, text="3. Stop Processing", command=self.stop_stream, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        deverb_frame = ctk.CTkFrame(self)
        deverb_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(deverb_frame, text="De-reverb Mix (0% = Original, 100% = De-reverbed)").pack()
        self.deverb_mix_slider = ctk.CTkSlider(deverb_frame, from_=0.0, to=1.0, variable=self.deverb_mix)
        self.deverb_mix_slider.pack(pady=5, padx=20, fill="x")

        self.calib_status_label = ctk.CTkLabel(self, text="Status: Not Calibrated")
        self.calib_status_label.pack(pady=5)

        # --- Plotting Setup ---
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2, sharey=self.ax1)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)

        self.line_in, = self.ax1.plot(np.zeros(self.plot_buffer_size))
        self.line_out, = self.ax2.plot(np.zeros(self.plot_buffer_size))
        self.ax1.set_title("Live Input (Reverberant)")
        self.ax2.set_title("Live Output (De-reverbed)")
        self.ax3.set_title("Room Impulse Response (Echo Pattern)")
        self.ax4.set_title("Inverse Filter Frequency Response")

        for ax in [self.ax1, self.ax2]:
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)
            
        for ax in [self.ax3, self.ax4]:
            ax.grid(True)

        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.ani = None

    def calibrate_room(self):
        duration = 1.5
        self.calib_status_label.configure(text="Status: Recording clap...", text_color="orange")
        self.update_idletasks() # Force GUI update

        try:
            # Record a short, sharp sound (like a clap)
            impulse_rec = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='float32')
            sd.wait()
            impulse_rec = impulse_rec.flatten()

            # Find the start of the clap and extract the Room Impulse Response (RIR)
            start_index = np.argmax(np.abs(impulse_rec))
            # We'll use a 0.5 second RIR estimate
            rir_len = int(0.5 * self.fs)
            rir_est = impulse_rec[start_index : start_index + rir_len]
            self.room_impulse_response = rir_est / np.max(np.abs(rir_est))

            # --- Calculate the Inverse Filter ---
            # This filter, when convolved with the RIR, approximates a single impulse (canceling the echo)
            desired_len = len(self.room_impulse_response)
            rir_fft = np.fft.rfft(self.room_impulse_response, n=2 * desired_len)
            
            # Regularization prevents division by zero and limits amplification of frequencies
            epsilon = 1e-4 
            inv_rir_fft = np.conj(rir_fft) / (np.abs(rir_fft)**2 + epsilon)
            self.inverse_filter = np.fft.irfft(inv_rir_fft, n=2 * desired_len)[:desired_len]
            self.inverse_filter /= np.max(np.abs(self.inverse_filter)) # Normalize

            # **IMPORTANT**: Initialize the overlap buffer for the overlap-save algorithm
            # Its size must be the filter length minus 1
            self.overlap_buffer = np.zeros(len(self.inverse_filter) - 1, dtype=np.float32)

            # --- Update the static plots ---
            self.ax3.clear()
            self.ax3.plot(self.room_impulse_response)
            self.ax3.set_title("Room Impulse Response")
            self.ax3.set_xlabel("Samples")
            self.ax3.set_ylabel("Amplitude")
            self.ax3.grid(True)

            self.ax4.clear()
            w, h = freqz(self.inverse_filter, 1, fs=self.fs, worN=2048)
            self.ax4.plot(w, 20 * np.log10(np.abs(h) + 1e-9))
            self.ax4.set_title("Inverse Filter Frequency Response")
            self.ax4.set_ylabel("Magnitude (dB)")
            self.ax4.set_xlabel("Frequency (Hz)")
            self.ax4.grid(True)
            self.canvas.draw()

            self.calib_status_label.configure(text="Status: Calibrated! Ready to start.", text_color="green")
            self.start_button.configure(state="normal")
        except Exception as e:
            self.calib_status_label.configure(text=f"Status: Calibration Failed. Try again. {e}", text_color="red")

    def audio_callback(self, indata, outdata, frames, time, status):
        """ This is the core audio processing function, called repeatedly by the audio stream. """
        if status:
            print(status, flush=True)

        input_block = indata[:, 0].astype(np.float32)

        # If not calibrated yet, or filter is not ready, just pass audio through
        if self.inverse_filter is None or self.overlap_buffer is None:
            outdata[:] = indata
            processed_block = input_block
        else:
            # --- OVERLAP-SAVE CONVOLUTION METHOD ---
            # This is the correct way to apply a long filter to a real-time audio stream.
            
            # 1. Create the processing block by concatenating the previous overlap with the new data
            conv_input = np.concatenate((self.overlap_buffer, input_block))

            # 2. Determine the mix between original and de-reverbed signal
            mix = self.deverb_mix.get()
            
            if mix > 0: # Only do convolution if we need the de-reverbed signal
                # Convolve with the inverse filter using 'valid' mode. The output is the part of
                # the convolution that is not tainted by edge effects.
                deverbed_signal = fftconvolve(conv_input, self.inverse_filter, mode='valid')
                
                # Mix the original (input_block) with the processed signal.
                # 'input_block' is correctly time-aligned with the output of the 'valid' convolution.
                processed_block = (input_block * (1 - mix)) + (deverbed_signal * mix)
            else: # If mix is 0, no need to compute the expensive convolution
                processed_block = input_block
            
            # 3. Update the overlap buffer for the *next* callback.
            # It consists of the last (filter_length - 1) samples of the current convolution input.
            # THIS IS THE CORRECTED LINE:
            self.overlap_buffer = conv_input[len(input_block):]

        # Ensure the output is within the valid [-1, 1] range to prevent clipping.
        np.clip(processed_block, -1.0, 1.0, out=processed_block)

        outdata[:] = processed_block.reshape(-1, 1)
        
        # Add data to queues for plotting in the main GUI thread
        self.plot_queue_in.put(input_block)
        self.plot_queue_out.put(processed_block)

    def animate_plots(self, frame):
        """ Updates the live waveform plots. """
        try:
            # Update Input Plot
            while not self.plot_queue_in.empty():
                self.plot_data_in.extend(self.plot_queue_in.get_nowait())
            self.line_in.set_ydata(self.plot_data_in)
            
            # Update Output Plot
            while not self.plot_queue_out.empty():
                self.plot_data_out.extend(self.plot_queue_out.get_nowait())
            self.line_out.set_ydata(self.plot_data_out)
            
            # Only return artists that have changed for blitting efficiency
            return self.line_in, self.line_out
        except queue.Empty:
            return self.line_in, self.line_out

    def start_stream(self):
        if self.stream is None:
            try:
                self.stream = sd.Stream(
                    samplerate=self.fs,
                    channels=1, 
                    callback=self.audio_callback,
                    blocksize=self.blocksize,
                    latency='low' # Request low latency for more responsive feel
                )
                self.stream.start()

                # Start the plot animation
                self.ani = animation.FuncAnimation(self.fig, self.animate_plots, blit=True, interval=30, cache_frame_data=False)
                self.canvas.draw()

                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal")
                self.calibrate_button.configure(state="disabled")
                self.calib_status_label.configure(text="Status: Processing...", text_color="cyan")
            except Exception as e:
                self.calib_status_label.configure(text=f"Error starting stream: {e}", text_color="red")
                if self.stream:
                    self.stream.close()
                self.stream = None

    def stop_stream(self):
        if self.stream is not None:
            # Stop the animation and the audio stream
            if self.ani:
                self.ani.event_source.stop()
                self.ani = None

            self.stream.stop()
            self.stream.close()
            self.stream = None

            # Reset plot deques to zeros
            self.plot_data_in.clear()
            self.plot_data_out.clear()
            self.plot_data_in.extend(np.zeros(self.plot_buffer_size))
            self.plot_data_out.extend(np.zeros(self.plot_buffer_size))

            # Trigger one last draw to show the cleared plots
            self.line_in.set_ydata(self.plot_data_in)
            self.line_out.set_ydata(self.plot_data_out)
            self.canvas.draw()

            # Reset button states
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.calibrate_button.configure(state="normal")
            self.calib_status_label.configure(text="Status: Stopped. Ready to start again.", text_color="white")

    def on_closing(self):
        """ Gracefully shut down the app. """
        self.stop_stream()
        self.destroy()

if __name__ == "__main__":
    app = LiveAudioEnhancerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()