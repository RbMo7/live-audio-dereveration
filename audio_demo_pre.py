import customtkinter as ctk
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import filedialog, messagebox
from scipy.signal import fftconvolve, freqz, welch, square
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Main Application ---
class AudioDeverbDemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DSP Signal Restoration Demonstrator")
        self.geometry("1100x800")

        # --- App Data ---
        self.fs = 44100
        self.clean_signal = None
        self.reverberant_signal = None
        self.restored_signal = None
        self.room_impulse_response = None
        self.inverse_filter = None

        # --- GUI Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # Configure row for plot frame expansion

        # --- Top Control Frame ---
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        top_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # --- NEW: Signal Generation Frame ---
        gen_frame = ctk.CTkFrame(top_frame)
        gen_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(gen_frame, text="1. Generate Signal", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        ctk.CTkButton(gen_frame, text="Sine Wave", command=self.generate_sine_wave).pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(gen_frame, text="Square Wave", command=self.generate_square_wave).pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(gen_frame, text="Speech Sample", command=self.load_embedded_sample).pack(fill="x", padx=5, pady=2)

        # --- Processing Frame ---
        proc_frame = ctk.CTkFrame(top_frame)
        proc_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(proc_frame, text="2. Process Signal", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.process_button = ctk.CTkButton(proc_frame, text="Apply De-reverberation Filter", command=self.process_audio, state="disabled")
        self.process_button.pack(fill="both", expand=True, padx=5, pady=2)

        # --- NEW: Playback Frame ---
        play_frame = ctk.CTkFrame(top_frame)
        play_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(play_frame, text="3. Play Audio", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.play_clean_button = ctk.CTkButton(play_frame, text="Play Original Clean", command=lambda: self.play_audio(self.clean_signal), state="disabled")
        self.play_clean_button.pack(fill="x", padx=5, pady=2)
        self.play_echo_button = ctk.CTkButton(play_frame, text="Play With Echo", command=lambda: self.play_audio(self.reverberant_signal), state="disabled")
        self.play_echo_button.pack(fill="x", padx=5, pady=2)
        self.play_restored_button = ctk.CTkButton(play_frame, text="Play Restored", command=lambda: self.play_audio(self.restored_signal), state="disabled")
        self.play_restored_button.pack(fill="x", padx=5, pady=2)
        
        # Plotting Frame
        plot_frame = ctk.CTkFrame(self)
        plot_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)

        # --- Plotting Setup ---
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.init_plots()

    def init_plots(self):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        
        self.ax1.set_title("Original Clean Signal")
        self.ax2.set_title("Signal with Echo Added")
        self.ax3.set_title("Restored Signal (After Filter)")
        self.ax4.set_title("Power Spectral Density")
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]: ax.grid(True)
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _generate_and_apply_echo(self):
        """Helper to apply a standard echo to the current self.clean_signal."""
        # Create a simple Room Impulse Response (RIR) for the filter calculation
        rir = np.zeros(int(0.5 * self.fs))
        rir[0] = 1.0
        echo_delay_samples = int(0.25 * self.fs)
        rir[echo_delay_samples] = 0.6
        self.room_impulse_response = rir

        # --- FIX: Manually create the echo instead of using convolution ---
        # 1. Create a delayed version of the clean signal
        delayed_signal = np.zeros_like(self.clean_signal)
        # The np.roll function is a simple way to create a delay
        delayed_signal[echo_delay_samples:] = self.clean_signal[:-echo_delay_samples]
        
        # 2. Add the original and delayed signals together
        self.reverberant_signal = self.clean_signal + (delayed_signal * 0.6)
        
        # 3. Normalize the result to prevent clipping (amplitude > 1.0)
        if np.max(np.abs(self.reverberant_signal)) > 0:
            self.reverberant_signal /= np.max(np.abs(self.reverberant_signal))
        # --- END OF FIX ---

        # The rest of the function remains the same
        self.inverse_filter = self._calculate_inverse_filter(self.room_impulse_response)
        
        self.update_initial_plots()
        self.process_button.configure(state="normal")
        self.play_clean_button.configure(state="normal")
        self.play_echo_button.configure(state="normal")
        self.play_restored_button.configure(state="disabled")

    def generate_sine_wave(self):
        self.init_plots()
        self.fs = 22050
        t = np.linspace(0, 1.5, int(1.5 * self.fs), endpoint=False)
        self.clean_signal = 0.8 * np.sin(2 * np.pi * 440 * t)
        self._generate_and_apply_echo()

    def generate_square_wave(self):
        self.init_plots()
        self.fs = 22050
        t = np.linspace(0, 1.5, int(1.5 * self.fs), endpoint=False)
        self.clean_signal = 0.5 * square(2 * np.pi * 440 * t)
        self._generate_and_apply_echo()

    def load_embedded_sample(self):
        self.init_plots()
        self.fs = 22050
        t = np.linspace(0, 1.5, int(1.5 * self.fs), endpoint=False)
        sig1 = np.sin(2 * np.pi * 330 * t) * np.exp(-t*2)
        sig2 = np.sin(2 * np.pi * 880 * t) * np.exp(-(t-0.5)**2 / 0.1)
        self.clean_signal = (sig1 + sig2 * 0.5)
        self.clean_signal /= np.max(np.abs(self.clean_signal))
        self._generate_and_apply_echo()
    
    def update_initial_plots(self):
        """Updates the first two plots after a signal is generated."""
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.ax4.clear()
        self.init_plots() # Re-add titles and grids
        
        # FIX: Plot only the first 1000 samples to "zoom in" on the waveform
        plot_len = min(1000, len(self.clean_signal))
        
        self.ax1.plot(self.clean_signal[:plot_len])
        self.ax2.plot(self.reverberant_signal[:plot_len])
        
        self.canvas.draw()

    def _calculate_inverse_filter(self, rir):
        rir_len = len(rir)
        rir_fft = np.fft.rfft(rir, n=2 * rir_len)
        epsilon = 1e-3
        inv_rir_fft = np.conj(rir_fft) / (np.abs(rir_fft)**2 + epsilon)
        inv_filter = np.fft.irfft(inv_rir_fft, n=2 * rir_len)
        return inv_filter / np.max(np.abs(inv_filter)) if np.max(np.abs(inv_filter)) > 0 else inv_filter

    def process_audio(self):
        """Applies the inverse filter to the reverberant signal and updates plots."""
        if self.reverberant_signal is None or self.inverse_filter is None:
            messagebox.showwarning("Warning", "Please load a demo scenario first.")
            return

        # Apply the de-reverberation filter
        self.restored_signal = fftconvolve(self.reverberant_signal, self.inverse_filter, mode='same')
        
        # Normalize the final signal to prevent clipping
        if np.max(np.abs(self.restored_signal)) > 0:
            self.restored_signal /= np.max(np.abs(self.restored_signal))
        
        # Update the final plots
        self.ax3.clear()

        # FIX: Plot only the first 1000 samples to "zoom in" on the waveform
        plot_len = min(1000, len(self.restored_signal))
        self.ax3.plot(self.restored_signal[:plot_len])
        
        self.ax3.set_title("Restored Signal (After Filter)")
        self.ax3.grid(True)
        
        # Update PSD plot
        self.ax4.clear()
        f_rev, pxx_rev = welch(self.reverberant_signal, self.fs, nperseg=1024)
        f_res, pxx_res = welch(self.restored_signal, self.fs, nperseg=1024)
        self.ax4.semilogy(f_rev, pxx_rev, label='With Echo', alpha=0.7)
        self.ax4.semilogy(f_res, pxx_res, label='Restored', linewidth=1.5)
        self.ax4.set_title("Power Spectral Density")
        self.ax4.set_xlabel("Frequency (Hz)")
        self.ax4.set_ylabel("Power")
        self.ax4.legend()
        self.ax4.grid(True)
        
        self.canvas.draw()
        
        self.play_restored_button.configure(state="normal")    

    def play_audio(self, data):
        if data is not None:
            sd.play(data, self.fs)
        else:
            messagebox.showwarning("Playback Error", "No audio data available to play.")


if __name__ == "__main__":
    app = AudioDeverbDemoApp()
    app.mainloop()