import customtkinter as ctk
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import filedialog, messagebox
from scipy.signal import butter, cheby1, firwin, sosfiltfilt, filtfilt, spectrogram, freqz, freqz_sos
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import io
import requests

# Set the appearance mode
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class AdvancedAudioFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Advanced DSP Audio Filtering")
        self.geometry("1250x800")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- App Data ---
        self.original_sound = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.fs = 44100
        self.active_filter_type = None

        # --- GUI Panels ---
        self.input_panel = ctk.CTkFrame(self, width=200)
        self.input_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.plot_panel = ctk.CTkFrame(self)
        self.plot_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")
        self.plot_panel.grid_rowconfigure(0, weight=1)
        self.plot_panel.grid_columnconfigure(0, weight=1)
        
        self.control_panel = ctk.CTkFrame(self, width=240)
        self.control_panel.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        # --- Input Widgets ---
        ctk.CTkLabel(self.input_panel, text="Input & Playback", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10, padx=10)
        ctk.CTkButton(self.input_panel, text="Load Audio File", command=self.load_audio).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(self.input_panel, text="Load Sample Audio", command=self.load_sample_audio).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(self.input_panel, text="Record 5 Seconds", command=self.record_audio).pack(fill="x", padx=10, pady=5)
        
        self.play_original_btn = ctk.CTkButton(self.input_panel, text="Play Original", command=lambda: self.play_audio(self.original_sound))
        self.play_original_btn.pack(fill="x", padx=10, pady=(15, 5))
        
        self.play_noisy_btn = ctk.CTkButton(self.input_panel, text="Play Noisy", command=lambda: self.play_audio(self.noisy_signal))
        self.play_noisy_btn.pack(fill="x", padx=10, pady=5)

        self.play_filtered_btn = ctk.CTkButton(self.input_panel, text="Play Filtered", command=lambda: self.play_audio(self.filtered_signal))
        self.play_filtered_btn.pack(fill="x", padx=10, pady=5)
        
        self.save_button = ctk.CTkButton(self.input_panel, text="Save Filtered Audio", command=self.save_audio)
        self.save_button.pack(fill="x", padx=10, pady=(15, 5))

        # --- Noise Widgets ---
        ctk.CTkLabel(self.input_panel, text="Noise Type", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5), padx=10)
        self.noise_type = ctk.StringVar(value="White")
        ctk.CTkRadioButton(self.input_panel, text="White", variable=self.noise_type, value="White").pack(anchor="w", padx=20, pady=2)
        ctk.CTkRadioButton(self.input_panel, text="Pink", variable=self.noise_type, value="Pink").pack(anchor="w", padx=20, pady=2)
        ctk.CTkRadioButton(self.input_panel, text="Brown", variable=self.noise_type, value="Brown").pack(anchor="w", padx=20, pady=2)
        ctk.CTkButton(self.input_panel, text="Add Noise", command=self.add_noise).pack(fill="x", padx=10, pady=10)

        # --- Control Widgets ---
        ctk.CTkLabel(self.control_panel, text="Filter Controls", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10, padx=10)
        ctk.CTkLabel(self.control_panel, text="Filter Order:").pack(anchor="w", padx=10)
        self.order_entry = ctk.CTkEntry(self.control_panel)
        self.order_entry.insert(0, "4")
        self.order_entry.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(self.control_panel, text="Cutoff Frequency (Hz):").pack(anchor="w", padx=10)
        self.cutoff_slider = ctk.CTkSlider(self.control_panel, from_=100, to=8000, command=self.update_cutoff_label)
        self.cutoff_slider.set(3000)
        self.cutoff_slider.pack(fill="x", padx=10, pady=5)
        self.cutoff_label = ctk.CTkLabel(self.control_panel, text="3000 Hz")
        self.cutoff_label.pack()

        ctk.CTkButton(self.control_panel, text="Apply Butterworth", command=self.apply_butterworth).pack(fill="x", padx=10, pady=(15,5))
        ctk.CTkButton(self.control_panel, text="Apply Chebyshev", command=self.apply_chebyshev).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(self.control_panel, text="Apply FIR Filter", command=self.apply_fir).pack(fill="x", padx=10, pady=5)
        
        self.realtime_update = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_panel, text="Update filter on change", variable=self.realtime_update).pack(pady=10)

        ctk.CTkLabel(self.control_panel, text="Analysis", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        self.snr_label = ctk.CTkLabel(self.control_panel, text="SNR: N/A", font=ctk.CTkFont(size=14))
        self.snr_label.pack(pady=5)

        # --- Plotting Setup ---
        plt.style.use("seaborn-v0_8-darkgrid")
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.update_plots()
        
        self.order_entry.bind("<Return>", self.realtime_filter_update)
        self.cutoff_slider.bind("<ButtonRelease-1>", self.realtime_filter_update)

    def realtime_filter_update(self, event=None):
        if self.realtime_update.get():
            # When a control changes, re-apply the last used filter
            if self.active_filter_type == "butter":
                self.apply_butterworth()
            elif self.active_filter_type == "cheby":
                self.apply_chebyshev()
            elif self.active_filter_type == "fir":
                self.apply_fir()
            else: # If no filter is active yet, just update the analysis plots
                self.update_plots()


    def load_audio(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        if not filepath: return
        self._load_data(filepath)

    def load_sample_audio(self):
        url = "https://github.com/librosa/librosa/raw/main/docs/audio/trumpet.wav"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self._load_data(io.BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Download Error", f"Failed to download sample audio: {e}")

    def _load_data(self, source):
        try:
            data, fs = sf.read(source)
            self.original_sound = data[:, 0] if data.ndim > 1 else data
            self.original_sound = self.original_sound / np.max(np.abs(self.original_sound))
            self.fs = fs
            self.noisy_signal = None
            self.filtered_signal = None
            self.active_filter_type = None
            self.cutoff_slider.configure(to=self.fs / 2 - 1)
            self.update_plots()
        except Exception as e:
            messagebox.showerror("File Error", f"Error loading file: {e}")

    def record_audio(self):
        try:
            self.fs = 44100
            self.original_sound = sd.rec(int(5 * self.fs), samplerate=self.fs, channels=1).flatten()
            sd.wait()
            self.original_sound = self.original_sound / np.max(np.abs(self.original_sound))
            self.noisy_signal = None
            self.filtered_signal = None
            self.active_filter_type = None
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Recording Error", f"Recording failed. Is a microphone connected? {e}")

    def add_noise(self):
        if self.original_sound is None: return
        noise_type = self.noise_type.get()
        n = len(self.original_sound)
        
        if noise_type == "White":
            noise = np.random.randn(n)
        else:
            fft_white = np.fft.rfft(np.random.randn(n))
            frequencies = np.fft.rfftfreq(n, 1/self.fs)
            frequencies[0] = 1e-6
            if noise_type == "Pink":
                fft_pink = fft_white / np.sqrt(frequencies)
                noise = np.fft.irfft(fft_pink, n=n)
            elif noise_type == "Brown":
                fft_brown = fft_white / frequencies
                noise = np.fft.irfft(fft_brown, n=n)

        signal_power = np.mean(self.original_sound**2)
        noise_power = np.mean(noise**2)
        target_noise_power = signal_power / (10**(15 / 10))
        scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
        
        self.noisy_signal = self.original_sound + scaled_noise
        self.filtered_signal = None
        self.snr_label.configure(text=f"Before: {self.calculate_snr(self.original_sound, self.noisy_signal):.2f} dB")
        self.update_plots()

    def _validate_order(self):
        try:
            order = int(self.order_entry.get())
            if order <= 0:
                messagebox.showerror("Invalid Input", "Filter order must be a positive whole number.")
                return None
            return order
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Input", "Please enter a valid whole number for the filter order.")
            return None

    def apply_butterworth(self):
        order = self._validate_order()
        if order is None or self.noisy_signal is None:
            if self.noisy_signal is None: messagebox.showwarning("Warning", "Please add noise first.")
            return
        self.active_filter_type = "butter"
        
        cutoff = self.cutoff_slider.get()
        sos = butter(order, cutoff, btype='low', fs=self.fs, output='sos')
        self.filtered_signal = sosfiltfilt(sos, self.noisy_signal)
        self.update_plots_after_filter("Butter")

    def apply_chebyshev(self):
        order = self._validate_order()
        if order is None or self.noisy_signal is None:
            if self.noisy_signal is None: messagebox.showwarning("Warning", "Please add noise first.")
            return
        self.active_filter_type = "cheby"
        
        cutoff = self.cutoff_slider.get()
        sos = cheby1(order, rp=0.5, Wn=cutoff, btype='low', fs=self.fs, output='sos')
        self.filtered_signal = sosfiltfilt(sos, self.noisy_signal)
        self.update_plots_after_filter("Cheby")

    def apply_fir(self):
        order = self._validate_order()
        if order is None or self.noisy_signal is None:
            if self.noisy_signal is None: messagebox.showwarning("Warning", "Please add noise first.")
            return
        self.active_filter_type = "fir"

        fir_order = order * 10
        if fir_order % 2 == 0: fir_order += 1
        
        cutoff = self.cutoff_slider.get()
        taps = firwin(fir_order, cutoff, fs=self.fs, pass_zero='lowpass')
        
        # --- THIS IS THE CORRECTED LINE ---
        self.filtered_signal = filtfilt(taps, 1, self.noisy_signal)
        
        self.update_plots_after_filter("FIR")

    def save_audio(self):
        if self.filtered_signal is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not filepath: return
        sf.write(filepath, self.filtered_signal, self.fs)

    def play_audio(self, data):
        if data is not None: sd.play(data, self.fs)

    def update_cutoff_label(self, value):
        self.cutoff_label.configure(text=f"{int(value)} Hz")

    def calculate_snr(self, signal, noisy_signal):
        noise = noisy_signal - signal
        return 10 * np.log10(np.var(signal) / np.var(noise))
    
    def update_plots_after_filter(self, ftype):
        self.snr_label.configure(text=f"After ({ftype}): {self.calculate_snr(self.original_sound, self.filtered_signal):.2f} dB")
        self.update_plots()

    def update_plots(self):
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax2 = self.fig.add_subplot(3, 1, 2)
        ax3 = self.fig.add_subplot(3, 2, 5)
        ax4 = self.fig.add_subplot(3, 2, 6)
        
        # --- Waveform Plot ---
        if self.original_sound is not None:
            plot_len = min(4000, len(self.original_sound))
            t_axis = np.arange(plot_len) / self.fs
            if self.noisy_signal is not None:
                ax1.plot(t_axis, self.noisy_signal[:plot_len], 'b-', alpha=0.6, label="Noisy")
            if self.filtered_signal is not None:
                ax1.plot(t_axis, self.filtered_signal[:plot_len], 'r-', linewidth=1.5, label="Filtered")
            ax1.plot(t_axis, self.original_sound[:plot_len], 'k-', alpha=0.8, label="Original")
            ax1.legend(loc='upper right')
        ax1.set_title("Time-Domain Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        # --- Spectrogram Plot ---
        plot_data = self.original_sound
        title_str = "Spectrogram (Original)"
        if self.filtered_signal is not None:
            plot_data = self.filtered_signal
            title_str = "Spectrogram (Filtered)"
        elif self.noisy_signal is not None:
            plot_data = self.noisy_signal
            title_str = "Spectrogram (Noisy)"
        
        if plot_data is not None and len(plot_data) > 256:
            f, t, Sxx = spectrogram(plot_data, self.fs, nperseg=256, noverlap=128)
            ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading='gouraud', cmap='viridis')
        ax2.set_title(title_str)
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')

        # --- Filter Analysis Plots ---
        order = self._validate_order()
        if order is not None:
            cutoff = self.cutoff_slider.get()
            
            # Butterworth
            sos_butter = butter(order, cutoff, btype='low', fs=self.fs, output='sos')
            w, h_butter = freqz_sos(sos_butter, worN=8000, fs=self.fs)
            ax3.plot(w, 20 * np.log10(abs(h_butter) + 1e-9), label=f'Butterworth (Order {order})')
            ax4.plot(w, np.unwrap(np.angle(h_butter)), label=f'Butterworth')

            # FIR
            fir_order = order * 10
            if fir_order % 2 == 0: fir_order += 1
            taps_fir = firwin(fir_order, cutoff, fs=self.fs, pass_zero='lowpass')
            w, h_fir = freqz(taps_fir, [1], worN=8000, fs=self.fs)
            ax3.plot(w, 20 * np.log10(abs(h_fir) + 1e-9), label=f'FIR (Order {fir_order-1})')
            ax4.plot(w, np.unwrap(np.angle(h_fir)), label=f'FIR')
            
            ax3.axvline(cutoff, color='r', linestyle='--', alpha=0.7, label='Cutoff Freq')
        else:
            ax3.text(0.5, 0.5, 'Enter a valid filter order', ha='center', va='center', transform=ax3.transAxes)

        ax3.set_title("Filter Frequency Response")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_ylim(-100, 5)
        ax3.legend()
        ax3.grid(True)

        ax4.set_title("Filter Phase Response")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Phase (radians)")
        ax4.legend()
        ax4.grid(True)
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

if __name__ == "__main__":
    app = AdvancedAudioFilterApp()
    app.mainloop()