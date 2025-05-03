import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class AudioNoiserApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Noise Generator")
        master.geometry("900x700")
        
        # Set theme style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize components
        self.audio_path = None
        self.original_audio = None
        self.original_sr = None
        self.noisy_audio = None
        self.playing_thread = None
        self.play_original_stop = False
        self.play_noisy_stop = False
        
        # Create UI components
        self.create_ui()
    
    def create_ui(self):
        """
        Create the user interface
        """
        # Main container frame with padding
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Upper section: Audio upload and controls
        upper_frame = ttk.LabelFrame(main_frame, text="Audio Control")
        upper_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Audio upload button
        upload_frame = ttk.Frame(upper_frame)
        upload_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.upload_button = ttk.Button(upload_frame, text="Upload Audio", command=self.upload_audio)
        self.upload_button.pack(side=tk.TOP, pady=5)
        
        self.audio_path_label = ttk.Label(upload_frame, text="No audio selected", wraplength=200)
        self.audio_path_label.pack(side=tk.TOP, pady=5)
        
        # Noise controls section
        controls_frame = ttk.LabelFrame(upper_frame, text="Noise Controls")
        controls_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Noise type
        noise_type_frame = ttk.Frame(controls_frame)
        noise_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(noise_type_frame, text="Noise Type:").pack(side=tk.LEFT, padx=5)
        
        self.noise_type = tk.StringVar(value="white")
        noise_types = ["white", "pink", "brown", "impulsive", "static", "ambient"]
        self.noise_type_dropdown = ttk.Combobox(noise_type_frame, textvariable=self.noise_type, 
                                               values=noise_types, state="readonly", width=15)
        self.noise_type_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Noise intensity
        intensity_frame = ttk.Frame(controls_frame)
        intensity_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(intensity_frame, text="Noise Intensity:").pack(side=tk.LEFT, padx=5)
        
        self.intensity_var = tk.DoubleVar(value=25.0)
        self.intensity_scale = ttk.Scale(intensity_frame, from_=1, to=100, 
                                        variable=self.intensity_var, length=200)
        self.intensity_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.intensity_label = ttk.Label(intensity_frame, text="25.0%")
        self.intensity_label.pack(side=tk.LEFT, padx=5)
        
        # Update the label when slider changes
        self.intensity_var.trace_add("write", self.update_intensity_label)
        
        # Action buttons
        action_frame = ttk.Frame(upper_frame)
        action_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.apply_noise_button = ttk.Button(action_frame, text="Apply Noise", 
                                           command=self.apply_noise, state=tk.DISABLED)
        self.apply_noise_button.pack(side=tk.TOP, pady=5)
        
        self.save_button = ttk.Button(action_frame, text="Save Noisy Audio", 
                                     command=self.save_audio, state=tk.DISABLED)
        self.save_button.pack(side=tk.TOP, pady=5)
        
        self.reset_button = ttk.Button(action_frame, text="Reset", command=self.reset_app)
        self.reset_button.pack(side=tk.TOP, pady=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Processing Status")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=880, mode='determinate')
        self.progress_bar.pack(padx=5, pady=5, fill=tk.X)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)
        
        # Audio display section
        audio_display_frame = ttk.LabelFrame(main_frame, text="Audio Comparison")
        audio_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Original audio
        original_frame = ttk.LabelFrame(audio_display_frame, text="Original Audio")
        original_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Waveform canvas for original audio
        self.original_fig = plt.Figure(figsize=(4, 2), dpi=100)
        self.original_canvas = FigureCanvasTkAgg(self.original_fig, master=original_frame)
        self.original_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Playback controls for original audio
        original_controls = ttk.Frame(original_frame)
        original_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_original_button = ttk.Button(original_controls, text="Play", 
                                             command=self.play_original, state=tk.DISABLED)
        self.play_original_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_original_button = ttk.Button(original_controls, text="Stop", 
                                             command=self.stop_original, state=tk.DISABLED)
        self.stop_original_button.pack(side=tk.LEFT, padx=5)
        
        # Noisy audio
        noisy_frame = ttk.LabelFrame(audio_display_frame, text="Noisy Audio")
        noisy_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Waveform canvas for noisy audio
        self.noisy_fig = plt.Figure(figsize=(4, 2), dpi=100)
        self.noisy_canvas = FigureCanvasTkAgg(self.noisy_fig, master=noisy_frame)
        self.noisy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Playback controls for noisy audio
        noisy_controls = ttk.Frame(noisy_frame)
        noisy_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_noisy_button = ttk.Button(noisy_controls, text="Play", 
                                          command=self.play_noisy, state=tk.DISABLED)
        self.play_noisy_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_noisy_button = ttk.Button(noisy_controls, text="Stop", 
                                          command=self.stop_noisy, state=tk.DISABLED)
        self.stop_noisy_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        audio_display_frame.grid_columnconfigure(0, weight=1)
        audio_display_frame.grid_columnconfigure(1, weight=1)
        audio_display_frame.grid_rowconfigure(0, weight=1)
    
    def update_intensity_label(self, *args):
        """
        Update the intensity label when slider changes
        """
        value = self.intensity_var.get()
        self.intensity_label.config(text=f"{value:.1f}%")
    
    def upload_audio(self):
        """
        Upload and display an audio file
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg")]
        )
        
        if not file_path:
            return
        
        try:
            self.audio_path = file_path
            
            # Load audio file
            self.update_status("Loading audio file...")
            self.progress_bar['value'] = 10
            self.master.update_idletasks()
            
            audio, sr = librosa.load(file_path, sr=None)
            self.original_audio = audio
            self.original_sr = sr
            
            self.progress_bar['value'] = 80
            
            # Display audio info
            duration = len(audio) / sr
            self.audio_path_label.config(text=f"File: {os.path.basename(file_path)}\nDuration: {duration:.2f}s\nSample Rate: {sr}Hz")
            
            # Display original waveform
            self.display_waveform(audio, self.original_fig, self.original_canvas)
            
            # Enable apply noise button
            self.apply_noise_button.config(state=tk.NORMAL)
            self.play_original_button.config(state=tk.NORMAL)
            self.stop_original_button.config(state=tk.NORMAL)
            
            # Update status
            self.progress_bar['value'] = 100
            self.update_status("Audio loaded successfully. Select noise type and intensity, then click 'Apply Noise'.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
            self.progress_bar['value'] = 0
            self.update_status("Error loading audio")
    
    def display_waveform(self, audio, figure, canvas):
        """
        Display audio waveform on canvas
        """
        figure.clear()
        ax = figure.add_subplot(111)
        
        # Plot waveform
        ax.plot(np.linspace(0, len(audio)/self.original_sr, len(audio)), audio, color='blue', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.3)
        
        figure.tight_layout()
        canvas.draw()
    
    def play_audio(self, audio, sr, stop_var_name):
        """
        Thread function to play audio
        """
        try:
            import sounddevice as sd
            sd.play(audio, sr)
            
            # Check if we need to stop playing
            while sd.get_stream().active:
                if getattr(self, stop_var_name):
                    sd.stop()
                    break
                time.sleep(0.1)
            
            # Reset stop flag
            setattr(self, stop_var_name, False)
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def play_original(self):
        """
        Play original audio
        """
        if self.original_audio is None:
            return
        
        self.play_original_stop = False
        self.playing_thread = threading.Thread(
            target=self.play_audio, 
            args=(self.original_audio, self.original_sr, 'play_original_stop')
        )
        self.playing_thread.daemon = True
        self.playing_thread.start()
        
        self.update_status("Playing original audio...")
    
    def stop_original(self):
        """
        Stop playing original audio
        """
        self.play_original_stop = True
        self.update_status("Playback stopped.")
    
    def play_noisy(self):
        """
        Play noisy audio
        """
        if self.noisy_audio is None:
            return
        
        self.play_noisy_stop = False
        self.playing_thread = threading.Thread(
            target=self.play_audio, 
            args=(self.noisy_audio, self.original_sr, 'play_noisy_stop')
        )
        self.playing_thread.daemon = True
        self.playing_thread.start()
        
        self.update_status("Playing noisy audio...")
    
    def stop_noisy(self):
        """
        Stop playing noisy audio
        """
        self.play_noisy_stop = True
        self.update_status("Playback stopped.")
    
    def apply_noise(self):
        """
        Apply the selected noise to the audio
        """
        if self.original_audio is None:
            return
        
        try:
            noise_type = self.noise_type.get()
            intensity = self.intensity_var.get() / 100.0  # Convert percentage to 0-1 scale
            
            self.update_status(f"Applying {noise_type} noise...")
            self.progress_bar['value'] = 10
            self.master.update_idletasks()
            
            # Apply noise based on selected type
            noisy_audio = self.add_noise(self.original_audio, noise_type, intensity)
            
            self.progress_bar['value'] = 80
            self.master.update_idletasks()
            
            # Store noisy audio
            self.noisy_audio = noisy_audio
            
            # Display noisy waveform
            self.display_waveform(noisy_audio, self.noisy_fig, self.noisy_canvas)
            
            # Enable save and play buttons
            self.save_button.config(state=tk.NORMAL)
            self.play_noisy_button.config(state=tk.NORMAL)
            self.stop_noisy_button.config(state=tk.NORMAL)
            
            # Update status
            self.progress_bar['value'] = 100
            self.update_status(f"{noise_type.title()} noise applied. You can save the noisy audio or try different settings.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply noise: {str(e)}")
            self.progress_bar['value'] = 0
            self.update_status("Error during noise application")
    
    def add_noise(self, audio, noise_type, intensity):
        """
        Add noise to the audio
        
        Args:
            audio: Audio signal
            noise_type: Type of noise to apply
            intensity: Intensity of noise (0-1)
            
        Returns:
            noisy_audio: Audio with noise applied
        """
        # Make a copy to avoid modifying the original
        noisy_audio = audio.copy()
        
        # Calculate the appropriate noise level
        signal_power = np.mean(audio ** 2)
        
        if noise_type == "white":
            # White noise (uniform across the frequency spectrum)
            noise = np.random.normal(0, 1, len(audio))
            noise_power = np.mean(noise ** 2)
            scaling_factor = np.sqrt(signal_power * intensity / noise_power)
            noise = scaling_factor * noise
            noisy_audio = noisy_audio + noise
            
        elif noise_type == "pink":
            # Pink noise (power inversely proportional to frequency)
            noise = np.random.normal(0, 1, len(audio))
            
            # Apply pink filter by working in frequency domain
            N = len(noise)
            X = np.fft.rfft(noise)
            S = np.fft.rfftfreq(N)
            S[S == 0] = 1  # Avoid division by zero
            
            # Pink noise has 1/f power spectrum (amplitude proportional to 1/sqrt(f))
            X = X / np.sqrt(S)
            noise = np.fft.irfft(X, n=N)
            
            # Normalize and scale
            noise = noise / np.std(noise)
            noise_power = np.mean(noise ** 2)
            scaling_factor = np.sqrt(signal_power * intensity / noise_power)
            noise = scaling_factor * noise
            
            # Add noise
            noisy_audio = noisy_audio + noise
            
        elif noise_type == "brown":
            # Brown noise (power decreases with frequency squared)
            noise = np.random.normal(0, 1, len(audio))
            
            # Apply brown filter by working in frequency domain
            N = len(noise)
            X = np.fft.rfft(noise)
            S = np.fft.rfftfreq(N)
            S[S == 0] = 1  # Avoid division by zero
            
            # Brown noise has 1/f^2 power spectrum
            X = X / S
            noise = np.fft.irfft(X, n=N)
            
            # Normalize and scale
            noise = noise / np.std(noise)
            noise_power = np.mean(noise ** 2)
            scaling_factor = np.sqrt(signal_power * intensity / noise_power)
            noise = scaling_factor * noise
            
            # Add noise
            noisy_audio = noisy_audio + noise
            
        elif noise_type == "impulsive":
            # Impulsive noise (random spikes)
            num_impulses = int(len(audio) * intensity * 0.05)  # Adjust frequency of impulses
            impulse_positions = np.random.randint(0, len(audio), num_impulses)
            impulse_amplitudes = np.random.uniform(0.5, 1.0, num_impulses) * intensity * 2
            
            for pos, amp in zip(impulse_positions, impulse_amplitudes):
                # Add positive or negative impulse
                noisy_audio[pos] = noisy_audio[pos] + (np.random.choice([-1, 1]) * amp)
            
        elif noise_type == "static":
            # Static noise (like radio static)
            # Mix of impulsive and white noise
            
            # Add white noise component
            white_noise = np.random.normal(0, 1, len(audio))
            white_noise_power = np.mean(white_noise ** 2)
            white_scaling = np.sqrt(signal_power * intensity * 0.3 / white_noise_power)
            noisy_audio = noisy_audio + white_scaling * white_noise
            
            # Add crackling/popping component
            num_crackles = int(len(audio) * intensity * 0.02)
            crackle_positions = np.random.randint(0, len(audio), num_crackles)
            crackle_lengths = np.random.randint(5, 50, num_crackles)
            crackle_amplitudes = np.random.uniform(0.2, 0.8, num_crackles) * intensity
            
            for pos, length, amp in zip(crackle_positions, crackle_lengths, crackle_amplitudes):
                end_pos = min(pos + length, len(audio))
                noisy_audio[pos:end_pos] = noisy_audio[pos:end_pos] + amp * np.random.normal(0, 1, end_pos - pos)
        
        elif noise_type == "ambient":
            # Ambient noise (simulating background noise)
            # Combination of colored noise with a low-pass filter
            
            # Create pink noise base
            noise = np.random.normal(0, 1, len(audio))
            N = len(noise)
            X = np.fft.rfft(noise)
            S = np.fft.rfftfreq(N)
            S[S == 0] = 1
            X = X / np.sqrt(S)  # Pink noise spectrum
            noise = np.fft.irfft(X, n=N)
            
            # Apply low-pass filter to simulate ambient characteristics
            b, a = self.butter_lowpass(1000, self.original_sr, order=2)
            noise = self.apply_filter(noise, b, a)
            
            # Normalize and apply intensity
            noise = noise / np.std(noise)
            noise_power = np.mean(noise ** 2)
            scaling_factor = np.sqrt(signal_power * intensity / noise_power)
            noise = scaling_factor * noise
            
            # Add to original audio
            noisy_audio = noisy_audio + noise
        
        # Clip values to avoid distortion
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return noisy_audio
    
    def butter_lowpass(self, cutoff, fs, order=5):
        """
        Design a low-pass filter for ambient noise simulation
        """
        from scipy import signal
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def apply_filter(self, data, b, a):
        """
        Apply a filter to data
        """
        from scipy import signal
        return signal.filtfilt(b, a, data)
    
    def save_audio(self):
        """
        Save the noisy audio
        """
        if self.noisy_audio is None:
            return
        
        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get original filename without extension
        original_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        noise_type = self.noise_type.get()
        intensity = self.intensity_var.get()
        
        # Create default filename
        default_filename = f"{original_name}_{noise_type}_{int(intensity)}pct_{timestamp}.wav"
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("FLAC files", "*.flac"),
                ("All files", "*.*")
            ],
            initialfile=default_filename
        )
        
        if save_path:
            try:
                self.update_status("Saving audio file...")
                self.progress_bar['value'] = 50
                
                sf.write(save_path, self.noisy_audio, self.original_sr)
                
                self.progress_bar['value'] = 100
                messagebox.showinfo("Success", f"Noisy audio saved successfully to:\n{save_path}")
                self.update_status("File saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio: {str(e)}")
                self.update_status("Error saving file.")
    
    def update_status(self, message):
        """
        Update status message
        """
        self.status_label.config(text=message)
        self.master.update_idletasks()
    
    def reset_app(self):
        """
        Reset the application state
        """
        # Stop any playing audio
        self.play_original_stop = True
        self.play_noisy_stop = True
        
        # Reset variables
        self.audio_path = None
        self.original_audio = None
        self.original_sr = None
        self.noisy_audio = None
        
        # Reset UI elements
        self.audio_path_label.config(text="No audio selected")
        
        # Clear waveform displays
        self.original_fig.clear()
        self.original_canvas.draw()
        self.noisy_fig.clear()
        self.noisy_canvas.draw()
        
        # Reset noise controls
        self.noise_type.set("white")
        self.intensity_var.set(25.0)
        
        # Reset buttons
        self.apply_noise_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.play_original_button.config(state=tk.DISABLED)
        self.stop_original_button.config(state=tk.DISABLED)
        self.play_noisy_button.config(state=tk.DISABLED)
        self.stop_noisy_button.config(state=tk.DISABLED)
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.update_status("Application reset. Ready to start.")

def main():
    # Set high DPI awareness for Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    root = tk.Tk()
    app = AudioNoiserApp(root)
    root.mainloop()

if __name__ == '__main__':
    main() 