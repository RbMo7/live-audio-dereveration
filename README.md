DSP Audio De-reverberation Toolkit

This project provides a suite of Python applications to demonstrate and solve the problem of audio reverberation (echo) using fundamental Digital Signal Processing (DSP) techniques. The core of the solution is based on modeling room acoustics as a Linear Time-Invariant (LTI) system and applying deconvolution via an inverse filter.

The repository contains two main applications:

    audio_demo_pre.py: A demonstration tool that uses pre-generated signals (sine, square, speech) to clearly visualize the process of adding and removing a known, artificial echo.

    audio_processor.py: A real-time application that attempts to remove actual room echo from a live microphone feed after a manual calibration step.

How It Works: The Theory

The application is built on the principle that a reverberant signal is the result of a clean audio source being convolved with the room's unique acoustic signature, known as the Room Impulse Response (RIR).
Echoed Signal=Clean Signal∗RIR

By applying the Convolution Theorem, we can move to the frequency domain where this relationship becomes simple multiplication. To remove the echo, we can therefore perform deconvolution by convolving the echoed signal with a calculated inverse filter. This project uses a regularized inverse filter to ensure stability and avoid amplifying noise.
Restored Signal=Echoed Signal∗Inverse Filter
Features
audio_demo_pre.py (Demonstration Tool)

    Signal Generation: Create clean test signals including pure Sine Waves, Square Waves (rich in harmonics), and a complex Speech-like sample.

    Controlled Echo Simulation: Adds a consistent, artificial echo to any generated signal to create a perfect test case.

    One-Click Restoration: Applies the calculated inverse filter to remove the artificial echo.

    Detailed Visualization: A four-quadrant plot shows:

        The original clean signal.

        The signal with the echo added.

        The final, restored signal.

        A Power Spectral Density (PSD) plot comparing the echoed and restored signals to analyze the frequency-domain effects.

    Audio Playback: Dedicated buttons to play the clean, echoed, and restored audio for auditory comparison.

audio_processor.py (Live Processor)

    Real-Time Calibration: A "Clap to Calibrate" feature records the room's acoustic signature (RIR).

    Live Inverse Filtering: Applies the calculated inverse filter to the live microphone audio stream.

    Live Visualization: Real-time plots show the incoming (reverberant) and outgoing (cleaned) audio waveforms.

    Echo Test Mode: Includes a built-in test to simulate and remove a fake echo, allowing for testing in any environment.

Setup and Installation
1. Clone the Repository

git clone <your-repository-url>
cd <your-repository-name>

2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

On Windows:

python -m venv myenv
myenv\Scripts\activate

On macOS/Linux:

python3 -m venv myenv
source myenv/bin/activate

3. Install Dependencies

Install all the required packages using the requirements.txt file.

pip install -r requirements.txt

How to Run the Applications

Make sure your virtual environment is activated before running.
To run the demonstration tool:

python audio_demo_pre.py

Usage:

    Click one of the "Generate Signal" buttons (e.g., "Sine Wave").

    Observe the "Original" and "With Echo" plots.

    Click "Apply De-reverberation Filter".

    Observe the "Restored Signal" and "Power Spectral Density" plots.

    Use the playback buttons to hear the results.

[Insert Screenshot of the Demonstration App Here]
To run the live processor:

python audio_processor.py

Usage:

    To test in a real room, click "Clap to Calibrate" and make a single, loud clap near your microphone.

    Alternatively, click "OR: Run Echo Test" to use the built-in test case.

    Click "Start Processing" and speak into your microphone.

    Observe the live plots and listen to the processed output from your speakers.

    Click "Stop Processing" when finished.

[Insert Screenshot of the Live Processor App Here]
Limitations

    Known RIR Required: The system is not "blind" and relies on a clean measurement of the Room Impulse Response.

    Noise Amplification: The inverse filtering process can amplify existing background noise.

    Static Environment: The calibration is only valid for a static room. Any change in the room's acoustics requires re-calibration.

    Positional Dependence: The filter is specific to the exact location of the microphone and sound source during calibration.
