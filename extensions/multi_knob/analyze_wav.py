import wave
import os
from pathlib import Path
from tabulate import tabulate
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def read_wav(filepath):
    with wave.open(filepath, 'rb') as wav:
        # Get basic properties
        info = {
            'filename': os.path.basename(filepath),
            'channels': wav.getnchannels(),
            'sample_width': wav.getsampwidth(),
            'frame_rate': wav.getframerate(),
            'frames': wav.getnframes(),
            'duration': wav.getnframes() / wav.getframerate()
        }
        
        # Read the actual audio data
        frames = wav.readframes(wav.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # Normalize the audio data to [-1, 1]
        audio_data = audio_data / 32768.0
        
        return info, audio_data

def plot_comparison(di_info, di_data, drive_info, drive_data, output_dir):
    # Calculate sample difference
    sample_diff = len(di_data) - len(drive_data)
    ms_diff = sample_diff / di_info['frame_rate'] * 1000
    
    # Create aligned drive signal
    if sample_diff > 0:
        aligned_drive = np.pad(drive_data, (sample_diff, 0))
    else:
        aligned_drive = drive_data[-len(di_data):]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Original Waveform Comparison',
            'Aligned Signals (Drive pre-padded)',
            'First 1000 samples',
            'Last 1000 samples (aligned)',
        ),
        specs=[[{"colspan": 2}, None],
               [{"colspan": 2}, None],
               [{}, {}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add traces for original signals
    fig.add_trace(
        go.Scatter(
            y=di_data,
            name='DI',
            line=dict(color='blue', width=1),
            opacity=0.7,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=drive_data,
            name=f'{drive_info["filename"]} ({ms_diff:.1f}ms shorter)',
            line=dict(color='red', width=1),
            opacity=0.7,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add traces for aligned signals
    fig.add_trace(
        go.Scatter(
            y=di_data,
            name='DI',
            line=dict(color='blue', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=aligned_drive,
            name=f'{drive_info["filename"]} (aligned)',
            line=dict(color='red', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add traces for first 1000 samples
    fig.add_trace(
        go.Scatter(
            y=di_data[:1000],
            name='DI',
            line=dict(color='blue', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=drive_data[:1000],
            name=drive_info['filename'],
            line=dict(color='red', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add traces for last 1000 samples (aligned)
    fig.add_trace(
        go.Scatter(
            y=di_data[-1000:],
            name='DI',
            line=dict(color='blue', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            y=aligned_drive[-1000:],
            name=f'{drive_info["filename"]} (aligned)',
            line=dict(color='red', width=1),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        width=1500,
        title_text=f"Waveform Comparison: DI vs {drive_info['filename']}",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Sample")
    fig.update_yaxes(title_text="Amplitude")
    
    # Save the interactive HTML file
    output_file = os.path.join(output_dir, f'comparison_DI_vs_{drive_info["filename"].replace(" ", "_")}')
    fig.write_html(output_file + '.html')

# Get all wav files in the directory
directory = "extensions/HRD III Drive channel modeling"
wav_files = sorted([f for f in Path(directory).glob("*.wav")])

# Analyze all files and store audio data
results = []
audio_data = {}
max_length = 0

for wav_file in wav_files:
    info, data = read_wav(str(wav_file))
    results.append(info)
    audio_data[info['filename']] = data
    max_length = max(max_length, len(data))

# Create table
headers = ['Filename', 'Frames', 'Duration (s)', 'Sample Rate', 'Channels']
table_data = [[
    r['filename'],
    r['frames'],
    f"{r['duration']:.3f}",
    r['frame_rate'],
    r['channels']
] for r in results]

# Print table
print("\nWAV File Analysis:")
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Print length differences
print("\nLength Differences (compared to shortest file):")
min_frames = min(r['frames'] for r in results)
for r in results:
    diff = r['frames'] - min_frames
    if diff > 0:
        print(f"{r['filename']}: +{diff} frames ({diff/r['frame_rate']*1000:.2f}ms longer)")

# Get DI file info and data
di_file = next(r for r in results if r['filename'] == 'DI.wav')
di_data = audio_data[di_file['filename']]

# Create individual comparisons
output_dir = 'wav_comparison'
for info in results:
    if info['filename'] != 'DI.wav':
        drive_data = audio_data[info['filename']]
        plot_comparison(di_file, di_data, info, drive_data, output_dir)
        print(f"\nCreated interactive comparison plot for DI vs {info['filename']}") 