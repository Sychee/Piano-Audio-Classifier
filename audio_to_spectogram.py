
def load_and_get_stats(filename):
    """Reads .wav file and returns data, sampling frequency, and length (time) of audio clip."""

    import scipy.io.wavfile as siow
    sampling_rate, amplitude_vector = siow.read(filename)

    wav_length = amplitude_vector.shape[0] / sampling_rate

    return sampling_rate, amplitude_vector, wav_length

def plot_wav_curve(filename, sampling_rate, amplitude_vector, wav_length):
    """Plots amplitude curve for a particular audio clip."""

    import matplotlib.pyplot as plt
    import numpy as np
    time = np.linspace(0, wav_length, amplitude_vector.shape[0])

    plt.plot(time, amplitude_vector)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'{filename} - viewed at {sampling_rate} samples/sec')
    plt.show()

def split_audio_into_chunks(sampling_rate, amplitude_vector, chunk_size):
    """Reshape data (amplitude vector) into many chunks of chunk_size miliseconds. Returns reshaped data and leftover data not grouped."""
    
    col_size = int(chunk_size / ((1 / sampling_rate) * 1000))
    whole = int(len(amplitude_vector) / col_size)
    first_partition_index = whole*col_size
    first_partition = amplitude_vector[:first_partition_index]
    second_partition = amplitude_vector[first_partition_index:]
    return first_partition.reshape((whole, col_size)), second_partition

def apply_fourier_transform(chunked_audio):
    """Apply fourier transform to chunked audio snippets to break up each chunk into vector of scores for each frequency band. Aggregates score vectors for each snippet into spectogram to be fed into neural network."""
    pass

if __name__ == '__main__':
    sampling_rate, amplitude_vector, wav_length = load_and_get_stats('hello.wav')
    data, leftovers = split_audio_into_chunks(sampling_rate, amplitude_vector, 20)