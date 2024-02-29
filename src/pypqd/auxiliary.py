import numpy as np

# Minimum disturbance duration for sag, swell and interruption, in cycles
DISTURB_MIN = 2
# Left and right time slot without disturbance, in cycles
DISTURB_EDGE = 1

# Form vectors with constant or random values
def build_vector(interval, rows):
    if len(interval)==1:
        return np.full((rows, 1), interval[0])
    elif len(interval)==2:
        return np.random.uniform(interval[0], interval[1], (rows, 1))

# Forms a matrix with the discrete time base for the calculation of the signals
def build_time_matrix(signals_quant, signal_duration, sampling_freq):
    nodes = int(signal_duration*sampling_freq)
    discrete_time_vector = np.linspace(0, signal_duration, nodes)
    return np.full((signals_quant, nodes), discrete_time_vector)

# Matrix with fixed unit step signals
def unit_step_fixed(interval, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    t = time_matrix[0]
    t1 = interval[1][0]
    t2 = interval[1][1]
    step = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    matrix = np.full((signals_quant, nodes), step)
    return matrix

# Matrix with random unit step signals
def unit_step_random(signal_freq, signal_duration, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    # Obtains the vectors with the start and end times of the disturbance
    duration_min = DISTURB_MIN/signal_freq
    edge = DISTURB_EDGE/signal_freq
    start_range = (edge, signal_duration-duration_min-edge)
    start = np.random.uniform(start_range[0], start_range[1], (signals_quant, 1))
    duration = np.random.uniform(duration_min, signal_duration-edge-start, (signals_quant, 1))
    stop = start + duration
    # Form the matrix
    matrix = np.zeros((signals_quant, nodes))
    for i in range(signals_quant):
        t = time_matrix[i]
        t1 = start[i][0]
        t2 = stop[i][0]
        matrix[i] = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    return matrix      

# Matrix with fixed unit step signals for impulsive transients
def unit_step_fixed_impulsive(interval, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    t = time_matrix[0]
    t1 = interval[1][0]
    t2 = t1 + 0.001
    step = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    matrix = np.full((signals_quant, nodes), step)
    return matrix, np.full((signals_quant, 1), t1)

# Matrix with random unit step signals for impulsive transient
def unit_step_random_impulsive(signal_freq, signal_duration, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    # Obtains the vectors with the start and end times of the disturbance
    duration = 0.001
    start_range = (1/signal_freq, signal_duration-1/signal_freq-duration)
    start = np.random.uniform(start_range[0], start_range[1], (signals_quant, 1))
    stop = start + duration
    # Form the matrix
    matrix = np.zeros((signals_quant, nodes))
    for i in range(signals_quant):
        t = time_matrix[i]
        t1 = start[i][0]
        t2 = stop[i][0]
        matrix[i] = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    return matrix, start      

# Matrix with fixed unit step signals for oscillatory transients
def unit_step_fixed_oscillatory(interval, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    t = time_matrix[0]
    t1 = interval[1][0]
    t2 = interval[1][1]
    step = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    matrix = np.full((signals_quant, nodes), step)
    return matrix, np.full((signals_quant, 1), t1)

# Matrix with random unit step signals for oscillatory transient
def unit_step_random_oscillatory(signal_freq, signal_duration, time_matrix):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    # Obtains the vectors with the start and end times of the disturbance
    duration_min = 0.5/signal_freq
    duration_max = 3/signal_freq
    start_range = (0, signal_duration-duration_max)
    start = np.random.uniform(start_range[0], start_range[1], (signals_quant, 1))
    duration = np.random.uniform(duration_min, duration_max, (signals_quant, 1))
    stop = start + duration
    # Form the matrix
    matrix = np.zeros((signals_quant, nodes))
    for i in range(signals_quant):
        t = time_matrix[i]
        t1 = start[i][0]
        t2 = stop[i][0]
        matrix[i] = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    return matrix, start     

# Matrix with fixed unit step signals for notch
def unit_step_fixed_notch(interval, time_matrix, c, n, fundamental_freq):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    t = time_matrix[0]
    t1 = interval[1][0] + (1/(c*fundamental_freq))*n
    t2 = interval[1][1] + (1/(c*fundamental_freq))*n
    step = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    matrix = np.full((signals_quant, nodes), step)
    return matrix

# Matrix with random unit step signals for notch
def unit_step_random_notch(interval, time_matrix, c, n, fundamental_freq):
    signals_quant = time_matrix.shape[0]
    nodes = time_matrix.shape[1]
    # Obtains the vectors with the start and end times of the disturbance
    duration_min = 0.01/fundamental_freq
    duration_max = 0.05/fundamental_freq
    start_range = (0, (1-0.01*c)/(c*fundamental_freq))
    start = np.random.uniform(start_range[0], start_range[1], (signals_quant, 1))
    duration = np.random.uniform(duration_min, 1/(c*fundamental_freq)-start, (signals_quant, 1))
    mask = duration > duration_max
    duration[mask] = duration_max
    stop = start + duration
    # Form the matrix
    matrix = np.zeros((signals_quant, nodes))
    for i in range(signals_quant):
        t = time_matrix[i]
        t1 = start[i][0] + (1/(c*fundamental_freq))*n
        t2 = stop[i][0] + (1/(c*fundamental_freq))*n
        matrix[i] = np.piecewise(t,[t<t1, (t>=t1) & (t<t2), t>=t2],[0, 1, 0])
    return matrix  
