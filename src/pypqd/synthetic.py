"""Module that provides functions for synthetic signals generation."""
import numpy as np
from pypqd.auxiliary import (build_vector, build_time_matrix, unit_step_fixed,
                             unit_step_random, unit_step_fixed_notch, unit_step_random_notch,
                             unit_step_fixed_impulsive, unit_step_random_impulsive,
                             unit_step_fixed_oscillatory, unit_step_random_oscillatory) 

# Normal signal dataset generator
def normal(amplitude=(1, ), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1):
    """Generates normal synthetic signals.
    
    Parameters
    ----------
    amplitude : tuple, optional
        Normal signal amplitude, in volts or per unit. The default is (1, ).
    fundamental_freq : int, optional
        Normal signal fundamental frequency, in hertz. The default is 50.
    phase_angle : tuple, optional
        Normal signal phase angle, in radians. The default is (-np.pi, np.pi).
    signal_duration : float, optional
        Synthetic signal duration, in seconds. The default is 0.2.
    sampling_freq : int, optional
        Sampling frequency of the synthetic signal, in hertz. The default is 3200.
    signals_quant : int, optional
        Amount of synthetic signals generated. The default is 1.

    Returns
    -------
    signals_list : list of dictionaries
        A list in which each normal signal generated is represented with a dictionary.
    
    Notes
    -----
    The normal signal is generated according to the model proposed in [1]_:
        
        .. math:: v(t) = A\sin(\omega t - \phi).  
        
    References
    ----------
    .. [1] Igual, R., Medrano, C., Arcega, F. J., & Mantescu, G. (2018). Integral
       mathematical model of power quality disturbances. Proceedings of International
       Conference on Harmonics and Quality of Power, ICHQP, 2018-May, 1–6.
       https://doi.org/10.1109/ICHQP.2018.8378902
    
    Examples
    --------
    >>> signal = synthetic.normal()
    >>> time, samples = signal[0]["Time"], signal[0]["Values"]
    >>> plt.plot(time, samples)
    
    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Multiply the pure signal by the amplitude
    complete_signals = amplitude_vector*pure_signals
    # Output formatting
    signals_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Normal",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle": phase_angle_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signals_list.append(single_signal)
    return signals_list

# Sag signal dataset generator
def sag(amplitude=(1,), fundamental_freq=50,phase_angle=(-np.pi, np.pi),
        signal_duration=0.2, sampling_freq=3200, signals_quant=1,
        sag_magnitude = (0.1, 0.9), sag_interval=("random", (0.02, 0.18))):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (1,).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    sag_magnitude : TYPE, optional
        DESCRIPTION. The default is (0.1, 0.9).
    sag_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0.02, 0.18)).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    sag_magnitude_vector = build_vector(sag_magnitude, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration,
                                             sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-
                          phase_angle_vector)
    # Disturbances
    if sag_interval[0]=="random":
        unit_step = unit_step_random(fundamental_freq, signal_duration,
                                     discrete_time_matrix)
    elif sag_interval[0]=="fixed":
        unit_step = unit_step_fixed(sag_interval, discrete_time_matrix)
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*pure_signals*(1-sag_magnitude_vector*
                                                      unit_step)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Sag",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Sag magnitude" : sag_magnitude_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Interruption signal dataset generator
def interruption(amplitude=(1,), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1,
        interruption_magnitude = (0.9, 1), interruption_interval=("random", (0.02, 0.18))):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (1,).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    interruption_magnitude : TYPE, optional
        DESCRIPTION. The default is (0.9, 1).
    interruption_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0.02, 0.18)).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    interruption_magnitude_vector = build_vector(interruption_magnitude, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    if interruption_interval[0]=="random":
        unit_step = unit_step_random(fundamental_freq, signal_duration, discrete_time_matrix)
    elif interruption_interval[0]=="fixed":
        unit_step = unit_step_fixed(interruption_interval, discrete_time_matrix)
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*pure_signals*(1-interruption_magnitude_vector*unit_step)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Interruption",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Interruption magnitude" : interruption_magnitude_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Swell signal dataset generator
def swell(amplitude=(1,), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1,
        swell_magnitude = (0.1, 0.8), swell_interval=("random", (0.02, 0.18))):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (1,).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    swell_magnitude : TYPE, optional
        DESCRIPTION. The default is (0.1, 0.8).
    swell_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0.02, 0.18)).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    swell_magnitude_vector = build_vector(swell_magnitude, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    if swell_interval[0]=="random":
        unit_step = unit_step_random(fundamental_freq, signal_duration, discrete_time_matrix)
    elif swell_interval[0]=="fixed":
        unit_step = unit_step_fixed(swell_interval, discrete_time_matrix)
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*pure_signals*(1+swell_magnitude_vector*unit_step)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Swell",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Swell magnitude" : swell_magnitude_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Voltage fluctuation signal dataset generator
def fluctuation(amplitude=(0.95, 1.05), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1,
        fluctuation_amplitude = (0.05, 0.1), fluctuation_freq=(8, 25)):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (0.95, 1.05).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    fluctuation_amplitude : TYPE, optional
        DESCRIPTION. The default is (0.05, 0.1).
    fluctuation_freq : TYPE, optional
        DESCRIPTION. The default is (8, 25).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    fluct_amplitude_vector = build_vector(fluctuation_amplitude, signals_quant)
    fluct_freq_vector = build_vector(fluctuation_freq, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    disturb_signal = np.sin(2*np.pi*fluct_freq_vector*discrete_time_matrix)
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*pure_signals*(1+fluct_amplitude_vector*disturb_signal)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Voltage fluctuation",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Fluctuation amplitude" : fluct_amplitude_vector[i][0],
                         "Fluctuation frequency" : fluct_freq_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Harmonic signal dataset generator
def harmonics(amplitude=(0.95, 1.05), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1, harmonic_number=(3, 5, 7),
        harmonic_amplitude=(0.05, 0.15), harmonic_phase=(-np.pi, np.pi)):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (0.95, 1.05).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    harmonic_number : TYPE, optional
        DESCRIPTION. The default is (3, 5, 7).
    harmonic_amplitude : TYPE, optional
        DESCRIPTION. The default is (0.05, 0.15).
    harmonic_phase : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).

    Returns
    -------
    signals_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    disturb_signal = np.zeros_like(discrete_time_matrix)
    for i in range(len(harmonic_number)):
        amplitude = build_vector(harmonic_amplitude, signals_quant)
        n = harmonic_number[i]
        phase = build_vector(harmonic_phase, signals_quant)
        disturb_signal += amplitude*np.sin(n*2*np.pi*fundamental_freq*discrete_time_matrix-phase)
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*(pure_signals + disturb_signal)
    # Output formatting
    signals_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Harmonics",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Harmonic number" : harmonic_number,
                         "Sampling frequency" : sampling_freq}
        signals_list.append(single_signal)
    return signals_list

# Notch signal dataset generator
def notch(amplitude=(0.95, 1.05), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1, notch_constant=[1, 2, 4, 6],
        notch_amplitude=(0.1, 0.4), notch_interval=("random", (0, 0.001))):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (0.95, 1.05).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    notch_constant : TYPE, optional
        DESCRIPTION. The default is [1, 2, 4, 6].
    notch_amplitude : TYPE, optional
        DESCRIPTION. The default is (0.1, 0.4).
    notch_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0, 0.001)).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    notch_amplitude_vector = build_vector(notch_amplitude, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbance
    signe_matrix = np.sign(pure_signals)
    c = np.random.choice(notch_constant)
    disturb_signal = np.zeros_like(discrete_time_matrix)    
    for i in range(int(signal_duration*fundamental_freq*c+1)):
        if notch_interval[0]=="random":
            unit_step = unit_step_random_notch(notch_interval, discrete_time_matrix,
                                              c, i, fundamental_freq)
        elif notch_interval[0]=="fixed":
            unit_step = unit_step_fixed_notch(notch_interval, discrete_time_matrix,
                                              c, i, fundamental_freq)
        disturb_signal += unit_step
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*(pure_signals-signe_matrix*(notch_amplitude_vector*disturb_signal))
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Notch",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Notch amplitude" : notch_amplitude_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Impulsive transient signal dataset generator
def impulsive(amplitude=(0.95, 1.05), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1,
        impulsive_amplitude = (0.222, 1.11), impulsive_interval=("random", (0.02, 0.18))):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (0.95, 1.05).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    impulsive_amplitude : TYPE, optional
        DESCRIPTION. The default is (0.222, 1.11).
    impulsive_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0.02, 0.18)).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    impulsive_amplitude_vector = build_vector(impulsive_amplitude, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    if impulsive_interval[0]=="random":
        unit_step, t1 = unit_step_random_impulsive(fundamental_freq, signal_duration, discrete_time_matrix)
    elif impulsive_interval[0]=="fixed":
        unit_step, t1 = unit_step_fixed_impulsive(impulsive_interval, discrete_time_matrix)
    disturb_signal = impulsive_amplitude_vector*(np.exp(-750*(discrete_time_matrix-t1))-
                                                 np.exp(-344*(discrete_time_matrix-t1)))*unit_step
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*(pure_signals-disturb_signal)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Impulsive transient",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Impulsive transient amplitude" : impulsive_amplitude_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list

# Oscillatory transient signal dataset generator
def oscillatory(amplitude=(0.95, 1.05), fundamental_freq=50,
        phase_angle=(-np.pi, np.pi), signal_duration=0.2,
        sampling_freq=3200, signals_quant=1,
        oscillatory_amplitude = (0.1, 0.8), oscillatory_interval=("random", (0.01, 0.06)),
        oscillatory_phase_angle=(-np.pi, np.pi), oscillatory_freq=(300, 900),
        oscillatory_time_constant=(0.008, 0.040)):
    """
    

    Parameters
    ----------
    amplitude : TYPE, optional
        DESCRIPTION. The default is (0.95, 1.05).
    fundamental_freq : TYPE, optional
        DESCRIPTION. The default is 50.
    phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    signal_duration : TYPE, optional
        DESCRIPTION. The default is 0.2.
    sampling_freq : TYPE, optional
        DESCRIPTION. The default is 3200.
    signals_quant : TYPE, optional
        DESCRIPTION. The default is 1.
    oscillatory_amplitude : TYPE, optional
        DESCRIPTION. The default is (0.1, 0.8).
    oscillatory_interval : TYPE, optional
        DESCRIPTION. The default is ("random", (0.01, 0.06)).
    oscillatory_phase_angle : TYPE, optional
        DESCRIPTION. The default is (-np.pi, np.pi).
    oscillatory_freq : TYPE, optional
        DESCRIPTION. The default is (300, 900).
    oscillatory_time_constant : TYPE, optional
        DESCRIPTION. The default is (0.008, 0.040).

    Returns
    -------
    signal_list : TYPE
        DESCRIPTION.

    """
    # Constant or random vectors
    amplitude_vector = build_vector(amplitude, signals_quant)
    phase_angle_vector = build_vector(phase_angle, signals_quant)
    oscillatory_amplitude_vector = build_vector(oscillatory_amplitude, signals_quant)
    time_constant_vector = build_vector(oscillatory_time_constant, signals_quant)
    oscillatory_freq_vector = build_vector(oscillatory_freq, signals_quant)
    oscillatory_phase_vector = build_vector(oscillatory_phase_angle, signals_quant)
    # Discrete time matrix
    discrete_time_matrix = build_time_matrix(signals_quant, signal_duration, sampling_freq)
    # Compute the matrix with pure signal sin(wt + phi)
    pure_signals = np.sin(2*np.pi*fundamental_freq*discrete_time_matrix-phase_angle_vector)
    # Disturbances
    if oscillatory_interval[0]=="random":
        unit_step, t1 = unit_step_random_oscillatory(fundamental_freq, signal_duration, discrete_time_matrix)
    elif oscillatory_interval[0]=="fixed":
        unit_step, t1 = unit_step_fixed_oscillatory(oscillatory_interval, discrete_time_matrix)
    disturb_signal_factor1 = oscillatory_amplitude_vector*np.exp(-(discrete_time_matrix-t1)/time_constant_vector)
    disturb_signal_factor2 = np.sin(2*np.pi*oscillatory_freq_vector*(discrete_time_matrix-t1)-oscillatory_phase_vector)
    disturb_signal = disturb_signal_factor1*disturb_signal_factor2*unit_step
    # Combines the pure signal with the disturbance
    complete_signals = amplitude_vector*(pure_signals+disturb_signal)
    # Output formatting
    signal_list = []
    for i in range(signals_quant):
        single_signal = {"Label" : "Oscillatory transient",
                         "Time" : discrete_time_matrix[0],
                         "Values" : complete_signals[i],
                         "Amplitude" : amplitude_vector[i][0],
                         "Frequency" : fundamental_freq,
                         "Phase angle" : phase_angle_vector[i][0],
                         "Oscillatory transient amplitude" : oscillatory_amplitude_vector[i][0],
                         "Oscillatory transient frequency" : oscillatory_freq_vector[i][0],
                         "Oscillatory transient phase angle" : oscillatory_phase_vector[i][0],
                         "Sampling frequency" : sampling_freq}
        signal_list.append(single_signal)
    return signal_list
