%% MATLAB SIMULATION OF A RADAR and TARGET
 
%% Parameters 
% Computer parameters based on user requirements

fc = 12e9; %center frequency of radar
c = 3e8; %speed of light
lambda = c/fc; %wavelength of the signal

% compute the necessary sweep time tm based on maximum range
range_max = 200;
tm = 5.5*range2time(range_max,c);

% compute the bandwidth for the frequency modulated pulse from 
% the range resolution 
range_res = 1; 
bw = range2bw(range_res,c);
sweep_slope = bw/tm;

% determine the maximum doppler and beat frequency from previous 
% parameters
fr_max = range2beat(range_max,sweep_slope,c);
v_max = 230*1000/3600;
fd_max = speed2dop(2*v_max,lambda);
fb_max = fr_max+fd_max;

% the sampling rate is just either the bandwidth or the maximum 
% beat frequency in the signal (times two to maintain nyquist)
fs = max(2*fb_max,bw);

%% Radar Signal
% With all the information above, one can set up the FMCW waveform used
% in the radar system.

% Frequency modulated continuous wave
waveform = phased.FMCWWaveform('SweepTime',tm,'SweepBandwidth',bw,...
    'SampleRate',fs);


%% Target Model

% simulate a target 43 meters away from the radat with a speed of 96 km/h
trgt_dist = 43;
trgt_speed = 96*1000/3600;
trgt_rcs = db2pow(min(10*log10(trgt_dist)+5,20)); %radar cross section

target = phased.RadarTarget('MeanRCS',trgt_rcs,'PropagationSpeed',c,...
    'OperatingFrequency',fc);
trgtmotion = phased.Platform('InitialPosition',[trgt_dist;0;0.5],...
    'Velocity',[trgt_speed;0;0]);

%%
% The propagation model is assumed to be free space.

channel = phased.FreeSpace('PropagationSpeed',c,...
    'OperatingFrequency',fc,'SampleRate',fs,'TwoWayPropagation',true);

%% Radar System Setup

ant_aperture = 6.06e-4;                         % in square meter
ant_gain = aperture2gain(ant_aperture,lambda);  % in dB

tx_ppower = db2pow(5)*1e-3;                     % in watts
tx_gain = 9+ant_gain;                           % in dB

rx_gain = 15+ant_gain;                          % in dB
rx_nf = 4.5;                                    % in dB

transmitter = phased.Transmitter('PeakPower',tx_ppower,'Gain',tx_gain);
receiver = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
    'SampleRate',fs);

%% Radar speed and motion
% Simulate a moving radar (such as an aircraft) travelling 30 Km/h in the
% same direction as the target. the target is moving away therefore the
% relative velocity is ~60m/s

radar_speed = 30*1000/3600;
radarmotion = phased.Platform('InitialPosition',[0;0;0.5],...
    'Velocity',[radar_speed;0;0]);

%% Radar Signal Simulation

% Next, run the simulation loop. 

rng(2012);
Nsweep = 64; % number of sweeps for coherent processing interval

% This is the radar IQ data as complex double. the real part is the 
% in-phase data and imaginary part is the quadrature data (I + jQ)
xr = complex(zeros(waveform.SampleRate*waveform.SweepTime,Nsweep));

for m = 1:Nsweep
    % Update radar and target positions
    [radar_pos,radar_vel] = radarmotion(waveform.SweepTime);
    [tgt_pos,tgt_vel] = trgtmotion(waveform.SweepTime);

    % Transmit FMCW waveform
    sig = waveform();
    txsig = transmitter(sig);
    
    % Propagate the signal and reflect off the target
    txsig = channel(txsig,radar_pos,tgt_pos,radar_vel,tgt_vel);
    txsig = target(txsig);
    
    % Dechirp the received radar return
    txsig = receiver(txsig);    
    dechirpsig = dechirp(txsig,sig);
   
    xr(:,m) = dechirpsig;
end


%% Range and Doppler Estimation

rngdopresp = phased.RangeDopplerResponse('PropagationSpeed',c,...
    'DopplerOutput','Speed','OperatingFrequency',fc,'SampleRate',fs,...
    'RangeMethod','FFT','SweepSlope',sweep_slope,...
    'RangeFFTLengthSource','Property','RangeFFTLength',2048,...
    'DopplerFFTLengthSource','Property','DopplerFFTLength',256);

clf;
plotResponse(rngdopresp,xr);                     % Plot range Doppler map
axis([-v_max v_max 0 range_max])
clim = caxis;


