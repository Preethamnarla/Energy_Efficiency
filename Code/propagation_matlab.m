% propagation_matlab.m
% Run in MATLAB or Octave.
clear;
rng(123);

% Parameters
N = 10000;
d0 = 1;
freq_mhz = 2000;
dist = linspace(1, 500, N);
pl_exp = 3.5;
shadow_std_db = 4.0;

% FSPL and log-distance
fspl_db = 20*log10(freq_mhz) + 20*log10(dist) - 27.55;
extra = 10*(pl_exp - 2).*log10(dist);
PL = fspl_db + extra;

% Show path loss vs distance
figure;
plot(dist, PL);
xlabel('Distance (m)');
ylabel('Path loss (dB)');
title('Log-distance Path Loss');

% Shadowing samples
shadow = randn(N,1) * shadow_std_db;
figure;
hist(shadow, 50);
xlabel('Shadowing (dB)');
ylabel('Count');
title('Shadowing Distribution (Gaussian, dB)');

% Rayleigh fading power (exponential)
ray = exprnd(1, N, 1);
figure;
cdfplot(ray);
xlabel('Power (linear)');
ylabel('CDF');
title('Rayleigh fading power CDF');

% Save figures if desired
print('-dpng','pl_plot.png');
print('-dpng','shadow_hist.png');
print('-dpng','rayleigh_cdf.png');
