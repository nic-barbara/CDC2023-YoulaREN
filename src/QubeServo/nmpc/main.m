% Designing a nonlinear MPC controller for the Qube Servo 2
% system. 
% 
% This code was written by Ruigang Wang and adapted 
% by Nicholas Barbara for our CDC 2023 submission.
%
% Email: nicholas.barbara@sydney.edu.au

close all;
clear;
clc;

% Environment parameters
dt = 0.02; 
max_steps = 100; 
x0_lims = [pi; pi; 0.5; 0.5];

% NMPC horizon and initial plant conditions
H = 22;
verbose = false;
x0_data = load("initial_conditions.mat");
x0_batch = x0_data.x0;

% Set up a figure
T = max_steps;
figure();
lwidth = 0.25;
colour = 0.7*[1, 1, 1];

p1 = subplot(311);
hold on;
ylabel('theta')

p2 = subplot(312);
hold on;
ylabel('alpha')

p3 = subplot(313);
hold on;
ylabel('input');
xlabel('time');

% Loop over many initial conditions
Jtot = 0;
nsims = size(x0_batch,2);
for k = 1:nsims

    % Inform the user
    fprintf("Starting sim %d of %d... ", k, nsims)

    % Run the closed-loop system
    x0 = x0_batch(:,k);
    [J, xt, ut, tm] = nmpc(x0, dt, H, max_steps, verbose);
    
    % Print final cost
    Jtot = Jtot + J;
    fprintf("cost: %.2f\n", J);
    
    % Plot results
    T = max_steps;
    plot(p1, tm,xt(1,:),'Linewidth',lwidth,"Color",colour); 
    plot(p2, tm,xt(2,:),'Linewidth',lwidth,"Color",colour); 
    plot(p3, tm,ut,'Linewidth',lwidth,"Color",colour); 
end

% Plot extras
plot(p1, [0,T*dt],[0.0, 0.0],'k-.');

plot(p2, [0,T*dt],[pi, pi],'k-.');
plot(p2, [0,T*dt],-[pi, pi],'k-.');

plot(p3, [0,T*dt],[20.0, 20.0],'r-.');
plot(p3, [0,T*dt],-[20.0, 20.0],'r-.');

% Print the final cost
fprintf("Final NMPC cost: %.2f\n", Jtot/nsims);

