% Qube Servo NMPC
%
% Run the closed-loop system with NMPC controller

function [J,xt,ut,tm] = nmpc(x0,dt,H,T,verbose)

    % Inputs
    if nargin < 5, verbose = true; end

    % Simulation setup
    nx = 4; 
    nu = 1;
    ny = 2;
    tm = dt*(1:T); 
    J = 0.0; 

    % High-gain observer matrix
    e = 2*dt;
    L = [1/e 0; 0 1/e; 1/e^2 0; 0 1/e^2];

    % Log system and observer states, controls
    xt = zeros(nx, T); 
    xo = zeros(nx, T);
    ut = zeros(nu, T);

    % Initialise controls randomly
    xh = zeros(nx, 1);
    ug = 0.5*randn(nu,H);
    u0 = zeros(nu,1);

    % Noise for the dynamics
    sw = 1e-2 * (dt/0.02);
    sv = 1e-2 * (dt/0.02);

    % Loop time
    for t = 1:T

        % Take measurement and update state estimate
        y0 = x0(1:2) + sv*randn(ny,1);
        if t == 1
            xh(1:2) = y0(1:2);
        else
            xh = next_obsv_state(xh, u0, y0, dt, L);
        end

        % Compute optimal controls and use the first one
        up = ocp(ug,xh,dt,H);
        u0 = up(:,1);

        % Log current state and controls
        xt(:,t) = x0; 
        xo(:,t) = xh;
        ut(:,t) = u0;

        % Update cost and run dynamics
        J = J + stage_cost(x0,u0);

        % Update system state        
        x0 = next_state(x0,u0,dt) + sw*randn(nx,1);

        % Store the controls from t = 2:H for next init
        ug = [up(:,2:end), up(:,end)];

        % Display states for convenience
        if verbose, disp([t,x0(1:2)', u0,J/t]); end
    end

    % Normalise cost by number of time samples
    J = J / T;
end