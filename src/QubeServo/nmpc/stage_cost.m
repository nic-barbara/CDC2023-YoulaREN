% Qube Servo NMPC
%
% Compute cost of a single step

function J = stage_cost(x,u)

    % Cost function weights
    q1 = 5.0;
    q2 = 10.0;
    r1 = 0.01;

    % Cos/sin cost function to wrap angles
    J = 2*q1*(1-cos(x(1))) + 2*q2*(1+cos(x(2))) + r1*u^2;
end