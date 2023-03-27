% Qube Servo NMPC
%
% Discretise the dynamics with RK4

function x1 = next_state(x0,u0,dt)
    k1 = qube(x0,u0);
    k2 = qube(x0+dt/2*k1,u0);
    k3 = qube(x0+dt/2*k2,u0);
    k4 = qube(x0+dt*k3,u0);
    x1 = x0 + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end