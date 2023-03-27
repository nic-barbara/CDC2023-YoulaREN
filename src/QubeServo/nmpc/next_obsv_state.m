% Qube Servo NMPC
%
% High-gain observer state dynamics

function xh1 = next_obsv_state(xh, u, y, dt, L)

    k1 = cts_obsv_dynamics(xh, u, y, L);
    k2 = cts_obsv_dynamics(xh + dt/2*k1, u, y, L);
    k3 = cts_obsv_dynamics(xh + dt/2*k2, u, y, L);
    k4 = cts_obsv_dynamics(xh + dt*k3, u, y, L);

    xh1 = xh + dt/6*(k1 + 2*k2 + 2*k3 + k4);

end

function xhd = cts_obsv_dynamics(xh, u, y, L)
    xhd = qube(xh, u) + L*(y - xh(1:2));
end