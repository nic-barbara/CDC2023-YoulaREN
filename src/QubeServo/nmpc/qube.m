% Continuous-time system dynamics of the Qube Servo 2
% rotary-arm pendulum system

function dx = qube(x,u)

    % Motor
    Rm = 8.4; 
    km = 0.042;

    % Rotary arm
    Mr = 0.095; 
    Lr = 0.085; 
    Jr = Mr*Lr^2/12; 
    Dr = 0.0015;

    % Pendulum arm
    mp = 0.024; 
    Lp = 0.129; 
    Jp = mp*Lp^2/12; 
    Dp = 0.0005;
    g = 9.81; 
    
    % Parallel axis theorem
    Jr = Jr + 0.25*Mr*Lr^2; 
    Jp =Jp + 0.25*mp*Lp^2;

    % Extract states
    ca = cos(x(2)); 
    sa = sin(x(2)); 
    td = x(3); 
    ad = x(4);

    % Model coefficients
    a1 = mp*Lr^2 + Jp*sa^2 + Jr;
    a2 = 0.5*mp*Lp*Lr*ca; 
    a3 = 2*Jp*sa*ca*td*ad - 0.5*mp*Lp*Lr*sa*ad^2 + (Dr+km^2/Rm)*td;
    b1 = km/Rm;
    c1 = 0.5*mp*Lp*Lr*ca;
    c2 = Jp;
    c3 = -Jp*ca*sa*td^2+0.5*mp*Lp*g*sa+Dp*ad;

    % Solve for accelerations
    denom = 1/(a1*c2-a2*c1);
    ddt = denom*(c2*(b1*u - a3) + a2*c3);
    dda = denom*(-c1*(b1*u - a3) - a1*c3);

    % Return state derivative
    dx=[td; ad; ddt; dda];
end