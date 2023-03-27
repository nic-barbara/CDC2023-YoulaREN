% Qube Servo NMPC
%
% Compute the optimal control sequence for a horizon
% using nonlinear MPC.

function up = ocp(ug,x0,dt,H)

    % Solver options
    options = optimset('Display','off',...
                    'TolFun', 1e-4,...
                    'MaxIter', 2000,...
                    'Algorithm', 'interior-point',...
                    'AlwaysHonorConstraints', 'bounds',...
                    'FinDiffType', 'forward',...
                    'HessFcn', [],...
                    'Hessian', 'bfgs',...
                    'HessMult', [],...
                    'InitBarrierParam', 0.1,...
                    'InitTrustRegionRadius', sqrt(size(ug,1)*size(ug,2)),...
                    'MaxProjCGIter', 2*size(ug,1)*size(ug,2),...
                    'ObjectiveLimit', -1e20,...
                    'ScaleProblem', 'obj-and-constr',...
                    'SubproblemAlgorithm', 'cg',...
                    'TolProjCG', 1e-3,...
                    'TolProjCGAbs', 1e-8);
    
    % Constraint on control input is 20 V
    umax = 20;

    % Define (empty) constraint arrays and bounds
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = -umax * ones(1,H);
    ub = umax * ones(1,H);

    % Solve for optimal controls numerically, with "warm restarts"
    up = fmincon(@(u) cost(u,x0,dt,H), ug, A,b,Aeq,beq,lb,ub,...
                 @(u) cons(u,x0,dt,H), options);
end

% Helper function to compute the cost
function J = cost(ug,x0,dt,H)
    J = 0.0;
    for t = 1:H
        u0 = ug(:,t);
        J = J + stage_cost(x0,u0);
        x0 = next_state(x0,u0,dt);
    end
end

% Add constraints (no need for this problem)
function [c,ceq] = cons(ug,x0,dt,H)
    c=[]; ceq=[];
end

