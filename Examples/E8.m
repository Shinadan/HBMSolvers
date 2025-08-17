%% Example 8: 1D Heat Equation - All HBM Options + ode45
clear; clc;

%% Spatial discretization
N = 20;                         
[D, x] = cheb(N);               
D2 = D^2;
D2 = D2(2:N, 2:N);             
x_interior = x(2:N);            

%% Initial condition
U0 = sin((pi/2)*(x_interior+1)) + 0.5*sin(pi*(x_interior+1));  

%% Time span
tspan = [0 1];

%% HBM Solver Options
opts_fixed_pic  = struct('method', 'picard', 'h', 0.0001, 'M', 4);
opts_fixed_qlm  = struct('method', 'qlm', 'h', 0.01, 'M', 4);
opts_adapt_pic  = struct('method', 'picard', 'h', 0.01, 'M', 4, 'tol', 1e-12, 'fac', 0.99);
opts_adapt_qlm  = struct('method', 'qlm', 'h', 0.01, 'M', 4, 'tol', 1e-12, 'fac', 0.99);

%% --- Fixed-step HBM Picard
tic;
sol_hbm_pic = hbmivp1(@(t,U) D2*U, tspan, U0, opts_fixed_pic);
time_hbm_pic = toc;

%% --- Fixed-step HBM QLM
tic;
sol_hbm_qlm = hbmivp1(@(t,U) D2*U, tspan, U0, opts_fixed_qlm);
time_hbm_qlm = toc;

%% --- Adaptive HBM Picard
tic;
sol_ahbm_pic = ahbmivp1(@(t,U) D2*U, tspan, U0, opts_adapt_pic);
time_ahbm_pic = toc;

%% --- Adaptive HBM QLM
tic;
sol_ahbm_qlm = ahbmivp1(@(t,U) D2*U, tspan, U0, opts_adapt_qlm);
time_ahbm_qlm = toc;

%% --- Include boundary zeros
U_hbm_pic    = [zeros(size(sol_hbm_pic.Y,1),1) sol_hbm_pic.Y zeros(size(sol_hbm_pic.Y,1),1)];
U_hbm_qlm    = [zeros(size(sol_hbm_qlm.Y,1),1) sol_hbm_qlm.Y zeros(size(sol_hbm_qlm.Y,1),1)];
U_ahbm_pic   = [zeros(size(sol_ahbm_pic.Y,1),1) sol_ahbm_pic.Y zeros(size(sol_ahbm_pic.Y,1),1)];
U_ahbm_qlm   = [zeros(size(sol_ahbm_qlm.Y,1),1) sol_ahbm_qlm.Y zeros(size(sol_ahbm_qlm.Y,1),1)];

%% --- ODE45 Solver
disp('Running ode45 solver...');
odefun = @(t,U) D2*U;
U0_vec = U0(:);
tic;
[t45, U45] = ode45(odefun, tspan, U0_vec, odeset('RelTol',1e-12,'AbsTol',1e-14));
time_ode45 = toc;
U45_full = [zeros(size(U45,1),1) U45 zeros(size(U45,1),1)];

%% --- Exact solution for each solver (matching time points)
[X_hbm_pic, T_hbm_pic]       = meshgrid(x, sol_hbm_pic.t);
U_exact_hbm_pic               = exp(-((pi/2)^2)*T_hbm_pic).*sin((pi/2)*(X_hbm_pic+1)) ...
                                + 0.5*exp(-(pi^2)*T_hbm_pic).*sin(pi*(X_hbm_pic+1));

[X_hbm_qlm, T_hbm_qlm]       = meshgrid(x, sol_hbm_qlm.t);
U_exact_hbm_qlm               = exp(-((pi/2)^2)*T_hbm_qlm).*sin((pi/2)*(X_hbm_qlm+1)) ...
                                + 0.5*exp(-(pi^2)*T_hbm_qlm).*sin(pi*(X_hbm_qlm+1));

[X_ahbm_pic, T_ahbm_pic]     = meshgrid(x, sol_ahbm_pic.t);
U_exact_ahbm_pic              = exp(-((pi/2)^2)*T_ahbm_pic).*sin((pi/2)*(X_ahbm_pic+1)) ...
                                + 0.5*exp(-(pi^2)*T_ahbm_pic).*sin(pi*(X_ahbm_pic+1));

[X_ahbm_qlm, T_ahbm_qlm]     = meshgrid(x, sol_ahbm_qlm.t);
U_exact_ahbm_qlm              = exp(-((pi/2)^2)*T_ahbm_qlm).*sin((pi/2)*(X_ahbm_qlm+1)) ...
                                + 0.5*exp(-(pi^2)*T_ahbm_qlm).*sin(pi*(X_ahbm_qlm+1));

[X_ode45, T_ode45]           = meshgrid(x, t45);
U_exact_ode45                 = exp(-((pi/2)^2)*T_ode45).*sin((pi/2)*(X_ode45+1)) ...
                                + 0.5*exp(-(pi^2)*T_ode45).*sin(pi*(X_ode45+1));

%% --- Compute Max Absolute Errors (matching time points)
maxerr_hbm_pic   = max(max(abs(U_hbm_pic - U_exact_hbm_pic)));
maxerr_hbm_qlm   = max(max(abs(U_hbm_qlm - U_exact_hbm_qlm)));
maxerr_ahbm_pic  = max(max(abs(U_ahbm_pic - U_exact_ahbm_pic)));
maxerr_ahbm_qlm  = max(max(abs(U_ahbm_qlm - U_exact_ahbm_qlm)));
maxerr_ode45     = max(max(abs(U45_full - U_exact_ode45)));


%% --- Display results
fprintf('Max Abs Errors and Computational Time:\n');
fprintf('HBM-Picard    : %.3e, Time = %.4f s, Points = %d\n', ...
    maxerr_hbm_pic, time_hbm_pic, length(sol_hbm_pic.t));
fprintf('HBM-QLM        : %.3e, Time = %.4f s, Points = %d\n', ...
    maxerr_hbm_qlm, time_hbm_qlm, length(sol_hbm_qlm.t));
fprintf('Adaptive HBM-Picard    : %.3e, Time = %.4f s, Points = %d\n', ...
    maxerr_ahbm_pic, time_ahbm_pic, length(sol_ahbm_pic.t));
fprintf('Adaptive HBM-QLM       : %.3e, Time = %.4f s, Points = %d\n', ...
    maxerr_ahbm_qlm, time_ahbm_qlm, length(sol_ahbm_qlm.t));
fprintf('ode45                  : %.3e, Time = %.4f s, Points = %d\n', ...
    maxerr_ode45, time_ode45, length(t45));

%% --- Plot Error Distributions
figure;
surf(X_hbm_pic, T_hbm_pic, abs(U_hbm_pic - U_exact_hbm_pic)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Error: HBM-Picard');

figure;
surf(X_hbm_qlm, T_hbm_qlm, abs(U_hbm_qlm - U_exact_hbm_qlm)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Error: HBM-QLM');

figure;
surf(X_ahbm_pic, T_ahbm_pic, abs(U_ahbm_pic - U_exact_ahbm_pic)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Error: Adaptive HBM Picard');

figure;
surf(X_ahbm_qlm, T_ahbm_qlm, abs(U_ahbm_qlm - U_exact_ahbm_qlm)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Error: Adaptive HBM QLM');

figure;
surf(X_ode45, T_ode45, abs(U45_full - U_exact_ode45)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Error: ode45');
