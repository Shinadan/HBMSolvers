%% Example 4: Third-Order Linear ODE
% y''' = -y, y(0)=1, y'(0)=-1, y''(0)=1
clear; clc;

% Define exact solution
exact = @(t) exp(-t);

% Define the ODE function for HBM solvers
f3 = @(t, y, yp, ypp) -y;

% Initial conditions
y0 = 1; yp0 = -1; ypp0 = 1;

% Time span
tspan = [0, 15];

%% HBM Solver Options
opts_fixed = struct('h', 0.05, 'method', 'picard', 'M', 4);
opts_adapt = struct('h', 0.05, 'method', 'picard', 'M', 4, 'tol', 1e-15, 'fac', 0.99);

%% --- Fixed-Step HBM ---
disp('Running HBM fixed-step solver (Picard & QLM)...');
opts_fixed.method = 'picard';
tic;
sol_hbm_picard = hbmivp3(f3, tspan, y0, yp0, ypp0, opts_fixed);
time_hbm_picard = toc;

opts_fixed.method = 'qlm';
tic;
sol_hbm_qlm = hbmivp3(f3, tspan, y0, yp0, ypp0, opts_fixed);
time_hbm_qlm = toc;

%% --- Adaptive HBM ---
disp('Running adaptive HBM solver (Picard & QLM)...');
opts_adapt.method = 'picard';
tic;
sol_ahbm_picard = ahbmivp3(f3, tspan, y0, yp0, ypp0, opts_adapt);
time_ahbm_picard = toc;

opts_adapt.method = 'qlm';
tic;
sol_ahbm_qlm = ahbmivp3(f3, tspan, y0, yp0, ypp0, opts_adapt);
time_ahbm_qlm = toc;

%% --- MATLAB solvers ---
ode_opts = odeset('RelTol',1e-13,'AbsTol',1e-15);
tic;
[t45, y45] = ode45(@(t,Y) [Y(2); Y(3); -Y(1)], tspan, [y0; yp0; ypp0], ode_opts);
time_ode45 = toc;

tic;
[t15s, y15s] = ode15s(@(t,Y) [Y(2); Y(3); -Y(1)], tspan, [y0; yp0; ypp0], ode_opts);
time_ode15s = toc;

%% --- Compute max absolute errors ---
err_hbm_pic  = max(abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)));
err_hbm_qlm  = max(abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)));
err_ahbm_pic = max(abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)));
err_ahbm_qlm = max(abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)));
err_ode45   = max(abs(y45(:,1) - exact(t45)));
err_ode15s  = max(abs(y15s(:,1) - exact(t15s)));

%% --- Display results ---
fprintf('\nExample 4: Third-Order Linear ODE Solver Comparison\n');
fprintf('Method               | Time (s)   | Points     | Max Abs Error\n');
fprintf('---------------------------------------------------------------\n');
fprintf('HBM Picard           | %.4f     | %d       | %.3e\n', time_hbm_picard, length(sol_hbm_picard.t), err_hbm_pic);
fprintf('HBM QLM              | %.4f     | %d       | %.3e\n', time_hbm_qlm, length(sol_hbm_qlm.t), err_hbm_qlm);
fprintf('Adaptive HBM Picard  | %.4f     | %d       | %.3e\n', time_ahbm_picard, length(sol_ahbm_picard.t), err_ahbm_pic);
fprintf('Adaptive HBM QLM     | %.4f     | %d       | %.3e\n', time_ahbm_qlm, length(sol_ahbm_qlm.t), err_ahbm_qlm);
fprintf('ode45                | %.4f     | %d       | %.3e\n', time_ode45, length(t45), err_ode45);
fprintf('ode15s               | %.4f     | %d       | %.3e\n', time_ode15s, length(t15s), err_ode15s);

%% --- Plot Error ---
figure;
semilogy(sol_hbm_picard.t, abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)), 'b-o', ...
         sol_hbm_qlm.t, abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)), 'r-s', ...
         sol_ahbm_picard.t, abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)), 'g-^', ...
         sol_ahbm_qlm.t, abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)), 'm-d', ...
         t45, abs(y45(:,1) - exact(t45)), 'k--', ...
         t15s, abs(y15s(:,1) - exact(t15s)), 'c-.', 'LineWidth', 1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error'); title('Error Comparison for Example 4');
