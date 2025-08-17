%% Example 3: Third-Order IVP with Exact Solution
clear; clc;

% Exact solution and RHS
exact = @(t) 1 + 0.5*log((2+t)./(2-t));
f = @(t, y, yp, ypp) yp .* (2*t.*ypp + yp);

% Time span and initial conditions
tspan = [0, 1.5];
y0 = 1; yp0 = 0.5; ypp0 = 0;

% Solver options
opts_fixed = struct('h', 0.01, 'method', 'picard', 'M', 4);
opts_adapt = struct('h', 0.01, 'method', 'picard', 'M', 4, 'tol', 1e-12, 'fac', 0.99);

%% --- Fixed-Step HBM ---
opts_fixed.method = 'picard';
tic; sol_hbm_picard = hbmivp3(f, tspan, y0, yp0, ypp0, opts_fixed); time_hbm_pic = toc;

opts_fixed.method = 'qlm';
tic; sol_hbm_qlm = hbmivp3(f, tspan, y0, yp0, ypp0, opts_fixed); time_hbm_qlm = toc;

%% --- Adaptive HBM ---
opts_adapt.method = 'picard';
tic; sol_ahbm_picard = ahbmivp3(f, tspan, y0, yp0, ypp0, opts_adapt); time_ahbm_pic = toc;

opts_adapt.method = 'qlm';
tic; sol_ahbm_qlm = ahbmivp3(f, tspan, y0, yp0, ypp0, opts_adapt); time_ahbm_qlm = toc;

%% --- MATLAB solvers ---
ode_opts = odeset('RelTol',1e-12,'AbsTol',1e-14);
rhs = @(t,Y) [Y(2); Y(3); Y(2).*(2*t*Y(3) + Y(2))];

tic; [t45,Y45] = ode45(rhs, tspan, [y0; yp0; ypp0], ode_opts); time_ode45 = toc;
tic; [t15s,Y15s] = ode15s(rhs, tspan, [y0; yp0; ypp0], ode_opts); time_ode15s = toc;

%% --- Compute max errors vs exact solution ---
err_hbm_pic    = max(abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)));
err_hbm_qlm    = max(abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)));
err_ahbm_pic   = max(abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)));
err_ahbm_qlm   = max(abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)));
err_ode45      = max(abs(Y45(:,1) - exact(t45)));
err_ode15s     = max(abs(Y15s(:,1) - exact(t15s)));

%% --- Display summary table ---
fprintf('\nExample 3: Third-Order IVP Solver Comparison\n');
fprintf('%-20s | %-10s | %-10s | %-12s\n', 'Method', 'Time (s)', 'Points', 'Max Abs Error');
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'HBM Picard', time_hbm_pic, length(sol_hbm_picard.t), err_hbm_pic);
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'HBM QLM', time_hbm_qlm, length(sol_hbm_qlm.t), err_hbm_qlm);
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'Adaptive HBM Picard', time_ahbm_pic, length(sol_ahbm_picard.t), err_ahbm_pic);
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'Adaptive HBM QLM', time_ahbm_qlm, length(sol_ahbm_qlm.t), err_ahbm_qlm);
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'ode45', time_ode45, length(t45), err_ode45);
fprintf('%-20s | %-10.4f | %-10d | %.3e\n', 'ode15s', time_ode15s, length(t15s), err_ode15s);

%% --- Plot solutions ---
figure;
plot(sol_hbm_picard.t, sol_hbm_picard.Y, 'bo-', ...
     sol_hbm_qlm.t, sol_hbm_qlm.Y, 'ro-', ...
     sol_ahbm_picard.t, sol_ahbm_picard.Y, 'gs-', ...
     sol_ahbm_qlm.t, sol_ahbm_qlm.Y, 'm^-', ...
     t45, Y45(:,1), 'k--', t15s, Y15s(:,1), 'c-.', 'LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$','interpreter', 'latex'); ylabel('$y(t)$', 'interpreter', 'latex'); title('Example 3: Third-order IVP Solution');

%% --- Plot errors ---
figure;
semilogy(sol_hbm_picard.t, abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)), 'bo-', ...
         sol_hbm_qlm.t, abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)), 'ro-', ...
         sol_ahbm_picard.t, abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)), 'gs-', ...
         sol_ahbm_qlm.t, abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)), 'm^-', ...
         t45, abs(Y45(:,1) - exact(t45)), 'k--', ...
         t15s, abs(Y15s(:,1) - exact(t15s)), 'c-.', 'LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error'); title('Error Comparison for Example 3');
