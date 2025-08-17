clear; clc;
%% Example 2: Van der Pol Equation
% y'' - mu*(1-y^2)*y' + y = 0, y(0) = 2, y'(0) = 0
mu = 1; %mu can be adjusted to change the stiffness of the problem
tspan = [0, 40];
y0 = 2;
yp0 = 0;

% Right-hand side for HBM solvers
f2 = @(t, y, yp) mu*(1 - y.^2).*yp - y;

%% --- HBM Solver Options ---
opts_fixed = struct('h', 0.01, 'method', 'picard', 'M', 4);
opts_adapt = struct('h', 0.01, 'method', 'picard', 'M', 4, 'tol', 1e-12, 'fac', 0.99);

%% --- Fixed-Step HBM ---
disp('Running HBM fixed-step solver (Picard & QLM)...');

tic;
sol_hbm_picard = hbmivp2(f2, tspan, y0, yp0, opts_fixed);
time_hbm_pic = toc;

opts_fixed.method = 'qlm';
tic;
sol_hbm_qlm = hbmivp2(f2, tspan, y0, yp0, opts_fixed);
time_hbm_qlm = toc;

%% --- Adaptive HBM ---
disp('Running adaptive HBM solver (Picard & QLM)...');

opts_adapt.method = 'picard';
tic;
sol_ahbm_picard = ahbmivp2(f2, tspan, y0, yp0, opts_adapt);
time_ahbm_pic = toc;

opts_adapt.method = 'qlm';
tic;
sol_ahbm_qlm = ahbmivp2(f2, tspan, y0, yp0, opts_adapt);
time_ahbm_qlm = toc;

%% --- MATLAB Native Solvers ---
disp('Running ode45 and ode15s as reference...');
ode_opts = odeset('RelTol',1e-12,'AbsTol',1e-14);

tic;
[t45, y45] = ode45(@(t,Y) [Y(2); mu*(1-Y(1).^2).*Y(2) - Y(1)], tspan, [y0; yp0], ode_opts);
time_ode45 = toc;

tic;
[t15s, y15s] = ode15s(@(t,Y) [Y(2); mu*(1-Y(1).^2).*Y(2) - Y(1)], tspan, [y0; yp0], ode_opts);
time_ode15s = toc;

%% --- Interpolate HBM solutions onto ode45 grid for comparison ---
y_hbm_pic_interp  = interp1(sol_hbm_picard.t, sol_hbm_picard.Y, t45);
y_hbm_qlm_interp  = interp1(sol_hbm_qlm.t, sol_hbm_qlm.Y, t45);
y_ahbm_pic_interp = interp1(sol_ahbm_picard.t, sol_ahbm_picard.Y, t45);
y_ahbm_qlm_interp = interp1(sol_ahbm_qlm.t, sol_ahbm_qlm.Y, t45);
y_ode15s_interp  = interp1(t15s, y15s(:,1), t45);

%% --- Compute deviations from ode45 ---
dev_hbm_pic  = max(abs(y_hbm_pic_interp - y45(:,1)));
dev_hbm_qlm  = max(abs(y_hbm_qlm_interp - y45(:,1)));
dev_ahbm_pic = max(abs(y_ahbm_pic_interp - y45(:,1)));
dev_ahbm_qlm = max(abs(y_ahbm_qlm_interp - y45(:,1)));
dev_ode15s   = max(abs(y_ode15s_interp - y45(:,1)));

%% --- Display Summary Table ---
fprintf('\nVan der Pol Equation - Solver Comparison (Deviation from ode45)\n');
fprintf('%-22s | %-10s | %-10s | %-10s\n', 'Method','Time (s)','Points','Max Deviation');
fprintf('---------------------------------------------------------------\n');
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'HBM Picard', time_hbm_pic, length(sol_hbm_picard.t), dev_hbm_pic);
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'HBM QLM', time_hbm_qlm, length(sol_hbm_qlm.t), dev_hbm_qlm);
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'Adaptive HBM Picard', time_ahbm_pic, length(sol_ahbm_picard.t), dev_ahbm_pic);
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'Adaptive HBM QLM', time_ahbm_qlm, length(sol_ahbm_qlm.t), dev_ahbm_qlm);
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'ode45', time_ode45, length(t45), 0); % reference
fprintf('%-22s | %-10.4f | %-10d | %.3e\n', 'ode15s', time_ode15s, length(t15s), dev_ode15s);

%% --- Plot Solution Comparison ---
figure;
plot(sol_hbm_picard.t, sol_hbm_picard.Y, 'bo-', ...
     sol_hbm_qlm.t, sol_hbm_qlm.Y, 'ro-', ...
     sol_ahbm_picard.t, sol_ahbm_picard.Y, 'gs-', ...
     sol_ahbm_qlm.t, sol_ahbm_qlm.Y, 'm^-', ...
     t45, y45(:,1), 'k--', t15s, y15s(:,1), 'c-.', 'LineWidth', 1.2);
legend('HBM Picard','HBM QLM','Adaptive HBM Picard','Adaptive HBM QLM','ode45','ode15s');
xlabel('t'); ylabel('y(t)'); title('Van der Pol Equation: Solution Comparison');

%% --- Plot Deviations ---
figure;
semilogy(t45, abs(y_hbm_pic_interp - y45(:,1)), 'b-', ...
         t45, abs(y_hbm_qlm_interp - y45(:,1)), 'r-', ...
         t45, abs(y_ahbm_pic_interp - y45(:,1)), 'g-', ...
         t45, abs(y_ahbm_qlm_interp - y45(:,1)), 'm-', ...
         t45, abs(y_ode15s_interp - y45(:,1)), 'c-', 'LineWidth', 1.2);
legend('HBM Picard','HBM QLM','Adaptive HBM Picard','Adaptive HBM QLM','ode15s');
xlabel('t'); ylabel('Deviation from ode45'); title('Van der Pol Equation: Deviations');
grid on;
