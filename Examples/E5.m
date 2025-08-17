clear; clc;
%% Example 5: Bessel-type Second-Order ODE
% y'' + (1/t)*y' + (1 - 0.25/t^2)*y = 0
tspan = [1, 10];
y0 = sqrt(2/pi)*sin(1);
yp0 = (2*cos(1) - sin(1))/sqrt(2*pi);

% Define ODE in the standard HBM form: y'' = f(t,y,yp)
f5 = @(t, y, yp) -(1./t).*yp - (t.^2 - 0.25)./(t.^2).*y;

%% HBM Solver Options
opts_fixed = struct('h', 0.01, 'method', 'picard', 'M', 4);
opts_adapt = struct('h', 0.01, 'method', 'picard', 'M', 4, 'tol', 1e-14, 'fac', 0.99);

%% --- Fixed-Step HBM ---
tic; sol_hbm_picard = hbmivp2(f5, tspan, y0, yp0, opts_fixed); time_hbm_picard = toc;
opts_fixed.method = 'qlm';
tic; sol_hbm_qlm = hbmivp2(f5, tspan, y0, yp0, opts_fixed); time_hbm_qlm = toc;

%% --- Adaptive HBM ---
opts_adapt.method = 'picard';
tic; sol_ahbm_picard = ahbmivp2(f5, tspan, y0, yp0, opts_adapt); time_ahbm_picard = toc;
opts_adapt.method = 'qlm';
tic; sol_ahbm_qlm = ahbmivp2(f5, tspan, y0, yp0, opts_adapt); time_ahbm_qlm = toc;

%% --- MATLAB Native Solvers ---
ode_opts = odeset('RelTol',1e-12,'AbsTol',1e-14);
tic;
[t45, y45] = ode45(@(t,Y) [Y(2); f5(t,Y(1),Y(2))], tspan, [y0; yp0], ode_opts);
time_ode45 = toc;
tic;
[t15s, y15s] = ode15s(@(t,Y) [Y(2); f5(t,Y(1),Y(2))], tspan, [y0; yp0], ode_opts);
time_ode15s = toc;

%% --- Exact Solution ---
exact = @(t) sqrt(2./(pi*t)).*sin(t);

%% --- Compute Max Absolute Error ---
maxerr_hbm_pic = max(abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)));
maxerr_hbm_qlm = max(abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)));
maxerr_ahbm_pic = max(abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)));
maxerr_ahbm_qlm = max(abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)));
maxerr_ode45 = max(abs(y45(:,1) - exact(t45)));
maxerr_ode15s = max(abs(y15s(:,1) - exact(t15s)));

%% --- Display Results ---
fprintf('\nExample 5: Bessel-type Second-Order ODE\n');
fprintf('Method               | Max Abs Error | Points | Time (s)\n');
fprintf('--------------------------------------------------------\n');
fprintf('HBM Picard           | %.3e | %d | %.4f\n', maxerr_hbm_pic, length(sol_hbm_picard.t), time_hbm_picard);
fprintf('HBM QLM              | %.3e | %d | %.4f\n', maxerr_hbm_qlm, length(sol_hbm_qlm.t), time_hbm_qlm);
fprintf('Adaptive HBM Picard  | %.3e | %d | %.4f\n', maxerr_ahbm_pic, length(sol_ahbm_picard.t), time_ahbm_picard);
fprintf('Adaptive HBM QLM     | %.3e | %d | %.4f\n', maxerr_ahbm_qlm, length(sol_ahbm_qlm.t), time_ahbm_qlm);
fprintf('ode45                | %.3e | %d | %.4f\n', maxerr_ode45, length(t45), time_ode45);
fprintf('ode15s               | %.3e | %d | %.4f\n', maxerr_ode15s, length(t15s), time_ode15s);

%% --- Plot Solutions ---
figure;
plot(sol_hbm_picard.t, sol_hbm_picard.Y, 'bo-', ...
     sol_hbm_qlm.t, sol_hbm_qlm.Y, 'ro-', ...
     sol_ahbm_picard.t, sol_ahbm_picard.Y, 'gs-', ...
     sol_ahbm_qlm.t, sol_ahbm_qlm.Y, 'm^-', ...
     t45, y45(:,1), 'k--', t15s, y15s(:,1), 'c-.', ...
     sol_hbm_picard.t, exact(sol_hbm_picard.t), 'k-', 'LineWidth', 1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s','Exact');
xlabel('$t$', 'interpreter', 'latex'); ylabel('$y(t)$', 'interpreter', 'latex'); title('Example 5: Bessel-type ODE Solution Comparison');

%% --- Plot Error ---
figure;
semilogy(sol_hbm_picard.t, abs(sol_hbm_picard.Y - exact(sol_hbm_picard.t)), 'bo-', ...
         sol_hbm_qlm.t, abs(sol_hbm_qlm.Y - exact(sol_hbm_qlm.t)), 'ro-', ...
         sol_ahbm_picard.t, abs(sol_ahbm_picard.Y - exact(sol_ahbm_picard.t)), 'gs-', ...
         sol_ahbm_qlm.t, abs(sol_ahbm_qlm.Y - exact(sol_ahbm_qlm.t)), 'm^-', ...
         t45, abs(y45(:,1) - exact(t45)), 'k--', ...
         t15s, abs(y15s(:,1) - exact(t15s)), 'c-.','LineWidth', 1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error'); title('Example 5: Error Distribution');
