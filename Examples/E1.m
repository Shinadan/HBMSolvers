clear; clc;
%% Example 1: First-order nonlinear IVP
% y' = -2y^2 + y + 1,  y(0) = 0
% Exact: y(t) = (exp(3t) - 1) / (exp(3t) + 2)

f = @(t, y) -2*y.^2 + y + 1;
exact = @(t) (exp(3*t) - 1) ./ (exp(3*t) + 2);
tspan = [0, 3];
y0 = 0;

% Solver settings
opts_picard = struct('method', 'picard', 'h', 0.01, 'M', 4);
opts_qlm    = struct('method', 'qlm',    'h', 0.01, 'M', 4);
ode_opts    = odeset('RelTol',1e-13, 'AbsTol',1e-15);

results = {};

%% --- HBM (Picard) ---
tic;
sol_hbmp = hbmivp1(f, tspan, y0, opts_picard);
time_hbmp = toc;
err_hbmp = abs(sol_hbmp.Y - exact(sol_hbmp.t));
results{end+1} = {'HBM-Picard', time_hbmp, max(err_hbmp), length(sol_hbmp.t)};

%% --- HBM (QLM) ---
tic;
sol_hbmq = hbmivp1(f, tspan, y0, opts_qlm);
time_hbmq = toc;
err_hbmq = abs(sol_hbmq.Y - exact(sol_hbmq.t));
results{end+1} = {'HBM-QLM', time_hbmq, max(err_hbmq), length(sol_hbmq.t)};

%% --- Adaptive HBM (Picard) ---
tic;
sol_ahbmp = ahbmivp1(f, tspan, y0, opts_picard);
time_ahbmp = toc;
err_ahbmp = abs(sol_ahbmp.Y - exact(sol_ahbmp.t));
results{end+1} = {'Adaptive HBM-Picard', time_ahbmp, max(err_ahbmp), length(sol_ahbmp.t)};

%% --- Adaptive HBM (QLM) ---
tic;
sol_ahbmq = ahbmivp1(f, tspan, y0, opts_qlm);
time_ahbmq = toc;
err_ahbmq = abs(sol_ahbmq.Y - exact(sol_ahbmq.t));
results{end+1} = {'Adaptive HBM-QLM', time_ahbmq, max(err_ahbmq), length(sol_ahbmq.t)};

%% --- ODE45 ---
tic;
[t45, y45] = ode45(f, tspan, y0, ode_opts);
time_ode45 = toc;
err_ode45 = abs(y45 - exact(t45));
results{end+1} = {'ode45', time_ode45, max(err_ode45), length(t45)};

%% --- ODE15s ---
tic;
[t15s, y15s] = ode15s(f, tspan, y0, ode_opts);
time_ode15s = toc;
err_ode15s = abs(y15s - exact(t15s));
results{end+1} = {'ode15s', time_ode15s, max(err_ode15s), length(t15s)};

%% --- Plot Solutions ---
figure;
plot(sol_hbmp.t, sol_hbmp.Y, 'b-o', ...
     sol_hbmq.t, sol_hbmq.Y, 'r-s', ...
     sol_ahbmp.t, sol_ahbmp.Y, 'g-.*', ...
     sol_ahbmq.t, sol_ahbmq.Y, 'm-+', ...
     t45, y45, 'c--', ...
     t15s, y15s, 'k-.', ...
     sol_hbmp.t, exact(sol_hbmp.t), 'k-', 'LineWidth', 1.2);
legend('HBM-P', 'HBM-Q', 'AHBM-P', 'AHBM-Q', 'ode45', 'ode15s', 'Exact', ...
       'Location', 'best');
xlabel('$t$', 'interpreter', 'latex'); ylabel('$y(t)$','interpreter', 'latex'); title('Solution Comparison: Example 1');

%% --- Plot Errors ---
figure;
semilogy(sol_hbmp.t, err_hbmp, 'b-o', ...
         sol_hbmq.t, err_hbmq, 'r-s', ...
         sol_ahbmp.t, err_ahbmp, 'g-.*', ...
         sol_ahbmq.t, err_ahbmq, 'm-+', ...
         t45, err_ode45, 'c--', ...
         t15s, err_ode15s, 'k-.', 'LineWidth', 1.2);
legend('HBM-P', 'HBM-Q', 'AHBM-P', 'AHBM-Q', 'ode45', 'ode15s', ...
       'Location', 'best');
xlabel('$t$','interpreter', 'latex'); ylabel('Absolute Error'); title('Error Comparison: Example 1');

%% --- Display Summary Table ---
fprintf('\n%-20s | %-10s | %-15s | %-10s\n', 'Method', 'Time (s)', 'Max Abs Error', 'Points');
fprintf('%s\n', repmat('-', 1, 65));
for i = 1:length(results)
    fprintf('%-20s | %-10.4f | %-15.3e | %-10d\n', ...
        results{i}{1}, results{i}{2}, results{i}{3}, results{i}{4});
end
