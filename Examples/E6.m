%% Example 6: Coupled Linear ODEs (y1'' = -y1 + y2, y2'' = y1 - y2)
clear; clc;

f_coupled = @(t, y, yp) [ -y(1) + y(2); y(1) - y(2) ];
tspan = [0, 10];
y0 = [1; -1];
yp0 = [0; 0];

%% Solver options
opts_fixed = struct('h', 0.01, 'method', 'picard', 'M', 4);
opts_adapt = struct('h', 0.01, 'method', 'picard', 'M', 4, 'tol', 1e-15, 'fac', 0.99);

%% --- Fixed-Step HBM ---
tic; sol_hbm_pic = hbmivp2(f_coupled, tspan, y0, yp0, opts_fixed); time_hbm_pic = toc;
opts_fixed.method = 'qlm';
tic; sol_hbm_qlm = hbmivp2(f_coupled, tspan, y0, yp0, opts_fixed); time_hbm_qlm = toc;

%% --- Adaptive HBM ---
opts_adapt.method = 'picard';
tic; sol_ahbm_pic = ahbmivp2(f_coupled, tspan, y0, yp0, opts_adapt); time_ahbm_pic = toc;
opts_adapt.method = 'qlm';
tic; sol_ahbm_qlm = ahbmivp2(f_coupled, tspan, y0, yp0, opts_adapt); time_ahbm_qlm = toc;

%% --- MATLAB Native Solvers ---
ode_opts = odeset('RelTol',1e-13,'AbsTol',1e-15);
tic; [t45, Y45] = ode45(@(t,Y) [Y(2); -Y(1)+Y(3); Y(4); Y(1)-Y(3)], tspan, [y0(1); yp0(1); y0(2); yp0(2)], ode_opts); time_ode45 = toc;
tic; [t15s, Y15s] = ode15s(@(t,Y) [Y(2); -Y(1)+Y(3); Y(4); Y(1)-Y(3)], tspan, [y0(1); yp0(1); y0(2); yp0(2)], ode_opts); time_ode15s = toc;

%% --- Exact solutions ---
exact_y1 = @(t) cos(sqrt(2)*t);
exact_y2 = @(t) -cos(sqrt(2)*t);

%% --- Compute max absolute errors ---
err_hbm_pic_y1  = max(abs(sol_hbm_pic.Y(:,1) - exact_y1(sol_hbm_pic.t)));
err_hbm_qlm_y1  = max(abs(sol_hbm_qlm.Y(:,1) - exact_y1(sol_hbm_qlm.t)));
err_ahbm_pic_y1 = max(abs(sol_ahbm_pic.Y(:,1) - exact_y1(sol_ahbm_pic.t)));
err_ahbm_qlm_y1 = max(abs(sol_ahbm_qlm.Y(:,1) - exact_y1(sol_ahbm_qlm.t)));

err_hbm_pic_y2  = max(abs(sol_hbm_pic.Y(:,2) - exact_y2(sol_hbm_pic.t)));
err_hbm_qlm_y2  = max(abs(sol_hbm_qlm.Y(:,2) - exact_y2(sol_hbm_qlm.t)));
err_ahbm_pic_y2 = max(abs(sol_ahbm_pic.Y(:,2) - exact_y2(sol_ahbm_pic.t)));
err_ahbm_qlm_y2 = max(abs(sol_ahbm_qlm.Y(:,2) - exact_y2(sol_ahbm_qlm.t)));

err_ode45_y1  = max(abs(Y45(:,1) - exact_y1(t45)));
err_ode15s_y1 = max(abs(Y15s(:,1) - exact_y1(t15s)));
err_ode45_y2  = max(abs(Y45(:,3) - exact_y2(t45)));
err_ode15s_y2 = max(abs(Y15s(:,3) - exact_y2(t15s)));

%% --- Display table for y1 ---
fprintf('\nExample 6: Coupled Linear ODEs - Max Abs Error (y1)\n');
fprintf('Method               | Max Abs Error | Points | Time (s)\n');
fprintf('----------------------------------------------------------\n');
fprintf('HBM Picard           | %.3e | %d | %.4f\n', err_hbm_pic_y1, length(sol_hbm_pic.t), time_hbm_pic);
fprintf('HBM QLM              | %.3e | %d | %.4f\n', err_hbm_qlm_y1, length(sol_hbm_qlm.t), time_hbm_qlm);
fprintf('Adaptive HBM Picard  | %.3e | %d | %.4f\n', err_ahbm_pic_y1, length(sol_ahbm_pic.t), time_ahbm_pic);
fprintf('Adaptive HBM QLM     | %.3e | %d | %.4f\n', err_ahbm_qlm_y1, length(sol_ahbm_qlm.t), time_ahbm_qlm);
fprintf('ode45                | %.3e | %d | %.4f\n', err_ode45_y1, length(t45), time_ode45);
fprintf('ode15s               | %.3e | %d | %.4f\n', err_ode15s_y1, length(t15s), time_ode15s);

%% --- Display table for y2 ---
fprintf('\nExample 6: Coupled Linear ODEs - Max Abs Error (y2)\n');
fprintf('Method               | Max Abs Error | Points | Time (s)\n');
fprintf('----------------------------------------------------------\n');
fprintf('HBM Picard           | %.3e | %d | %.4f\n', err_hbm_pic_y2, length(sol_hbm_pic.t), time_hbm_pic);
fprintf('HBM QLM              | %.3e | %d | %.4f\n', err_hbm_qlm_y2, length(sol_hbm_qlm.t), time_hbm_qlm);
fprintf('Adaptive HBM Picard  | %.3e | %d | %.4f\n', err_ahbm_pic_y2, length(sol_ahbm_pic.t), time_ahbm_pic);
fprintf('Adaptive HBM QLM     | %.3e | %d | %.4f\n', err_ahbm_qlm_y2, length(sol_ahbm_qlm.t), time_ahbm_qlm);
fprintf('ode45                | %.3e | %d | %.4f\n', err_ode45_y2, length(t45), time_ode45);
fprintf('ode15s               | %.3e | %d | %.4f\n', err_ode15s_y2, length(t15s), time_ode15s);

%% --- Plot error distribution for y1 ---
figure;
semilogy(sol_hbm_pic.t, abs(sol_hbm_pic.Y(:,1) - exact_y1(sol_hbm_pic.t)), 'b-o', ...
         sol_hbm_qlm.t, abs(sol_hbm_qlm.Y(:,1) - exact_y1(sol_hbm_qlm.t)), 'r-o', ...
         sol_ahbm_pic.t, abs(sol_ahbm_pic.Y(:,1) - exact_y1(sol_ahbm_pic.t)), 'g-s', ...
         sol_ahbm_qlm.t, abs(sol_ahbm_qlm.Y(:,1) - exact_y1(sol_ahbm_qlm.t)), 'm-^', ...
         t45, abs(Y45(:,1) - exact_y1(t45)), 'k--', ...
         t15s, abs(Y15s(:,1) - exact_y1(t15s)), 'c-.','LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error ($y_1(t)$)', 'interpreter', 'latex'); 
title('Example 6: Error Distribution $y_1(t)$', 'interpreter', 'latex');

%% --- Plot error distribution for y2 ---
figure;
semilogy(sol_hbm_pic.t, abs(sol_hbm_pic.Y(:,2) - exact_y2(sol_hbm_pic.t)), 'b-o', ...
         sol_hbm_qlm.t, abs(sol_hbm_qlm.Y(:,2) - exact_y2(sol_hbm_qlm.t)), 'r-o', ...
         sol_ahbm_pic.t, abs(sol_ahbm_pic.Y(:,2) - exact_y2(sol_ahbm_pic.t)), 'g-s', ...
         sol_ahbm_qlm.t, abs(sol_ahbm_qlm.Y(:,2) - exact_y2(sol_ahbm_qlm.t)), 'm-^', ...
         t45, abs(Y45(:,3) - exact_y2(t45)), 'k--', ...
         t15s, abs(Y15s(:,3) - exact_y2(t15s)), 'c-.','LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error ($y_2(t)$)', 'interpreter', 'latex' );
title('Example 6: Error Distribution $y_2(t)$', 'interpreter', 'latex');
