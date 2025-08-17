clear; clc;

%% Example 7: Stiff Second-Order ODE
% y'' = -100y + 99*sin(t), y(0) = 1, y'(0) = 11
f7 = @(t, y, yp) -100*y + 99*sin(t);
exact7 = @(t) cos(10*t) + sin(10*t) + sin(t);

tspan = [0, 5];
y0 = 1;
yp0 = 11;

%% HBM Options
opts_fixed = struct('method', 'picard', 'h', 0.001);
opts_adapt = struct('method', 'picard', 'h', 0.001);

%% --- Fixed-Step HBM ---
tic;
sol_hbm_picard = hbmivp2(f7, tspan, y0, yp0, opts_fixed);
time_hbm_picard = toc;

opts_fixed.method = 'qlm';
tic;
sol_hbm_qlm = hbmivp2(f7, tspan, y0, yp0, opts_fixed);
time_hbm_qlm = toc;

%% --- Adaptive HBM ---
opts_adapt.method = 'picard';
tic;
sol_ahbm_picard = ahbmivp2(f7, tspan, y0, yp0, opts_adapt);
time_ahbm_picard = toc;

opts_adapt.method = 'qlm';
tic;
sol_ahbm_qlm = ahbmivp2(f7, tspan, y0, yp0, opts_adapt);
time_ahbm_qlm = toc;

%% --- MATLAB Native Solvers ---
ode_opts = odeset('RelTol',1e-12,'AbsTol',1e-14);
tic;
[t45, y45] = ode45(@(t,Y) [Y(2); -100*Y(1)+99*sin(t)], tspan, [y0; yp0], ode_opts);
time_ode45 = toc;

tic;
[t15s, y15s] = ode15s(@(t,Y) [Y(2); -100*Y(1)+99*sin(t)], tspan, [y0; yp0], ode_opts);
time_ode15s = toc;

%% --- Compute Max Errors ---
y_hbm_pic = sol_hbm_picard.Y;
y_hbm_qlm = sol_hbm_qlm.Y;
y_ahbm_pic = sol_ahbm_picard.Y;
y_ahbm_qlm = sol_ahbm_qlm.Y;

err_hbm_pic = max(abs(y_hbm_pic - exact7(sol_hbm_picard.t)));
err_hbm_qlm = max(abs(y_hbm_qlm - exact7(sol_hbm_qlm.t)));
err_ahbm_pic = max(abs(y_ahbm_pic - exact7(sol_ahbm_picard.t)));
err_ahbm_qlm = max(abs(y_ahbm_qlm - exact7(sol_ahbm_qlm.t)));

err_ode45 = max(abs(y45(:,1) - exact7(t45)));
err_ode15s = max(abs(y15s(:,1) - exact7(t15s)));

%% --- Display Table ---
fprintf('\nExample 7: Stiff Second-Order ODE - Solver Comparison\n');
fprintf('%-22s | %-12s | %-10s | %-12s\n','Method','Time (s)','Points','Max Abs Error');
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','HBM Picard',time_hbm_picard,length(sol_hbm_picard.t),err_hbm_pic);
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','HBM QLM',time_hbm_qlm,length(sol_hbm_qlm.t),err_hbm_qlm);
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','Adaptive HBM Picard',time_ahbm_picard,length(sol_ahbm_picard.t),err_ahbm_pic);
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','Adaptive HBM QLM',time_ahbm_qlm,length(sol_ahbm_qlm.t),err_ahbm_qlm);
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','ode45',time_ode45,length(t45),err_ode45);
fprintf('%-22s | %-12.4f | %-10d | %-12.3e\n','ode15s',time_ode15s,length(t15s),err_ode15s);

%% --- Plot Solution ---
figure;
plot(sol_hbm_picard.t,y_hbm_pic,'b-o',sol_hbm_qlm.t,y_hbm_qlm,'r-s',...
     sol_ahbm_picard.t,y_ahbm_pic,'g-^',sol_ahbm_qlm.t,y_ahbm_qlm,'m-d',...
     t45,y45(:,1),'k--',t15s,y15s(:,1),'c-.','LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('$y(t)$', 'interpreter', 'latex'); title('Example 7: Stiff ODE - Solution Comparison');

%% --- Plot Errors ---
figure;
semilogy(sol_hbm_picard.t,abs(y_hbm_pic-exact7(sol_hbm_picard.t)),'bo-',...
         sol_hbm_qlm.t,abs(y_hbm_qlm-exact7(sol_hbm_qlm.t)),'r-s',...
         sol_ahbm_picard.t,abs(y_ahbm_pic-exact7(sol_ahbm_picard.t)),'g-^',...
         sol_ahbm_qlm.t,abs(y_ahbm_qlm-exact7(sol_ahbm_qlm.t)),'m-d',...
         t45,abs(y45(:,1)-exact7(t45)),'k--',t15s,abs(y15s(:,1)-exact7(t15s)),'c-.','LineWidth',1.2);
legend('HBM-P','HBM-Q','AHBM-P','AHBM-Q','ode45','ode15s');
xlabel('$t$', 'interpreter', 'latex'); ylabel('Absolute Error'); title('Example 7: Errors');
