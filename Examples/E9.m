%% Example 9: Heat equation with source on x in [-1,1]
%  y_t = y_xx + f(x,t),  y(\pm 1,t)=0,  y(x,0)=0
%  Exact: y(x,t) = (1 - e^{-t}) * sin((pi/2)*(x+1))
%  f(x,t) = [ e^{-t} + (pi/2)^2 * (1 - e^{-t}) ] * sin((pi/2)*(x+1))

clear; clc;

%% ---- Spatial discretization (Chebyshev, Trefethen's cheb.m) ----
N = 20;                              % total Chebyshev points (including boundaries)
[D, x] = cheb(N);                    % requires cheb.m
D2 = D^2;

% Enforce Dirichlet BCs by removing boundary rows/cols
D2_in = D2(2:N, 2:N);                % Laplacian on interior
x_in  = x(2:N);                      % interior grid


% Forcing term on the interior
Ffun = @(t) ( exp(-t) + ((pi/2)^2)*(1 - exp(-t)) ) .* sin( (pi/2)*(x_in + 1) );

% ODE system: U'(t) = D2_in * U(t) + F(t), with U(0) = 0
odefun = @(t, U) D2_in*U + Ffun(t);

% Exact solution on interior
u_exact_in = @(t) (1 - exp(-t)) .* sin( (pi/2)*(x_in + 1) );

% Time span
tspan = [0 1];

% Initial condition (interior only)
U0 = zeros(size(x_in));

%% ---- HBM options ----
opts_fixed_pic = struct('method','picard','h',1e-4,'M',4);
opts_fixed_qlm = struct('method','qlm'   ,'h',1e-2,'M',4);

opts_adapt_pic = struct('method','picard','h',1e-2,'M',4,'tol',1e-12,'fac',0.99);
opts_adapt_qlm = struct('method','qlm'   ,'h',1e-2,'M',4,'tol',1e-12,'fac',0.99);

ode45_opts = odeset('RelTol',1e-12,'AbsTol',1e-14);

%% ---- Run HBM: fixed Picard ----
tic;
sol_hbm_pic = hbmivp1(odefun, tspan, U0, opts_fixed_pic);
time_hbm_pic = toc;

U_hbm_pic_full = [zeros(size(sol_hbm_pic.Y,1),1), sol_hbm_pic.Y, zeros(size(sol_hbm_pic.Y,1),1)];
[X_pic, T_pic] = meshgrid(x, sol_hbm_pic.t);
U_exact_pic = (1 - exp(-T_pic)) .* sin( (pi/2)*(X_pic + 1) );
maxerr_hbm_pic = max(abs(U_hbm_pic_full(:) - U_exact_pic(:)));
pts_hbm_pic = numel(sol_hbm_pic.t);

%% ---- Run HBM: fixed QLM ----
tic;
sol_hbm_qlm = hbmivp1(odefun, tspan, U0, opts_fixed_qlm);
time_hbm_qlm = toc;

U_hbm_qlm_full = [zeros(size(sol_hbm_qlm.Y,1),1), sol_hbm_qlm.Y, zeros(size(sol_hbm_qlm.Y,1),1)];
[X_qlm, T_qlm] = meshgrid(x, sol_hbm_qlm.t);
U_exact_qlm = (1 - exp(-T_qlm)) .* sin( (pi/2)*(X_qlm + 1) );
maxerr_hbm_qlm = max(abs(U_hbm_qlm_full(:) - U_exact_qlm(:)));
pts_hbm_qlm = numel(sol_hbm_qlm.t);

%% ---- Run HBM: adaptive Picard ----
tic;
sol_ahbm_pic = ahbmivp1(odefun, tspan, U0, opts_adapt_pic);
time_ahbm_pic = toc;

U_ahbm_pic_full = [zeros(size(sol_ahbm_pic.Y,1),1), sol_ahbm_pic.Y, zeros(size(sol_ahbm_pic.Y,1),1)];
[X_ap, T_ap] = meshgrid(x, sol_ahbm_pic.t);
U_exact_ap = (1 - exp(-T_ap)) .* sin( (pi/2)*(X_ap + 1) );
maxerr_ahbm_pic = max(abs(U_ahbm_pic_full(:) - U_exact_ap(:)));
pts_ahbm_pic = numel(sol_ahbm_pic.t);

%% ---- Run HBM: adaptive QLM ----
tic;
sol_ahbm_qlm = ahbmivp1(odefun, tspan, U0, opts_adapt_qlm);
time_ahbm_qlm = toc;

U_ahbm_qlm_full = [zeros(size(sol_ahbm_qlm.Y,1),1), sol_ahbm_qlm.Y, zeros(size(sol_ahbm_qlm.Y,1),1)];
[X_aq, T_aq] = meshgrid(x, sol_ahbm_qlm.t);
U_exact_aq = (1 - exp(-T_aq)) .* sin( (pi/2)*(X_aq + 1) );
maxerr_ahbm_qlm = max(abs(U_ahbm_qlm_full(:) - U_exact_aq(:)));
pts_ahbm_qlm = numel(sol_ahbm_qlm.t);

%% ---- Run ode45 for comparison ----
tic;
[t45, U45] = ode45(odefun, tspan, U0, ode45_opts);
time_ode45 = toc;

U45_full = [zeros(size(U45,1),1), U45, zeros(size(U45,1),1)];
[X45, T45] = meshgrid(x, t45);
U_exact_45 = (1 - exp(-T45)) .* sin( (pi/2)*(X45 + 1) );
maxerr_ode45 = max(abs(U45_full(:) - U_exact_45(:)));
pts_ode45 = numel(t45);

%% ---- Display summary ----
fprintf('\nExample 9: Heat equation with source — Errors, time, points\n');
fprintf('%-25s  %-10s  %-10s  %-12s\n','Method','Time (s)','Points','Max Abs Error');
fprintf('%-25s  %-10.4f  %-10d  %-12.3e\n','HBM Picard', time_hbm_pic,  pts_hbm_pic,  maxerr_hbm_pic);
fprintf('%-25s  %-10.4f  %-10d  %-12.3e\n','HBM QLM',   time_hbm_qlm,  pts_hbm_qlm,  maxerr_hbm_qlm);
fprintf('%-25s  %-10.4f  %-10d  %-12.3e\n','AHBM Picard',time_ahbm_pic, pts_ahbm_pic, maxerr_ahbm_pic);
fprintf('%-25s  %-10.4f  %-10d  %-12.3e\n','AHBM QLM',   time_ahbm_qlm, pts_ahbm_qlm, maxerr_ahbm_qlm);
fprintf('%-25s  %-10.4f  %-10d  %-12.3e\n','ode45',             time_ode45,    pts_ode45,    maxerr_ode45);

%% ---- Visualize error surfaces for each method ----
figure; surf(X_pic, T_pic, abs(U_hbm_pic_full - U_exact_pic)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('HBM Picard error');

figure; surf(X_qlm, T_qlm, abs(U_hbm_qlm_full - U_exact_qlm)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('HBM QLM error');

figure; surf(X_ap, T_ap, abs(U_ahbm_pic_full - U_exact_ap)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Adaptive HBM Picard error');

figure; surf(X_aq, T_aq, abs(U_ahbm_qlm_full - U_exact_aq)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('Adaptive HBM QLM error');

figure; surf(X45, T45, abs(U45_full - U_exact_45)); shading interp; colorbar;
xlabel('$x$','interpreter', 'latex'); ylabel('$t$','interpreter', 'latex'); zlabel('|Error|'); title('ode45 error');
