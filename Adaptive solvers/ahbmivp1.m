% AUTHOR:
%   SD Oloniiju, Rhodes University, South Africa
function sol = ahbmivp1(odefunc, tspan, y0, opts)
    % ahbmivp1 - Adaptive Hybrid Block Method solver for first-order IVPs
    %
    % USAGE:
    %   sol = ahbmivp1(odefunc, tspan, y0)
    %   sol = ahbmivp1(odefunc, tspan, y0, opts)
    %
    % INPUTS:
    %   odefunc   - function handle @(t, y), returning (d x 1) column vector
    %   tspan     - [t0, tf]
    %   y0        - initial condition (column vector)
    %   opts      - optional struct, can include: h, M, method, tol, fac
    %
    % OUTPUTS:
    %   sol.t     - time grid at main nodes
    %   sol.Y     - solution at each main node (N x d)

    if nargin < 4 || isempty(opts)
        opts = struct();  % Make optional
    end

    % === Allowed optional fields ===
    allowedFields = {'h','M','method','tol','fac'};

    % Check for unrecognized fields
    optFields = fieldnames(opts);
    unknown = setdiff(optFields, allowedFields);
    if ~isempty(unknown)
        error('Unknown option field(s): %s', strjoin(unknown, ', '));
    end

    % Default values
    h        = getfielddefault(opts, 'h', 0.1);
    M        = getfielddefault(opts, 'M', 3);
    method   = getfielddefault(opts, 'method', 'picard');
    tol      = getfielddefault(opts, 'tol', 1e-15);
    fac      = getfielddefault(opts, 'fac', 0.9);

    Tol = 1e-13;       % linearization iteration tolerance
    maxIter = 100;     % max iterations per block for linearization

    % Precompute block matrices and parameters
    [p, A]    = getEquispacedMatrices(M);    % main collocation matrix
    A_low     = getEquispacedMatricesLow(M); % lower-order collocation matrix for error est
    s         = M + 1;                        % nodes per block
    d         = length(y0);                   % system dimension

    % Initialize solution storage
    tcur = tspan(1);
    yn   = y0(:);
    n    = 1;
    Y    = [];
    T    = [];

    % Main adaptive stepping loop
    while tcur < tspan(2)
        % Adjust step size to avoid overshoot
        if tcur + h > tspan(2)
            h = tspan(2) - tcur;
        end

        % Collocation nodes for current block
        tnp = tcur + h * p(:);

        % Initial guess for intrastep values
        ynp = repmat(yn.', s, 1); % s x d matrix

        if strcmpi(method, 'picard')
            % --- Adaptive Picard iteration ---
            for iter = 1:maxIter
                yprev = ynp;

                % Evaluate f(t,y) at block nodes
                Fnp = zeros(s, d);
                for i = 1:s
                    Fnp(i, :) = odefunc(tnp(i), ynp(i, :).').';
                end

                % Update internal nodes
                Yn  = repmat(yn.', M, 1);
                Ynp = Yn + h * (A * Fnp);
                ynp = [yn.'; Ynp];

                % Check convergence
                if norm(ynp - yprev, inf) < Tol
                    break
                end
            end

        elseif strcmpi(method, 'qlm')
            % --- Adaptive Quasi-Linearization method ---
            for iter = 1:maxIter
                yprev = ynp;

                % Evaluate f and Jacobian at each node
                F = zeros(s, d);
                J = zeros(s, d, d);
                for i = 1:s
                    yi = ynp(i, :).'; % column vector
                    F(i, :) = odefunc(tnp(i), yi).';
                    J(i, :, :) = numericalJacobian(odefunc, tnp(i), yi);
                end

                % Assemble global LHS and RHS for QLM linear system
                LHS = zeros(d*M, d*M);
                RHS = zeros(d*M, 1);

                for ii = 1:d
                    Ri = repmat(yn(ii), M, 1);

                    for jj = 1:d
                        Jblock = squeeze(J(:, jj, ii));
                        D = diag(Jblock(2:end));
                        Aij = -h * A(:, 2:end) * D;
                        if ii == jj
                            Aij = eye(M) + Aij;
                        end

                        row_idx = (ii-1)*M + (1:M);
                        col_idx = (jj-1)*M + (1:M);
                        LHS(row_idx, col_idx) = Aij;

                        Ri = Ri + h * A(:,1) * (J(1, jj, ii) * yn(jj));
                    end

                    nonlinear_term = F(:, ii);
                    for jj = 1:d
                        nonlinear_term = nonlinear_term - J(:, jj, ii) .* ynp(:, jj);
                    end
                    Ri = Ri + h * A * nonlinear_term;

                    idx = (ii-1)*M + (1:M);
                    RHS(idx) = Ri;
                end

                % Solve for internal node values update
                Yflat = LHS \ RHS;
                Ynp_block = reshape(Yflat, M, d);
                ynp = [yn.'; Ynp_block];

                % Check convergence
                if norm(ynp - yprev, inf) < Tol
                    break
                end

                if iter == maxIter
                    warning('QLM did not converge at t=%.5g (block %d).', tcur, n);
                end
            end
        else
            error('Unknown method "%s". Use "picard" or "qlm".', method);
        end

        % Local error estimate (embedded lower order)
        Fnp = zeros(s, d);
        for i = 1:s
            Fnp(i, :) = odefunc(tnp(i), ynp(i, :).').';
        end
        est = h * ((A(end,:) - A_low(end,:)) * Fnp);
        est = norm(est, inf);

        % Accept/reject step
        if est < tol
            % Accept step
            yn = ynp(end, :).';
            tcur = tcur + h;

            Y(:, :, n) = ynp;
            T(:, n) = tnp;

            h = min(2*h, tspan(2) - tcur);
            n = n + 1;
        else
            % Reject step
            h = fac * h * (tol/est)^(1/(M+2));
        end
    end

    % Assemble main output grid and solution
    [~, ~, N] = size(Y);
    tgrid = zeros(N, 1);
    Ymain = zeros(N, d);

    tgrid(1) = tspan(1);
    Ymain(1,:) = y0.';

    for k = 1:N-1
        tgrid(k+1) = T(end, k);
        Ymain(k+1,:) = squeeze(Y(end, :, k));
    end

    % Package output
    sol.t = tgrid;
    sol.Y = Ymain;
end


%% === Generate Coefficient Matrices for Equally Spaced Points ===
function [p, A] = getEquispacedMatrices(M)
    p = (0:M)' / M;
    Afull = generateCoefficientMatrix(p,1);
    A = Afull(2:end, :);
end

%% === Generate Coefficient Matrices for Lower Order (Embedded) ===
function [A] = getEquispacedMatricesLow(M)
    p = (0:M)' / M;
    Afull = generateCoeffMatrixLowerOrder(p,1);
    A = Afull(2:end, :);
end

%% === Numerical Jacobian ===
function J = numericalJacobian(f, t, y)
    d = length(y);
    J = zeros(d);
    eps = 1e-8;
    for j = 1:d
        e = zeros(d,1); e(j) = 1;
        J(:,j) = (f(t, y + eps*e) - f(t, y - eps*e)) / (2*eps);
    end
end

%% === Generate n-Fold Integration Matrix using Cauchy's Formula ===
function A = generateCoefficientMatrix(p, num_integrations)
    if num_integrations < 1 || floor(num_integrations) ~= num_integrations
        error('Number of integrations must be a positive integer.');
    end

    M = length(p) - 1;
    A = zeros(M + 1, M + 1);
    n = num_integrations;
    pre_factor = 1 / factorial(n - 1);

    for j_idx = 1:(M + 1)
        j = j_idx - 1;
        num_poly_coeffs = 1;
        den_scalar = 1;
        for k_idx = 1:(M + 1)
            k = k_idx - 1;
            if k ~= j
                num_poly_coeffs = conv(num_poly_coeffs, [1, -p(k_idx)]);
                den_scalar = den_scalar * (p(j_idx) - p(k_idx));
            end
        end
        lagrange_coeffs = num_poly_coeffs / den_scalar;

        for i_idx = 1:(M + 1)
            upper_limit = p(i_idx);
            k = n - 1;
            term_coeffs = zeros(1, k + 1);
            for l = 0:k
                power = l;
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k-power)) * ((-1)^power);
            end

            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);
            indef_integral_coeffs = polyint(integrand_coeffs);
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);
            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

%% === Generate Lower Order Coefficient Matrix ===
function A = generateCoeffMatrixLowerOrder(p, num_integrations)
    if num_integrations < 1 || floor(num_integrations) ~= num_integrations
        error('Number of integrations must be a positive integer.');
    end

    M = length(p) - 1;
    A = zeros(M+1, M+1);
    n = num_integrations;
    pre_factor = 1 / factorial(n - 1);

    for j_idx = 1:(M)
        j = j_idx - 1;
        num_poly_coeffs = 1;
        den_scalar = 1;
        for k_idx = 1:(M)
            k = k_idx - 1;
            if k ~= j
                num_poly_coeffs = conv(num_poly_coeffs, [1, -p(k_idx)]);
                den_scalar = den_scalar * (p(j_idx) - p(k_idx));
            end
        end
        lagrange_coeffs = num_poly_coeffs / den_scalar;

        for i_idx = 1:(M + 1)
            upper_limit = p(i_idx);
            k = n - 1;
            term_coeffs = zeros(1, k + 1);
            for l = 0:k
                power = l;
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k-power)) * ((-1)^power);
            end

            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);
            indef_integral_coeffs = polyint(integrand_coeffs);
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);
            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

%function to get default optional values
function val = getfielddefault(S, field, default)
    if isfield(S, field)
        val = S.(field);
    else
        val = default;
    end
end
