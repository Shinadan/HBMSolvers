% AUTHOR:
%   SD Oloniiju, Rhodes University, South Africa
function sol = hbmivp1(odefunc, tspan, y0, opts)
    % hbmivp1 - Hybrid Block Method solver for first-order IVPs
    %
    % USAGE:
    %   sol = hbmivp1(odefunc, tspan, y0)
    %   sol = hbmivp1(odefunc, tspan, y0, opts)
    %
    % INPUTS:
    %   odefunc   - function handle @(t, y), returning (d x 1) column vector
    %   tspan     - [t0, tf]
    %   y0        - initial condition (column vector)
    %   opts      - optional struct, can include: h, M, method
    %
    % OUTPUTS:
    %   sol.t     - time grid at main nodes
    %   sol.Y     - solution at each main node (N x d)

    if nargin < 4 || isempty(opts)
        opts = struct();  % Make optional
    end

    % Allowed optional fields
    allowedFields = {'h','M','method'};
    optFields = fieldnames(opts);
    unknown = setdiff(optFields, allowedFields);
    if ~isempty(unknown)
        error('Unknown option field(s): %s', strjoin(unknown, ', '));
    end

    % Default values
    h        = getfielddefault(opts, 'h', 0.01);
    M        = getfielddefault(opts, 'M', 3);
    method   = getfielddefault(opts, 'method', 'picard');
    Tol = 1e-13; % iteration tolerance
    maxIter = 100; % max iterations per block

    % Input validation
    if numel(tspan) ~= 2
        error('Input "tspan" must be a 2-element vector specifying [t0, tf].');
    end
    testval = odefunc(tspan(1), y0(:));
    if numel(testval) ~= numel(y0)
        error('odefunc must return a vector of the same length as y0.');
    end

    tgrid = tspan(1):h:tspan(2);
    N = length(tgrid);
    d = length(y0);
    s = M + 1;

    % Preallocate solution arrays
    if d == 1
        Y = zeros(s, 1, N-1);
    else
        Y = zeros(s, d, N-1);
    end
    t = zeros(s, N-1);

    % Get nodes and matrix
    [p, A] = getEquispacedMatrices(M);

    yn = y0(:);

    switch lower(method)
        case 'picard'
            % -------------------
            % Picard solver
            % -------------------
            for n = 1:N-1
                t(:,n) = tgrid(n) + h * p(:);
                tnp = t(:,n);

                ynp = repmat(yn.', s, 1); % initial guess

                for iter = 1:maxIter
                    yprev = ynp;

                    Fnp = zeros(s, d);
                    for i = 1:s
                        Fnp(i, :) = odefunc(tnp(i), ynp(i, :).');
                    end

                    Yn = repmat(yn.', M, 1);
                    Ynp = Yn + h * (A * Fnp);
                    ynp = [yn.'; Ynp];

                    if norm(ynp - yprev, inf) < Tol
                        break;
                    end
                    if iter == maxIter
                        warning('Picard iteration did not converge at step %d.', n);
                    end
                end

                if d == 1
                    Y(:,1,n) = ynp;
                else
                    Y(:,:,n) = ynp;
                end

                yn = ynp(end, :).';
            end

        case 'qlm'
            % -------------------
            % QLM solver
            % -------------------
            for n = 1:N-1
                t(:,n) = tgrid(n) + h * p(:);
                tnp = t(:,n);
                ynp = repmat(yn.', s, 1);

                for iter = 1:maxIter
                    yprev = ynp;

                    F = zeros(s, d);
                    J = zeros(s, d, d);

                    for i = 1:s
                        F(i,:) = odefunc(tnp(i), ynp(i,:).');
                        J(i,:,:) = numericalJacobian(odefunc, tnp(i), ynp(i,:).');
                    end

                    LHS = zeros(d*M, d*M);
                    RHS = zeros(d*M, 1);

                    for i = 1:d
                        Ri = repmat(yn(i), M, 1);

                        for j = 1:d
                            Jblock = squeeze(J(:,j,i));
                            D = diag(Jblock(2:end));
                            Aij = -h * A(:,2:end) * D;
                            if i == j, Aij = eye(M) + Aij; end

                            row_idx = (i-1)*M + (1:M);
                            col_idx = (j-1)*M + (1:M);
                            LHS(row_idx, col_idx) = Aij;

                            Ri = Ri + h * A(:,1) * (J(1,j,i) * yn(j));
                        end

                        nonlinear_term = F(:,i);
                        for j = 1:d
                            nonlinear_term = nonlinear_term - J(:,j,i) .* ynp(:,j);
                        end
                        Ri = Ri + h * A * nonlinear_term;

                        idx = (i-1)*M + (1:M);
                        RHS(idx) = Ri;
                    end

                    Ynp = LHS \ RHS;
                    ynp = [yn.'; reshape(Ynp, M, d)];

                    if norm(ynp - yprev, inf) < Tol
                        break;
                    end
                    if iter == maxIter
                        warning('Block %d: QLM did not converge.', n);
                    end
                end

                if d == 1
                    Y(:,1,n) = ynp;
                else
                    Y(:,:,n) = ynp;
                end

                yn = ynp(end, :).';
            end
        otherwise
            error('Unknown method "%s". Use "picard" or "qlm".', method);
    end

    % Extract main nodes solution
    Ymain = zeros(N, d);
    Ymain(1,:) = y0.';
    for n = 1:N-1
        if d == 1
            Ymain(n+1,:) = Y(end,1,n);
        else
            Ymain(n+1,:) = Y(end,:,n);
        end
    end

    % Package output
    sol.t = tgrid.';
    sol.Y = Ymain;
end

% ---- Helper functions ----
function J = numericalJacobian(f, t, y)
    d = length(y); J = zeros(d);
    eps = 1e-8;
    for j = 1:d
        e = zeros(d,1); e(j) = 1;
        J(:,j) = (f(t, y + eps*e) - f(t, y - eps*e)) / (2*eps);
    end
end

function [p, A] = getEquispacedMatrices(M)
    p = (0:M)' / M;
    Afull = generateCoefficientMatrix(p,1);
    A = Afull(2:end, :);
end

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

function val = getfielddefault(S, field, default)
    if isfield(S, field)
        val = S.(field);
    else
        val = default;
    end
end
