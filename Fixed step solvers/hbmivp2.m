% AUTHOR:
%   SD Oloniiju, Rhodes University, South Africa
function sol = hbmivp2(odefunc, tspan, y0, yprime0, opts)
    % hbmivp2 - Solves second-order IVPs using Hybrid Block Method.
    %
    % USAGE:
    %   sol = hbmivp2(f, tspan, y0, yprime0)
    %   sol = hbmivp2(f, tspan, y0, yprime0, opts)
    %
    % INPUTS:
    %   odefunc   - Function handle: f(t, y, y')
    %   tspan     - [t0, tf]
    %   y0        - Initial condition y(t0)
    %   yprime0   - Initial derivative y'(t0)
    %   opts      - Optional struct: h, M, method
    %
    % OUTPUTS:
    %   sol.t     - Vector of main time grid
    %   sol.Y     - Solution y at main grid
    %   sol.Y1    - First derivative y' at main grid

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    allowedFields = {'h', 'M', 'method'};
    optFields = fieldnames(opts);
    unknown = setdiff(optFields, allowedFields);
    if ~isempty(unknown)
        error('Unknown option field(s): %s', strjoin(unknown, ', '));
    end

    h = getfielddefault(opts, 'h', 0.01);
    M = getfielddefault(opts, 'M', 3);
    method = getfielddefault(opts, 'method', 'picard');

    Tol = 1e-13;   % tolerance for iteration convergence
    maxIter = 100; % max iterations per block

    % Validate inputs
    if numel(tspan) ~= 2
        error('tspan must be a 2-element vector [t0, tf].');
    end
    if numel(y0) ~= numel(yprime0)
        error('Initial y0 and yprime0 must be same length vectors.');
    end
    d = numel(y0);
    testval = odefunc(tspan(1), y0(:), yprime0(:));
    if numel(testval) ~= d
        error('odefunc must return a vector of same length as y0.');
    end

    % Time grid
    tgrid = tspan(1):h:tspan(2);
    N = length(tgrid);
    s = M + 1;

    % Get block matrices
    [p, A, B] = getEquispacedMatrices(M);

    % Preallocate storage
    Y = zeros(s, d, N-1);
    Y1 = zeros(s, d, N-1);
    t = zeros(s, N-1);

    % Initial conditions
    yn = y0(:);
    yn1 = yprime0(:);

    switch lower(method)
        case 'picard'
            % Picard solver
            for n = 1:N-1
                t(:, n) = tgrid(n) + h * p(:);
                tnp = t(:, n);

                ynp = repmat(yn.', s, 1);
                ynp1 = repmat(yn1.', s, 1);

                for iter = 1:maxIter
                    yprev = ynp;
                    y1prev = ynp1;

                    Fnp = zeros(s, d);
                    for i = 1:s
                        Fnp(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).');
                    end

                    Yn1 = repmat(yn1.', M, 1);
                    Yn = repmat(yn.', M, 1);

                    Ynp = Yn + h * diag(p(2:end)) * Yn1 + (h^2) * (B * Fnp);
                    Ynp1 = Yn1 + h * (A * Fnp);

                    ynp = [yn.'; Ynp];
                    ynp1 = [yn1.'; Ynp1];

                    err = max(norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf));
                    if err < Tol
                        break;
                    end
                    if iter == maxIter
                        warning('Block %d: Picard iteration did not converge in maxIter.', n);
                    end
                end

                Y(:, :, n) = ynp;
                Y1(:, :, n) = ynp1;

                yn = ynp(end, :).';
                yn1 = ynp1(end, :).';
            end

        case 'qlm'
            % QLM solver
            for n = 1:N-1
                t(:, n) = tgrid(n) + h * p(:);
                tnp = t(:, n);

                ynp = repmat(yn.', s, 1);
                ynp1 = repmat(yn1.', s, 1);

                for iter = 1:maxIter
                    yprev = ynp;
                    y1prev = ynp1;

                    F = zeros(s, d);
                    Jy = zeros(s, d, d);
                    Jyp = zeros(s, d, d);

                    for i = 1:s
                        F(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).');
                        [Jy(i, :, :), Jyp(i, :, :)] = numericalJacobian(odefunc, tnp(i), ynp(i, :).', ynp1(i, :).');
                    end

                    L11 = zeros(d * M, d * M);
                    L12 = zeros(d * M, d * M);
                    L21 = zeros(d * M, d * M);
                    L22 = zeros(d * M, d * M);
                    R1 = zeros(d * M, 1);
                    R2 = zeros(d * M, 1);

                    for i = 1:d
                        Ry = repmat(yn(i), M, 1) + h * p(2:end) * yn1(i);
                        Ry1 = repmat(yn1(i), M, 1);

                        for j = 1:d
                            JY = squeeze(Jy(:, j, i));
                            JYP = squeeze(Jyp(:, j, i));

                            Dy = diag(JY(2:end));
                            Dyp = diag(JYP(2:end));

                            B11 = -(h^2) * B(:, 2:end) * Dy;
                            B12 = -(h^2) * B(:, 2:end) * Dyp;
                            A21 = -h * A(:, 2:end) * Dy;
                            A22 = -h * A(:, 2:end) * Dyp;

                            if i == j
                                B11 = eye(M) + B11;
                                A22 = eye(M) + A22;
                            end

                            row_idx = (i - 1) * M + (1:M);
                            col_idx = (j - 1) * M + (1:M);

                            L11(row_idx, col_idx) = B11;
                            L12(row_idx, col_idx) = B12;
                            L21(row_idx, col_idx) = A21;
                            L22(row_idx, col_idx) = A22;

                            Ry = Ry + (h^2) * B(:, 1) * (Jy(1, j, i) * yn(j)) + (h^2) * B(:, 1) * (Jyp(1, j, i) * yn1(j));
                            Ry1 = Ry1 + h * A(:, 1) * (Jy(1, j, i) * yn(j)) + h * A(:, 1) * (Jyp(1, j, i) * yn1(j));
                        end

                        nonlinear_term = F(:, i);
                        for j = 1:d
                            nonlinear_term = nonlinear_term - Jy(:, j, i) .* ynp(:, j) - Jyp(:, j, i) .* ynp1(:, j);
                        end

                        Ry = Ry + (h^2) * B * nonlinear_term;
                        Ry1 = Ry1 + h * A * nonlinear_term;

                        idx = (i - 1) * M + (1:M);
                        R1(idx) = Ry;
                        R2(idx) = Ry1;
                    end

                    LHS = [L11, L12; L21, L22];
                    RHS = [R1; R2];

                    Ynp_combined = LHS \ RHS;
                    Ynp = reshape(Ynp_combined(1:d * M), M, d);
                    Ynp1 = reshape(Ynp_combined(d * M + 1:end), M, d);

                    ynp = [yn.'; Ynp];
                    ynp1 = [yn1.'; Ynp1];

                    err = max(norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf));
                    if err < Tol
                        break;
                    end
                    if iter == maxIter
                        warning('Block %d: QLM did not converge in maxIter.', n);
                    end
                end

                Y(:, :, n) = ynp;
                Y1(:, :, n) = ynp1;

                yn = ynp(end, :).';
                yn1 = ynp1(end, :).';
            end

        otherwise
            error('Unknown method "%s". Use "picard" or "qlm".', method);
    end

    % Extract main nodes solution
    Ymain = zeros(N, d);
    Y1main = zeros(N, d);
    Ymain(1, :) = y0.';
    Y1main(1, :) = yprime0.';
    for n = 1:N-1
        Ymain(n + 1, :) = Y(end, :, n);
        Y1main(n + 1, :) = Y1(end, :, n);
    end
    sol.t = tgrid.';
    sol.Y = Ymain;
    sol.Y1 = Y1main;

    
end

% -------- Helper functions --------
function [p, A, B] = getEquispacedMatrices(M)
    p = (0:M)' / M;
    Afull = generateCoefficientMatrix(p, 1);
    Bfull = generateCoefficientMatrix(p, 2);
    A = Afull(2:end, :);
    B = Bfull(2:end, :);
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
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k - power)) * ((-1)^power);
            end
            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);
            indef_integral_coeffs = polyint(integrand_coeffs);
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);
            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

function [Jy, Jyp] = numericalJacobian(odefunc, t, y, yp, epsilon)
    if nargin < 5, epsilon = 1e-8; end
    d = length(y);
    Jy = zeros(d);
    Jyp = zeros(d);
    for j = 1:d
        ey = zeros(d, 1); ey(j) = 1;
        Jy(:, j) = (odefunc(t, y + epsilon * ey, yp) - odefunc(t, y - epsilon * ey, yp)) / (2 * epsilon);
        Jyp(:, j) = (odefunc(t, y, yp + epsilon * ey) - odefunc(t, y, yp - epsilon * ey)) / (2 * epsilon);
    end
end

function val = getfielddefault(S, field, default)
    if isfield(S, field)
        val = S.(field);
    else
        val = default;
    end
end