% AUTHOR:
%       SD Oloniiju, Rhodes University, South Africa
function sol = hbmivp3(odefunc, tspan, y0, yp0, ypp0, opts)
    % hbmivp3 - Hybrid Block Method solver for third-order IVPs.
    %
    % SYNTAX:
    %   sol = hbmivp3(odefunc, tspan, y0, yp0, ypp0)
    %   sol = hbmivp3(..., opts)
    %
    % INPUTS:
    %   odefunc   - function handle: f(t, y, y', y''), returns (d x 1)
    %   tspan     - [t0, tf], time interval
    %   y0        - initial value of y (d x 1)
    %   yp0       - initial value of y' (d x 1)
    %   ypp0      - initial value of y'' (d x 1)
    %   opts      - optional arguments (struct)
    %
    % OUTPUTS:
    %   sol.t     - time vector at main grid points
    %   sol.Y     - solution y at main points (N x d)
    %   sol.Y1    - first derivative y' at main points (N x d)
    %   sol.Y2    - second derivative y'' at main points (N x d)
    %
    % OPTIONS (opts):
    %   h         - Block step size (default: 0.01)
    %   M         - Number of intra-block collocation points (default: 3)
    %   method    - 'picard' or 'qlm' (default: 'picard')

    if nargin < 6 || isempty(opts)
        opts = struct();  % Make optional
    end

    allowedFields = {'h', 'M', 'method'};
    optFields = fieldnames(opts);
    unknown = setdiff(optFields, allowedFields);
    if ~isempty(unknown)
        error('Unknown option field(s): %s', strjoin(unknown, ', '));
    end

    % Retrieve options with defaults
    h       = getfielddefault(opts, 'h', 0.01);
    M       = getfielddefault(opts, 'M', 3);
    method  = getfielddefault(opts, 'method', 'picard');

    Tol = 1e-13;   % tolerance for iteration convergence
    maxIter = 100; % max iterations per block

    % Validate input sizes and function output dimension
    d = numel(y0);
    testval = odefunc(tspan(1), y0(:), yp0(:), ypp0(:));
    if numel(testval) ~= d
        error('odefunc output dimension does not match initial condition dimension.');
    end
    if numel(tspan) ~= 2
        error('tspan must be a 2-element vector [t0 tf].');
    end

    % Create time grid of main points
    tgrid = tspan(1):h:tspan(2);
    N = length(tgrid);       % number of main points
    s = M + 1;               % number of points per block (including start)

    % Preallocate solution arrays
    Ymain = zeros(N, d);
    Y1main = zeros(N, d);
    Y2main = zeros(N, d);

    % Set initial conditions for first main point
    Ymain(1,:) = y0.';
    Y1main(1,:) = yp0.';
    Y2main(1,:) = ypp0.';

    % Get equispaced block matrices (p, A, B, C)
    [p, A, B, C] = getEquispacedMatrices(M);

    % Initialize values for first block start
    yn = y0(:);
    yn1 = yp0(:);
    yn2 = ypp0(:);

    % ----
    % Choose the solver method and embed the entire solver loop inline
    switch lower(method)
        case 'picard'
            % --- Block Picard solver ---
            for n = 1:N-1
                t_block = tgrid(n) + h * p(:); % time nodes in current block

                % Initial guess: constant extrapolation of initial values
                ynp = repmat(yn.', s, 1);   % y values (s x d)
                ynp1 = repmat(yn1.', s, 1); % y' values
                ynp2 = repmat(yn2.', s, 1); % y'' values

                for iter = 1:maxIter
                    yprev = ynp;
                    y1prev = ynp1;
                    y2prev = ynp2;

                    % Evaluate right-hand side at all collocation points
                    Fnp = zeros(s, d);
                    for i = 1:s
                        Fnp(i,:) = odefunc(t_block(i), ynp(i,:).', ynp1(i,:).', ynp2(i,:).');
                    end

                    % Diagonal matrices of intra-block collocation points (excluding first node)
                    P1 = diag(p(2:end));
                    P2 = diag(p(2:end).^2);

                    % Values at start of block, replicated for M points
                    Yn  = repmat(yn.', M, 1);
                    Yn1 = repmat(yn1.', M, 1);
                    Yn2 = repmat(yn2.', M, 1);

                    % Picard iteration formulas for inner points (size M x d)
                    Ynp  = Yn  + h*P1*Yn1 + 0.5*(h^2)*P2*Yn2 + (h^3) * (C * Fnp);
                    Ynp1 = Yn1 + h*P1*Yn2 + (h^2) * (B * Fnp);
                    Ynp2 = Yn2 + h * (A * Fnp);

                    % Add back initial point at top to get full s x d matrices
                    ynp  = [yn.'; Ynp];
                    ynp1 = [yn1.'; Ynp1];
                    ynp2 = [yn2.'; Ynp2];

                    % Check convergence: max norm of updates over y, y', y''
                    err = max([norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf), norm(ynp2 - y2prev, inf)]);
                    if err < Tol
                        break;
                    end

                    if iter == maxIter
                        warning('Block %d: Picard iteration did not converge within maxIter.', n);
                    end
                end

                % Save block solution
                Ymain(n+1,:) = ynp(end, :);
                Y1main(n+1,:) = ynp1(end, :);
                Y2main(n+1,:) = ynp2(end, :);

                % Update initial values for next block
                yn = ynp(end, :).';
                yn1 = ynp1(end, :).';
                yn2 = ynp2(end, :).';
            end

        case 'qlm'
            % --- Block Quasilinearization Method (QLM) ---
            for n = 1:N-1
                t_block = tgrid(n) + h * p(:);

                % Initial guess
                ynp = repmat(yn.', s, 1);
                ynp1 = repmat(yn1.', s, 1);
                ynp2 = repmat(yn2.', s, 1);

                for iter = 1:maxIter
                    yprev = ynp;
                    y1prev = ynp1;
                    y2prev = ynp2;

                    % Evaluate function and Jacobians at collocation points
                    F = zeros(s, d);
                    Jy = zeros(s, d, d);
                    Jyp = zeros(s, d, d);
                    Jypp = zeros(s, d, d);

                    for i = 1:s
                        F(i,:) = odefunc(t_block(i), ynp(i,:).', ynp1(i,:).', ynp2(i,:).');
                        [Jy(i,:,:), Jyp(i,:,:), Jypp(i,:,:)] = numericalJacobian(odefunc, t_block(i), ynp(i,:).', ynp1(i,:).', ynp2(i,:).');
                    end

                    % Assemble block linear system for updates
                    % Matrices sizes: d*M by d*M
                    L11 = zeros(d*M,d*M); L12 = zeros(d*M,d*M); L13 = zeros(d*M,d*M);
                    L21 = zeros(d*M,d*M); L22 = zeros(d*M,d*M); L23 = zeros(d*M,d*M);
                    L31 = zeros(d*M,d*M); L32 = zeros(d*M,d*M); L33 = zeros(d*M,d*M);
                    R1 = zeros(d*M,1);
                    R2 = zeros(d*M,1);
                    R3 = zeros(d*M,1);

                    for i = 1:d
                        % Precompute vectors for block RHS for each equation i
                        Ry = repmat(yn(i), M, 1) + h * (p(2:end)) * yn1(i) + 0.5*(h^2) * (p(2:end).^2) * yn2(i);
                        Ry1 = repmat(yn1(i), M, 1) + h * (p(2:end)) * yn2(i);
                        Ry2 = repmat(yn2(i), M, 1);

                        for j = 1:d
                            JY = squeeze(Jy(:,j,i));     % s x 1
                            JYP = squeeze(Jyp(:,j,i));
                            JYPP = squeeze(Jypp(:,j,i));

                            Dy = diag(JY(2:end));
                            Dyp = diag(JYP(2:end));
                            Dypp = diag(JYPP(2:end));

                            % Left hand side matrices with identity added for diagonal blocks
                            C11 = -(h^3)*C(:,2:end)*Dy;
                            C12 = -(h^3)*C(:,2:end)*Dyp;
                            C13 = -(h^3)*C(:,2:end)*Dypp;

                            B21 = -(h^2)*B(:,2:end)*Dy;
                            B22 = -(h^2)*B(:,2:end)*Dyp;
                            B23 = -(h^2)*B(:,2:end)*Dypp;

                            A31 = -h*A(:,2:end)*Dy;
                            A32 = -h*A(:,2:end)*Dyp;
                            A33 = -h*A(:,2:end)*Dypp;

                            if i == j
                                C11 = eye(M) + C11;
                                B22 = eye(M) + B22;
                                A33 = eye(M) + A33;
                            end

                            row_idx = (i-1)*M + (1:M);
                            col_idx = (j-1)*M + (1:M);

                            L11(row_idx, col_idx) = C11;
                            L12(row_idx, col_idx) = C12;
                            L13(row_idx, col_idx) = C13;

                            L21(row_idx, col_idx) = B21;
                            L22(row_idx, col_idx) = B22;
                            L23(row_idx, col_idx) = B23;

                            L31(row_idx, col_idx) = A31;
                            L32(row_idx, col_idx) = A32;
                            L33(row_idx, col_idx) = A33;

                            % Add contributions from Jacobians at first node
                            Ry = Ry + (h^3) * C(:,1) * (Jy(1,j,i)*yn(j) + Jyp(1,j,i)*yn1(j) + Jypp(1,j,i)*yn2(j));
                            Ry1 = Ry1 + (h^2) * B(:,1) * (Jy(1,j,i)*yn(j) + Jyp(1,j,i)*yn1(j) + Jypp(1,j,i)*yn2(j));
                            Ry2 = Ry2 + h * A(:,1) * (Jy(1,j,i)*yn(j) + Jyp(1,j,i)*yn1(j) + Jypp(1,j,i)*yn2(j));
                        end

                        % Nonlinear residual term for right hand side
                        nonlinear_term = F(:,i);
                        for j = 1:d
                            nonlinear_term = nonlinear_term - Jy(:,j,i).*ynp(:,j) - Jyp(:,j,i).*ynp1(:,j) - Jypp(:,j,i).*ynp2(:,j);
                        end

                        Ry = Ry + (h^3)*C*nonlinear_term;
                        Ry1 = Ry1 + (h^2)*B*nonlinear_term;
                        Ry2 = Ry2 + h*A*nonlinear_term;

                        idx = (i-1)*M + (1:M);
                        R1(idx) = Ry;
                        R2(idx) = Ry1;
                        R3(idx) = Ry2;
                    end

                    % Assemble full system and solve
                    LHS = [L11, L12, L13;
                           L21, L22, L23;
                           L31, L32, L33];
                    RHS = [R1; R2; R3];

                    sol_block = LHS \ RHS;

                    Ynp = reshape(sol_block(1:d*M), M, d);
                    Ynp1 = reshape(sol_block(d*M+1:2*d*M), M, d);
                    Ynp2 = reshape(sol_block(2*d*M+1:end), M, d);

                    % Update full vectors with initial values
                    ynp  = [yn.'; Ynp];
                    ynp1 = [yn1.'; Ynp1];
                    ynp2 = [yn2.'; Ynp2];

                    % Check convergence
                    err = max([norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf), norm(ynp2 - y2prev, inf)]);
                    if err < Tol
                        break;
                    end

                    if iter == maxIter
                        warning('Block %d: QLM did not converge within maxIter.', n);
                    end
                end

                % Save solution for block
                Ymain(n+1,:) = ynp(end, :);
                Y1main(n+1,:) = ynp1(end, :);
                Y2main(n+1,:) = ynp2(end, :);

                % Prepare initial conditions for next block
                yn = ynp(end, :).';
                yn1 = ynp1(end, :).';
                yn2 = ynp2(end, :).';
            end

        otherwise
            error('Unknown method "%s". Choose "picard" or "qlm".', method);
    end

    % Package output
    sol.t = tgrid.';
    sol.Y = Ymain;
    sol.Y1 = Y1main;
    sol.Y2 = Y2main;
end


%% === Helper Functions ===

function [p, A, B, C] = getEquispacedMatrices(M)
    % Returns collocation points and integration matrices for the block method
    % Inputs:
    %   M - number of intra-block collocation points
    % Outputs:
    %   p - collocation points (M+1 x 1)
    %   A, B, C - coefficient matrices for 1st, 2nd, 3rd integrals of Lagrange polynomials

    p = (0:M)' / M;              % equispaced points in [0,1]
    Afull = generateCoefficientMatrix(p,1);
    Bfull = generateCoefficientMatrix(p,2);
    Cfull = generateCoefficientMatrix(p,3);

    % Remove first row (corresponds to initial point)
    A = Afull(2:end, :);
    B = Bfull(2:end, :);
    C = Cfull(2:end, :);
end

function A = generateCoefficientMatrix(p, num_integrations)
    % Generates the matrix of coefficients for n-fold integration of Lagrange polynomials
    % p: collocation points vector (M+1 x 1)
    % num_integrations: integer > 0, number of integrations

    if num_integrations < 1 || floor(num_integrations) ~= num_integrations
        error('Number of integrations must be a positive integer.');
    end

    M = length(p) - 1;
    A = zeros(M + 1, M + 1);
    n = num_integrations;
    pre_factor = 1 / factorial(n - 1);

    for j_idx = 1:(M + 1)
        j = j_idx - 1;
        % Compute Lagrange polynomial coefficients for node j
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

            % Compute polynomial (p_i - s)^(n-1) using binomial theorem
            k = n - 1;
            term_coeffs = zeros(1, k + 1);
            for l = 0:k
                power = l;
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k - power)) * ((-1)^power);
            end

            % Convolve lagrange poly with (p_i - s)^(n-1)
            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);

            % Integrate polynomial (antiderivative)
            indef_integral_coeffs = polyint(integrand_coeffs);

            % Evaluate definite integral from 0 to p_i
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);

            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

%% === JACOBIAN MATRIX ===
function [Jy, Jyp, Jypp] = numericalJacobian(odefunc, t, y, yp, ypp, epsilon)
%numericalJacobian3 - Computes Jacobians of f(t, y, y', y'') w.r.t y, y', y''
%   using central finite differences
%
% INPUTS:
%   odefunc - function handle: f(t, y, y', y'')
%   t       - scalar time
%   y       - column vector of y
%   yp      - column vector of y'
%   ypp     - column vector of y''
%   epsilon - (optional) perturbation step size (default 1e-8)
%
% OUTPUTS:
%   Jy   - Jacobian ∂f/∂y   (d × d)
%   Jyp  - Jacobian ∂f/∂y'  (d × d)
%   Jypp - Jacobian ∂f/∂y'' (d × d)

    if nargin < 6
        epsilon = 1e-8;
    end

    d = length(y);
    Jy = zeros(d, d);
    Jyp = zeros(d, d);
    Jypp = zeros(d, d);

    % ∂f/∂y
    for j = 1:d
        ey = zeros(d,1); ey(j) = 1;

        f_plus = odefunc(t, y + epsilon*ey, yp, ypp);
        f_minus = odefunc(t, y - epsilon*ey, yp, ypp);

        Jy(j, :) = (f_plus - f_minus) / (2 * epsilon);
    end

    % ∂f/∂y'
    for j = 1:d
        eyp = zeros(d,1); eyp(j) = 1;

        f_plus = odefunc(t, y, yp + epsilon*eyp, ypp);
        f_minus = odefunc(t, y, yp - epsilon*eyp, ypp);

        Jyp(j, :) = (f_plus - f_minus) / (2 * epsilon);
    end

    % ∂f/∂y''
    for j = 1:d
        eypp = zeros(d,1); eypp(j) = 1;

        f_plus = odefunc(t, y, yp, ypp + epsilon*eypp);
        f_minus = odefunc(t, y, yp, ypp - epsilon*eypp);

        Jypp(j, :) = (f_plus - f_minus) / (2 * epsilon);
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



