% AUTHOR:
%   SD Oloniiju, Rhodes University, South Africa
function sol = ahbmivp2(odefunc, tspan, y0, yprime0, opts)
%AHBMIVP2  Adaptive Hybrid Block Method for second-order IVPs (Picard / QLM)
%
% sol = ahbmivp2(odefunc, tspan, y0, yprime0)
% sol = ahbmivp2(..., opts)
%
% Inputs:
%  - odefunc  : f(t,y,y') (column output of size d×1)
%  - tspan    : [t0, tf]
%  - y0       : initial y(t0) (vector or scalar)
%  - yprime0  : initial y'(t0)
%  - opts     : optional struct with fields (all optional)
%       h       - initial step (default 0.01)
%       M       - degree (M+1 nodes per block) (default 3)
%       method  - 'picard' (default) or 'qlm'
%       tol     - local error tolerance for adaptivity (default 1e-15)
%       fac     - safety factor for rejected steps (default 0.9)
%
% Outputs (struct):
%  sol.t   - vector of main-node times (column)
%  sol.Y   - solution y at main nodes (length(sol.t) x d)
%  sol.Y1  - solution y' at main nodes (length(sol.t) x d)

    %% ---------- handle opts and defaults ----------
    if nargin < 5 || isempty(opts)
        opts = struct();
    end
    allowedFields = {'h','M','method','tol','fac'};
    flds = fieldnames(opts);
    bad = setdiff(flds, allowedFields);
    if ~isempty(bad)
        error('Unknown option field(s): %s', strjoin(bad, ', '));
    end

    h       = getfielddefault(opts, 'h', 0.01);
    M       = getfielddefault(opts, 'M', 3);
    method  = getfielddefault(opts, 'method', 'picard');
    tol     = getfielddefault(opts, 'tol', 1e-12);   % local error tolerance (adaptivity)
    fac     = getfielddefault(opts, 'fac', 0.9);     % safety factor

    Tol     =  1e-13;   % iteration tolerance, set at 1e-13 
    maxIter = 100; %max iterations per block (set at 100)

    % basic input validation
    if numel(tspan) ~= 2, error('tspan must be [t0, tf].'); end
    d = numel(y0);
    testv = odefunc(tspan(1), y0(:), yprime0(:));
    if numel(testv) ~= d, error('odefunc must return length(y0) output.'); end

    %% ---------- precompute matrices ----------
    [p, A, B] = getEquispacedMatrices(M);              % p size (s x 1), A (s x (M+1)) 
    % For error estimate, we need low-order matrices. 
    [A_low, B_low] = getEquispacedMatricesLow(M);

    s = M + 1;    % nodes per block
    tf = tspan(2);

    %% ---------- allocation for adaptive stepping ----------
    tcur = tspan(1);
    yn = y0(:);        % column vector
    yn1 = yprime0(:);  % column vector

    nblock = 1;
    Yblocks = [];   % s x d x nblocks
    Y1blocks = [];  % s x d x nblocks
    Tblocks = [];   % s x nblocks

    %% ---------- adaptive time-stepping ----------
    while tcur < tf
        % avoid overshoot
        if tcur + h > tf
            h = tf - tcur;
            if h == 0, break; end
        end

        % collocation times in this block
        tnp = tcur + h * p(:);   % s x 1

        % initial guess: constant across block
        ynp = repmat(yn.', s, 1);    % s x d
        ynp1 = repmat(yn1.', s, 1);  % s x d

        switch lower(method)
        case 'picard'
            % -------- Picard iterations for second order --------
            for iter = 1:maxIter
                yprev = ynp; y1prev = ynp1;

                % evaluate f at nodes
                Fnp = zeros(s, d);
                for i = 1:s
                    Fnp(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).'); 
                end

                % Picard update for y' and y
                Yn1 = repmat(yn1.', M, 1);  % M x d
                Yn  = repmat(yn.', M, 1);   % M x d

                % Ynp (internal) uses double integral B and single integral A
                Ynp  = Yn + h * diag(p(2:end)) * Yn1 + (h^2) * (B * Fnp);   % M x d
                Ynp1 = Yn1 + h * (A * Fnp);                                 % M x d

                ynp  = [yn.'; Ynp];    % s x d
                ynp1 = [yn1.'; Ynp1];  % s x d

                err = max(norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf));
                if err < Tol
                    break;
                end
                if iter == maxIter
                    warning('Block %d: Picard did not converge in maxIter.', nblock);
                end
            end

        case 'qlm'
            % -------- QLM iterations for second order --------
            for iter = 1:maxIter
                yprev = ynp; y1prev = ynp1;

                % Evaluate function and Jacobians at all nodes
                F = zeros(s, d);
                Jy = zeros(s, d, d);
                Jyp = zeros(s, d, d);
                for i = 1:s
                    yi = ynp(i, :).';
                    ypi = ynp1(i, :).';
                    F(i, :) = odefunc(tnp(i), yi, ypi).';
                    [Jy(i,:,:), Jyp(i,:,:)] = numericalJacobian(odefunc, tnp(i), yi, ypi);
                end

                % Assemble block LHS and RHS
                L11 = zeros(d*M, d*M); L12 = zeros(d*M, d*M);
                L21 = zeros(d*M, d*M); L22 = zeros(d*M, d*M);
                R1 = zeros(d*M, 1); R2 = zeros(d*M, 1);

                for ii = 1:d
                    Ry = repmat(yn(ii), M, 1) + h * (p(2:end)) * yn1(ii);
                    Ry1 = repmat(yn1(ii), M, 1);

                    for jj = 1:d
                        JY = squeeze(Jy(:, jj, ii));
                        JYP = squeeze(Jyp(:, jj, ii));

                        Dy = diag(JY(2:end));
                        Dyp = diag(JYP(2:end));

                        B11 = -(h^2) * B(:,2:end) * Dy;
                        B12 = -(h^2) * B(:,2:end) * Dyp;
                        A21 = -h * A(:,2:end) * Dy;
                        A22 = -h * A(:,2:end) * Dyp;

                        if ii == jj
                            B11 = eye(M) + B11;
                            A22 = eye(M) + A22;
                        end

                        % Place into global matrices
                        row_idx = (ii-1)*M + (1:M);
                        col_idx = (jj-1)*M + (1:M);
                        L11(row_idx, col_idx) = B11;
                        L12(row_idx, col_idx) = B12;
                        L21(row_idx, col_idx) = A21;
                        L22(row_idx, col_idx) = A22;

                        % contributions from first node terms
                        Ry = Ry + (h^2) * B(:,1) * (Jy(1,jj,ii) * yn(jj)) + (h^2) * B(:,1) * (Jyp(1,jj,ii) * yn1(jj));
                        Ry1 = Ry1 + h * A(:,1) * (Jy(1,jj,ii) * yn(jj)) + h * A(:,1) * (Jyp(1,jj,ii) * yn1(jj));
                    end

                    % nonlinear correction
                    nonlinear_term = F(:,ii);
                    for jj = 1:d
                        nonlinear_term = nonlinear_term - Jy(:, jj, ii) .* ynp(:, jj) - Jyp(:, jj, ii) .* ynp1(:, jj);
                    end
                    Ry = Ry + (h^2) * B * nonlinear_term;
                    Ry1 = Ry1 + h * A * nonlinear_term;

                    idx = (ii-1)*M + (1:M);
                    R1(idx) = Ry;
                    R2(idx) = Ry1;
                end

                % Solve global system
                LHS = [L11, L12; L21, L22];
                RHS = [R1; R2];
                Yflat = LHS \ RHS;

                Ynp = reshape(Yflat(1:d*M), M, d);
                Ynp1 = reshape(Yflat(d*M+1:end), M, d);

                ynp = [yn.'; Ynp];
                ynp1 = [yn1.'; Ynp1];

                err = max(norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf));
                if err < Tol
                    break;
                end
                if iter == maxIter
                    warning('Block %d: QLM did not converge in maxIter.', nblock);
                end
            end

        otherwise
            error('Unknown method "%s". Use "picard" or "qlm".', method);
        end

        %% ---------- error estimation (embedded lower order) ----------
        % recompute F at converged ynp 
        Fnp = zeros(s, d);
        for i = 1:s
            Fnp(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).');
        end

        % estimate for y (uses B) and y' (uses A); 
        est_y  = (h^2) * ((B(end, :) - B_low(end, :)) * Fnp);  % 1 x d
        est_y1 =  h      * ((A(end, :) - A_low(end, :)) * Fnp); % 1 x d

        est = max(norm(est_y, inf), norm(est_y1, inf));  % scalar norm

        %% ---------- accept / reject step ----------
        if est < tol
            % accept step
            yn  = ynp(end, :).';
            yn1 = ynp1(end, :).';
            tcur = tcur + h;

            % store block
            Yblocks(:, :, nblock)  = ynp;
            Y1blocks(:, :, nblock) = ynp1;
            Tblocks(:, nblock)     = tnp;

            % attempt to grow step
            h = min(2*h, tf - tcur);
            nblock = nblock + 1;
        else
            % reject: shrink
            h = fac * h * (tol/est)^(1/(M+3));
            % reattempt with smaller h (same tcur)
        end
    end % adaptive while

    %% ---------- build outputs at main nodes ----------
    if isempty(Yblocks)
        % no steps (h was 0 or tspan degenerate)
        sol.t = tspan(1);
        sol.Y = y0(:).';
        sol.Y1 = yprime0(:).';
        return;
    end

    [~, ~, Nblocks] = size(Yblocks);
    tgrid = zeros(Nblocks, 1);
    Ymain = zeros(Nblocks, d);
    Y1main = zeros(Nblocks, d);

    tgrid(1) = tspan(1);
    Ymain(1, :) = y0(:).';
    Y1main(1,:) = yprime0(:).';

    for k = 1:Nblocks-1
        tgrid(k+1) = Tblocks(end, k);         % last collocation time in block k
        Ymain(k+1, :) = squeeze(Yblocks(end, :, k));
        Y1main(k+1,:) = squeeze(Y1blocks(end, :, k));
    end

    % pack into output
    sol.t  = tgrid;
    sol.Y  = Ymain;
    sol.Y1 = Y1main;
end

function [p, A, B] = getEquispacedMatrices(M)
    p = (0:M)' / M;
    Afull = generateCoefficientMatrix(p,1);
    Bfull = generateCoefficientMatrix(p,2);
    A = Afull(2:end, :);
    B = Bfull(2:end, :);
end

%% === Generate n-Fold Integration Matrix using Cauchy's Formula ===
function A = generateCoefficientMatrix(p, num_integrations)
    % Helper function to generate the coefficient matrix A by evaluating
    % the n-fold integral of Lagrange polynomials using Cauchy's formula
    % for repeated integration.
    % FORMULA: (1/(n-1)!) * integral from 0 to p_i of (p_i - s)^(n-1) * l_j(s) ds

    % --- Input Validation ---
    if num_integrations < 1 || floor(num_integrations) ~= num_integrations
        error('Number of integrations must be a positive integer.');
    end

    % Get the degree M (length of p is M+1)
    M = length(p) - 1;
    
    % Initialize the matrix with zeros
    A = zeros(M + 1, M + 1);
    
    % Calculate the constant pre-factor 1/(n-1)!
    n = num_integrations;
    pre_factor = 1 / factorial(n - 1);
    
    % Loop over the columns (j) of the matrix A, representing each Lagrange poly l_j(s).
    for j_idx = 1:(M + 1)
        j = j_idx - 1; % 0-based index from the problem definition
    
        % --- Step 1: Construct the coefficients of the j-th Lagrange polynomial, l_j(s) ---
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
        
        % Loop over the rows (i) of the matrix A, representing the upper limit p_i.
        for i_idx = 1:(M + 1)
            upper_limit = p(i_idx);
            
            % --- Step 2: Construct the polynomial (p_i - s)^(n-1) ---
            % We use the binomial theorem. The polynomial is in the variable 's'.
            % For (a-x)^k, the coefficients are given by nchoosek(k,j)*a^(k-j)*(-1)^j
            k = n - 1;
            term_coeffs = zeros(1, k + 1);
            for l = 0:k
                % The coefficient for s^l in (p_i - s)^k is nchoosek(k,l) * p_i^(k-l) * (-1)^l
                % MATLAB stores polynomials from highest power to lowest.
                power = l;
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k-power)) * ((-1)^power);
            end

            % --- Step 3: Convolve the two polys to get the full integrand ---
            % conv(A, B) is equivalent to polynomial multiplication.
            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);
            
            % --- Step 4: Perform the definite integral from 0 to p_i ---
            % First, find the indefinite integral (antiderivative)
            indef_integral_coeffs = polyint(integrand_coeffs);
            
            % Then, evaluate at the limits using polyval
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);
            
            % --- Step 5: Apply the pre-factor and store in the matrix A ---
            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

%% === Generate Coefficient Matrices for Equally Spaced Points ===
function [A,B] = getEquispacedMatricesLow(M)
    p = (0:M)' / M;
    Afull = generateCoeffMatrixLowerOrder(p,1);
    Bfull = generateCoeffMatrixLowerOrder(p,2);
    A = Afull(2:end, :);
    B = Bfull(2:end, :);
end

%% === Generate n-Fold Integration Matrix using Cauchy's Formula ===
function A = generateCoeffMatrixLowerOrder(p, num_integrations)
    % Helper function to generate the coefficient matrix A of lower order 
    % approximation by evaluating the n-fold integral of Lagrange polynomials 
    % using Cauchy's formula for repeated integration.
    % FORMULA: (1/(n-1)!) * integral from 0 to p_i of (p_i - s)^(n-1) * l_j(s) ds

    % --- Input Validation ---
    if num_integrations < 1 || floor(num_integrations) ~= num_integrations
        error('Number of integrations must be a positive integer.');
    end

    % Get the degree M (length of p is M+1)
    M = length(p) - 1;
    
    % Initialize the matrix with zeros
    A = zeros(M+1, M+1);
    
    % Calculate the constant pre-factor 1/(n-1)!
    n = num_integrations;
    pre_factor = 1 / factorial(n - 1);
    
    % Loop over the columns (j) of the matrix A, representing each Lagrange poly l_j(s).
    for j_idx = 1:(M)
        j = j_idx - 1; % 0-based index from the problem definition
    
        % --- Step 1: Construct the coefficients of the j-th Lagrange polynomial, l_j(s) ---
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
        
        % Loop over the rows (i) of the matrix A, representing the upper limit p_i.
        for i_idx = 1:(M + 1)
            upper_limit = p(i_idx);
            
            % --- Step 2: Construct the polynomial (p_i - s)^(n-1) ---
            % We use the binomial theorem. The polynomial is in the variable 's'.
            % For (a-x)^k, the coefficients are given by nchoosek(k,j)*a^(k-j)*(-1)^j
            k = n - 1;
            term_coeffs = zeros(1, k + 1);
            for l = 0:k
                % The coefficient for s^l in (p_i - s)^k is nchoosek(k,l) * p_i^(k-l) * (-1)^l
                % MATLAB stores polynomials from highest power to lowest.
                power = l;
                term_coeffs(k - power + 1) = nchoosek(k, power) * (upper_limit^(k-power)) * ((-1)^power);
            end

            % --- Step 3: Convolve the two polys to get the full integrand ---
            % conv(A, B) is equivalent to polynomial multiplication.
            integrand_coeffs = conv(lagrange_coeffs, term_coeffs);
            
            % --- Step 4: Perform the definite integral from 0 to p_i ---
            % First, find the indefinite integral (antiderivative)
            indef_integral_coeffs = polyint(integrand_coeffs);
            
            % Then, evaluate at the limits using polyval
            integral_value = polyval(indef_integral_coeffs, upper_limit) - polyval(indef_integral_coeffs, 0);
            
            % --- Step 5: Apply the pre-factor and store in the matrix A ---
            A(i_idx, j_idx) = pre_factor * integral_value;
        end
    end
end

function [Jy, Jyp] = numericalJacobian(odefunc, t, y, yp, epsilon)
    if nargin < 5, epsilon = 1e-8; end
    d = length(y);
    Jy = zeros(d); Jyp = zeros(d);
    for j = 1:d
        ey = zeros(d,1); ey(j) = 1;
        Jy(:,j) = (odefunc(t, y+epsilon*ey, yp) - odefunc(t, y-epsilon*ey, yp)) / (2*epsilon);
        Jyp(:,j) = (odefunc(t, y, yp+epsilon*ey) - odefunc(t, y, yp-epsilon*ey)) / (2*epsilon);
    end
end


%% ----------------------------
%% Get default field
%% ----------------------------
function val = getfielddefault(S, field, default)
    if isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
    else
        val = default;
    end
end
