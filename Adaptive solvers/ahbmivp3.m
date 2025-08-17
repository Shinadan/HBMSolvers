% AUTHOR:
%   SD Oloniiju, Rhodes University, South Africa
function sol = ahbmivp3(odefunc, tspan, y0, yprime0, yprimeprime0, opts)
%AHBMIVP3  Adaptive Hybrid Block Method for third-order IVPs (Picard / QLM)
%
% sol = ahbmivp3(odefunc, tspan, y0, yprime0, yprimeprime0)
% sol = ahbmivp3(..., opts)
%
% Inputs:
%  - odefunc      : f(t, y, y', y'') (column output of size d×1)
%  - tspan        : [t0, tf]
%  - y0           : initial y(t0) (vector or scalar)
%  - yprime0      : initial y'(t0)
%  - yprimeprime0 : initial y''(t0)
%  - opts         : optional struct with fields (all optional)
%       h       - initial step (default 0.01)
%       M       - degree (M+1 nodes per block) (default 3)
%       method  - 'picard' (default) or 'qlm'
%       tol     - local error tolerance for adaptivity (default 1e-15)
%       fac     - safety factor for rejected steps (default 0.9)
%
% Outputs (struct):
%  sol.t   - vector of main-node times (column)
%  sol.Y   - solution y at main nodes (length(sol.t) x d)
%  sol.Y1  - solution y' at main nodes
%  sol.Y2  - solution y'' at main nodes

    %% ---------- handle opts and defaults ----------
    if nargin < 6 || isempty(opts)
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
    tol     = getfielddefault(opts, 'tol', 1e-15);   % Local error tolerance
    fac     = getfielddefault(opts, 'fac', 0.9);     % Safety factor

    Tol     =  1e-13;   % Iteration tolerance
    maxIter = 100;      % Max iterations per block

    % Basic input validation
    if numel(tspan) ~= 2, error('tspan must be [t0, tf].'); end
    d = numel(y0);
    testv = odefunc(tspan(1), y0(:), yprime0(:), yprimeprime0(:));
    if numel(testv) ~= d, error('odefunc must return length(y0) output.'); end

    %% ---------- precompute matrices ----------
    [p, A, B, C] = getEquispacedMatrices(M);
    [A_low, B_low, C_low] = getEquispacedMatricesLow(M);

    s = M + 1;
    tf = tspan(2);

    %% ---------- allocation for adaptive stepping ----------
    tcur = tspan(1);
    yn  = y0(:);
    yn1 = yprime0(:);
    yn2 = yprimeprime0(:);

    nblock = 1;
    Yblocks  = []; % s x d x nblocks
    Y1blocks = []; % s x d x nblocks
    Y2blocks = []; % s x d x nblocks
    Tblocks  = []; % s x nblocks

    %% ---------- adaptive time-stepping loop ----------
    while tcur < tf
        % Avoid overshooting the final time
        if tcur + h > tf
            h = tf - tcur;
            if h < 1e-15, break; end % Prevent tiny final steps
        end

        % Collocation times in the current block
        tnp = tcur + h * p(:);

        % Initial guess: constant across the block
        ynp  = repmat(yn.', s, 1);
        ynp1 = repmat(yn1.', s, 1);
        ynp2 = repmat(yn2.', s, 1);

        switch lower(method)
        case 'picard'
            % -------- Picard Iterations for Third Order --------
            for iter = 1:maxIter
                yprev = ynp; y1prev = ynp1; y2prev = ynp2;

                % Evaluate f at all nodes
                Fnp = zeros(s, d);
                for i = 1:s
                    Fnp(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).', ynp2(i, :).');
                end

                % Replicate initial values for M inner points
                Yn  = repmat(yn.', M, 1);
                Yn1 = repmat(yn1.', M, 1);
                Yn2 = repmat(yn2.', M, 1);

                % Precompute Taylor series terms
                P1 = diag(p(2:end));
                P2 = diag(p(2:end).^2);

                % Picard update for y, y', y'' (internal nodes only)
                Ynp  = Yn  + h*P1*Yn1 + 0.5*(h^2)*P2*Yn2 + (h^3) * (C * Fnp);
                Ynp1 = Yn1 + h*P1*Yn2 + (h^2) * (B * Fnp);
                Ynp2 = Yn2 + h * (A * Fnp);

                % Assemble full block solution
                ynp  = [yn.'; Ynp];
                ynp1 = [yn1.'; Ynp1];
                ynp2 = [yn2.'; Ynp2];

                err = max([norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf), norm(ynp2 - y2prev, inf)]);
                if err < Tol
                    break;
                end
                if iter == maxIter
                    warning('Block %d: Picard did not converge in maxIter.', nblock);
                end
            end

        case 'qlm'
             % -------- QLM Iterations for Third Order --------
            for iter = 1:maxIter
                yprev = ynp; y1prev = ynp1; y2prev = ynp2;

                % Evaluate function and Jacobians at all collocation points
                F = zeros(s, d);
                Jy = zeros(s, d, d); Jyp = zeros(s, d, d); Jypp = zeros(s, d, d);
                for i = 1:s
                    yi = ynp(i, :).'; ypi = ynp1(i, :).'; yppi = ynp2(i, :).';
                    F(i, :) = odefunc(tnp(i), yi, ypi, yppi).';
                    [Jy(i,:,:), Jyp(i,:,:), Jypp(i,:,:)] = numericalJacobian(odefunc, tnp(i), yi, ypi, yppi);
                end

                % Assemble block LHS and RHS
                L11 = zeros(d*M); L12 = zeros(d*M); L13 = zeros(d*M);
                L21 = zeros(d*M); L22 = zeros(d*M); L23 = zeros(d*M);
                L31 = zeros(d*M); L32 = zeros(d*M); L33 = zeros(d*M);
                R1 = zeros(d*M, 1); R2 = zeros(d*M, 1); R3 = zeros(d*M, 1);

                for ii = 1:d
                    Ry  = repmat(yn(ii), M, 1) + h * p(2:end) * yn1(ii) + 0.5*(h^2) * (p(2:end).^2) * yn2(ii);
                    Ry1 = repmat(yn1(ii), M, 1) + h * p(2:end) * yn2(ii);
                    Ry2 = repmat(yn2(ii), M, 1);

                    for jj = 1:d
                        JY = squeeze(Jy(:, jj, ii)); JYP = squeeze(Jyp(:, jj, ii)); JYPP = squeeze(Jypp(:, jj, ii));
                        Dy = diag(JY(2:end)); Dyp = diag(JYP(2:end)); Dypp = diag(JYPP(2:end));

                        C11 = -(h^3)*C(:,2:end)*Dy;   C12 = -(h^3)*C(:,2:end)*Dyp;   C13 = -(h^3)*C(:,2:end)*Dypp;
                        B21 = -(h^2)*B(:,2:end)*Dy;   B22 = -(h^2)*B(:,2:end)*Dyp;   B23 = -(h^2)*B(:,2:end)*Dypp;
                        A31 = -h*A(:,2:end)*Dy;     A32 = -h*A(:,2:end)*Dyp;     A33 = -h*A(:,2:end)*Dypp;
                        
                        if ii == jj
                            C11 = eye(M) + C11;
                            B22 = eye(M) + B22;
                            A33 = eye(M) + A33;
                        end
                        
                        row_idx = (ii-1)*M + (1:M); col_idx = (jj-1)*M + (1:M);
                        L11(row_idx, col_idx) = C11; L12(row_idx, col_idx) = C12; L13(row_idx, col_idx) = C13;
                        L21(row_idx, col_idx) = B21; L22(row_idx, col_idx) = B22; L23(row_idx, col_idx) = B23;
                        L31(row_idx, col_idx) = A31; L32(row_idx, col_idx) = A32; L33(row_idx, col_idx) = A33;
                        
                        % Contributions from Jacobians at first node (t_n)
                        J_at_n = Jy(1,jj,ii)*yn(jj) + Jyp(1,jj,ii)*yn1(jj) + Jypp(1,jj,ii)*yn2(jj);
                        Ry  = Ry  + (h^3) * C(:,1) * J_at_n;
                        Ry1 = Ry1 + (h^2) * B(:,1) * J_at_n;
                        Ry2 = Ry2 + h     * A(:,1) * J_at_n;
                    end
                    
                    % Nonlinear residual term for right hand side
                    nonlinear_term = F(:,ii);
                    for jj = 1:d
                        nonlinear_term = nonlinear_term - Jy(:,jj,ii).*ynp(:,jj) - Jyp(:,jj,ii).*ynp1(:,jj) - Jypp(:,jj,ii).*ynp2(:,jj);
                    end
                    
                    Ry  = Ry  + (h^3)*C*nonlinear_term;
                    Ry1 = Ry1 + (h^2)*B*nonlinear_term;
                    Ry2 = Ry2 + h*A*nonlinear_term;

                    idx = (ii-1)*M + (1:M);
                    R1(idx) = Ry; R2(idx) = Ry1; R3(idx) = Ry2;
                end

                % Assemble and solve the global system
                LHS = [L11, L12, L13; L21, L22, L23; L31, L32, L33];
                RHS = [R1; R2; R3];
                Yflat = LHS \ RHS;
                
                Ynp = reshape(Yflat(1:d*M), M, d);
                Ynp1 = reshape(Yflat(d*M+1:2*d*M), M, d);
                Ynp2 = reshape(Yflat(2*d*M+1:end), M, d);

                ynp  = [yn.'; Ynp];
                ynp1 = [yn1.'; Ynp1];
                ynp2 = [yn2.'; Ynp2];

                err = max([norm(ynp - yprev, inf), norm(ynp1 - y1prev, inf), norm(ynp2 - y2prev, inf)]);
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

        %% ---------- Error Estimation (Embedded Lower Order) ----------
        % Recompute F at the converged solution for safety
        Fnp = zeros(s, d);
        for i = 1:s
            Fnp(i, :) = odefunc(tnp(i), ynp(i, :).', ynp1(i, :).', ynp2(i, :).');
        end

        % Estimates for y, y', and y'' using difference between high/low order matrices
        est_y  = (h^3) * ((C(end, :) - C_low(end, :)) * Fnp);  % 1 x d
        est_y1 = (h^2) * ((B(end, :) - B_low(end, :)) * Fnp);  % 1 x d
        est_y2 =  h    * ((A(end, :) - A_low(end, :)) * Fnp);  % 1 x d

        % The final error estimate is the max norm over all components
        est = max([norm(est_y, inf), norm(est_y1, inf), norm(est_y2, inf)]);

        %% ---------- Accept / Reject Step and Step-Size Control ----------
        if est < tol
            % --- ACCEPT STEP ---
            % Update solution to the end of the block
            yn  = ynp(end, :).';
            yn1 = ynp1(end, :).';
            yn2 = ynp2(end, :).';
            tcur = tcur + h;

            % Store the entire block's solution and times
            Yblocks(:, :, nblock)  = ynp;
            Y1blocks(:, :, nblock) = ynp1;
            Y2blocks(:, :, nblock) = ynp2;
            Tblocks(:, nblock)     = tnp;

            % Increase step size for next attempt (up to a factor of 2)
            h = min(2 * h, tf - tcur);
            
            nblock = nblock + 1;
        else
            % --- REJECT STEP ---
            % Reduce step size and re-attempt the same block
            h = fac * h * (tol / est)^(1 / (M + 4));
        end
    end % adaptive while

    %% ---------- Build Output Solution Structure ----------
    if isempty(Yblocks)
        % Handle cases with no successful steps
        sol.t = tspan(1);
        sol.Y = y0(:).'; sol.Y1 = yprime0(:).'; sol.Y2 = yprimeprime0(:).';
        return;
    end
    
    [~, ~, Nblocks] = size(Yblocks);
    % We store the solution at the start of each block + the final point
    tgrid = zeros(Nblocks, 1);
    Ymain = zeros(Nblocks, d); Y1main = zeros(Nblocks, d); Y2main = zeros(Nblocks, d);
    
    tgrid(1) = tspan(1);
    Ymain(1, :) = y0(:).';
    Y1main(1,:) = yprime0(:).';
    Y2main(1,:) = yprimeprime0(:).';
    
    for k = 1:Nblocks-1
        tgrid(k+1)   = Tblocks(end, k);
        Ymain(k+1, :)  = Yblocks(end, :, k);
        Y1main(k+1, :) = Y1blocks(end, :, k);
        Y2main(k+1, :) = Y2blocks(end, :, k);
    end

    % Pack into the output solution struct
    sol.t  = tgrid;
    sol.Y  = Ymain;
    sol.Y1 = Y1main;
    sol.Y2 = Y2main;
end


%% --- Helper Functions ---

function [p, A, B, C] = getEquispacedMatrices(M)
    p = (0:M)' / M;
    Afull = generateCoefficientMatrix(p, 1);
    Bfull = generateCoefficientMatrix(p, 2);
    Cfull = generateCoefficientMatrix(p, 3);
    A = Afull(2:end, :);
    B = Bfull(2:end, :);
    C = Cfull(2:end, :);
end

function [A_low, B_low, C_low] = getEquispacedMatricesLow(M)
    p = (0:M)' / M;
    
    Afull_low = generateCoeffMatrixLowerOrder(p, 1);
    Bfull_low = generateCoeffMatrixLowerOrder(p, 2);
    Cfull_low = generateCoeffMatrixLowerOrder(p, 3);
    
    A_low = Afull_low(2:end, :);
    B_low = Bfull_low(2:end, :);
    C_low = Cfull_low(2:end, :);
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

function [Jy, Jyp, Jypp] = numericalJacobian(odefunc, t, y, yp, ypp, epsilon)
    if nargin < 6, epsilon = 1e-8; end
    d = length(y);
    Jy = zeros(d); Jyp = zeros(d); Jypp = zeros(d);
    
    for j = 1:d
        ey = zeros(d,1); ey(j) = 1;
        
        Jy(:,j) = (odefunc(t, y+epsilon*ey, yp, ypp) - odefunc(t, y-epsilon*ey, yp, ypp)) / (2*epsilon);
        Jyp(:,j) = (odefunc(t, y, yp+epsilon*ey, ypp) - odefunc(t, y, yp-epsilon*ey, ypp)) / (2*epsilon);
        Jypp(:,j) = (odefunc(t, y, yp, ypp+epsilon*ey) - odefunc(t, y, yp, ypp-epsilon*ey)) / (2*epsilon);
    end
end

function val = getfielddefault(S, field, default)
    if isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
    else
        val = default;
    end
end

