# HBM Solvers

Hybrid Block Method (HBM) solvers for initial value problems (IVPs).

## Overview
`HBMIVP` is a suite of solvers based on the hybrid block method for solving ordinary differential equations (ODEs) and partial differential equations (PDEs). The implementation includes both **Picard** and **Quasilinearization (QLM)** schemes, with options for **fixed-step** and **adaptive-step** integration. The solvers are designed to handle stiff and non-stiff systems, as well as higher-order IVPs, in a structured and efficient way.

## Features
- Solvers for first, second, and third-order IVPs.
- Picard and Quasilinearization (QLM) methods.
- Fixed-step and adaptive-step implementations.
- Support for stiff and non-stiff problems.
- Examples ranging from simple test equations to PDEs (e.g., the heat equation).
- High-accuracy results validated against analytical solutions and MATLAB’s `ode45`.

## Usage
A typical workflow:

% Define problem

f = @(t,y) -100*y + 99*sin(t);     % RHS of ODE

y0 = [1; 11];                      % Initial conditions

% Solver options

opts = struct('method','qlm','h',1e-2,'M',4);

% Call solver

[t, y] = hbmivp(f, [0, 1], y0, opts);

% Compare with analytical solution

y_exact = cos(10*t) + sin(10*t) + sin(t);

plot(t, y(:,1), 'b-', t, y_exact, 'r--')

legend('HBM solution', 'Exact solution')

## Examples
The examples folder contains:
- Examples 1 to 7: Test problems for first- to third-order IVPs.
- Examples 8 and 9: PDEs solved using spectral discretization in space and HBM in time.
Comparison with MATLAB’s `ode45` and `ode15s` is included for examples.

## Contributing
Contributions are welcome! Please open issues or submit pull requests to:
- Add new examples
- Improve solver efficiency
- Extend to fractional or stochastic IVPs/PDEs

## Citation
If you use HBMIVP in your research, please cite:

@misc{hbmivp2025,
 
  author       = {Shina D Oloniiju},
  
  title        = {HBMIVP: Hybrid block method solvers for IVPs},
  
  year         = {2025},
  
  howpublished = {\url{https://[github.com/Shinadan/HBMSolvers](https://github.com/Shinadan/HBMSolvers)}}

}

