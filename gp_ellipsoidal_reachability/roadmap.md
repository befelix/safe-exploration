
## Roadmap

#### `Cart_Pole System`
- [x] dynamics
- [x] constraints
- [x] gaussian angle to gaussian cos/sin representation (see PILCO)
- [x] Find stabilizing LQR setting
- [ ] Check dynamics (might be wrong right now)
- [x] Fix and test normalization

#### `InvertedPendulum`
- [ ] Adjust to changes made in interface 

#### `Static/Dynamic Exploration`
- [ ] Fix bugs

#### `SafeMPC`
- [x] performance and safety trajectory share multiple controls
- [x] Implement "Cautious MPC" 
- [x] custom cost function
- [x] fix issues with feasbility / change settings to reduce primal infeasibility fast, always return feasible solution if one was found

#### `episode_runner`
- [x] Allow for option to sample starting points inside safeset which map back into safeset

#### 'GPs'
- [x] Make it possible to exclude state space dims for inputs of the dynamics (important for cart-pole)




#### `Future work: Build small library for casadi MPC (Later possibly acados)`
- [ ] addition and multiplication of kernels and their casadi representations
- [ ] kernel and hyperparameter specification in config instead of GP class
