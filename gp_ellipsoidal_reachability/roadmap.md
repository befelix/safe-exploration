
## Roadmap

#### `Cart_Pole System`
- [x] dynamics
- [ ] constraints
- [x] gaussian angle to gaussian cos/sin representation (see PILCO)
- [ ] Find stabilizing LQR setting
- [ ] Check dynamics (might be wrong right now)
- [ ] Fix and test normalization

#### `SafeMPC`
- [ ] performance and safety trajectory share multiple controls
- [ ] implement kim wabersich (zeilinger) approach 
- [x] custom cost function
- [ ] fix issues with feasbility / change settings to reduce primal infeasibility fast, always return feasible solution if one was found

#### `episode_runner`
- [x] Allow for option to sample starting points inside safeset which map back into safeset

#### 'GPs'
- [ ] Make it possible to exclude state space dims for inputs of the dynamics (important for cart-pole)
- [ ] addition and multiplication of kernels and their casadi representations
- [ ] kernel and hyperparameter specification in config instead of GP class
