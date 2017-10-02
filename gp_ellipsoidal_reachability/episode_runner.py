# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:11:23 2017

@author: tkoller
"""
from environments import InvertedPendulum
import warnings

DEFAULT_EPISODE_OPTIONS = {"n_ep": 6, "n_steps": 80, "n_scenarios": 1, "n_steps_init": 50}
DEFAULT_SAVE_OPTIONS = {}
DEFAULT_ENV_OPTIONS= {"env_name": "InvertedPendulum"}

def run(env_options = None, episode_options = None, controller_options = None, 
        save_options = None)
    """ Run episode setting """
        
    n_ep, n_step, n_scenarios, n_steps_init = _process_episode_options(episode_options)
    env, visualize = _create_env(env_options)
    
    if n_scenarios > 1:
        raise NotImplementedError("For now we don't support multiple experiments!)
        
    X,y = do_rollout(env, n_steps, visualize = visualize)
    for i in range(n_ep):
        
        solver = create_solver(env, controller_options, X, y)
        xx, yy = do_rollout(env, n_steps, solver = solver, visualize = visualize)
        
        X = np.vstack((X,xx))
        y = np.vstack((y,yy))
        
        
def _process_episode_options(episode_options = None):
    """ Return default options replaced by the specified input episode options 
    
    Merge the default episode options with 
    Parameters
    ----------
    episode_options: dict
        The episode_options chosen by the user    
    """
    
    if episode_options is None:
        opts = DEFAULT_EPISODE_OPTIONS
    else:
        raise NotImplementedError()
        
    n_ep = episode_options["n_ep"]
    n_steps = episode_options["n_steps"]
    n_scenarios = episode_options["n_scenarios"]
    n_steps_init = episode_options["n_steps_init"]
    
    return n_ep, n_step, n_scenarios, n_steps_init
    
def _create_env(env_options = None):
    """ Given a set of options, create an environment """
    
    if env_options is None:
        return InvertedPendulum(), True
    else:
        raise NotImplementedError("Need to implement this!")
        
def do_rollout(env, n_steps, solver = None, relative_dynamics = True, visualize = True, verbosity = 1):
    """
    
    """
    
    state = env.reset()
    target = env.get_target()
    
    n_successful = 0
    xx = np.zeros((1,env.ns+env.nu))
    yy= np.zeros((1,env.ns))
    for i in range(n_steps):
        if visualize:
            env.render()
        
        if solver is None:
            action = env.random_action()
        else:
            t_start_solver = time.time()
            action = solver.get_action(x0_mu = state.reshape((-1,1)),x0_sigm = x0_sigm,Target = target)
            t_end_solver = time.time()
            t_solver = t_end_solver - t_start_solver
            
            if verbosity > 0
                print("total time solver in ms: {}".format(t_solver))

        action,observation = env.step(action)
        if done:
            break

        state_action = np.hstack((state,action))
        xx = np.vstack((xx,state_action))
        if relative_dynamics:
            yy = np.vstack((yy,observation - state))
        else:
            yy = np.vstack((yy,observation))
            
        n_successful += 1
        state = observation
        
    if n_successful == 0:
        warnings.warn("Agent survived 0 steps, cannot collect data")
        xx = []
        yy = []
    else:
        xx = xx[1:,:]
        yy = yy[1:,:]
        
    print("Agent survived {} steps".format(n_successful))

    return xx,yy
    
    
if __name__ == "__main__":
    