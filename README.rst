=============================================================
 Learning-based Model Predictive Control for Safe Exploration
=============================================================

.. image:: https://circleci.com/gh/befelix/safe-exploration/tree/master.svg?style=svg
    :target: https://circleci.com/gh/befelix/safe-exploration/tree/master
    
.. image:: https://travis-ci.com/befelix/safe-exploration.svg?branch=master
    :target: https://travis-ci.com/befelix/safe-exploration

This code accompanies the following paper:

.. [1] T. Koller, F. Berkenkamp, M. Turchetta, A. Krause,
  `Learning-based Model Predictive Control for Safe Exploration <https://arxiv.org/abs/1803.08287>`_
  in Proc. of the Conference on Decision and Control (CDC), 2018

Installation
------------

Install the library including all dependencies with.

::

  pip install -e ".[test,visualization,ssm_gpy,ssm_pytroch]"
  
  
`test` for the testing tools. `visualization` for visualizations such as matplotlib. `ssm_gpy` and `ssm_pytorch` for state space models based on `GPy` or `PyTorch`, respectively.

Experiments can be run using the `experiments/run.py` script.

Test can be run using pytest. There are also more sophisticated style tests in 
`scripts/test_code.sh`.
