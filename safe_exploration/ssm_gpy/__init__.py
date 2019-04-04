# -*- coding: utf-8 -*-


try:
	import GPy
except:
	raise ImportError("Subpackage ssm_gpy requires optional dependency GPy")


from .gaussian_process import *
from .gp_models_old import *