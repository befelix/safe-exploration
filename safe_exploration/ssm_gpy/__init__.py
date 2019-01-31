# -*- coding: utf-8 -*-


try:
	import GPy
except:
	raise ImportError("Subpackage ssm_gpy requires optional dependency GPy")