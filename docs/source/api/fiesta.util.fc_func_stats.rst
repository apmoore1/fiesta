fiesta.util.fc_func_stats
=========================
Given a `Fixed Confidence (FC) method <./fixed_confidence.html>`_ 
like :func:`~fiesta.fiesta.TTTS` it will run that function with the 
given keyword arguments *N* times and will report the summary 
statistics: min, mean, max evaluations and percentage of times the FC function 
correctly found the best model based on the ``correct_model_index`` argument.

.. automodule:: fiesta.util
   :members: fc_func_stats
   :noindex: