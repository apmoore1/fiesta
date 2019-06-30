Fixed Budget (FB)
=================
These model selection functions are to be used in the FB setting which is framed as the following problem:

Given a budget :math:`T` and a set of models :math:`N` can we find the best 
model :math:`N^*` using only :math:`T` model evaluations (runs). 

The budget :math:`T` can be spent either fairly across all of the models 
e.g. :math:`T=10` and :math:`N=5` we would give each of the :math:`N` 
models :math:`2` evaluations each. This fair approach is the approach 
taken in `fiesta.non_adaptive_fb <./fiesta.non_adaptive_fb.html>`_ and is the standard way of 
evaluating models in Natural Language Processing.

.. toctree::
    ./fiesta.non_adaptive_fb
    ./fiesta.sequential_halving