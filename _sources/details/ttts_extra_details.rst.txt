In more detail after each of the :math:`N` models are evaluated :math:`3` times
the following two steps are repeated until one of the models is better 
than the rest by a certain confidence level:

1. :math:`2` models are sampled from the :math:`N` models where the sampling is 
   weighted based on the performance of the models. 
2. Between the :math:`2` models :math:`1` is choosen randomly and is evaluated