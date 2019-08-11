It breaks the problem down into rounds of :math:`\log_2N`, where in each round 
half of the worse performing models are removed from the canditae models 
:math:`S` until only the best performing model is left (At the start the number 
of candiate models :math:`S` = :math:`N`). In each round each model is
evaluated :math:`\lfloor\frac{T}{|S|\lceil\log_2N\rceil}\rfloor`. The models 
performance each round is based on the models evaluations across all of the 
rounds. In the last round of 2 models each model is 
evaluated :math:`2^{\lceil\log_2N\rceil}-1` times. 

*Example:*

:math:`T=16`, :math:`N=4`:

+-------+----------------------+---------------+
| Round | Candidate Models     |  #Evaluations |
+=======+======================+===============+
| 1     | S = {m1, m2, m3, m4} | 2             |
+-------+----------------------+---------------+
| 2     | S = {m2, m4}         | 4             |
+-------+----------------------+---------------+
| output| S = {m2}             |               |
+-------+----------------------+---------------+