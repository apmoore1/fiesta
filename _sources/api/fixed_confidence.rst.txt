Fixed Confidence (FC)
=====================
:Problem: .. include:: ../details/fc_problem.rst

:Caveat: All of the Fixed Confidence methods assume that the evaluations scores produced 
         by the models follow a Gaussian (normal) distribution. For a great guide on 
         knowing what distribution your evaluation metric would produce see 
         `Dror and Reichart, 2018 guide <https://arxiv.org/pdf/1809.01448.pdf>`_.

.. toctree::
    :caption: Methods

    ./fiesta.non_adaptive_fc
    ./fiesta.TTTS