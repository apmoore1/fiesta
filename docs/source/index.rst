.. FIESTA documentation master file, created by
   sphinx-quickstart on Sun Jun  2 19:57:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FIESTA Package
==============
FIESTA (Fast IdEntification of State-of-The-Art) package as the name suggests 
allows you to easily find the **best model** from a set of models taking 
both **data splits** and **random seeds** into account. 

This package is also 
flexible enough to only consider the **random seeds** and keep the **data 
splits** fixed which may be required for evaluation competitions like 
`SemEval <http://alt.qcri.org/semeval2019/index.php?id=tasks>`_.

FIESTA looks at two different model selection scenarios; **Fixed Budget** and 
**Fixed Confidence**.

Fixed Budget (FB)
+++++++++++++++++

Problem
-------
.. include:: ./details/fb_problem.rst

Methods
-------

`fiesta.non_adaptive_fb <./api/fiesta.non_adaptive_fb.html>`_
#############################################################
.. include:: ./details/non_adaptive_fb_explaination.rst

`fiesta.sequential_halving <./api/fiesta.sequential_halving.html>`_
###################################################################
.. include:: ./details/sequential_halving_explaination.rst

For more details on this approach see the 
`Sequential Halving documentation <./api/fiesta.sequential_halving.html>`_.

.. include:: ./details/sequential_halving_note.rst

Fixed Confidence (FC)
+++++++++++++++++++++

Problem
-------
.. include:: ./details/fc_problem.rst

Methods
-------

`fiesta.non_adaptive_fc <./api/fiesta.non_adaptive_fc.html>`_
#############################################################
.. include:: ./details/non_adaptive_fc_explaination.rst

`fiesta.TTTS <./api/TTTS.html>`_
#############################################################
.. include:: ./details/ttts_explaination.rst

.. include:: ./details/ttts_note.rst

For more details on this approach see the 
`TTTS documentation <./api/fiesta.TTTS.html>`_.

Installing
==========

.. code-block:: bash 

    pip install fiesta-nlp

.. toctree::
    :maxdepth: 4
    :caption: Package Reference

    ./api/model_selectors
    ./api/fiesta.util



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
