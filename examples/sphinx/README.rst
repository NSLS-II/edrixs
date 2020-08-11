Examples
========
Sections termed `Original examples`_ and `Extended examples`_ provide
calculations for different models. We provide `Pedagogical examples`_ separately
at the bottom of the page.

Original examples
=================

The original examples from 
`Y. Wang et al., Comput. Phys. Commun 243, 151-165 (2019) <https://doi.org/10.1016/j.cpc.2019.04.018>`_
are kept in our `GitHub repository <https://github.com/NSLS-II/edrixs/tree/master/examples/cpc>`_.
These are scripts that compute XAS and RIXS calculations for particular physical
models including:

* Single atom model for `Ni <https://github.com/NSLS-II/edrixs/tree/master/examples/cpc/single_atom>`_. See section 5.1.
* A two-site `Ir-Ir cluster <https://github.com/NSLS-II/edrixs/tree/master/examples/cpc/two_site_cluster>`_ model.
  See section 5.2. This allows the modeling of dimer excitations in RIXS where electrons transition between atoms.
  We used this model in our `manuscript on Ba5AlIr2O11 <https://doi.org/10.1103/PhysRevLett.122.106401>`_.
* `Anderson impurity model <https://github.com/NSLS-II/edrixs/tree/master/examples/cpc/two_site_cluster>`_.
  See section 5.3. This addresses Ba2YOsO6. 

Extended examples
=================
Additional examples are available
`here <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS>`_.
Thus far we have addressed.

* Single atom and Anderson impurity models for
  `Ba2YOsO6 <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/Ba2YOsO6>`_
* Two-site cluster for
  `Ba3InIr2O9 <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/Ba3InIr2O9>`_
* Single atom model for
  `La2NiO4 <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/La2NiO4>`_
  taken from 
  `G. Fabbris et al., Phys. Rev. Lett. 118, 156402 (2017)
  <https://doi.org/10.1103/PhysRevLett.118.156402>`_ extended to show how
  spin direction can inflence XAS and RIXS spectra
* Single atom model for
  `LaNiO3 heterostructures <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/LaNiO3_thin>`_
  taken from 
  `G. Fabbris et al., Phys. Rev. Lett. 117, 147401 (2016)
  <https://doi.org/10.1103/PhysRevLett.117.147401>`_.
  showing how spin direction can influence XAS and RIXS spectra.
* Heavy fermion example for `plutonium O-edge <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/Pu_O45>`_
* Single atom model for
  `pyrochlore iridates <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/Pyrochlore_227>`_
  taking into account differently oriented octahedra.
* Single atom model for O-edge RIXS on
  `URu2Si2 from <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/URu2Si2>`_
  `L. Andrew Wray et al., Phys. Rev. Lett. 114, 236401 (2015)
  <https://doi.org/10.1103/PhysRevLett.114.236401>`_ including both 5f and 5d
  orbitals.
* Single atom model for 
  `L-edge RIXS on URu2Si2 <https://github.com/NSLS-II/edrixs/tree/master/examples/more/RIXS/U_L3>`_
  from
  `Y. Wang et al., Phys. Rev. B 96, 085146 (2017)
  <https://doi.org/10.1103/PhysRevB.96.085146>`_

Pedagogical examples
====================

The examples below aim to provide pedagogical information
on how edrixs works. So far we have covered
:ref:`sphx_glr_auto_examples_example_0_ed_calculator.py`,
:ref:`sphx_glr_auto_examples_example_1_crystal_field.py`,
:ref:`sphx_glr_auto_examples_example_2_AIM_XAS.py`, and
:ref:`sphx_glr_auto_examples_example_3_GS_analysis.py`.
