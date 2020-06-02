.. _pythontips:

**************************
Tips for python and edrixs
**************************

In the design of edrixs, we made a deliberate choice to use
`python <http://www.python.org>`_ for the application programming interface. This is
because of its readability, easy of use and flexibility to run and combine it
in many different ways.

The standard way to run a python script ``myscript.py`` is::

     python myscript.py

This will generate the outputs of the script such as plots you choose to save
and print statements will be returned into the terminal. You can save the print
output by simply redirecting the output to a file::

     python myscript.py > myoutput.txt

For more exploratory usage,  `IPython <http://ipython.org>`_ or
`Jupyter <http://www.jupyter.org>`_
can be very useful. (See :ref:`edrixsanddocker` for invoking jupyter over docker.)
After starting IPython by typing ``ipython`` at the command line
you might find it useful to execute your script via::

     %run -i myscript.py

The *interactive* flag ``-i`` means that all the variables and functions you loaded
will be available to you at the command line. These can be straightforwardly printed
or inspected via the ``?`` or ``??`` flags, which show you the object documentation
and the object code respectively.::

     from edrixs import cd_cubic_d
     cd_cubic_d??

Including ``%matplotlib widget`` in your script
will facilitate interactive plots. All these interactive options are also available
in the rich outputs possible within the
`juptyer lab <https://jupyterlab.readthedocs.io/en/stable/>`_ interface.

A brute force option to look at a variable deep within the code is to use a debugger::

     python3 -m pdb myscript.py

Use ``import pdb; pdb.set_trace()`` to set the place where you want to enter the
debugger. See `here <https://docs.python.org/3/library/pdb.html>`_ for more details.

If you are feeling even braver, you can browse the Fortran code which does the
heavyweight computation. Either in the source edrixs directory or via the online
`edrixs repo <http://www.github.com/NSLS-II/edrixs>`_.
