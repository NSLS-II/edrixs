*********************
edrixs and docker
*********************

Run edrixs in a docker container
--------------------------------

To make life easier, we have built a docker image based on Ubuntu Linux (18.04) for edrixs, so you don't need to struggle with the installation anymore.
The docker image can be used on any OS as long as the `docker <https://www.docker.com/>`_ application is available.
Follow these steps to use the docker image:

* Install the `docker <https://www.docker.com/>`_ application on your system and `learn how to use it <https://docs.docker.com/get-started/>`_.

* Once Docker is running, create a directory to store data in your host OS and launch a container to run edrixs::

    $ mkdir /dir/on/your/host/os   # A directory on your host OS
    $ docker pull edrixs/edrixs    # pull latest version
    $ docker run -it -u rixs -w /home/rixs -v /dir/on/your/host/os:/home/rixs/data edrixs/edrixs

  it will take a while to pull the image from `Docker Hub <https://cloud.docker.com/repository/docker/edrixs/edrixs/>`_ for the first time, while, it will launch the local one very fast at the next time.

  * ``-u rixs`` means using a default user ``rixs`` to login the Ubuntu Linux, the password is ``rixs``.

  * ``-v /dir/on/your/host/os:/home/rixs/data`` means mounting the directory ``/dir/on/your/host/os`` from your host OS to ``/home/rixs/data`` on the Ubuntu Linux in the container.

* In the container, you can play with edrixs as you are using an isolated Ubuntu Linux system. After launching the container, you will see ``data`` and ``edrixs_examples`` in ``/home/rixs`` directory. If you want to save the data from edrixs calculations to your host system, you need to work in ``/home/rixs/data`` directory::

    $ cd /home/rixs/data
    $ cp -r ../edrixs_examples .

    Play with edrixs ...

  Note that any changes outside ``/home/rixs/data`` will be lost when this container stops. You can only use your host OS to make interactive plots. Use ``sudo apt-get install`` to install software packages if they are needed.

* Type ``exit`` in the container to exit. You can delete all the stopped containers by::

    $ docker rm $(docker ps -a -q)

* If you do not need the image anymore, you can delete it by::

    $ docker rmi edrixs/edrixs

Connect to docker python session with Jupyter
----------------------------------------------

`Jupyter <https://jupyter.org/>`_  is a popular way to integrate your code with rich output including plots. You may find this working mode particularly useful for exploratory work, when you try many different approaches to calculations or analysis.

* To use this follow the steps above, but pass an additional command ``-p 8888`` when you launch the container i.e.::

    $ docker run -it -p 8888:8888 -u rixs -w /home/rixs -v /dir/on/your/host/os:/home/rixs/data edrixs/edrixs

  from within the container initiate the jupyter session as::

    $ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  This will return a URL that you can enter in a browser on your host machine. If you want to use interactive `ipython widgets <https://ipywidgets.readthedocs.io/en/stable/>`_ and `plotting <https://github.com/matplotlib/jupyter-matplotlib>`_, replace ``edrixs/edrixs`` with ``edrixs/edrixs_interactive``. This downloads a (larger) container with the required additional packages.
