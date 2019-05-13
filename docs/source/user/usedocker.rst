*********************
Run edrixs in docker
*********************
To make life easier, we have built a docker image based on Ubuntu Linux (18.04) for edrixs, so you don't need to struggle with the installation anymore. 
The docker image can be used on any OS as long as the `docker <https://www.docker.com/>`_ application are available.
Follow these steps to use the docker image:

* Install the `docker <https://www.docker.com/>`_ application on your system and `learn how to use it <https://docs.docker.com/get-started/>`_.

* Once the docker is running, create a directory to store data in your host OS and launch a container to run edrixs::

    $ mkdir /dir/on/your/host/os   # A directory on your host OS
    $ docker run -it -u rixs -w /home/rixs -v /dir/on/your/host/os:/home/rixs/data laowang2017/edrixs

  it will take a while to pull the image from `Docker Hub <https://cloud.docker.com/repository/docker/laowang2017/edrixs/>`_ for the first time, while, it will launch the local one very fast at the next time. 

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

    $ docker rmi laowang2017/edrixs   
