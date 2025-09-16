FROM ubuntu:18.04

RUN apt-get clean
RUN apt-get update
RUN apt-get install -qy \
    git python3 python3-setuptools python3-dev python3-pip
RUN apt-get install -y cmake libblas-dev liblapack-dev gfortran
RUN apt-get install -y g++ gcc

RUN pip install torch
RUN pip install opencv
RUN pip install typing 
RUN pip install torch_dimcheck
RUN pip install torchtyping
RUN pip install disk

RUN git clone https://github.com/ducha-aiki/pydegensac.git
WORKDIR pydegensac
RUN python3 ./setup.py build

CMD cp -R /pydegensac/build/lib.linux-x86_64-3.6/pydegensac /home/