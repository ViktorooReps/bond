FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git

RUN git clone https://github.com/ViktorooReps/bond.git
RUN cd bond