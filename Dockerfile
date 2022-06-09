FROM ubuntu:latest

MAINTAINER Daria Doroshkova
RUN apt-get update -y 
COPY . /opt/pythonproject
WORKDIR /opt/pythonproject
RUN apt install -y python3-pip 
RUN pip3 install -r requirements.txt 
CMD python3 ap.py

