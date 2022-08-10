## Basic Docker Commands

```bash
## Installing Docker
sudo apt-get update
sudo apt-get install docker.io -y

# Hello World Docker
sudo docker run hello-world

# Puliing existing image
sudo docker pull busybox

# Running (Simple) Container
sudo docker run busybox echo "hello from busybox"

#######################################################
#
#       Containers
#
#######################################################

# List Containers
sudo docker ps
sudo docker ps -a

# Running (Persistent)/(Interactive terminal) Container
sudo docker run -it busybox


# Remove/Erasing exited container
sudo docker rm CONTAINER_ID

# Remove all containers
sudo docker container prune


#######################################################
#
#       Images
#
#######################################################

# List images
sudo docker images

# Remove Images
sudo docker rmi -f IMAGE_ID



#######################################################
#
#     Advanced Containers Images
#
#######################################################

sudo docker pull httpd

# Run docker in daemon mode (in background) with port mapping
# -d    daemon mode
# -p    HOST_PORT:CONTAINER_PORT
# -v    maps directory of local host to directory of the container
# -P    Docker will select a random port

sudo docker run -d -p 80:80 httpd

# COOL MONITORING
docker netdata


#######################################################
#
#     Creating Docker Files
#
#######################################################

# Build Docker File Definition
sudo docker build -t REPOSITORY_NAME:VERSION_NAME .

# Check Sample Dockerfile for an example of a Dockerfile


#######################################################
#
#   Docker Monitoring
#
#######################################################

# Real-time Stats
sudo docker stats

# Real-time Historical Stats
cAdvisor in github\

# Docker logs
sudo docker logs NAME
```

## Dockerfile.example

```Dockerfile
# Use the official image as a parent image.
FROM centos:latest

# Declare the maintainer of the Dockerfile
MAINTAINER Prasanjit-Singh www.binpipe.org

# Some backward incompatibility workarounds
RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

#Install Apache Webserver
RUN yum -y install httpd

# Set the working directory to webroot
WORKDIR /var/www/html

# Copy the code to the webroot directory 
COPY html /var/www/html

# Run the command to launch Apache Webserver daemon
CMD ["/usr/sbin/httpd", "-D", "FOREGROUND"]

# Expose port 80  for the website 
EXPOSE 80
```
