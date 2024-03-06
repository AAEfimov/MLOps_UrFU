# Download base image ubuntu 18.04
FROM ubuntu:18.04

# LABEL about the custom image
LABEL maintainer="efimov.alexey.a@gmail.com"
LABEL version="1.0"
LABEL description="docker agent for jenkins"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

RUN apt-get update && apt-get install -y \
  git \
  subversion \
  gcc \
  inetutils-tools \
  net-tools \
  iputils-ping \
  iproute2 \
  make \
  vim \
  mc \
  libc6-i386 \
  lib32ncurses5 \
  lib32tinfo5 \
  lib32z1 \
  g++ \
  clang \
  python3 \
  python3-venv \
  libssl1.0-dev \
  flex \
  gawk \
  zip \
  bison

