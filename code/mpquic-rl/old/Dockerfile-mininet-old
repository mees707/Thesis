#FROM iwaseyusuke/mininet:ubuntu-20.04
#FROM containernet/containernet
#COPY id_rsa.pub ~/.ssh/authorized_keys

#RUN apt-get update && \
#      apt-get -y install sudo
#
#RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
#
#USER docker
#RUN echo 'root:Docker!' | chpasswd
#RUN echo 'docker:docker' | chpasswd

#RUN ./mount_tmpfs.sh



FROM ubuntu:22.04

USER root
WORKDIR /root

COPY ENTRYPOINT.sh /

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dnsutils \
    ifupdown \
    iproute2 \
    iptables \
    iputils-ping \
    mininet \
    net-tools \
    openvswitch-switch \
    openvswitch-testcontroller \
    tcpdump \
    vim \
    x11-xserver-utils \
    xterm \
    python3 \
    python3-pip \
	git \
 && rm -rf /var/lib/apt/lists/* \
 && touch /etc/network/interfaces \
 && chmod +x /ENTRYPOINT.sh

EXPOSE 6633 6653 6640

#go1.20.4
#go1.16.15
#COPY mpquic-rl/
RUN curl -OL https://golang.org/dl/go1.12.2.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.12.2.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"

#RUN DEBIAN_FRONTEND=noninteractive go get github.com/lucas-clemente/quic-go && cd ~/go/src/github.com/lucas-clemente/quic-go &&\
# git remote add mp-quic https://github.com/qdeconinck/mp-quic.git && git fetch mp-quic && git checkout conext17 && go get -t -u ./...


#RUN DEBIAN_FRONTEND=noninteractive go get github.com/lucas-clemente/quic-go
#RUN go get github.com/quic-go/quic-go
#RUN mv ~/go/src/github.com/quic-go/quic-go ~/go/src/github.com/lucas-clemente/quic-go
#RUN cd ~/go/src/github.com/lucas-clemente/quic-go &&\
#git remote add mp-quic https://github.com/qdeconinck/mp-quic.git && git fetch mp-quic && git checkout conext17

#RUN curl -OL https://redirector.gvt1.com/edgedl/go/go1.9.2.linux-amd64.tar.gz
#RUN tar -C /usr/local -xzf go1.9.2.linux-amd64.tar.gz
#ENV PATH="$PATH:/usr/local/go/bin"

ENTRYPOINT ["/ENTRYPOINT.sh"]