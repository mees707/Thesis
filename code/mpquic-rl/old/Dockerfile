#entry copied from https://github.com/rixsilverith/mininet-docker/tree/main
FROM ubuntu:22.04

USER root
WORKDIR /root

COPY ENTRYPOINT.sh /

#RUN yes| unminimize #unsure if needed

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
	nano\
	git \
	sudo\
	openssh-server\
 && rm -rf /var/lib/apt/lists/* \
 && touch /etc/network/interfaces \
 && chmod +x /ENTRYPOINT.sh
 
RUN useradd -rm -d /home/mininet -s /bin/bash -g root -G sudo -u 1001 -p $(openssl passwd -1 mininet) mininet
ADD --chown=mininet mininet.tar.gz /home/mininet/
 
#ensure we can connect to the internet
RUN sysctl -w net.ipv4.ip_forward=1

EXPOSE 6633 6653 6640 22

#go1.20.4
#go1.16.15

#Install go
RUN curl -OL https://golang.org/dl/go1.12.2.linux-amd64.tar.gz
RUN tar -C /home/mininet/ -xzf go1.12.2.linux-amd64.tar.gz #/usr/local
ENV PATH="$PATH:/home/mininet/go/bin"

# Install deb packages
#ADD mptcp_debs /mptcp_debs/
#COPY enable-mtcp.sh /
#CMD ["/enable-mtcp.sh"]

#add mininet user

RUN usermod -aG sudo mininet

#set root password
RUN echo 'root:root' | chpasswd

#RUN mkdir /var/run/sshd
#
#RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y 
#RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed -ri 's/PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config
#RUN sed -ri 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

WORKDIR /home/mininet

COPY id_rsa.pub /home/mininet/.ssh/authorized_keys
RUN chown -R mininet: /home/mininet/.ssh
RUN chmod 644 /home/mininet/.ssh/authorized_keys
COPY id_rsa.pub /root/.ssh/authorized_keys
COPY id_rsa.pub /etc/ssh/authorized_keys
RUN chmod 644 /etc/ssh/authorized_keys

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo "alias python=/usr/bin/python3" >> ~/.bash_profile
RUN echo 'alias python=/usr/bin/python3' >> ~/.bashrc
RUN cp /usr/bin/python3 /usr/bin/python

COPY mpquic-rl/central_service/minitopo /home/mininet/git/minitopo/src/

ENTRYPOINT ["/ENTRYPOINT.sh"]
#USER mininet