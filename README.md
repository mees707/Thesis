# Thesis


Useful links:

https://github.com/pijuskri/MPQUIC-meta-learning/tree/main
previous person that worked on this project


https://github.comthe/sajibtariq/dashframework
dash evaluation framework

https://github.com/thomaswpp/mpquic-sbd
Multipath QUIC where the experiments will be performed


https://mininet.org/vm-setup-notes/
mininet setup guide

https://mininet.org/walkthrough/
mininet walkthrough



how to launch vm:

open terminal in folder location that the disc is located

run:
sudo qemu-system-x86_64 -m 2048 kvm-mpquic.vmdk -net nic,model=virtio -net user,net=192.168.101.0/24,hostfwd=tcp::8022-:22


ssh -X -p 8022 mininet@localhost


scp ~/.ssh/id_rsa.pub mininet@localhost:~/
