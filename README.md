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


**DO NOT USE RSA AS SSH KEY, DEPRICATED**
setting up ssh key if this is not done before:

generating the key (host):
ssh-keygen -t ed25519

moving the key from host to vm (the vm should be running, executed on host):
scp -P 8022 ~/.ssh/id_ed25519.pub mininet@localhost:/tmp

ssh into vm(host):
ssh -p 8022 mininet@localhost


move file inside of vm to correct place(vm)
cat /tmp/id_ed25519.pub >> ~/.ssh/authorized_keys
this might give an error such as file not found, in that case:
mkdir -p ~/.ssh
and try again

after this is done, log off and try (host)
ssh -p 8022 mininet@localhost
again, should not be a password promt anymore



