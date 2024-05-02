# Thesis


Useful links:
ttps://multipath-quic.org/2017/12/09/artifacts-available.html 

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
**MAKE SURE TO FOLLOW THESE STEPS, NOT THE ONES FROM MININET.ORG**
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





NOT YET CLEAR ABOUT THESE STEPS:
# Create if it does not exists yet
The second one is to mount the file measurement location on the RAM. This should limit the impact of the hard disk on the results.
$ sudo mkdir /mnt/tmpfs
# This should be run each time the VM is restarted
$ sudo mount -t tmpfs -o size=256M tmpfs /mnt/tmpfs
steps dont work so far


going to keep the date on here so i know where i am
2 may:
trying to get the dash framework to work, should not be too hard except for installing some packages
dash framework VM download does not work, will probably email the original creator asking if he still has it somewhere

build the "MPQUIC-SBD: Multipath QUIC (MPQUIC) with support of Shared Bottleneck Detection (SBD)" framework, dont know how it is usefull yet tho...




















