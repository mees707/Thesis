# Thesis
conda activate mpquicrl
cd code/mpquic-rl

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
sudo qemu-system-x86_64 -m 2048 kvm-mpquic.vmdk -net nic,model=virtio -net user,net=192.168.101.0/24,hostfwd=tcp::2222-:22


**DO NOT USE RSA AS SSH KEY, DEPRICATED**
**MAKE SURE TO FOLLOW THESE STEPS, NOT THE ONES FROM MININET.ORG**
setting up ssh key if this is not done before:

generating the key (host):
ssh-keygen -t ed25519

moving the key from host to vm (the vm should be running, executed on host):
scp -P 2222 ~/.ssh/id_ed25519.pub mininet@localhost:/tmp

ssh into vm(host):
ssh -p 2222 mininet@localhost


move file inside of vm to correct place(vm)
cat /tmp/id_ed25519.pub >> ~/.ssh/authorized_keys
this might give an error such as file not found, in that case:
mkdir -p ~/.ssh
and try again

after this is done, log off and try (host)
ssh -p 2222 mininet@localhost
again, should not be a password promt anymore





NOT YET CLEAR ABOUT THESE STEPS:
# Create if it does not exists yet
The second one is to mount the file measurement location on the RAM. This should limit the impact of the hard disk on the results.
$ sudo mkdir /mnt/tmpfs
# This should be run each time the VM is restarted
$ sudo mount -t tmpfs -o size=256M tmpfs /mnt/tmpfs
steps dont work so far



running the model: dont forget to conda activate mpquicrl



going to keep the date on here so i know where i am

2 may:
trying to get the dash framework to work, should not be too hard except for installing some packages
dash framework VM download does not work, will probably email the original creator asking if he still has it somewhere

build the "MPQUIC-SBD: Multipath QUIC (MPQUIC) with support of Shared Bottleneck Detection (SBD)" framework, dont know how it is usefull yet tho...


3 may:
have set up a conda env called mpquicrl to get all the correct package versions and pyhthon version that is needed by mpquic-rl, the requirements.txt is kind off a mess and it took quite a while to get all packages working.
including a yml file after i get things working should make this easier to work on
need to figure out now what to run and how to run the model, since there is no explanation as to what does what anywhere. need to create a proper README with instructions as to how to run these things, since right now this is costing a lot of time to figure out.
some changed made to MPQUIC-meta-learning

inside Code/main/mpquic-rl/central_service/centraltrainer/basic_thread.py, there is a utils.logger import config_logger, this does not seem to work, changed it to python_utils.logger import Logged. should function the same.
not sure about this fix, need to look into this if i run into issues: https://sherpa.readthedocs.io/en/latest/overview/utilities.html and https://stackoverflow.com/questions/61742333/can-not-import-logger-from-utils-error

main.py is moved from Code/main/mpquic-rl/central_service/main.py to Code/main/mpquic-rl/main.py, since the imports inside this file seem to be directed as if main is located there.

the requirement file has a package called avalanche-rl, this package needs to be downloaded via pip install git+https://github.com/ContinualAI/avalanche-rl.git, NOT what is currently in the requirement.txt

there is a runs.zip file located in Code/main/mpquic-rl/central_service/, this file needs to be extracted at Code/main/mpquic-rl/runs


4 may:
placed mpquic-rl in code github, moved the runs folder
fixed runs.zip location, extracted to proper file

the fix for the config_logger did not work. this is not a package but another file, now seems to be resolved (changed "from utils.logger import config_logger" to "from central_service.utils.logger import config_logger")

6 may:
fixed all previous issues (for now), and placed the middleware folder inside of the VM in attempt to fix the following issue:

FileNotFoundError: [Errno 2] No such file or directory: 'ssh -p 2222 mininet@localhost "stdbuf -oL killall ~/go/bin/middleware"': 'ssh -p 2222 mininet@localhost "stdbuf -oL killall ~/go/bin/middleware"'
i dont understand where this error is coming from, it should not be looking for  a file but run this command.


7 may:
since i dont know a solution to the problem from 6 may, lets try to get the original mpquic-rl by kakanis running.

8 may:
running this command: qemu-img convert -O qcow2 vm-mpquic-sbd.vmdk vm-mpquic-sbd.qcow2
seems to fix the broken VM, but only works on windows.

9 may:
got the working VM from pijus, this work is done for now
solving dependancy/version/depricated issues for the repository is now the main focus.
using python 3.6 is not an option anymore since gym is depricated for python 3.7, along with quite a few other packages.
now switched to using python 3.9 which required manually installing packages. i am creating this in a conda env so it should be easy to save and then recreate once it is working.

10 may:
environment is done, also created a short tutorial for setting up the VM (on windows)


13 may:
working on improving the QoE

14 may:
shortcut to correct place 
cd  code\mpquic-rl\mpquic-rl\central_service
python main.py
training model using minrrt to set a baseline. Dont understand fully yet why it starts at trace 9, and what should be used when training a new model, since it seems to be running on models that are created before.
no need to seperately safe the plots, they get saved but are in gitignore, saved under central_service\logs\log number

due to a mistake in the code somewhere the logs get saved in two different file locations, one in the main directory and the other in the correct central service folder. needs to get fixed.