#sshpass -p "mininet" rsync -azP -e 'ssh -p 2222' --exclude={'*/venv/*','*/proxy_module/*','*.so','go.sum','proxy_module.h'} ./mpquic-sbd mininet@172.23.160.1:/home/mininet/Workspace/
sshpass -p "mininet" rsync -azP -e 'ssh -p 2222' --exclude={'*/venv/*','*.so','go.sum','proxy_module.h'} ./mpquic-sbd mininet@172.23.160.1:/home/mininet/Workspace/
./ssh_exec.sh "cd ~/Workspace/mpquic-sbd/ && export GOPATH=~/go && export GOBIN=~/go/bin && time ./build.sh" # go env
#./ssh_exec.sh "cd ~/go/src/github.com/mkanakis/middleware/ && go build && cp middleware ~/go/bin/middleware"