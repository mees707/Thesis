sshpass -p "mininet" rsync -azP -e 'ssh -p 2222' --exclude={'*/venv/*','*.so','go.sum','proxy_module.h'} ./video/conv_video/dash/* mininet@172.23.160.1:/home/mininet/dash/video/run