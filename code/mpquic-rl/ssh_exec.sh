#!/bin/bash
User="mininet"
Host="172.23.160.1"
Port="2222"

ssh -p $Port $User@$Host "source /etc/profile; source ~/.profile; $1"