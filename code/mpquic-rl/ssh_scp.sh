#!/bin/bash
User="mininet"
Host="172.23.160.1"
SendDir="~/go/src/github.com/mkanakis/middleware/"
Port="2222"

scp -P 2222 "$1" $User@$Host:"$2"



#ssh -p $Port $User@$Host "$1"