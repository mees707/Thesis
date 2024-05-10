#!/bin/bash

User="mininet"
Host="localhost"
SourceDir="central_service/minitopo/."
SendDir="~/git/minitopo/src/"

scp -P 2222 -r $SourceDir $User@$Host:$SendDir