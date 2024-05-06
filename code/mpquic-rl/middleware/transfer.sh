#!/bin/bash

User="mininet"
Host="localhost"
SendDir="~/go/src/github.com/mkanakis/middleware/"

scp -P 2222 interface.go pubsub.go middleware.go zserver.go $User@$Host:$SendDir
