#!/bin/bash

User="mininet"
Host="localhost"
SendDir="~/go/src/github.com/lucas-clemente/quic-go/"

scp -P 2222 scheduler.go $User@$Host:$SendDir
scp -P 2222 zclient.go $User@$Host:$SendDir
scp -P 2222 zpublisher.go $User@$Host:$SendDir


# scp example/client_browse_deptree/zpublisher.go $User@$Host:$SendDir/example/client_browse_deptree/
scp -P 2222 example/client_browse_deptree/main.go $User@$Host:$SendDir/example/client_browse_deptree/
# scp example/main.go $User@$Host:$SendDir/example/


# scp congestion/bdw_stats.go $User@$Host:$SendDir/congestion 

# Add PATH to stream
scp -P 2222 session.go $User@$Host:$SendDir
scp -P 2222 stream.go $User@$Host:$SendDir
scp -P 2222 h2quic/server.go $User@$Host:$SendDir/h2quic/