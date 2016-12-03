#!/bin/bash
CONTAINERNAME="tensorflow"
docker stop $CONTAINERNAME
docker rm $CONTAINERNAME
docker run --name $CONTAINERNAME -t -d -v $(pwd):/code -p 6006:6006 -p 8889:8888 tensorflow/tensorflow