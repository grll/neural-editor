#!/bin/bash
env >> /etc/environement
service ssh start

exec "$@"