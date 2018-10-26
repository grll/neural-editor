#!/bin/bash
set -e
    env >> /etc/environement
    service ssh start

exec "$@"