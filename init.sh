echo "#!/bin/bash -l" >> /etc/mypython.sh
echo "source /etc/environement" >> /etc/mypython.sh
echo "/opt/conda/envs/pytorch-py27/bin/python \"$@\"" >> /etc/mypython.sh
chmod 755 /etc/mypython.sh
env >> etc/environement