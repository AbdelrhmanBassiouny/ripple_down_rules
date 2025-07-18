FROM python:3.12 AS base

SHELL [ "/bin/bash" , "-c" ]

RUN apt-get update && \
    apt install graphviz graphviz-dev -y && \
    python -m pip install --upgrade pip && \
    pip install vcstool

## Set the user to the non-root user
#ARG USERNAME=root
#
## Switch to non-root user
#USER $USERNAME
ENV HOME=/root

# Create overlay workspace
WORKDIR $HOME/workspace/src
COPY ./packages.repos .

# Install virtualenv and virtualenvwrapper
RUN python3 -m venv $HOME/.virtualenvs/venv && \
    source $HOME/.virtualenvs/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install virtualenvwrapper

# Add virtualenvwrapper to bashrc
RUN echo "export WORKON_HOME=$HOME/.virtualenvs" >> $HOME/.bashrc && \
    echo "export VIRTUALENVWRAPPER_PYTHON=$HOME/.virtualenvs/venv/bin/python" >> $HOME/.bashrc && \
    echo "export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.virtualenvs/venv/bin/virtualenv" >> $HOME/.bashrc && \
    echo "source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh" >> $HOME/.bashrc

ENV WORKON_HOME=$HOME/.virtualenvs
ENV VIRTUALENVWRAPPER_PYTHON=$HOME/.virtualenvs/venv/bin/python
ENV VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.virtualenvs/venv/bin/virtualenv

RUN vcs import --input ./packages.repos --recursive

# Create a virtual environment for rdr
RUN source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    mkvirtualenv --system-site-packages rdr -p python3 && \
    workon rdr && \
    pip install --upgrade pip

RUN cd $HOME/workspace/src/ripple_down_rules && \
    git pull && \
    source $HOME/.virtualenvs/venv/bin/virtualenvwrapper.sh && \
    workon rdr && \
    pip install -r requirements-dev-ci.txt && \
    pip install pytest mypy flake8 black isort && \
    pip install .
