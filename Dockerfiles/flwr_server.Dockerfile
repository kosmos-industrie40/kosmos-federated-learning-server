# Build a virtualenv using the appropriate Debian release
# * Install python3-venv for the built-in Python3 venv module (not installed by default)
# * Install gcc libpython3-dev to compile C Python modules
# * Update pip to support bdist_wheel
FROM python:3.8.0 AS build-venv
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update
RUN apt-get install --no-install-suggests --no-install-recommends --yes gcc git
RUN python3 -m venv /app/venv
RUN /app/venv/bin/python --version
RUN /app/venv/bin/pip install --upgrade pip

# Build the virtualenv as a separate step: Only re-execute this step when requirements.txt changes
# Build command: python setup.py bdist_wheel
RUN /app/venv/bin/pip install setuptools==41.0.0
RUN /app/venv/bin/pip install -r /app/requirements.txt

#If .git folder doesn't exist while setup.py install is running an error will occur:
#Make sure you're either building from a fully intact git repository or PyPI tarballs.
COPY src/ /app/src
COPY setup.cfg /app
COPY setup.py /app
RUN /app/venv/bin/python setup.py install
RUN rm -rf /app/.git


FROM python:3.8.0
RUN apt update
RUN apt install --yes texlive texlive-latex-extra texlive-fonts-recommended dvipng  cm-super
RUN mkdir /app
WORKDIR /app
#USER nonroot:nonroot
COPY --from=build-venv /app/venv ./venv
COPY ./src ./src
WORKDIR /app/src/fl_server/
EXPOSE 8080
EXPOSE 50052:50052
ENTRYPOINT ["/app/venv/bin/python", "flwr_server.py"]
