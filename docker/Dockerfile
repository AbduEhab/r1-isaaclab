# syntax=docker/dockerfile:1.7
#
# Isaac Sim 6.2.0 + Isaac Lab v3.0.0, environment only.
#
#   docker build -f docker/Dockerfile -t isaac-env:6.2 .
#   docker run --gpus all -it -v ${PWD}:/workspace isaac-env:6.2
#
# Deliberately contains no project code and no daemon: mount the repository into /workspace and
# run it with Isaac Lab's own launcher. The point of the image is a pinned, portable Isaac Sim and
# Isaac Lab, which is the part that is expensive and version-sensitive to reproduce.

ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
ARG ISAACLAB_TAG=v3.0.0

FROM ${ISAACSIM_IMAGE}

ARG ISAACLAB_TAG
ARG ISAACSIM_ROOT=/isaac-sim
ARG ISAACLAB_PATH=/opt/IsaacLab

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    ISAACSIM_ROOT=${ISAACSIM_ROOT} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Corporate TLS interception: when a CA bundle is passed as a build secret, trust it for git,
# pip and uv. Without the secret the image builds on an unfiltered network as usual.
#   docker build --secret id=corp_ca,src=C:\certs\corporate-roots.pem -f docker/Dockerfile -t isaac-env:6.2 .
RUN --mount=type=secret,id=corp_ca \
    if [ -s /run/secrets/corp_ca ]; then \
        cp /run/secrets/corp_ca /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# Isaac Lab from the same tag the Windows machine runs, linked to the container's Isaac Sim.
RUN git clone --depth 1 --branch "${ISAACLAB_TAG}" https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
 && ln -sfn "${ISAACSIM_ROOT}" "${ISAACLAB_PATH}/_isaac_sim" \
 && find "${ISAACLAB_PATH}" -type f -name "*.sh" -exec sed -i 's/\r$//' {} + \
 && chmod 755 "${ISAACLAB_PATH}/isaaclab.sh"

# Installs the Isaac Lab packages (editable), torch cu130, Newton, RSL-RL and their dependencies
# into Isaac Sim's bundled Python. The long step: several GB from download.pytorch.org and PyPI.
RUN cd "${ISAACLAB_PATH}" && ./isaaclab.sh --install newton,rl

# Environment image: do not inherit Isaac Sim's headless launcher, so `docker run <image> <cmd>`
# runs the command directly. The simulator itself is launched explicitly.
ENTRYPOINT []
WORKDIR /workspace
# Isaac Sim's headless launcher checks this; harmless for a shell-entrypoint image.
ENV ACCEPT_EULA=Y
CMD ["/bin/bash"]




