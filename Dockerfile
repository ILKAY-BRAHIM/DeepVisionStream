FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# Optional: install additional packages if needed
# RUN apt-get update && apt-get install -y vim git
# RUN ls bin/bash /opt/nvidia/deepstream/deepstream-7.1/sources/
WORKDIR /
COPY ./initialize.sh  /
RUN  /bin/bash /initialize.sh
WORKDIR /opt/nvidia/deepstream/deepstream-7.1/sources/
CMD [ "/bin/bash" ]