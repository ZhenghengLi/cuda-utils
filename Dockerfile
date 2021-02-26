# build ############################################################
FROM zhenghengli/ubuntu-devel:20.04-cuda11.0.3 AS builder

# build and install
ARG SOURCE_DIR=/opt/cuda-utils_src
ARG BUILD_DIR=/opt/cuda-utils_build
ARG INSTALL_DIR=/opt/cuda-utils
ADD $PWD $SOURCE_DIR
RUN set -x \
    ## build and install
    && cmake -S $SOURCE_DIR -B $BUILD_DIR -G Ninja -D CMAKE_BUILD_TYPE=Release \
    && cmake --build $BUILD_DIR --parallel $(nproc) \
    && cmake --install $BUILD_DIR --prefix $INSTALL_DIR \
    ## clean
    && rm -rf $SOURCE_DIR \
    && rm -rf $BUILD_DIR

# deploy ############################################################
FROM zhenghengli/ubuntu-runtime:20.04-cuda11.0.3

# copy from builder
ARG INSTALL_DIR=/opt/cuda-utils
COPY --from=builder $INSTALL_DIR $INSTALL_DIR

# add labels
ARG SOURCE_COMMIT
ARG COMMIT_MSG
ARG BUILD_TIME
LABEL description="CUDA Utilities" \
    maintainer="Zhengheng Li <zhenghenge@gmail.com>" \
    source_commit="$SOURCE_COMMIT" \
    commit_msg="$COMMIT_MSG" \
    build_time="$BUILD_TIME"

# set runtime environment variables
ENV PATH=$INSTALL_DIR/bin:$INSTALL_DIR/scripts:$PATH \
    LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH \
    SOURCE_COMMIT="$SOURCE_COMMIT" \
    COMMIT_MSG="$COMMIT_MSG" \
    BUILD_TIME="$BUILD_TIME"

CMD ["/bin/bash"]
