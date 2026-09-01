FROM gcr.io/oss-fuzz-base/base-builder-python
RUN pip3 install --upgrade pip
COPY . $SRC/coola
COPY .clusterfuzzlite/build.sh $SRC/build.sh
WORKDIR $SRC/coola
