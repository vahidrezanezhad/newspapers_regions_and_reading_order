all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp sbb_newspapers/sbb_newspapers.py ~/bin/sbb_newspapers
       chmod +x ~/bin/sbb_newspapers
