pip3 install pandas
pip3 install numpy
pip3 install tensorflow
pip3 install tensorflow-hub
# https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
# pip3 install googletrans
git clone https://github.com/BoseCorp/py-googletrans.git
cd ./py-googletrans
python3 setup.py install
cd ..

export TFHUB_CACHE_DIR=models_cache
mkdir $TFHUB_CACHE_DIR
