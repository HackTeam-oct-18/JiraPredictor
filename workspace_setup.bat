pip install -U pip virtualenv
virtualenv --system-site-packages -p python ./venv
.\venv\Scripts\activate

pip install --upgrade pip
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.10.0-cp36-cp36m-win_amd64.whl
pip install pandas numpy sklearn tensorflow-hub yandex_translater

set TFHUB_CACHE_DIR=models_cache
mkdir %TFHUB_CACHE_DIR%

echo tfhub cache dir set to %TFHUB_CACHE_DIR%
