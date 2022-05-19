The generator.py programs are used for creating tensorflow datasets from .tfrecord files. The imput images are croped or padded to the 128x64 size mentiond in the theses.
The used packages can be used with the Anaconda .ymal file.
Used packages and their version:
# Name                    Version                   Build  Channel
_tflow_select             2.1.0                       gpu
abseil-cpp                20210324.2           hd77b12b_0
absl-py                   0.15.0             pyhd3eb1b0_0
aiohttp                   3.8.1            py39h2bbff1b_1
aiosignal                 1.2.0              pyhd3eb1b0_0
astor                     0.8.1            py39haa95532_0
asttokens                 2.0.5              pyhd8ed1ab_0    conda-forge
astunparse                1.6.3                      py_0
async-timeout             4.0.1              pyhd3eb1b0_0
attrs                     21.4.0             pyhd3eb1b0_0
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                        py_2    conda-forge
backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
blas                      1.0                         mkl
blinker                   1.4              py39haa95532_0
brotlipy                  0.7.0           py39h2bbff1b_1003
ca-certificates           2021.10.8            h5b45459_0    conda-forge
cachetools                4.2.2              pyhd3eb1b0_0
certifi                   2021.10.8        py39hcbf5309_2    conda-forge
cffi                      1.15.0           py39h2bbff1b_1
charset-normalizer        2.0.4              pyhd3eb1b0_0
click                     8.0.4            py39haa95532_0
colorama                  0.4.4              pyhd3eb1b0_0
cryptography              3.4.8            py39h71e12ea_0
cudatoolkit               11.3.1               h59b6b97_2
cudnn                     8.2.1                cuda11.3_0
cycler                    0.11.0                   pypi_0    pypi
dataclasses               0.8                pyh6d0b6a4_7
debugpy                   1.6.0            py39h415ef7b_0    conda-forge
decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
dill                      0.3.4                    pypi_0    pypi
entrypoints               0.4                pyhd8ed1ab_0    conda-forge
executing                 0.8.3              pyhd8ed1ab_0    conda-forge
flatbuffers               2.0.0                h6c2663c_0
fonttools                 4.32.0                   pypi_0    pypi
frozenlist                1.2.0            py39h2bbff1b_0
gast                      0.4.0              pyhd3eb1b0_0
giflib                    5.2.1                h62dcd97_0
google-auth               2.6.0              pyhd3eb1b0_0
google-auth-oauthlib      0.4.1                      py_2
google-pasta              0.2.0              pyhd3eb1b0_0
googleapis-common-protos  1.56.0                   pypi_0    pypi
grpcio                    1.42.0           py39hc60d5dd_0
h5py                      3.6.0            py39h3de5c98_0
hdf5                      1.10.6               h7ebc959_0
icc_rt                    2019.0.0             h0cc432a_1
icu                       68.1                 h6c2663c_0
idna                      3.3                pyhd3eb1b0_0
importlib-metadata        4.11.3           py39haa95532_0
intel-openmp              2021.4.0          haa95532_3556
ipykernel                 6.13.0           py39h832f523_0    conda-forge
ipython                   8.2.0            py39hcbf5309_0    conda-forge
jedi                      0.18.1           py39hcbf5309_1    conda-forge
joblib                    1.1.0                    pypi_0    pypi
jpeg                      9d                   h2bbff1b_0
jupyter_client            7.2.2              pyhd8ed1ab_1    conda-forge
jupyter_core              4.9.2            py39hcbf5309_0    conda-forge
keras-preprocessing       1.1.2              pyhd3eb1b0_0
kiwisolver                1.4.2                    pypi_0    pypi
libcurl                   7.82.0               h86230a5_0
libpng                    1.6.37               h2a8f88b_0
libprotobuf               3.17.2               h23ce68f_1
libsodium                 1.0.18               h8d14728_1    conda-forge
libssh2                   1.9.0                h7a1dbc1_1
markdown                  3.3.4            py39haa95532_0
matplotlib                3.5.1                    pypi_0    pypi
matplotlib-inline         0.1.3              pyhd8ed1ab_0    conda-forge
mkl                       2021.4.0           haa95532_640
mkl-service               2.4.0            py39h2bbff1b_0
mkl_fft                   1.3.1            py39h277e83a_0
mkl_random                1.2.2            py39hf11a4ad_0
multidict                 5.1.0            py39h2bbff1b_2
nest-asyncio              1.5.5              pyhd8ed1ab_0    conda-forge
numpy                     1.21.5           py39h7a0a035_1
numpy-base                1.21.5           py39hca35cd5_1
oauthlib                  3.2.0              pyhd3eb1b0_0
openssl                   1.1.1n               h8ffe710_0    conda-forge
opt_einsum                3.3.0              pyhd3eb1b0_1
packaging                 21.3               pyhd8ed1ab_0    conda-forge
pandas                    1.4.2            py39h2e25243_1    conda-forge
parso                     0.8.3              pyhd8ed1ab_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.1.0                    pypi_0    pypi
pip                       21.2.4           py39haa95532_0
promise                   2.3                      pypi_0    pypi
prompt-toolkit            3.0.29             pyha770c72_0    conda-forge
protobuf                  3.17.2           py39hd77b12b_0
psutil                    5.9.0            py39hb82d6ee_1    conda-forge
pure_eval                 0.2.2              pyhd8ed1ab_0    conda-forge
pyasn1                    0.4.8              pyhd3eb1b0_0
pyasn1-modules            0.2.8                      py_0
pycparser                 2.21               pyhd3eb1b0_0
pygments                  2.11.2             pyhd8ed1ab_0    conda-forge
pyjwt                     2.1.0            py39haa95532_0
pyopenssl                 21.0.0             pyhd3eb1b0_1
pyparsing                 3.0.8              pyhd8ed1ab_0    conda-forge
pyreadline                2.1              py39haa95532_1
pysocks                   1.7.1            py39haa95532_0
python                    3.9.12               h6244533_0
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python-flatbuffers        1.12               pyhd3eb1b0_0
python_abi                3.9                      2_cp39    conda-forge
pytz                      2022.1             pyhd8ed1ab_0    conda-forge
pywin32                   303              py39hb82d6ee_0    conda-forge
pyzmq                     22.3.0           py39he46f08e_2    conda-forge
requests                  2.27.1             pyhd3eb1b0_0
requests-oauthlib         1.3.0                      py_0
rsa                       4.7.2              pyhd3eb1b0_1
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3            py39h0a974cb_0
setuptools                61.2.0           py39haa95532_0
six                       1.16.0             pyhd3eb1b0_1
sklearn                   0.0                      pypi_0    pypi
snappy                    1.1.8                h33f27b4_0
sqlite                    3.38.2               h2bbff1b_0
stack_data                0.2.0              pyhd8ed1ab_0    conda-forge
tensorboard               2.6.0                      py_1
tensorboard-data-server   0.6.0            py39haa95532_0
tensorboard-plugin-wit    1.6.0                      py_0
tensorflow                2.6.0           gpu_py39he88c5ba_0
tensorflow-addons         0.16.1                   pypi_0    pypi
tensorflow-base           2.6.0           gpu_py39hb3da07e_0
tensorflow-datasets       4.5.2                    pypi_0    pypi
tensorflow-estimator      2.6.0              pyh7b7c402_0
tensorflow-gpu            2.6.0                h17022bd_0
tensorflow-metadata       1.7.0                    pypi_0    pypi
termcolor                 1.1.0            py39haa95532_1
threadpoolctl             3.1.0                    pypi_0    pypi
tornado                   6.1              py39hb82d6ee_3    conda-forge
tqdm                      4.64.0                   pypi_0    pypi
traitlets                 5.1.1              pyhd8ed1ab_0    conda-forge
typeguard                 2.13.3                   pypi_0    pypi
typing-extensions         4.1.1                hd3eb1b0_0
typing_extensions         4.1.1              pyh06a4308_0
tzdata                    2022a                hda174b7_0
urllib3                   1.26.8             pyhd3eb1b0_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
werkzeug                  2.0.3              pyhd3eb1b0_0
wheel                     0.35.1             pyhd3eb1b0_0
win_inet_pton             1.1.0            py39haa95532_0
wincertstore              0.2              py39haa95532_2
wrapt                     1.13.3           py39h2bbff1b_2
yarl                      1.6.3            py39h2bbff1b_0
zeromq                    4.3.4                h0e60522_1    conda-forge
zipp                      3.7.0              pyhd3eb1b0_0
zlib                      1.2.11               hbd8134f_5