# vs-source

### DVDs were an error.

<br>

A wrapper for DVD file structure and ISO files.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB). <br><br> <br><br>

## How to install

Install `vssource` with the following command:

```sh
pip install vssource
```

Or if you want the latest git version, install it with this command:

```sh
pip install git+https://github.com/Jaded-Encoding-Thaumaturgy/vs-source.git
```

<br>

## Usage

After installation, functions can be loaded and used as follows:

```py
from vssource import IsoFile, DGIndexNV

# Indexing with D2VWitch
haruhi = IsoFile(r".\Suzumiya_2009_DVD\KABA_6001.ISO")
...
# Indexing with DGIndexNV
haruhi = IsoFile(r".\Suzumiya_2009_DVD\KABA_6001.ISO", DGIndexNV())

ep1, ep2, dvd_menu = haruhi.get_title(None, [(0, 7), (8, 15), -1])
...
```
