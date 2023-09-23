# vs-source

### DVDs are nice actually.
<br>

A wrapper for DVD file structure and ISO files.

Required plugins:
- dvdsrc
- or
- d2vsource and (dgindex or ([patched](https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b) d2vwitch))

Optional dependecies:
- The Chapters can be automatically be checked against libdvdnav using [dvdsrc_dvdnav_title_ptt_test](https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538)
- The determined dvdstrucut can automatically be double checked agaist libdvdread with dvdsrc

Getting a vs.AudioNode  **require** [dvdsrc](https://github.com/jsaowji/dvdsrc/)

The splits should behave just like mkvmerge chapter splits (split before the chapter, first chapter is 1)

So if you want the (first chapter, all other chapters after) [2].

Hitting I in mpv is helpful for determining chapter splits
```bash
mpv --dvd-device=<iso> dvd://<title>
# title are zero index in mpv
# vssource titles are starting from 1
# sometimes it is useful to to scale the osc down
# --script-opts=osc-scalewindowed=0.4,osc-scalefullscreen=0.4
```

It has not been tested throughly in production yet, so api, functionality, features and output is subject to change.

<br><br>
## How to install

Install `vssource` with the following command:

```sh
$ pip install vssource
```

Or if you want the latest git version, install it with this command:

```sh
$ pip install git+https://github.com/Irrational-Encoding-Wizardry/vs-source.git
```
<br>

## Usage

After installation, functions can be loaded and used as follows:

```py
from vssource import IsoFile, DGIndex
from vstools import set_output

# Autodetect what to index with
iso = IsoFile(".\DVD_VIDEO.ISO")

# Index with dvdsrc
iso = IsoFile(".\SOME_DVD_FOLDER", True)
print(iso)

title1 = iso.get_title(1)
print(title1)

title1.video().set_output(0)
title1.audio(0).set_output(1)

title1.dump_ac3("full_title.ac3", 0)

t1 = iso.get_title(1, [7, 13])

# these do the same thing
set_output(list(t1.split_video()) + list(t1.split_audio(0)))
t1.preview_output_split(0)

ep1,ep2,information = t1.split_video()
ep1a,ep2a,informationa = t1.split_audio(0)
a,b,c = t1.split_ac3(0)
#a,b,c[0] contain path to ac3 file
#a,b,c[1] contains in seconds of how much samples are there too much at the start
```

## Advanced Usage
```py
# Remove junk from the end 
t1 = iso.get_title(1,[7, 12])
t1.chapters[-1] -= 609
ep1,ep2,ep3 = t1.split_video()


#title 2 through 5 containe episodes
#remove "junk" from beginning
#you can quickly check 
for a in range(2,6):
    t = iso.get_title(a,splits=[])
    t.chapters[0] += 180
    t.preview_output_split()

# multi angle + rff mode
#japanese
t4 = iso.get_title(4, [5,10,15], angle_nr=1, rff_mode=2)
t4.preview_output_split(audio=1)
#italian
t4 = iso.get_title(4, [5,10,15], angle_nr=2, rff_mode=2)
t4.preview_output_split(audio=0)
```