# vs-source

### DVDs are nice actually.
<br>

A wrapper for DVD file structure and ISO files.

Required plugins:
- dvdsrc
- or
- d2vsource and (dgindex or ([patched](https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b) d2vwitch))

dgindex is recommended over d2vwitch because d2vwitch has problems.

dgindex can be used under linux with wine and requires binfmt and dgindex in path
chmod +x DGIndex.exe
sudo ln -s $(pwd)/DGIndex.exe /usr/bin/dgindex


Optional dependecies:
- The Chapters can be automatically be checked against libdvdnav using [dvdsrc_dvdnav_title_ptt_test](https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538)
- The determined dvdstrucut can automatically be double checked agaist libdvdread with dvdsrc

Getting a vs.AudioNode  **requires** [dvdsrc](https://github.com/jsaowji/dvdsrc/)

The splits should behave just like mkvmerge chapter splits (split before the chapter, first chapter is 1)

So if you want the (first chapter, all other chapters after) [2].

Output chapters always start at beginning (0) and end at last frame.

Hitting I in mpv is helpful for determining chapter splits
```bash
mpv --dvd-device=<iso> dvd://<title>
# title are zero index in mpv
# vssource titles are starting from 1
# sometimes it is useful to to scale the osc down
# --script-opts=osc-scalewindowed=0.4,osc-scalefullscreen=0.4
```

Also helpful [mpv-dvd-browser](https://github.com/CogentRedTester/mpv-dvd-browser)

It has not been tested throughly in production yet, so api, functionality, features and output is subject to change.

Audio support is still flaky, one known problem is if a complete title does not contain any seamless cut, which is mostly not the case on main feature.

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

# prints audio and chapter information
print(title1)

title1.video().set_output(0)
title1.audio(0).set_output(1)

title1.dump_ac3("full_title.ac3", 0)

t1 = title1
# -1 is replace with end i.e 15
ep1,ep2,ep3 = t1.split([6,11])
ep1,ep2,ep3 = t1.split_ranges([(1,5),(6,10),(11,15)])
ep1         = t1.split_range(1, 5)
ep2         = t1.split_range(6, 10)
ep3         = t1.split_range(11, -1)

# preview your splits
t1.preview(t1.split([6,11]))

ep1 = t1.split_range(1,5,audio=0)
print(ep1.chapters[:-1])
set_output([
    ep1.video,
    ep1.audio,
])

a = ep1.split_ac3(0)
#a[0] contain path to ac3 file
#a[1] contains in seconds of how much samples are there too much at the start
```

## Advanced Usage
```py

# Remove junk from the end 
t1 = iso.get_title(1)
t1.chapters[-1] -= 609
t1.preview(t1.split([7,12]))


#title 2 through 5 containe episodes
#remove "junk" from beginning
#you can quickly check with preview
for a in range(2,6):
    t = iso.get_title(a)
    t.chapters[0] += 180
    t.preview(t.split([]))

# multi angle + multi audio + rff mode
#japanese
a = iso.get_title(4, angle_nr=1, rff_mode=2).split([5,10,15],audio=1)

# ep 1 japanese
a[0].video.set_output(0)
a[0].audio.set_output(1)

#italian
b = iso.get_title(4, angle_nr=2, rff_mode=2).split([5,10,15],audio=0)

# ep 2 italian
b[1].video.set_output(0)
b[1].audio.set_output(1)

```