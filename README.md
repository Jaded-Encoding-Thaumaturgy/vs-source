# vs-source

### DVDs were an error, but can be nice to use with this package.

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
One of these plugins is required:

- dvdsrc2
- d2vSource and either DGIndex or [patched d2vwitch](https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b)

> _DGIndex is recommended over d2vwitch as the latter has various problems._
>
> It can also be used under linux with wine; notice that doing it requires ` binfmt` and `dgindex` in PATH.
>
> ```bash
> chmod +x DGIndex.exe
> sudo ln -s $(pwd)/DGIndex.exe /usr/bin/dgindex
> ```
>
> <br>

<br>

Optional dependecies:

- [dvdsrc_dvdnav_title_ptt_test](https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538) to automatically check the chapters against `libdvdnav`.
- [dvdsrc](https://github.com/jsaowji/dvdsrc/) to automatically double-check the determined dvdstrucut agaist libdvdread.
- [mpv](https://github.com/mpv-player/mpv) to determine chapter splits by loading the DVD and hitting I or using it like this:

  > ```bash
  > mpv --dvd-device=<iso> dvd://<title>
  > # mpv titles are zero-indexed
  > # vssource titles indices start from 1
  > # sometimes it is useful to to scale the osc down
  > # --script-opts=osc-scalewindowed=0.4,osc-scalefullscreen=0.4
  > ```

  Related to mpv, the [mpv-dvd-browser](https://github.com/CogentRedTester/mpv-dvd-browser) plugin can be useful for this too.

<br>

> Getting a vs.AudioNode and demuxing AC3 **requires** [dvdsrc2](https://github.com/jsaowji/dvdsrc2/)
>
> **The only codecs offically supported are: stereo 16bit LPCM and AC3**

## Usage

After installation, functions can be loaded and used as follows:

```py

from vssource import IsoFile, D2VWitch, DGIndex
from vstools import set_output

# Autodetect what to index with
iso = IsoFile('.\DVD_VIDEOS\Suzumiya_2009_DVD\KABA_6001.ISO')

# Force index with dgindex
iso = IsoFile('.\SOME_DVD_FOLDER\HARUHI', indexer=DGIndex)

title1 = iso.get_title(1)

# prints audio and chapter information
print(title1)

title1.video.set_output(0)
title1.audios[0].set_output(1)

title1.dump_ac3('full_title.ac3', 0)

# -1 is replace with end i.e 15
ep1, ep2, ep3 = title1.split_at([6, 11])
ep1, ep2, ep3 = title1.split_ranges([(1, 5), (6, 10), (11, 15)])
ep1           = title1.split_range(1, 5)
ep2           = title1.split_range(6, 10)
ep3           = title1.split_range(11, -1)

# preview your splits
title1.preview(title1.split_at([6,11]))

ep1 = title1.split_range(1,5,audio=0)

print(ep1.chapters[:-1])

set_output([
    ep1.video,
    ep1.audio,
])

a = ep1.ac3('/tmp/ep1.ac3',0)
# a is in seconds of how much samples are there too much at the start

## Advanced Usage

# Remove junk from the end
title1 = iso.get_title(1)
title1.chapters[-1] -= 609
title1.preview(title1.split_at([7, 12]))


#--------

splits = []
# title 2 through 5 containe episodes
# remove 'junk' from beginning
# you can quickly check with preview
for a in range(2,6):
    t = iso.get_title(a)
    t.chapters[0] += 180
    splits += [t.split_at([])]
    t.preview(splits[-1])

print(splits[0].ac3('/tmp/ep1.ac3'))
# 0.019955555555555556

#--------

# multi angle + multi audio + rff mode
# japanese
a = iso.get_title(4, angle_nr=1, rff_mode=2).split_at([5, 10, 15],audio=1)

# EP 1 japanese
a[0].video.set_output(0)
a[0].audio.set_output(1)

# italian
b = iso.get_title(4, angle_nr=2, rff_mode=2).split_at([5, 10, 15],audio=0)

# ep 2 italian
b[1].video.set_output(0)
b[1].audios[0].set_output(1)
```

The `Title.split_at` method should behave just like mkvmerge chapter splits (split before the chapter, first chapter is 1), so if you want the (first chapter, all other chapters after).

Output chapters always start with frame zero and end at last frame.
