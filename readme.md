# Shadertoy Utils!

These utilities can turn images and sounds into packed grayscale/luminance and 8bit PCM buffers... along with the code necessary to reproduce them in Shadertoy!

# Pre-requisites

You need scipy and PIL.

    pip install scipy Pillow

You also need ffmpeg. I have Windows, so I just downloaded the static builds and placed them in the same directory. If you're on *nix just change:

    ffmpegPath = "ffmpeg\\bin\\ffmpeg.exe"

to

    ffmpegPath = "ffmpeg"

or where ever ffmpeg is.

# How to use

## Images

This is the code I used to convert all the images for the [Angels cracktro](https://www.shadertoy.com/view/WljSR1):

    shader = imageToLumBuffer('angels1.png', [5.100, 8.500], [2.050, 4.60])
    shader = imageToLumBuffer('angels2.png', [4.5, 15.500], [1.71, 4.300], shader, 1)
    shader = imageToLumBuffer('angelshorse.png', [5.0, 5.0], [1.957, 1.750], shader, 2)
    shader = imageToLumBuffer('angels_scroll.png', [2.0, 4.0], [0.5, 2.0], shader, 3)

    file = open('shaderimage.txt', 'w')
    file.write(shader)
    file.close()

First parameter to `imageToLumBuffer()` is the filename. The second parameter is xy scale and the third parameter is the xy offset. The generated function will take those into account and return zero outside of those bounds to make your life easier... and so that you don't have to remember what works in your scenario for every particular image. If you wanna handle that yourself just use `[1.0, 1.0]` and `[0.0, 0.0]`.

Fourth parameter allows you to concatenate previously generated code with this one so that you can use all of your images in the same shader.

Fifth parameter is for images generated after the first one since they need their own unique IDs. Again, it's just for convenience so you don't have to manually change function/variable names.

Your images cannot in total (for one Shadertoy) use up more than 4096 uniforms (4096 * 4 pixels). Choose your resolutions wisely :)

## Sounds

This is again the code used for the above cracktro.

    shader = soundToShaderToy('angels.mp3', 'angels.wav', 2000, 0.8, 7.6)

    file = open("shadersound.txt", "w")
    file.write(shader)
    file.close()

The first two parameters are input and output files.

Third parameter is the final frequency. Some audio (e.g. low frequency chiptunes) may still sound acceptable at 1.2KHz... but other tracks may need higher frequency. You'll have to figure this out by listening to the produced .WAV, as that will be closest to what will playback inside Shadertoy.

The fourth and fifth parameters are start position and length of the audio clip you're choosing inside the original track.

As with the images, you're still bound by 4096 uniforms with each packing 4 8bit samples. How much length vs. frequency you want to pack in there is up to you. You'll have to get creative with the clip you want to include and how much frequency you'll need to reproduce it faithfully :)

### Sampling frequency vs. sample precision

Generally speaking, going from 16bit PCM to 8bit PCM isn't as detrimental as going from 44KHz to 8KHz, 2KHz or 1KHz. Funny how that works, eh? The more faithful the vibrations of the speaker diaphragm the better the reproduction.

For comparison, check out `madworld_8bit_highfreq.mp3` (which is 8bit PCM -- created with Audacity -- at 44KHz) with `madworld.wav` (which -- by default -- is 16bit PCM at 1.2KHz). While `madworld_8bit_highfreq.mp3` has a bit of static, the piano sounds WAY better than the same tune at 1.2KHz with higher precision samples.

# Known issues

Apparently ANGLE's OpenGL backend does not like large uniform arrays. '\_o.O_/`

# License

M.I.T. for the code.

Everything else belongs to their respective copyright holders.

Go nuts.