# Shadertoy Utils!

These utilities can turn images, sounds and text into packed buffers for consumption in GLSL... along with the code necessary to reproduce them in Shadertoy!

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

The following code turns an image to a consumable buffer:

    shader = imageToBuffer('graffiti_lowres.png', [1.0, 1.0], [0.0, 0.0], shader, 1, "r2g4b2")

    file = open('shaderimage.txt', 'w')
    file.write(shader)
    file.close()

First parameter to `imageToBuffer()` is the filename. The second parameter is xy scale and the third parameter is the xy offset. The generated function will take those into account and return zero outside of those bounds to make your life easier... and so that you don't have to remember what works in your scenario for every particular image. If you wanna handle that yourself just use `[1.0, 1.0]` and `[0.0, 0.0]`.

Fourth parameter allows you to concatenate previously generated code with this one so that you can use all of your images in the same shader.

Fifth parameter is for images generated after the first one since they need their own unique IDs. Again, it's just for convenience so you don't have to manually change function/variable names.

Sixth parameter is one of the following:

* **"bw"**: Every packed bit is either a black or a white value
* **"luma"**: Every packed byte is a grayscale value indicating brightness
* **"r2g4b2"**: Every byte packs trichromatic color information in r2g4b2 format

Your buffers cannot in total (for one Shadertoy) use up more than 4096 uniforms. Choose your resolutions wisely :)

## Sounds

This is the code used for the angels cracktro.

    shader = soundToShaderToy('angels.mp3', 'angels.wav', 2000, 0.8, 7.6)

    file = open("shadersound.txt", "w")
    file.write(shader)
    file.close()

The first two parameters are input and output files.

Third parameter is the final frequency. Some audio (e.g. low frequency chiptunes) may still sound acceptable at 1.2KHz... but other tracks may need higher frequency. You'll have to figure this out by listening to the produced .WAV, as that will be closest to what will playback inside Shadertoy.

The fourth and fifth parameters are start position and length of the audio clip you're choosing inside the original track.

The sixth parameter optionally creates a switch/case function instead of reading from the buffer. This may break compilation.

As with the images, you're still bound by 4096 uniforms with each packing 4 8bit samples. How much length vs. frequency you want to pack in there is up to you. You'll have to get creative with the clip you want to include and how much frequency you'll need to reproduce it faithfully :)

### Sampling frequency vs. sample precision

Generally speaking, going from 16bit PCM to 8bit PCM isn't as detrimental as going from 44KHz to 8KHz, 2KHz or 1KHz. Funny how that works, eh? The more faithful the vibrations of the speaker diaphragm the better the reproduction.

For comparison, check out `madworld_8bit_highfreq.mp3` (which is 8bit PCM -- created with Audacity -- at 44KHz) with `madworld.wav` (which -- by default -- is 16bit PCM at 1.2KHz). While `madworld_8bit_highfreq.mp3` has a bit of static, the piano sounds WAY better than the same tune at 1.2KHz with higher precision samples.

## Text

This code will turn the given text into a fontmap sampler for otaviogood's fontmap:

	textToBuffer("Yo dude!", shader, 0)

See [here](https://www.shadertoy.com/view/llcXRl) on how to use the fontmap sampler to render SDF text.

First parameter is the text to be encoded.

Second parameter is previous shader code.

Third parameter is the text ID in case of multiple text samplers.

# Known issues

Apparently ANGLE's OpenGL backend does not like large uniform arrays. '\\\_o.O\_/`

# License

M.I.T. for the code.

Everything else belongs to their respective copyright holders.

Go nuts.