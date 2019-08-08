"""
   Covered under the MIT license:

   Copyright (c) 2019 TooMuchVoltage Software Inc.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
"""

"""
   You need PIL and scipy: pip install scipy Pillow
   
   ... and you need ffmpeg too, so replace the following path ...
"""

# Replace the following with your ffmpeg path
ffmpegPath = "ffmpeg\\bin\\ffmpeg.exe"

import math
import subprocess
from scipy.io import wavfile as wav
from PIL import Image

"""   Unpack util   """
def writeUnpackVec4():
	shaderSource =  "// Ideally we'd use unpackUnorm4x8(), but this is not available\n"
	shaderSource += "vec4 unpackVec4(uint inp)\n{\n"
	shaderSource += "	float R = float (inp >> 24) / 256.0;\n"
	shaderSource += "	inp &= uint(0x00FFFFFF);\n"
	shaderSource += "	float G = float (inp >> 16) / 256.0;\n"
	shaderSource += "	inp &= uint(0x0000FFFF);\n"
	shaderSource += "	float B = float (inp >> 8) / 256.0;\n"
	shaderSource += "	inp &= uint(0x000000FF);\n"
	shaderSource += "	float A = float (inp) / 256.0;\n"
	shaderSource += "	return vec4 (R,G,B,A);\n}\n\n"
	return shaderSource

wroteUnpackUtilityForSound = False

"""   We're basically normalizing and packing 4 samples per uint. Samples are denormalized after unpacking.    """
"""   At the end of the day you have about 4096 uniforms to play with... so either reduce frequency or length. """
"""   Have fun! ;)                                                                                             """
def soundToShaderToy(inputMP3, outputWAV, freq, startSecond, lenSeconds):

	global wroteUnpackUtilityForSound, ffmpegPath

	# This normalizes the format we're dealing with to 16bit PCM... which will be converted shortly to 8bit PCM...
	subprocess.call([ffmpegPath,'-i',inputMP3,'-y','-ar',"%d" % freq,'-c:a','pcm_s16le','-ac','1','-ss','%.3fs' % startSecond,'-t','%.3fs' % lenSeconds,outputWAV])

	rate, data = wav.read(outputWAV)
	if len(data) % 4 != 0:
		for i in 4 - len(data) % 4:
			data.append (0.0)

	countItems = 0
	ABCD = 'A'
	constructedUint32 = 0
	
	shaderSource = ""
	
	if wroteUnpackUtilityForSound == False:
		shaderSource = writeUnpackVec4()
		wroteUnpackUtilityForSound = True

	shaderSource += "uint shaderBuf[%d] = uint[](" % (len(data)/4)

	for i in data:
		sampleNorm = ((i/32768.0) + 1.0) * 0.5
		if ABCD == 'A':
			constructedUint32 = int (sampleNorm * 255.0) << 24
			ABCD = 'B'
		elif ABCD == 'B':
			constructedUint32 += int (sampleNorm * 255.0) << 16
			ABCD = 'C'
		elif ABCD == 'C':
			constructedUint32 += int (sampleNorm * 255.0) << 8
			ABCD = 'D'
		else:
			constructedUint32 += int (sampleNorm * 255.0)
			ABCD = 'A'

		if ABCD == 'A':
			if countItems != len(data)-1:
				shaderSource += ("%uu," % constructedUint32)
			else:
				shaderSource += ("%uu);" % constructedUint32)
				break
		countItems += 1

	shaderSource += "\n\n";
	shaderSource += "vec2 mainSound( float time )\n{\n"
	shaderSource += "	int curTime = int(time*%.1f) %% %d;\n" % (float(freq),len(data))
	shaderSource += "	uint packedSample = shaderBuf[curTime/4];\n"
	shaderSource += "	vec4 unpackedSound = unpackVec4 (packedSample);\n"
	shaderSource += "	if (curTime % 4 == 0) return vec2 (unpackedSound.x * 2.0 - 1.0);\n"
	shaderSource += "	else if (curTime % 4 == 1) return vec2 (unpackedSound.y * 2.0 - 1.0);\n"
	shaderSource += "	else if (curTime % 4 == 2) return vec2 (unpackedSound.z * 2.0 - 1.0);\n"
	shaderSource += "	else return vec2 (unpackedSound.a * 2.0 - 1.0);\n}\n"
	
	return shaderSource

wroteUnpackUtility = False
def imageToLumBuffer(imgName, imgScale, imgOffset, shaderSoFar = "", imgId = 0):

	global wroteUnpackUtility

	shaderSource = ""
	
	if wroteUnpackUtility == False:
		shaderSource = writeUnpackVec4()
		wroteUnpackUtility = True
	
	im = Image.open(imgName)
	pix = im.load()
	imgWidth = im.size[0]
	if imgWidth % 4 != 0:
		imgWidth += 4 - (imgWidth % 4)
	imgHeight = im.size[1]
	pixCount = imgWidth * imgHeight
	dataSize = (imgWidth/4) * imgHeight # We're packing 4 luminance values into a single uint
	shaderSource += "#define IMG%d_WIDTH %d\n#define IMG%d_HEIGHT %d\nuint shaderBuf%d[%d] = uint[](" % (imgId, imgWidth, imgId, im.size[1], imgId, dataSize)

	countItems = 0
	ABCD = 'A'
	constructedUint32 = 0

	for j in range (0, imgHeight):
		for i in range (0, imgWidth):
			fetchPix = []
			if i >= im.size[0]:
				fetchPix = [0, 0, 0] # Black out where we fall over actual image
			else:
				fetchPix = pix[i,j]
			r = (fetchPix[0]/256.0)
			g = (fetchPix[1]/256.0)
			b = (fetchPix[2]/256.0)
			luminance = (r * 0.3) + (g * 0.59) + (b * 0.11)

			if ABCD == 'A':
				constructedUint32 = int (luminance * 255.0) << 24
				ABCD = 'B'
			elif ABCD == 'B':
				constructedUint32 += int (luminance * 255.0) << 16
				ABCD = 'C'
			elif ABCD == 'C':
				constructedUint32 += int (luminance * 255.0) << 8
				ABCD = 'D'
			else:
				constructedUint32 += int (luminance * 255.0)
				ABCD = 'A'

			if ABCD == 'A':
				if countItems != pixCount - 1:
					shaderSource += ("%uu," % constructedUint32)
				else:
					shaderSource += ("%uu);" % constructedUint32)
					break
			countItems += 1

	shaderSource += "\n\nfloat image%dBlit(in vec2 uv)\n{\n" % imgId
	shaderSource += "	uv = uv * vec2 (%.3f, %.3f) - vec2 (%.3f, %.3f);\n\n" % (imgScale[0], imgScale[1], imgOffset[0], imgOffset[1])
	shaderSource += "	if (uv.x < 0.0 || uv.x > 1.0) return 0.0;\n"
	shaderSource += "	if (uv.y < 0.0 || uv.y > 1.0) return 0.0;\n\n"
	shaderSource += "	int vi = int ((1.0 - uv.y) * float (IMG%d_HEIGHT));\n" % imgId
	shaderSource += "	int ui = int (uv.x * float (IMG%d_WIDTH));\n" % imgId
	shaderSource += "	uint fetchedSample = shaderBuf%d[vi*(IMG%d_WIDTH/4) + (ui/4)];\n" % (imgId, imgId)
	shaderSource += "	vec4 unpackedColor = unpackVec4 (fetchedSample);\n"
	shaderSource += "	if (ui % 4 == 0) return unpackedColor.x;\n"
	shaderSource += "	else if (ui % 4 == 1) return unpackedColor.y;\n"
	shaderSource += "	else if (ui % 4 == 2) return unpackedColor.z;\n"
	shaderSource += "	else return unpackedColor.a;\n}\n"

	return shaderSoFar+"\n"+shaderSource

# Convert all the images...

shader = imageToLumBuffer('angels1.png', [5.100, 8.500], [2.050, 4.60])
shader = imageToLumBuffer('angels2.png', [4.5, 15.500], [1.71, 4.300], shader, 1)
shader = imageToLumBuffer('angelshorse.png', [5.0, 5.0], [1.957, 1.750], shader, 2)
shader = imageToLumBuffer('angels_scroll.png', [2.0, 4.0], [0.5, 2.0], shader, 3)

file = open('shaderimage.txt', 'w')
file.write(shader)
file.close()

# Convert the sound...

# shader = soundToShaderToy('madworld.mp3', 'madworld.wav', 1220, 1.08, 12.0) # This is for the other shadertoy... ;)
shader = soundToShaderToy('angels.mp3', 'angels.wav', 2000, 0.8, 7.6)

file = open("shadersound.txt", "w")
file.write(shader)
file.close()