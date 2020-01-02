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
	shaderSource += "	float R = float (inp >> 24) / 255.0;\n"
	shaderSource += "	inp &= uint(0x00FFFFFF);\n"
	shaderSource += "	float G = float (inp >> 16) / 255.0;\n"
	shaderSource += "	inp &= uint(0x0000FFFF);\n"
	shaderSource += "	float B = float (inp >> 8) / 255.0;\n"
	shaderSource += "	inp &= uint(0x000000FF);\n"
	shaderSource += "	float A = float (inp) / 255.0;\n"
	shaderSource += "	return vec4 (R,G,B,A);\n}\n\n"
	return shaderSource

wroteUnpackUtilityForSound = False

"""   We're basically normalizing and packing 4 samples per uint. Samples are denormalized after unpacking.    """
"""   At the end of the day you have about 4096 uniforms to play with... so either reduce frequency or length. """
"""   Have fun! ;)                                                                                             """
def soundToShaderToy(inputMP3, outputWAV, freq, startSecond, lenSeconds, makeFunction = False):

	global wroteUnpackUtilityForSound, ffmpegPath

	# This normalizes the format we're dealing with to 16bit PCM... which will be converted shortly to 8bit PCM...
	subprocess.call([ffmpegPath,'-i',inputMP3,'-y','-ar',"%d" % freq,'-c:a','pcm_s16le','-ac','1','-ss','%.3fs' % startSecond,'-t','%.3fs' % lenSeconds,outputWAV])

	rate, data = wav.read(outputWAV)
	if len(data) % 4 != 0:
		for i in 4 - len(data) % 4:
			data.append (0.0)

	countItems = 0
	countT = 0
	ABCD = 'A'
	constructedUint32 = 0
	
	shaderSource = ""
	
	if wroteUnpackUtilityForSound == False:
		shaderSource = writeUnpackVec4()
		wroteUnpackUtilityForSound = True

	if makeFunction == True:
		timeToUintDict = dict()
	else:
		timeSeries = []

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
			if makeFunction == True:
				if constructedUint32 != 0:
					if constructedUint32 not in timeToUintDict:
						timeToUintDict[constructedUint32] = []
					timeToUintDict[constructedUint32].append (countT)
				countT += 1
				if countItems == len(data)-1:
					break
			else:
				timeSeries.append (constructedUint32)
				if countItems == len(data)-1:
					break
		countItems += 1

	if makeFunction == False:
		countItems = 0
		shaderSource += "uint shaderBuf[%d] = uint[](" % (len(data)/4)
		for constructedUint32 in timeSeries:
			if countItems != len (timeSeries) - 1:
				shaderSource += ("%uu," % constructedUint32)
			else:
				shaderSource += ("%uu);" % constructedUint32)
				break
			countItems += 1
	else:
		shaderSource += "uint getSample(int curT)\n{\n\tswitch (curT)\n\t{\n";
		for constructedUint32 in timeToUintDict:
			for countT in timeToUintDict[constructedUint32]:
				shaderSource += "	case %d:\n" % (countT);		
			shaderSource += "\t\treturn %uu;\n" % (constructedUint32);		
		shaderSource += "	}\n\treturn 0u;\n}\n";

	shaderSource += "\n\n";
	shaderSource += "vec2 mainSound( float time )\n{\n"
	shaderSource += "	int curTime = int(time*%.1f) %% %d;\n" % (float(freq),len(data))
	if makeFunction == True:
		shaderSource += "	uint packedSample = getSample(curTime/4);\n"
	else:
		shaderSource += "	uint packedSample = shaderBuf[curTime/4];\n"
	shaderSource += "	vec4 unpackedSound = unpackVec4 (packedSample);\n"
	shaderSource += "	if (curTime % 4 == 0) return vec2 (unpackedSound.x * 2.0 - 1.0);\n"
	shaderSource += "	else if (curTime % 4 == 1) return vec2 (unpackedSound.y * 2.0 - 1.0);\n"
	shaderSource += "	else if (curTime % 4 == 2) return vec2 (unpackedSound.z * 2.0 - 1.0);\n"
	shaderSource += "	else return vec2 (unpackedSound.a * 2.0 - 1.0);\n}\n"
	
	return shaderSource

wroteUnpackUtility = False
""" Modes can be "bw", "luma" and "r2g4b2" """
def imageToBuffer(imgName, imgScale, imgOffset, shaderSoFar = "", imgId = 0, mode = "luma"):

	global wroteUnpackUtility

	shaderSource = ""
	
	if wroteUnpackUtility == False:
		shaderSource = writeUnpackVec4()
		wroteUnpackUtility = True
	
	im = Image.open(imgName)
	pix = im.load()
	imgWidth = im.size[0]
	if mode == "bw":
		if imgWidth % 32 != 0:
			imgWidth += 32 - (imgWidth % 32)
	else:
		if imgWidth % 4 != 0:
			imgWidth += 4 - (imgWidth % 4)
	imgHeight = im.size[1]
	pixCount = imgWidth * imgHeight
	if mode == "bw":
		dataSize = (imgWidth/32) * imgHeight # We're packing 32 bit values into a single uint
	else:
		dataSize = (imgWidth/4) * imgHeight # We're packing 4 luminance/r2g4b2 values into a single uint
	shaderSource += "#define IMG%d_WIDTH %d\n#define IMG%d_HEIGHT %d\nuint shaderBuf%d[%d] = uint[](" % (imgId, imgWidth, imgId, im.size[1], imgId, dataSize)

	countItems = 0
	ABCD = 'A'
	countTo32 = 0
	constructedUint32 = 0

	for j in range (0, imgHeight):
		for i in range (0, imgWidth):
			fetchPix = []
			if i >= im.size[0]:
				fetchPix = [0, 0, 0] # Black out where we fall over actual image
			else:
				fetchPix = pix[i,j]
			luminance = 0.0
			r = (fetchPix[0]/255.0)
			g = (fetchPix[1]/255.0)
			b = (fetchPix[2]/255.0)
			if mode == "luma" or mode == "bw":
				luminance = r * 0.3 + g * 0.59 + b * 0.11
				pixVal = int (luminance * 255.0)
			else:
				pixVal = int (round(r * 3.0)) + (int(round(g * 15.0)) << 2) + (int(round(b * 3.0)) << 6);
				
			if mode == "bw":
				if countTo32 == 0:
					constructedUint32 = 0
				constructedUint32 += int(1 if luminance > 0.5 else 0) << countTo32
				countTo32 = (countTo32 + 1) % 32
			else:
				if ABCD == 'A':
					constructedUint32 = pixVal << 24
					ABCD = 'B'
				elif ABCD == 'B':
					constructedUint32 += pixVal << 16
					ABCD = 'C'
				elif ABCD == 'C':
					constructedUint32 += pixVal << 8
					ABCD = 'D'
				else:
					constructedUint32 += pixVal
					ABCD = 'A'

			if mode == "bw":
				if countTo32 == 0:
					if countItems != pixCount - 1:
						shaderSource += ("%uu," % constructedUint32)
					else:
						shaderSource += ("%uu);" % constructedUint32)
						break
			else:
				if ABCD == 'A':
					if countItems != pixCount - 1:
						shaderSource += ("%uu," % constructedUint32)
					else:
						shaderSource += ("%uu);" % constructedUint32)
						break
			countItems += 1

	if mode == "r2g4b2":
		shaderSource += "\n\nvec4 image%dBlit(in vec2 uv)\n{\n" % imgId
	else:
		shaderSource += "\n\nfloat image%dBlit(in vec2 uv)\n{\n" % imgId
	shaderSource += "	uv = uv * vec2 (%.3f, %.3f) - vec2 (%.3f, %.3f);\n\n" % (imgScale[0], imgScale[1], imgOffset[0], imgOffset[1])
	if mode == "r2g4b2":
		shaderSource += "	if (uv.x < 0.0 || uv.x > 1.0) return vec4(0.0);\n"
		shaderSource += "	if (uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);\n\n"
	else:
		shaderSource += "	if (uv.x < 0.0 || uv.x > 1.0) return 0.0;\n"
		shaderSource += "	if (uv.y < 0.0 || uv.y > 1.0) return 0.0;\n\n"
	shaderSource += "	int vi = int ((1.0 - uv.y) * float (IMG%d_HEIGHT));\n" % imgId
	shaderSource += "	int ui = int (uv.x * float (IMG%d_WIDTH));\n" % imgId
	shaderSource += "	uint fetchedSample = shaderBuf%d[vi*(IMG%d_WIDTH/%d) + (ui/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
	if mode == "bw":
		shaderSource += "	return ((fetchedSample & (1u << (ui % 32))) != 0u) ? 1.0 : 0.0;\n}\n"
	elif mode == "luma":
		shaderSource += "	vec4 unpackedColor = unpackVec4 (fetchedSample);\n"
		shaderSource += "	if (ui % 4 == 0) return unpackedColor.x;\n"
		shaderSource += "	else if (ui % 4 == 1) return unpackedColor.y;\n"
		shaderSource += "	else if (ui % 4 == 2) return unpackedColor.z;\n"
		shaderSource += "	else return unpackedColor.a;\n}\n"
	else:
		shaderSource += "	vec4 unpacked4Pixels = unpackVec4 (fetchedSample);\n"
		shaderSource += "	uint pixVal;\n"
		shaderSource += "	if (ui % 4 == 0) pixVal = uint (unpacked4Pixels.x * 255.0);\n"
		shaderSource += "	else if (ui % 4 == 1) pixVal = uint (unpacked4Pixels.y * 255.0);\n"
		shaderSource += "	else if (ui % 4 == 2) pixVal = uint (unpacked4Pixels.z * 255.0);\n"
		shaderSource += "	else pixVal = uint (unpacked4Pixels.a * 255.0);\n"
		shaderSource += "	return vec4 (float (pixVal & 3u) * 0.333333, float ((pixVal & 60u) >> 2) * 0.06666667, float ((pixVal & 192u) >> 6) * 0.333333, 1.0);\n}\n"

	return shaderSoFar+"\n"+shaderSource
	
def textToBuffer(textContent, shaderSoFar = "", textId = 0):
	shaderSource = shaderSoFar
	msg = textContent
	needsSpace = len(msg) % 4
	if needsSpace != 0:
		msg += " " * (4 - needsSpace)

	shaderSource += "\nconst uint textArr%d[%d] = uint[](" % (textId, len(msg) / 4)
	for i in range (0, len(msg) // 4):
		curWordIndex = i * 4
		curWordThing = 0
		curWordThing |= ord(msg[curWordIndex]) % 16 + ((15 - ord(msg[curWordIndex]) // 16) << 4)
		curWordIndex += 1
		curWordThing |= (ord(msg[curWordIndex]) % 16 + ((15 - ord(msg[curWordIndex]) // 16) << 4)) << 8
		curWordIndex += 1
		curWordThing |= (ord(msg[curWordIndex]) % 16 + ((15 - ord(msg[curWordIndex]) // 16) << 4)) << 16
		curWordIndex += 1
		curWordThing |= (ord(msg[curWordIndex]) % 16 + ((15 - ord(msg[curWordIndex]) // 16) << 4)) << 24
		if i == (len(msg) // 4) - 1:
			shaderSource += "%uu);\n" % (curWordThing)
		else:
			shaderSource += "%uu," % (curWordThing)
			
	shaderSource += "// Otavio Good's sampler for Shadertoy fonts\nvec4 SampleFontTex%d(vec2 uv)\n" % (textId)
	shaderSource += "{\n"
	shaderSource += "	vec2 fl = floor(uv + 0.5);\n"
	shaderSource += "	if (fl.y == 0.0) {\n"
	shaderSource += "		int charIndex = int(fl.x + 3.0);\n"
	shaderSource += "		int arrIndex = (charIndex / 4) %% %d;\n" % (len(msg) // 4)
	shaderSource += "		uint wordFetch = textArr%d[arrIndex];\n" % (textId)
	shaderSource += "		uint charFetch = (wordFetch >> ((charIndex % 4) * 8)) & 0x000000FFu;\n"
	shaderSource += "		float charX = float (int (charFetch & 0x0000000Fu)     );\n"
	shaderSource += "		float charY = float (int (charFetch & 0x000000F0u) >> 4);\n"
	shaderSource += "		fl = vec2(charX, charY);\n"
	shaderSource += "	}\n"
	shaderSource += "	uv = fl + fract(uv+0.5)-0.5;\n"
	shaderSource += "	return texture(iChannel2, (uv+0.5)*(1.0/16.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001) - 0.5;\n"
	shaderSource += "}\n"

	return shaderSource


# Convert all the images...

#shader = imageToBuffer('angels1.png', [5.100, 8.500], [2.050, 4.60])
#shader = imageToBuffer('angels2.png', [4.5, 15.500], [1.71, 4.300], shader, 1)
#shader = imageToBuffer('angelshorse.png', [5.0, 5.0], [1.957, 1.750], shader, 2)
#shader = imageToBuffer('angels_scroll.png', [2.0, 4.0], [0.5, 2.0], shader, 3)
shader = imageToBuffer('graffiti.png', [1.0, 1.0], [0.0, 0.0], "", 0, "r2g4b2")
shader = imageToBuffer('graffiti_lowres.png', [1.0, 1.0], [0.0, 0.0], shader, 1, "r2g4b2")
shader = textToBuffer("Yo dude!", shader)

file = open('shaderimage.txt', 'w')
file.write(shader)
file.close()

# Convert the sound...

# shader = soundToShaderToy('madworld.mp3', 'madworld.wav', 1220, 1.08, 12.0) # This is for the other shadertoy... ;)
# shader = soundToShaderToy('angels.mp3', 'angels.wav', 2000, 0.8, 7.6)
shader = soundToShaderToy('sotb1.mp3', 'sotb1.wav', 3200, 36.0, 5.1)

file = open("shadersound.txt", "w")
file.write(shader)
file.close()