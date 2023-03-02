"""
   Covered under the MIT license:

   Copyright (c) 2023 TooMuchVoltage Software Inc.

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
   You need PIL, scipy and plyfile: pip install scipy Pillow plyfile
   
   ... and you need ffmpeg too, so replace the following path ...
"""

# Replace the following with your ffmpeg path
ffmpegPath = "ffmpeg\\bin\\ffmpeg.exe"

import math
import subprocess
from scipy.io import wavfile as wav
from PIL import Image
from plyfile import PlyData, PlyElement
import numpy

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
	shaderSource += "	return vec2 (unpackVec4 (packedSample)[curTime & 3] * 2.0 - 1.0);\n}\n"
	
	return shaderSource

wroteUnpackUtility = False
""" Modes can be "bw", "luma" and "r2g4b2" """
""" Filter can be "nearest" and "bilinear" """
def imageToBuffer(imgName, imgScale, imgOffset, shaderSoFar = "", imgId = 0, mode = "luma", filter = "nearest"):

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
	if filter == "nearest":
		shaderSource += "	int vi = int ((1.0 - uv.y) * float (IMG%d_HEIGHT));\n" % imgId
		shaderSource += "	int ui = int (uv.x * float (IMG%d_WIDTH));\n" % imgId
		shaderSource += "	uint fetchedSample = shaderBuf%d[vi*(IMG%d_WIDTH/%d) + (ui/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
		if mode == "bw":
			shaderSource += "	return ((fetchedSample & (1u << (ui % 32))) != 0u) ? 1.0 : 0.0;\n}\n"
		elif mode == "luma":
			shaderSource += "	return unpackVec4 (fetchedSample)[ui & 3];\n}\n"
		else:
			shaderSource += "	uint pixVal = uint(unpackVec4 (fetchedSample)[ui & 3] * 255.0);\n"
			shaderSource += "	return vec4 (float (pixVal & 3u) * 0.333333, float ((pixVal & 60u) >> 2) * 0.06666667, float ((pixVal & 192u) >> 6) * 0.333333, 1.0);\n}\n"
	else:
		shaderSource += "	float vunorm = (1.0 - uv.y) * float (IMG%d_HEIGHT);\n" % imgId
		shaderSource += "	float uunorm = uv.x * float (IMG%d_WIDTH);\n" % imgId
		shaderSource += "	int vi = int (vunorm);\n"
		shaderSource += "	int ui = int (uunorm);\n"
		shaderSource += "	float vfract = fract (vunorm);\n"
		shaderSource += "	float ufract = fract (uunorm);\n"
		shaderSource += "	int ui_1 = min (ui + 1,IMG%d_WIDTH - 1);\n" % imgId
		shaderSource += "	int vi_1 = min (vi + 1,IMG%d_HEIGHT - 1);\n" % imgId
		shaderSource += "	uint fetchedSample00 = shaderBuf%d[vi*(IMG%d_WIDTH/%d) + (ui/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
		shaderSource += "	uint pixVal00 = uint(unpackVec4 (fetchedSample00)[ui & 3] * 255.0);\n"
		shaderSource += "	uint fetchedSample01 = shaderBuf%d[vi*(IMG%d_WIDTH/%d) + (ui_1/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
		shaderSource += "	uint pixVal01 = uint(unpackVec4 (fetchedSample01)[ui_1 & 3] * 255.0);\n"
		shaderSource += "	uint fetchedSample10 = shaderBuf%d[vi_1*(IMG%d_WIDTH/%d) + (ui/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
		shaderSource += "	uint pixVal10 = uint(unpackVec4 (fetchedSample10)[ui & 3] * 255.0);\n"
		shaderSource += "	uint fetchedSample11 = shaderBuf%d[vi_1*(IMG%d_WIDTH/%d) + (ui_1/%d)];\n" % (imgId, imgId, (32 if mode == "bw" else 4), (32 if mode == "bw" else 4))
		shaderSource += "	uint pixVal11 = uint(unpackVec4 (fetchedSample11)[ui_1 & 3] * 255.0);\n"
		if mode == "bw":
			shaderSource += "	float col00 = ((fetchedSample00 & (1u << (ui % 32))) != 0u) ? 1.0 : 0.0;\n"
			shaderSource += "	float col01 = ((fetchedSample01 & (1u << (ui_1 % 32))) != 0u) ? 1.0 : 0.0;\n"
			shaderSource += "	float col10 = ((fetchedSample10 & (1u << (ui % 32))) != 0u) ? 1.0 : 0.0;\n"
			shaderSource += "	float col11 = ((fetchedSample11 & (1u << (ui_1 % 32))) != 0u) ? 1.0 : 0.0;\n"
		elif mode == "luma":
			shaderSource += "	float col00 = unpackVec4 (fetchedSample00)[ui & 3];\n"
			shaderSource += "	float col01 = unpackVec4 (fetchedSample01)[ui_1 & 3];\n"
			shaderSource += "	float col10 = unpackVec4 (fetchedSample10)[ui & 3];\n"
			shaderSource += "	float col11 = unpackVec4 (fetchedSample11)[ui_1 & 3];\n"
		else:
			shaderSource += "	vec4 col00 = vec4 (float (pixVal00 & 3u) * 0.333333, float ((pixVal00 & 60u) >> 2) * 0.06666667, float ((pixVal00 & 192u) >> 6) * 0.333333, 1.0);\n"
			shaderSource += "	vec4 col01 = vec4 (float (pixVal01 & 3u) * 0.333333, float ((pixVal01 & 60u) >> 2) * 0.06666667, float ((pixVal01 & 192u) >> 6) * 0.333333, 1.0);\n"
			shaderSource += "	vec4 col10 = vec4 (float (pixVal10 & 3u) * 0.333333, float ((pixVal10 & 60u) >> 2) * 0.06666667, float ((pixVal10 & 192u) >> 6) * 0.333333, 1.0);\n"
			shaderSource += "	vec4 col11 = vec4 (float (pixVal11 & 3u) * 0.333333, float ((pixVal11 & 60u) >> 2) * 0.06666667, float ((pixVal11 & 192u) >> 6) * 0.333333, 1.0);\n"
		shaderSource += "	return mix (mix(col00, col01, ufract), mix(col10, col11, ufract), vfract);\n}\n"

	return shaderSoFar+"\n"+shaderSource

class bitArray:
	def __init__(self):
		self.outpArray = bytearray()
		self.outpByte = 0
		self.writtenBits = 0

	def append(self, bits, bitRange):
		if bitRange < (8 - self.writtenBits):
			bits <<= ((8 - self.writtenBits) - bitRange)
			self.outpByte |= bits
			self.writtenBits += bitRange
		else:
			topBitShift = (bitRange - (8 - self.writtenBits))
			bitsTop = (bits >> topBitShift)
			self.outpByte |= bitsTop
			self.outpArray += self.outpByte.to_bytes(1)
			self.outpByte = 0
			self.writtenBits = 0
			if topBitShift == 0:
				return
			bitsBottom = ((bits & (0xFF >> (8 - topBitShift))) << (8 - topBitShift))
			self.outpByte = bitsBottom
			self.writtenBits = topBitShift

	def report(self):
		if self.writtenBits == 0:
			return self.outpArray
		else:
			return self.outpArray + self.outpByte.to_bytes(1)

	def reportHexArray(self, itemType, itemNum = 0):
		outputBytes = self.report()
		if len(outputBytes) % 4 != 0:
			for i in range (0, 4 - (len(outputBytes) % 4)):
				outputBytes += (0).to_bytes(1)
		numInts = int (len(outputBytes) / 4)
		shaderUniforms = "\nuint %s%d[%d] = uint[](" % (itemType, itemNum, numInts)
		for i in range (0, numInts):
			curInt = (outputBytes[i * 4] << 24) + (outputBytes[i * 4 + 1] << 16) + (outputBytes[i * 4 + 2] << 8) + outputBytes[i * 4 + 3]
			if i == numInts - 1:
				shaderUniforms += ("%uu);" % (curInt))
			else:
				shaderUniforms += ("%uu," % (curInt))
		return shaderUniforms

"""   Unpack util   """
def writeSVOUnpackUtils(shaderSource):
	unpackUtils = """
bool rayBoxIntersectTime (vec3 l1,vec3 invm,vec3 bmin,vec3 bmax, out float tMin, out float tMax)
{
	vec3 bmin_l1 = (bmin - l1)*invm;
	vec3 bmax_l1 = (bmax - l1)*invm;
	vec3 minVec = min (bmin_l1, bmax_l1);
	vec3 maxVec = max (bmin_l1, bmax_l1);

	float tmin = max(max(minVec.x, minVec.y), minVec.z);
	float tmax = min(min(maxVec.x, maxVec.y), maxVec.z);

	bool retVal = ((tmax >= tmin) && (tmin < 1.0) && (tmax > 0.0));
	tMin = tmin;
	tMax = tmax;
	return retVal;
}

uint countSetBits(uint n)
{
	uint count = 0u;
	while (n != 0u) {
		count += (n & 1u);
		n >>= 1u;
	}
	return count;
}

uint countSetBitsBefore(uint n, uint comp)
{
	uint beforeMask = comp ^ (comp - 1u); // See: https://realtimecollisiondetection.net/blog/?p=78
	n &= (~beforeMask);
	uint count = 0u;
	while (n != 0u) {
		count += (n & 1u);
		n >>= 1u;
	}
	return count;
}
"""
	return shaderSource + unpackUtils

def tessellate(v1, v2, v3, rangeMeasure, points):
	e1 = [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
	e2 = [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]]
	e1sq = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]
	e2sq = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]
	if e1sq > rangeMeasure or e2sq > rangeMeasure:
		mid1 = [(v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5, (v1[2] + v2[2]) * 0.5]
		mid2 = [(v2[0] + v3[0]) * 0.5, (v2[1] + v3[1]) * 0.5, (v2[2] + v3[2]) * 0.5]
		mid3 = [(v1[0] + v3[0]) * 0.5, (v1[1] + v3[1]) * 0.5, (v1[2] + v3[2]) * 0.5]
		tessellate (v1, mid1, mid3, rangeMeasure, points)
		tessellate (mid1, v2, mid2, rangeMeasure, points)
		tessellate (mid1, mid2, mid3, rangeMeasure, points)
		tessellate (mid3, mid2, v3, rangeMeasure, points)
	else:
		points.append (v1)
		points.append (v2)
		points.append (v3)

wroteSVOUnpackUtility = False
""" BitStream SVO Compression """
def SVOToBitstream(plyFileName, shaderSoFar = "", svoNum = 0):
	global wroteSVOUnpackUtility
	shaderSource = ""
	shaderSource += shaderSoFar
	if wroteSVOUnpackUtility == False:
		shaderSource = writeSVOUnpackUtils(shaderSource)
		wroteSVOUnpackUtility = True

	voxXRange = 10
	voxYRange = 10
	voxZRange = 10

	pointDB = [[[[]for k in range(voxXRange * 4)] for j in range(voxYRange * 4)] for i in range(voxZRange * 4)]
	pointDBMin = []
	pointDBMax = []
	points = []
	firstTime = True
	plydata = PlyData.read(plyFileName)

	for i in range (0, len(plydata['vertex'])):
		curPoint = [plydata['vertex'][i]['x'], plydata['vertex'][i]['y'], plydata['vertex'][i]['z']]
		if firstTime == True:
			pointDBMin = [curPoint[0], curPoint[1], curPoint[2]]
			pointDBMax = [curPoint[0], curPoint[1], curPoint[2]]
			firstTime = False
		else:
			pointDBMin = [min (curPoint[0], pointDBMin[0]), min (curPoint[1], pointDBMin[1]), min (curPoint[2], pointDBMin[2])]
			pointDBMax = [max (curPoint[0], pointDBMax[0]), max (curPoint[1], pointDBMax[1]), max (curPoint[2], pointDBMax[2])]
			
	pointDBRange = [pointDBMax[0] - pointDBMin[0], pointDBMax[1] - pointDBMin[1], pointDBMax[2] - pointDBMin[2]]

	for i in plydata['face']['vertex_indices']:
		v1index = i[0]
		v2index = i[1]
		v3index = i[2]
		v1 = [plydata['vertex'][v1index]['x'], plydata['vertex'][v1index]['y'], plydata['vertex'][v1index]['z']]
		v2 = [plydata['vertex'][v2index]['x'], plydata['vertex'][v2index]['y'], plydata['vertex'][v2index]['z']]
		v3 = [plydata['vertex'][v3index]['x'], plydata['vertex'][v3index]['y'], plydata['vertex'][v3index]['z']]
		tessellate (v1, v2, v3, (pointDBRange[0] * pointDBRange[0] + pointDBRange[1] * pointDBRange[1] + pointDBRange[2] * pointDBRange[2]) * 0.00021, points)

	for curPoint in points:
		pointDBXCoord = int (((curPoint[0] - pointDBMin[0]) / pointDBRange[0]) * 0.999999 * voxXRange * 4)
		pointDBYCoord = int (((curPoint[1] - pointDBMin[1]) / pointDBRange[1]) * 0.999999 * voxYRange * 4)
		pointDBZCoord = int (((curPoint[2] - pointDBMin[2]) / pointDBRange[2]) * 0.999999 * voxZRange * 4)
		pointDB[pointDBXCoord][pointDBYCoord][pointDBZCoord] = True

	outStream = bitArray()

	for i in range (0, voxXRange):
		for j in range (0, voxYRange):
			for k in range (0, voxZRange):
				topBrick = False
				allMidBricks = []
				for ii in range (0, 2):
					for jj in range (0, 2):
						for kk in range (0, 2):
							midBrick = []
							for iii in range (0, 2):
								for jjj in range (0, 2):
									for kkk in range (0, 2):
										voxelX = i * 4 + ii * 2 + iii
										voxelY = j * 4 + jj * 2 + jjj
										voxelZ = k * 4 + kk * 2 + kkk
										if pointDB[voxelX][voxelY][voxelZ] == True:
											midBrick.append(True)
											topBrick = True
										else:
											midBrick.append(False)
							allMidBricks.append (midBrick)
				if topBrick == True:
					outStream.append (1, 1)
					for aMidBrick in allMidBricks:
						outStream.append (1 if True in aMidBrick else 0, 1)
					for aMidBrick in allMidBricks:
						if True in aMidBrick:
							for aVoxel in aMidBrick:
								outStream.append (1 if aVoxel == True else 0, 1)
				else:
					outStream.append (0, 1)

	shaderSource += "\nconst vec3 grid%dMin = vec3 (%0.2f, %0.2f, %0.2f);" % (svoNum, -voxXRange * 0.5, -voxYRange * 0.5, -voxZRange * 0.5)
	shaderSource += "\nconst vec3 grid%dMax = vec3 (%0.2f, %0.2f, %0.2f);" % (svoNum, voxXRange * 0.5, voxYRange * 0.5, voxZRange * 0.5)
	shaderSource += "\nconst vec3 grid%dRange = grid%dMax - grid%dMin;" % (svoNum, svoNum, svoNum)
	shaderSource += outStream.reportHexArray("svoObject", svoNum)
	shaderSource += "\nuint readBitsSVO%d (uint bitLoc, uint numBits) {" % (svoNum)
	shaderSource += "\n    uint wordLoc = bitLoc / 32u;"
	shaderSource += "\n    uint leftToRead = (32u - (bitLoc % 32u));"
	shaderSource += "\n    if (numBits <= leftToRead) {"
	shaderSource += "\n        uint shiftToMask = leftToRead - numBits;"
	shaderSource += "\n        uint masker = 0xFFFFFFFFu;"
	shaderSource += "\n        masker >>= uint(32u - numBits);"
	shaderSource += "\n        masker <<= shiftToMask;"
	shaderSource += "\n        uint value = (svoObject%d[wordLoc] & masker);" % (svoNum)
	shaderSource += "\n        value >>= shiftToMask;"
	shaderSource += "\n        return value;"
	shaderSource += "\n    } else {"
	shaderSource += "\n        uint bottomBits = numBits - leftToRead;"
	shaderSource += "\n        uint masker = 0xFFFFFFFFu;"
	shaderSource += "\n        masker >>= uint(32u - leftToRead);"
	shaderSource += "\n        uint topNum = (svoObject%d[wordLoc] & masker);" % (svoNum)
	shaderSource += "\n        uint bottomMasker = 0xFFFFFFFFu;"
	shaderSource += "\n        uint bottomShifter = uint(32u - bottomBits);"
	shaderSource += "\n        bottomMasker <<= bottomShifter;"
	shaderSource += "\n        uint value = (svoObject%d[wordLoc + 1u] & bottomMasker);" % (svoNum)
	shaderSource += "\n        uint bottomNum = (value >> bottomShifter);"
	shaderSource += "\n        return ((topNum << bottomBits) | bottomNum);"
	shaderSource += "\n    }"
	shaderSource += "\n}"
	shaderSource += "\n\nbool readLeafSVO%d (vec3 samplePos, vec3 sampleDir, out vec3 skipPos) {" % (svoNum)
	shaderSource += "\n    skipPos = vec3 (10000.0);"
	shaderSource += "\n    if ( any(lessThan(samplePos, grid%dMin)) || any(greaterThan(samplePos, grid%dMax)) ) return false;" % (svoNum, svoNum)
	shaderSource += "\n    uvec3 topBrickPos = uvec3 (samplePos - grid%dMin);" % (svoNum)
	shaderSource += "\n    uint topBrickId = topBrickPos.z + topBrickPos.y * uint(grid0Range.x) + topBrickPos.x * uint(grid0Range.y) * uint(grid0Range.z);"
	shaderSource += "\n    uint streamReadPos = 0u;"
	shaderSource += "\n    for (int i = 0; i < int(topBrickId); i++) {"
	shaderSource += "\n        uint isOcc = readBitsSVO%d (streamReadPos, 1u);" % (svoNum)
	shaderSource += "\n        streamReadPos += 1u;"
	shaderSource += "\n        if (isOcc == 1u) {"
	shaderSource += "\n            uint countMidBricks = countSetBits (readBitsSVO%d (streamReadPos, 8u));" % (svoNum)
	shaderSource += "\n            streamReadPos += (8u + countMidBricks * 8u);"
	shaderSource += "\n        }"
	shaderSource += "\n    }"
	shaderSource += "\n    uint topBrick = readBitsSVO%d (streamReadPos, 1u);" % (svoNum)
	shaderSource += "\n    if (topBrick == 0u) {"
	shaderSource += "\n        vec3 topBrickMin = grid%dMin + vec3 (topBrickPos);" % (svoNum)
	shaderSource += "\n        vec3 topBrickMax = topBrickMin + vec3 (1.0);"
	shaderSource += "\n        vec3 p1 = samplePos;"
	shaderSource += "\n        vec3 p2 = p1 + sampleDir * 2.0;"
	shaderSource += "\n        vec3 m = p2 - p1;"
	shaderSource += "\n        float tMin, tMax;"
	shaderSource += "\n        rayBoxIntersectTime (p1, vec3(1.0)/m, topBrickMin, topBrickMax, tMin, tMax);"
	shaderSource += "\n        skipPos = p1 + m * (tMax + 0.01);"
	shaderSource += "\n        return false;"
	shaderSource += "\n    }"
	shaderSource += "\n    streamReadPos += 1u;"
	shaderSource += "\n    uint midBricks = readBitsSVO%d (streamReadPos, 8u);" % (svoNum)
	shaderSource += "\n    streamReadPos += 8u;"
	shaderSource += "\n    vec3 topBrickMinCorner = grid%dMin + vec3 (topBrickPos);" % (svoNum)
	shaderSource += "\n    vec3 sampleRelativeToTopBrick = fract (samplePos);"
	shaderSource += "\n    uint checkMidBrickBit = 0x80u;"
	shaderSource += "\n    vec3 sampleRelativeToMidBrick = sampleRelativeToTopBrick;"
	shaderSource += "\n    vec3 midBrickPos = vec3 (0.0);"
	shaderSource += "\n    if ( sampleRelativeToTopBrick.x > 0.5 ) {"
	shaderSource += "\n        sampleRelativeToMidBrick.x -= 0.5;"
	shaderSource += "\n        midBrickPos.x = 0.5;"
	shaderSource += "\n        checkMidBrickBit >>= 4u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( sampleRelativeToTopBrick.y > 0.5 ) {"
	shaderSource += "\n        sampleRelativeToMidBrick.y -= 0.5;"
	shaderSource += "\n        midBrickPos.y = 0.5;"
	shaderSource += "\n        checkMidBrickBit >>= 2u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( sampleRelativeToTopBrick.z > 0.5 ) {"
	shaderSource += "\n        sampleRelativeToMidBrick.z -= 0.5;"
	shaderSource += "\n        midBrickPos.z = 0.5;"
	shaderSource += "\n        checkMidBrickBit >>= 1u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( (midBricks & checkMidBrickBit) == 0u ) {"
	shaderSource += "\n        vec3 midBrickMin = grid%dMin + vec3 (topBrickPos) + midBrickPos;" % (svoNum)
	shaderSource += "\n        vec3 midBrickMax = midBrickMin + vec3 (0.5);"
	shaderSource += "\n        vec3 p1 = samplePos;"
	shaderSource += "\n        vec3 p2 = p1 + sampleDir * 2.0;"
	shaderSource += "\n        vec3 m = p2 - p1;"
	shaderSource += "\n        float tMin, tMax;"
	shaderSource += "\n        rayBoxIntersectTime (p1, vec3(1.0)/m, midBrickMin, midBrickMax, tMin, tMax);"
	shaderSource += "\n        skipPos = p1 + m * (tMax + 0.01);"
	shaderSource += "\n        return false;"
	shaderSource += "\n    }"
	shaderSource += "\n    uint skipMidBricks = countSetBitsBefore (midBricks, checkMidBrickBit);"
	shaderSource += "\n    streamReadPos += (8u * skipMidBricks);"
	shaderSource += "\n    uint finalMidBrick = readBitsSVO%d (streamReadPos, 8u);" % (svoNum)
	shaderSource += "\n    uint checkVoxelBrickBit = 0x80u;"
	shaderSource += "\n    if ( sampleRelativeToMidBrick.x > 0.25 ) {"
	shaderSource += "\n        checkVoxelBrickBit >>= 4u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( sampleRelativeToMidBrick.y > 0.25 ) {"
	shaderSource += "\n        checkVoxelBrickBit >>= 2u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( sampleRelativeToMidBrick.z > 0.25 ) {"
	shaderSource += "\n        checkVoxelBrickBit >>= 1u;"
	shaderSource += "\n    }"
	shaderSource += "\n    if ( (checkVoxelBrickBit & finalMidBrick) != 0u ) return true;"
	shaderSource += "\n    skipPos = samplePos + sampleDir * 0.25;"
	shaderSource += "\n    return false;"
	shaderSource += "\n}"
	shaderSource += "\n\nbool traceRaySVO%d(vec3 p1, vec3 p2, out vec3 hitPos) {" % (svoNum)
	shaderSource += "\n    vec3 m = p2 - p1;"
	shaderSource += "\n    float hitMin, hitMax;"
	shaderSource += "\n    if ( !rayBoxIntersectTime (p1, vec3(1.0)/m, grid%dMin, grid%dMax, hitMin, hitMax) ) {" % (svoNum, svoNum)
	shaderSource += "\n        hitPos = vec3 (-1.0);"
	shaderSource += "\n        return false;"
	shaderSource += "\n    }"
	shaderSource += "\n    "
	shaderSource += "\n    hitMin += 0.00001;"
	shaderSource += "\n    hitMax -= 0.00001;"
	shaderSource += "\n    vec3 curPos = p1 + hitMin * m;"
	shaderSource += "\n    vec3 curDir = normalize (m);"
	shaderSource += "\n    vec3 skipPos = vec3 (0.0);"
	shaderSource += "\n    for (int i = 0; i != 100; i++) {"
	shaderSource += "\n        if (readLeafSVO%d (curPos, curDir, skipPos)) {" % (svoNum)
	shaderSource += "\n            hitPos = curPos;"
	shaderSource += "\n            return true;"
	shaderSource += "\n        }"
	shaderSource += "\n        if ( skipPos == vec3(10000.0) ) break;"
	shaderSource += "\n        curPos = skipPos;"
	shaderSource += "\n    }"
	shaderSource += "\n    return false;"
	shaderSource += "\n}"
	shaderSource += "\n\nvoid decodeSVO%d( inout vec4 fragColor, vec2 fragCoord ) {" % (svoNum)
	shaderSource += "\n    uvec2 pixCoord = uvec2 (fragCoord - vec2 (0.5));"
	shaderSource += "\n    if ( iFrame %% 60 == 0 && all (lessThan (pixCoord,uvec2 (uint(grid%dRange.x), uint(grid%dRange.y * grid%dRange.z)/2u))) ) {" % (svoNum, svoNum, svoNum)
	shaderSource += "\n        uint writeR = 0u, writeG = 0u, writeB = 0u, writeA = 0u;"
	shaderSource += "\n        vec3 baseCoord;"
	shaderSource += "\n        baseCoord.x = float (pixCoord.x %% uint(grid%dRange.x)) - float(uint(grid%dRange.x)/2u);" % (svoNum, svoNum)
	shaderSource += "\n        baseCoord.y = float (pixCoord.y %% uint(grid%dRange.y)) - float(uint(grid%dRange.y)/2u);" % (svoNum, svoNum)
	shaderSource += "\n        baseCoord.z = float (pixCoord.y / uint(grid%dRange.y)) * 2.0 - float(uint(grid%dRange.z)/2u);" % (svoNum, svoNum)
	shaderSource += "\n        for (uint zone = 0u; zone != 2u; zone++) {"
	shaderSource += "\n            for (uint k = 0u;k != 4u; k++) {"
	shaderSource += "\n                for (uint j = 0u;j != 4u; j++) {"
	shaderSource += "\n                    for (uint i = 0u;i != 4u; i++) {"
	shaderSource += "\n                        bool occupancy = readLeafSVO%d (baseCoord + vec3 (i, j, k) * 0.25 + vec3 (0.001, -0.001, float(zone) + 0.001));" % (svoNum)
	shaderSource += "\n                        if (!occupancy) continue;"
	shaderSource += "\n                        uint write = (1u << i);"
	shaderSource += "\n                        write <<= (j * 4u);"
	shaderSource += "\n                        if ( zone == 0u ) {"
	shaderSource += "\n                            if ( k < 2u ) {"
	shaderSource += "\n                                write <<= (k * 16u);"
	shaderSource += "\n                                writeR |= write;"
	shaderSource += "\n                            } else {"
	shaderSource += "\n                                write <<= ((k - 2u) * 16u);"
	shaderSource += "\n                                writeG |= write;"
	shaderSource += "\n                            }"
	shaderSource += "\n                        } else {"
	shaderSource += "\n                            if ( k < 2u ) {"
	shaderSource += "\n                                write <<= (k * 16u);"
	shaderSource += "\n                                writeB |= write;"
	shaderSource += "\n                            } else {"
	shaderSource += "\n                                write <<= ((k - 2u) * 16u);"
	shaderSource += "\n                                writeA |= write;"
	shaderSource += "\n                            }"
	shaderSource += "\n                        }"
	shaderSource += "\n                    }"
	shaderSource += "\n                }"
	shaderSource += "\n            }"
	shaderSource += "\n        }"
	shaderSource += "\n        fragColor = vec4 (uintBitsToFloat(writeR), uintBitsToFloat(writeG), uintBitsToFloat(writeB), uintBitsToFloat(writeA));"
	shaderSource += "\n    }"
	shaderSource += "\n    else {"
	shaderSource += "\n        fragColor = texelFetch (iChannel%d, ivec2(fragCoord - vec2 (0.5)), 0);" % (svoNum)
	shaderSource += "\n    }"
	shaderSource += "\n}"
	shaderSource += "\n\nbool occupancyReadGrid%d (vec3 samplePos) {" % (svoNum)
	shaderSource += "\n    if ( any(lessThan(samplePos, grid%dMin)) || any(greaterThan(samplePos, grid%dMax)) ) return false;" % (svoNum, svoNum)
	shaderSource += "\n    vec3 samplePosRel = samplePos - grid%dMin;" % (svoNum)
	shaderSource += "\n    uvec3 fetchPos = uvec3 (samplePosRel);"
	shaderSource += "\n    ivec2 sampleCoord = ivec2 (fetchPos.x, fetchPos.y + uint (fetchPos.z/2u) * uint(grid%dRange.y));" % (svoNum)
	shaderSource += "\n    vec4 fetchTexel = texelFetch (iChannel%d, sampleCoord, 0);" % (svoNum)
	shaderSource += "\n    uvec4 fetchTexelRGBA = uvec4 (floatBitsToUint(fetchTexel.x), floatBitsToUint(fetchTexel.y), floatBitsToUint(fetchTexel.z), floatBitsToUint(fetchTexel.a));"
	shaderSource += "\n    uvec3 checkBits = uvec3 (fract (samplePosRel) * 4.0);"
	shaderSource += "\n    uint shiftBits = 0u;"
	shaderSource += "\n    if (checkBits.z < 2u)"
	shaderSource += "\n        shiftBits = checkBits.x + checkBits.y * 4u + checkBits.z * 16u;"
	shaderSource += "\n    else"
	shaderSource += "\n        shiftBits = checkBits.x + checkBits.y * 4u + (checkBits.z - 2u) * 16u;"
	shaderSource += "\n    uint readUint;"
	shaderSource += "\n    if ((fetchPos.z % 2u) == 0u)"
	shaderSource += "\n        if (checkBits.z < 2u)"
	shaderSource += "\n            readUint = fetchTexelRGBA.x;"
	shaderSource += "\n        else"
	shaderSource += "\n            readUint = fetchTexelRGBA.y;"
	shaderSource += "\n    else"
	shaderSource += "\n        if (checkBits.z < 2u)"
	shaderSource += "\n            readUint = fetchTexelRGBA.z;"
	shaderSource += "\n        else"
	shaderSource += "\n            readUint = fetchTexelRGBA.a;"
	shaderSource += "\n    uint maskBits = (1u << shiftBits);"
	shaderSource += "\n    if ( (readUint & maskBits) == 0u ) return false;"
	shaderSource += "\n    return true;"
	shaderSource += "\n}"

	return shaderSource

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
#shader = imageToBuffer('graffiti.png', [1.0, 1.0], [0.0, 0.0], "", 0, "r2g4b2", "nearest")
#shader = imageToBuffer('graffiti_lowres.png', [1.0, 1.0], [0.0, 0.0], shader, 1, "r2g4b2", "nearest")
#shader = textToBuffer("Yo dude!", shader)

#file = open('shaderimage.txt', 'w')
#file.write(shader)
#file.close()

# Convert the sound...

# shader = soundToShaderToy('madworld.mp3', 'madworld.wav', 1220, 1.08, 12.0) # This is for the other shadertoy... ;)
# shader = soundToShaderToy('angels.mp3', 'angels.wav', 2000, 0.8, 7.6)
# shader = soundToShaderToy('sotb1.mp3', 'sotb1.wav', 3200, 36.0, 5.1)

#file = open("shadersound.txt", "w")
#file.write(shader)
#file.close()

shader = SVOToBitstream ("bun_zipper.ply")
shader = SVOToBitstream ("dragon_vrip_res4.ply", shader, 1)
shader = SVOToBitstream ("armadillo.ply", shader, 2)
file = open("everything.svo", "w")
file.write(shader)
file.close()
