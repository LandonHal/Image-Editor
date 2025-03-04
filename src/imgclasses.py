import os
import json
import zlib
from struct import unpack
from time import perf_counter
from math import floor
import matplotlib.pyplot as plt
import numpy 

# PNG hex representation ------------------
#|  -----------     PNG header  |   8 Bytes
#|  89 50 4e 47 0d 0a 1a 0a
#|  
#|  * <-- Chunks
#|
#|  End of File

#|  -----------     Chunk   |   12 Bytes + Chunk Data
#|  xx xx xx xx <-- Num of bytes in chunk data (unsigned int)
#|  xx xx xx xx <-- 4-byte chunk type code (ascii)
#|  * <-- Chunk data
#|  xx xx xx xx <-- CRC (should probably implement this)

#|  -----------     IHDR Image Header Chunk Data   |   13 Bytes 
#|  xx xx xx xx <-- Width in pixels
#|  xx xx xx xx <-- Height in pixels
#|  xx <-- Bit Depth
#|  xx <-- Color Type
#|  xx <-- Compression Method
#|  xx <-- Filter Method
#|  xx <-- Interlace Method

# All integers larger than one byte are big-endian encoded
# An additional "filter-type" byte is added to the beginning of every scanline. The filter-type byte is not considered part of the image data, but it is included in the datastream sent to the compression step.

# As it stands, this project does not work as the outputted image is blank. 

class Globals:
    Debug = None

# Time them functions
def timedFunc(step: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if Globals.Debug:
                print(f'Step {step} started')
                start = perf_counter()
                out = func(*args, **kwargs)
                print(f'Step {step} elapsed in {perf_counter() - start} seconds')
            else:
                out = func(*args, **kwargs)
            return out
        return wrapper
    return decorator

class Scanline:
    def __init__(self, filtertype: int, data: bytearray | list[tuple[int]]):
        self._filter = filtertype
        self._data = data # This is an array of tuple[int], or a bytearray depending on program progress

class Chunk:
    def __init__(self, bin: bytearray, size):
        self._size = size
        self._type = bin[4:8]
        self._data = bin[8:-4]
        self._crc = bin[-4:]

class Image: # This is a representation of a PNG. _chunks is compressed data, _scanlines is decompressed data
    def __init__(self, sig: bytes = None, bin: bytes = None): 
        self._signature = sig
        print(sig)
        self._chunks: list[Chunk] = list()
        self._scanlines: list[list[bytes]] = list()
        self._details = {
            "size": None,
            "dimensions": None,
            "bit-depth": None,
            "color-type": None,
            "compression-method": None,
            "filter-method": None,
            "interlace-method": None
        }

        if bin is not None:
            self.populate(bin)

    def details(self):
        for k in self._details.keys():
            print(f'Image with {k}: {self._details[k]}')
    
    def populate(self, bin: bytes):
        # populates image data given binary data
        self._details['size'] = self.__unpackChunks(bytearray(bin))
        self._details['dimensions'] = (unpack('>I', self._chunks[0]._data[0:4])[0], unpack('>I', self._chunks[0]._data[4:8])[0]) # Width, length
        self._details['bit-depth'] = self._chunks[0]._data[8]
        self._details['color-type'] = self._chunks[0]._data[9]
        self._details['compression-method'] = self._chunks[0]._data[10]
        self._details['filter-method'] = self._chunks[0]._data[11]
        self._details['interlace-method'] = self._chunks[0]._data[12]
        match self._details['color-type']:
            case 6:
                self._details['bpp'] = 4 * int(self._details['bit-depth'] / 8)
            case 4:
                self._details['bpp'] = 2 * int(self._details['bit-depth'] / 8)
            case 2:
                self._details['bpp'] = 3 * int(self._details['bit-depth'] / 8)
            case 0:
                self._details['bpp'] = 1 * int(self._details['bit-depth'] / 8)

        if self._details['color-type'] == 3 :
            raise Exception('Color type not supported')
        if not (self._details['bit-depth'] == 8 or self._details['bit-depth'] == 16):
            raise Exception('Bit depth is not supported')
        if not (self._details['interlace-method'] == 0):
            raise Exception('Interlacing is not supported')

        raw = b''.join([bytes(chunk._data) for chunk in self._chunks if chunk._type == b'IDAT'])
        print(f'Raw IDAT is {len(raw)} bytes')
        filtData = self.__decompress(raw) # Concatenated bytes of idat chunk(s) 
        self._scanlines = self.__defilter(filtData)

    def update(self):
        filtered = self.__filter(self._scanlines)
        compressed = self.__compress(filtered)
        self.__packChunks(compressed)
        
        print('Image updated')

    def export(self, name: str):
        # this creates the png file
        with open(os.path.join(os.path.dirname(__file__)[0:-3], name), 'wb') as fp:
            fp.write(self._signature)

            for chunk in self._chunks:
                print(f"Writing chunk: {chunk._type}, size of data: {len(chunk._data)}")
                fp.write(chunk._size.to_bytes(4, 'big'))
                fp.write(chunk._type)
                fp.write(chunk._data)
                fp.write(chunk._crc)

            fp.close()
        
    @timedFunc('Unpack Chunks')
    def __unpackChunks(self, binArray: bytearray) -> int: # Unpacks given binary to chunks in image, returns bytes read
        i = 0

        while True:
            chunkSize = unpack('>I', binArray[i:i + 4])[0] + 12 # Num unpacked is size of chunk in bytes 
            self._chunks.append(Chunk(binArray[i:i + chunkSize], chunkSize))

            if Globals.Debug:
                print(f"Size of last counted chunk: {chunkSize}")
                print(f'Chunk type: {self._chunks[-1]._type}')
                print(f"Counted chunks: {self._chunks.__len__()}\n")

            if (self._chunks[-1]._type == b'IEND'):
                break

            i += chunkSize

        return i
    
    @timedFunc("Decompress Data")
    def __decompress(self, bin: bytes) -> list[Scanline]: # Decompresses idat data, and constructs into scanlines 
        filtData = bytearray(zlib.decompress(bin))

        filtSize = len(filtData)
        if self._details['dimensions'][1] * (self._details['dimensions'][0] * self._details['bpp'] + 1) != filtSize:
            raise Exception('There are more/less bytes than expected.')
        if Globals.Debug:
            print(f'Decompressed image is {filtSize} bytes, which is {filtSize - self._details['size']} bytes more than the compressed file.')

        # Parses decompressed bytes into scanlines
        buffer = []
        i = 0
        for scanline in range(self._details['dimensions'][1]):
            filtType = filtData[i : i + 1] # filter type byte
            i += 1
            scanData = filtData[i : i + self._details['dimensions'][0] * self._details['bpp']] # scanline data
            i += self._details['dimensions'][0] * self._details['bpp']
            #for pxl in range(self._details['dimensions'][0]): Defunct thanks to filtering
             #   bufferLine.append(filtData[i: i + self._details['bpp']])
              #  i += self._details['bpp']
            buffer.append(Scanline(int.from_bytes(filtType), scanData)) # At this level, scanline data is a single bytearray

        for scanline in buffer:
            if len(scanline._data) != len(buffer[0]._data):
                raise Exception('Failed to read all pixels')
        if len(filtData[i:]): #data remains
            raise Exception('Failed to read all pixels')
        
        return buffer # in form list[Scanline]
    
    @timedFunc("Defilter Data")
    def __defilter(self, data: list[Scanline]) -> list[Scanline]: # TODO: add support for image-wide filters
        unfilt = data
            
        for i, scanline in enumerate(data):
            print(i+1, scanline._filter)
            match (scanline._filter):
                case 0:
                    unfilt[i]._data = (scanline._data)
                case 1:
                    unfilt[i]._data = (Algorithms.desub(scanline._data))
                case 2:
                    unfilt[i]._data = (Algorithms.deup(unfilt[-1]._data, scanline._data)) 
                case 3:
                    unfilt[i]._data = (Algorithms.deaverage(unfilt[-1]._data, scanline._data))
                case 4:
                    unfilt[i]._data = (Algorithms.depaeth(unfilt[-1]._data, scanline._data))

        debugimg = [scanline._data for scanline in unfilt]

        plt.imshow(numpy.array(debugimg).reshape((self._details['dimensions'][1], self._details['dimensions'][0], 4)))
        plt.show()
        plt.imshow

        # Seperates each scanline into pixels 
        for l, line in enumerate(unfilt):
            buffer = []
            i = 0

            for k in range(int(len(line._data) / self._details['bpp'])): # for each pixel
                px = line._data[i : i + self._details['bpp']]
                buffer.append(tuple(px))
                i += self._details['bpp']

            unfilt[l]._data = buffer



        return unfilt

    @timedFunc("Pack Chunks")
    def __packChunks(self, bytes):

        i = 0
        for chunk in self._chunks:
            if chunk._type == b'IDAT':
                if i == 0:
                    chunk._data = bytes
                else:
                    self._chunks.remove(chunk)

            chunk._size = len(chunk._data)

            i += 1

        #update IHDR chunk
        chunkIHDR = next(chunk for chunk in self._chunks if chunk._type == b'IHDR')

        chunkIHDR._data[0:4] = self._details['dimensions'][0].to_bytes(4, 'big')
        chunkIHDR._data[4:8] = self._details['dimensions'][1].to_bytes(4, 'big')
        chunkIHDR._data[8] = self._details['bit-depth']
        chunkIHDR._data[9] = self._details['color-type']
        chunkIHDR._data[10] = self._details['compression-method']
        chunkIHDR._data[11] = self._details['filter-method']
        chunkIHDR._data[12] = self._details['interlace-method']

    @timedFunc("Compress Data")
    def __compress(self, bin: bytes):
        decompData = zlib.compress(bin)
        if Globals.Debug:
            print(f'Compressed image is {len(decompData)} bytes, which is {len(bin) - len(decompData)} bytes less than the decompressed file.')
            return decompData

    @timedFunc("Filter Data")
    def __filter(self, data: list[list[list[bytearray]]]): # whole lotta lists
        filt: list[list[bytearray]] = []
        
        for i in range(len(data)):
            buffer = []
            scanData = bytearray(data[i][1]) # combines pixels in scanline

            """match int.from_bytes(data[i][0]): Legacy code
                case 0:
                    buffer.append(scanData)
                case 1:
                    buffer.append(Algorithms.sub(scanData)) # filt type is ignored ( first byte )
                case 2:
                    buffer.append(Algorithms.up(filt[-1][1], scanData))
                case 3:
                    buffer.append(Algorithms.average(filt[-1][1], scanData))
                case 4:
                    buffer.append(Algorithms.paeth(filt[-1][1], scanData))"""

            # Brute force filter selection. This is the official way to do it.
            if i == 0:
                buffer.append(b'\x00')
                buffer.append(scanData)
                filt.append(buffer)
                continue

            filters = {
                0: scanData,
                1: Algorithms.sub(scanData),
                2: Algorithms.up(filt[-1][1], scanData),
                3: Algorithms.average(filt[-1][1], scanData),
                4: Algorithms.paeth(filt[-1][1], scanData)
            }

            lens = []
            for k in filters.keys():
                filtered = filters[k]
                compressed = bytearray(zlib.compress(filtered))
                lens.append((len(compressed), k))

            lens.sort(key = lambda x: x[0]) # sort to find shortest compression data
            buffer.append(lens[0][1].to_bytes(1, 'big'))
            buffer.append(bytes(zlib.compress(filters[lens[0][1]])))
            filt.append(buffer)


        if len(filt) is not self._details['dimensions'][1]:
            raise Exception('Failed to filter all scanlines')
        return b''.join(sum(filt, []))

class Algorithms:

    def desub(scan: bytearray):
        temp = scan
        for i, byte in enumerate(scan):
            out = byte + temp[i - 1] #if temp[i - 1] exists else 0 <---pseudocode for later
            temp[i] = out & 0xff
        return temp

    def deup(prevScan: bytearray, scan: bytearray):
        temp = scan
        for i, byte in enumerate(scan):
            out = byte + prevScan[i]
            temp[i] = 0 #out & 0xff
        return temp

    def deaverage(prevScan: bytearray, scan: bytearray):
        temp = scan
        for i, byte in enumerate(scan):
            out = byte + floor(temp[i - 1] + prevScan[i] / 2)
            temp[i] = 0 #out & 0xff
        return temp

    def depaeth(prevScan: bytearray, scan: bytearray):
        temp = scan
        for i, byte in enumerate(scan):
            out = byte + Algorithms.__paeth(temp[i - 1], prevScan[i], prevScan[i - 1])
            temp[i] = 0 #out & 0xff
        return temp

    def __paeth(a, b, c):
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
                Pr = a
        elif pb <= pc:
            Pr = b
        else:
                Pr = c
        return Pr


with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as config:
    Globals.Debug = json.load(config)["debug"]

with open("C:\\Users\\halcombl2\\Desktop\\Coding II\\Image Editor\\Dice.png", 'rb') as fp:
    stream = fp.read()
    img = Image(bytearray(stream)[0:8], bytearray(stream)[8:])
img.details()
#img.update()
#img.export("gunk.png")
