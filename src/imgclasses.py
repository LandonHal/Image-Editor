import os
import json
import zlib
from struct import unpack
from time import perf_counter
from math import floor

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
#|  xx xx xx xx <-- CRC

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

class Globals:
    Debug = None

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

class Chunk:
    def __init__(self, bin: bytearray, size):
        self._size = size
        self._type = bin[4:8]
        self._data = bin[8:-4]
        self._crc = bin[-4:]

class Image: # This is a representation of a PNG
    def __init__(self, sig: bytes = None, bin: bytes = None): 
        self._signature = sig
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

        filtData = self.__decompress(b''.join([bytes(chunk._data) for chunk in self._chunks if chunk._type == b'IDAT'])) # Concatenated bytes of idat chunk(s) 
        self._scanlines = self.__defilter(filtData)

    def update(self):
        filtered = self.__filter(self._scanlines)
        compressed = self.__compress(b''.join([line[1] for line in filtered]))
        self.__packChunks(compressed)
        
        print('Image updated')

    def export(self):
        pass
        
    @timedFunc('Unpack Chunks')
    def __unpackChunks(self, binArray: bytearray) -> int: # Unpacks given binary to chunks in image, returns bytes read
        i = 0

        while True:
            chunkSize = unpack('>I', binArray[i:i + 4])[0] + 12 # Num unpacked is size of chunk in bytes 
            self._chunks.append(Chunk(binArray[i:i + chunkSize], chunkSize))

            if Globals.Debug:
                print(f"Size of last counted chunk: {chunkSize}")
                print(f"Counted chunks: {self._chunks.__len__()}\n")

            if (self._chunks[-1]._type == b'IEND'):
                break

            i += chunkSize

        return i
    
    @timedFunc("Decompress Data")
    def __decompress(self, bin: bytes):
        filtData = bytearray(zlib.decompress(bin))

        filtSize = len(filtData)
        if self._details['dimensions'][1] * (self._details['dimensions'][0] * self._details['bpp'] + 1) != filtSize:
            raise Exception('There are more/less bytes than expected.')
        if Globals.Debug:
            print(f'Decompressed image is {filtSize} bytes, which is {filtSize - self._details['size']} bytes more than the compressed file.')

        # parses decompressed bytes into scanlines
        buffer = []
        i = 0
        for line in range(self._details['dimensions'][1]):
            bufferLine = []
            bufferLine.append(filtData[i : i + 1])
            i += 1
            bufferLine.append(filtData[i : i + self._details['dimensions'][0] * self._details['bpp']])
            i += self._details['dimensions'][0] * self._details['bpp']
            #for pxl in range(self._details['dimensions'][0]): Defunct thanks to filtering
             #   bufferLine.append(filtData[i: i + self._details['bpp']])
              #  i += self._details['bpp']
            buffer.append(bufferLine)

        if len(filtData[i:]):
            raise Exception('Failed to read all pixels')
        
        return buffer # Pixels not seperated in return value
    
    @timedFunc("Defilter Data")
    def __defilter(self, data: list[list[bytearray]]) -> list[list[list[bytearray]]]: # TODO: add support for image-wide filters
        unfilt = [] 
            
        # Filters each scanline
        for i in range(len(data)):
            buffer = [data[i][0]] # filter type byte
            match int.from_bytes(data[i][0]):
                case 0:
                    buffer.append(data[i][1])
                case 1:
                    buffer.append(Algorithms.desub(data[i][1])) # filt type is ignored ( first byte )
                case 2:
                    buffer.append(Algorithms.deup(unfilt[-1][1], data[i][1]))
                case 3:
                    buffer.append(Algorithms.deaverage(unfilt[-1][1], data[i][1]))
                case 4:
                    buffer.append(Algorithms.depaeth(unfilt[-1][1], data[i][1]))

            unfilt.append(buffer)

        # Seperates each scanline into pixels 
        bD = int(self._details['bit-depth'] / 8)

        for line in unfilt:
            buffer = []
            i = 0
            for k in range(int(len(line[1]) / self._details['bpp'])): 
                buffer.append(line[1][i + self._details['bpp'] - 1])
                i += self._details['bpp']
            line[1] = buffer

        return unfilt

    @timedFunc("Pack Chunks")
    def __packChunks():
        pass

    @timedFunc("Compress Data")
    def __compress(self, bin: bytes):
        decompData = zlib.compress(bin)

    @timedFunc("Filter Data")
    def __filter(self, data: list[list[list[bytearray]]]): # whole lotta lists
        filt: list[list[bytearray]] = []

        
        for i in range(len(data)):
            buffer = [int.from_bytes(data[i][0])] # filter type byte
            scanData = bytearray(data[i][1]) # combines pixels

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

            filters = {
                0: scanData,
                1: lambda: Algorithms.sub(scanData),
                2: lambda: Algorithms.up(filt[-1][1], scanData),
                3: lambda: Algorithms.average(filt[-1][1], scanData),
                4: lambda: Algorithms.paeth(filt[-1][1], scanData)
            }

            lens = []
            for k in filters.keys():
                compressed = zlib.compress(bytes(filters[k]))
                lens.append((len(compressed)), compressed)

            lens.sort(key = lambda x: x[0])
            filt.append(lens[0][1])

        if len(filt) is not self._details['dimensions'][1]:
            raise Exception('Failed to filter all scanlines')

        return filt 

class Algorithms:

    def sub(scan: bytearray):
        temp = scan
        i = 1
        for byte in temp[1:]: # starting with second byte in pxl data
            byte -= temp[i - 1]
            i += 1
        return temp

    def desub(scan: bytearray):
        temp = scan
        i = 1
        for byte in temp[1:]: # starting with second byte in pxl data
            byte += temp[i - 1]
            i += 1
        return temp

    def up(prevScan: bytearray, scan: bytearray):
        temp = scan
        k = 0
        for byte in temp:
            byte -= prevScan[k]
        return temp

    def deup(prevScan: bytearray, scan: bytearray):
        temp = scan
        k = 0
        for byte in temp:
            byte += prevScan[k]
        return temp

    def average(prevScan: bytearray, scan: bytearray):
        temp = scan
        i = 1
        k = 0
        for byte in temp[1:]: # starting with second byte in pxl data
            byte -= floor(temp[i - 1] + prevScan[k] / 2)
        return temp
        
    def deaverage(prevScan: bytearray, scan: bytearray):
        temp = scan
        i = 1
        k = 0
        for byte in temp[1:]: # starting with second byte in pxl data
            byte += floor(temp[i - 1] + prevScan[k] / 2)
        return temp 

    def paeth(prevScan: bytearray, scan: bytearray):
        temp = scan
        i = 1
        k = 0
        for byte in temp: # starting with second byte in pxl data
            byte -= Algorithms.__paeth(temp[i - 1], prevScan[k], prevScan[k - 1])
        return temp 

    def depaeth(prevScan: bytearray, scan: bytearray):
        temp = scan
        i = 1
        k = 0
        for byte in temp: # starting with second byte in pxl data
            byte += Algorithms.__paeth(temp[i - 1], prevScan[k], prevScan[k - 1])
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
    img.update()
