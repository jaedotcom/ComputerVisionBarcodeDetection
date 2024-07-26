# Built in packages
import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import imageIO.png

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    

def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)

    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    
    return new_array


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            greyscale_pixel_array[y][x] = round(0.299*pixel_array_r[y][x] + 0.587 *pixel_array_g[y][x] + 0.114 *pixel_array_b[y][x])
    
    return greyscale_pixel_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    output_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min_value, max_value = computeMinAndMaxValues(pixel_array, image_width, image_height)

    if min_value == max_value:
        return output_pixel_array
    
    for i in range(image_height):
        for j in range(image_width):
            pixel_value = pixel_array[i][j]
            scaled_value = int((pixel_value - min_value) / (max_value - min_value) * 255 + 0.5)
            quantized_value = max(0, min(scaled_value, 255))
            output_pixel_array[i][j] = quantized_value
    
    return output_pixel_array
    

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = float('inf')  
    max_value = float('-inf')  

    for i in range(image_height):
        for j in range(image_width):
            pixel_value = pixel_array[i][j]
            if pixel_value < min_value:
                min_value = pixel_value
            if pixel_value > max_value:
                max_value = pixel_value
    
    return min_value, max_value
    

def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            gradient = (
                (-1.0 * (pixel_array[y-1][x-1] + pixel_array[y+1][x-1])) + (-2.0 * (pixel_array[y][x-1])) + 
                (pixel_array[y-1][x+1] + pixel_array[y+1][x+1]) + (2.0 * (pixel_array[y][x+1]))
            )
            final_gradient = gradient/8.0
            output[y][x] = abs(final_gradient)
    
    for i in range(image_height):
        for j in range(image_width):
            if (output[i][j] == 0):
                output[i][j] = 0.0
    
    return output


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height-1):
        for x in range(1, image_width-1):
            gradient = (
                (pixel_array[y-1][x-1] + pixel_array[y-1][x+1]) + (2 * (pixel_array[y-1][x])) +
                (-1 * (pixel_array[y+1][x-1] + pixel_array[y+1][x+1])) + (-2 * (pixel_array[y+1][x]))
                )
            final_gradient = gradient/8.0
            output[y][x] = abs(final_gradient)
    output = [[0.0 if pixel == 0 else pixel for pixel in row] for row in output]
    return output


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)

    kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    kernel_sum = 16 

    for y in range(image_height):
        for x in range(image_width):
            total = 0.0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    px = min(max(x + kx, 0), image_width - 1)
                    py = min(max(y + ky, 0), image_height - 1)
                    pixel_value = pixel_array[py][px]
                    kernel_value = kernel[ky + 1][kx + 1]
                    total += pixel_value * kernel_value

            result_array[y][x] = total / kernel_sum

    return result_array


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    dilated_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] > 0:
                dilated_image[y][x] = 1
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if 0 <= y + dy < image_height and 0 <= x + dx < image_width:
                            dilated_image[y + dy][x + dx] = 1
    
    return dilated_image


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    eroded_image = createInitializedGreyscalePixelArray(image_width, image_height)
    structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] > 0:
                eroded = True
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        se_x = x + j
                        se_y = y + i

                        if 0 <= se_x < image_width and 0 <= se_y < image_height:
                            if pixel_array[se_y][se_x] == 0:
                                eroded = False
                                break  
                        else:
                            eroded = False
                            break  

                if eroded:
                    eroded_image[y][x] = 1

    return eroded_image


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    
    new_queue = Queue()
    dict_ = dict();
    group = 1
    group_count = 0
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] != 0:
                pixel_array[y][x] = 255;
    p_array = [[0 for x in range(image_width+2)] for y in range(image_height+2)]
    
    for y in range(image_height):
        for x in range(image_width):
            p_array[y+1][x+1] = pixel_array[y][x]
    
    for y in range(1, image_height+1):
        for x in range(1, image_width+1):
            if p_array[y][x] == 255:
                p_array[y][x] = group
                group_count += 1
                
                new_queue.enqueue([y,x])
                
                while new_queue.isEmpty() == False:
                    poppedout = new_queue.dequeue()
                    
                    if p_array[poppedout[0]][poppedout[1]+1] == 255:
                        new_queue.enqueue([poppedout[0],poppedout[1]+1])
                        p_array[poppedout[0]][poppedout[1]+1] = group
                        group_count +=1
                        
                    if p_array[poppedout[0]][poppedout[1]-1] == 255:
                        new_queue.enqueue([poppedout[0],poppedout[1]-1])
                        p_array[poppedout[0]][poppedout[1]-1] = group
                        group_count +=1
                        
                    if p_array[poppedout[0]+1][poppedout[1]] == 255:
                        new_queue.enqueue([poppedout[0]+1,poppedout[1]])
                        p_array[poppedout[0]+1][poppedout[1]] = group
                        group_count +=1
                        
                    if p_array[poppedout[0]-1][poppedout[1]] == 255:
                        new_queue.enqueue([poppedout[0]-1,poppedout[1]])
                        p_array[poppedout[0]-1][poppedout[1]] = group
                        group_count +=1
                        
                dict_[group] = group_count
                group+=1
                group_count = 0
    
    for y in range(1, image_height+1):
        for x in range(1, image_width+1):
            pixel_array[y-1][x-1] = p_array[y][x]
    
    return (pixel_array, dict_)      


def getKey(val, dict_):
    for key, value in dict_.items():
        if val == value:
            return key     


def findBorder(group_array, dict_, image_height, image_width):
    big_num = None
    min_x, max_x, min_y, max_y = None, None, None, None

    for key in dict_:
        if big_num == None:
            big_num = dict_[key]
        if dict_[key] > big_num:
            big_num = dict_[key]  

    big_group = getKey(big_num, dict_)

    for y in range(image_height):
        for x in range(image_width):
            if group_array[y][x] == big_group:
                if min_x == None:
                    min_x, max_x, min_y, max_y = x,x,y,y
                if y > max_y:
                    max_y = y
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x

    return [min_x,max_x,min_y,max_y]


def separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]

    return new_array    
        

def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    filename = "Barcode2"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():

        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
 
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    vertical_sobel = computeVerticalEdgesSobelAbsolute(px_array, image_width, image_height)
    horizontal_sobel = computeHorizontalEdgesSobelAbsolute(px_array, image_width, image_height)
    
    for height in range(image_height):
        for width in range(image_width):
            px_array[height][width] = abs(abs(horizontal_sobel[height][width]) - abs(vertical_sobel[height][width]))
    
    for _ in range(5):
        px_array = computeGaussianAveraging3x3RepeatBorder(px_array, image_width, image_height)

    px_array = computeThresholdGE(px_array, 25, image_width, image_height)
   
    for _ in range(3):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    for _ in range(6):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    groupDict = computeConnectedComponentLabeling(px_array, image_width, image_height)
    group_array, dict_ = groupDict[0], groupDict[1]

    minMax_vals = findBorder(group_array, dict_, image_height, image_width)

    px_array = separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    bbox_min_x,bbox_max_x,bbox_min_y,bbox_max_y = minMax_vals[0],minMax_vals[1],minMax_vals[2],minMax_vals[3]


    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()