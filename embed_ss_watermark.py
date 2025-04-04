from __future__ import print_function
import argparse
import numpy as np
import scipy.fftpack
import time

import cv2
   
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho').astype(np.uint8)
 
def dct3(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')

def idct3(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho').astype(np.uint8)
 
def read_yuv_next_frame(file, width, height, type=np.uint8):
    # YUV 420 -> Y has 4 times info of U or V
    y_size = width * height
    u_size = (width * height) // 4
    v_size = (width * height) // 4
    
    y = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width)).astype(type)
    u = np.frombuffer(file.read(u_size), dtype=np.uint8).reshape((height//2, width//2)).astype(type)
    v = np.frombuffer(file.read(v_size), dtype=np.uint8).reshape((height//2, width//2)).astype(type)
    
    return y, u, v
    
def read_yuv_all(file, width, height, amount_of_frames, type=np.uint8):
    y = np.empty((amount_of_frames, height, width), dtype=type)
    u = np.empty((amount_of_frames, height//2, width//2), dtype=type)
    v = np.empty((amount_of_frames, height//2, width//2), dtype=type)

    for i in range(amount_of_frames):
        #print("Reading frame %d" % i)
        y[i], u[i], v[i] = read_yuv_next_frame(file, width, height, type=type)
        
    return y, u, v
    
def read_265_next_frame(file, width, height, type=np.uint8):
    # Assume file = cv2.VideoCapture(filename)
    ret, frame = file.read()
    if not ret: print("Something went wrong while reading file %s: %s" % (file, ret))
    
    # Still not same as ffmpeg decode.. :(
    frame_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV_I420)
    y = frame_yuv[:height,:]
    u = frame_yuv[height:height+height/4,:].reshape((height//2, width//2))
    v = frame_yuv[height+height/4:,:].reshape((height//2, width//2))
    
    #frame_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    #y = frame_yuv[:,:,0].astype(type)
    #u = cv2.resize(frame_yuv[:,:,1],(width//2, height//2), fx=0, fy=0, interpolation = cv2.INTER_NEAREST).astype(type)
    #v = cv2.resize(frame_yuv[:,:,2],(width//2, height//2), fx=0, fy=0, interpolation = cv2.INTER_NEAREST).astype(type)
    
    return y, u, v
    
def read_265_all(file, width, height, amount_of_frames, type=np.uint8):
    y = np.empty((amount_of_frames, height, width), dtype=type)
    u = np.empty((amount_of_frames, height//2, width//2), dtype=type)
    v = np.empty((amount_of_frames, height//2, width//2), dtype=type)

    for i in range(amount_of_frames):
        #print("Reading frame %d" % i)
        y[i], u[i], v[i] = read_265_next_frame(file, width, height, type=type)
        
    return y, u, v
        
def read_next_frame(file, width, height, type=np.uint8, is_yuv=True):
    if is_yuv:
        return read_yuv_next_frame(file, width, height, type=type)
    else:
        return read_265_next_frame(file, width, height, type=type)
        
def read_all(file, width, height, amount_of_frames, type=np.uint8, is_yuv=True):
    if is_yuv:
        return read_yuv_all(file, width, height, amount_of_frames, type=type)
    else:
        return read_265_all(file, width, height, amount_of_frames, type=type)
        
def write_yuv_frame(file, y, u, v, width, height):
    file.write(np.getbuffer(y.reshape(width * height)))
    file.write(np.getbuffer(u.reshape((width * height)//4)))
    file.write(np.getbuffer(v.reshape((width * height)//4)))
    
def write_yuv_all(file, y, u, v, width, height, amount_of_frames):
    for i in range(amount_of_frames):
        file.write(np.getbuffer(y[i].reshape(width * height)))
        file.write(np.getbuffer(u[i].reshape((width * height)//4)))
        file.write(np.getbuffer(v[i].reshape((width * height)//4)))

def generate_gaussian_sequence(watermark_size, seed):
    mu = 0
    sigma = 1
    np.random.seed(seed)
    return np.sign(np.random.normal(mu, sigma, watermark_size))

# Find largest N elements
def find_largest_coeffs(coeffs, watermark_size, skip_coefficients=1):
    coeffs = np.abs(coeffs)
    if skip_coefficients > 0:
        start_coeff_index = skip_coefficients
    else:
        start_coeff_index = len(coeffs) + skip_coefficients
    max_indices = np.argpartition(coeffs[skip_coefficients:], -watermark_size)[-watermark_size:] + start_coeff_index # Skip DC, get max indices
    max_indices_sorted = max_indices[np.argsort(coeffs[max_indices])[::-1]] # Sort the max indices on value
    #print(max_indices_sorted)
    #print(coeffs[max_indices_sorted])
    return max_indices_sorted

# Function changes N largest elements 
def find_and_change_largest_coeffs_not_dc_fast(coeffs, watermark_size, alpha, watermark_seq, skip_coefficients): 
    new_coeffs = np.copy(coeffs)
  
    # max or max(absolute)
    max_indices_sorted = find_largest_coeffs(coeffs, watermark_size, skip_coefficients)
    
    for i, (watermark_bit, max_index_i) in enumerate(zip(watermark_seq, max_indices_sorted)):
        max_coeff = new_coeffs[max_index_i]
        
        # Change in new_coeffs
        new_coeffs[max_index_i] = max_coeff * (1 + watermark_bit * alpha)
        #print("Changed coeff nr %d from %f to %f" % (max_index_i, max_coeff, new_coeffs[max_index_i]))        
        
    return new_coeffs
    
   
def embed_ss2_watermark_frame(y, watermark_seq, watermark_size, alpha, width, height, skip_coefficients):
    dct_vector = dct2(y).reshape(width * height)
    
    watermarked_dct_vector = find_and_change_largest_coeffs_not_dc_fast(dct_vector, watermark_size, alpha, watermark_seq, skip_coefficients)
    #watermarked_dct_vector = dct_vector
    
    watermarked_y = idct2(watermarked_dct_vector.reshape(height, width))
    
    return watermarked_y

def embed_ss2_watermark(input, output, width, height, amount_of_frames, watermark_size, alpha, seed, skip_coefficients):
    # Open files
    file = open(input, 'rb')
    output_file = open(output, 'wb')
    
    watermark_seq = generate_gaussian_sequence(watermark_size, seed)
    
    try:
        # Process frame per frame
        for frame in range(amount_of_frames):
            y, u, v = read_yuv_next_frame(file, width, height)
            
            y_watermarked = embed_ss2_watermark_frame(y, watermark_seq, watermark_size, alpha, width, height, skip_coefficients)
            
            #print("Processed frame %d" % frame)
            
            write_yuv_frame(output_file, y_watermarked, u, v, width, height)
    finally:
        file.close()
        output_file.close()
    
    
def embed_ss3_watermark_all(y, watermark_seq, watermark_size, alpha, width, height, amount_of_frames, skip_coefficients):
    dct_vector = dct3(y).reshape(width * height * amount_of_frames)
    
    watermarked_dct_vector = find_and_change_largest_coeffs_not_dc_fast(dct_vector, watermark_size, alpha, watermark_seq, skip_coefficients)
    #watermarked_dct_vector = dct_vector
    
    watermarked_y = idct3(watermarked_dct_vector.reshape(amount_of_frames, height, width))
    
    return watermarked_y
    
def embed_ss3_watermark(input, output, width, height, amount_of_frames, watermark_size, alpha, seed, skip_coefficients):
    # Open files
    file = open(input, 'rb')
    output_file = open(output, 'wb')
    
    watermark_seq = generate_gaussian_sequence(watermark_size, seed)
    
    try:
        y, u, v = read_yuv_all(file, width, height, amount_of_frames)
        
        y_watermarked = embed_ss3_watermark_all(y, watermark_seq, watermark_size, alpha, width, height, amount_of_frames, skip_coefficients)
        
        write_yuv_all(output_file, y_watermarked, u, v, width, height, amount_of_frames)
    finally:
        file.close()
        output_file.close()

#####################################################
# Main function
#####################################################
if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Embed SS watermark.')
    parser.add_argument('-i','--input', type=str, help='Input', required=True)
    parser.add_argument('-o','--output', type=str, help='Output', required=True)
    parser.add_argument('-w','--width', type=int, help='width', required=True)
    parser.add_argument('-hh','--height', type=int, help='height', required=True)
    parser.add_argument('-ws','--watermarksize', type=int, help='watermark size', required=True)
    parser.add_argument('-s','--seed', type=int, help='seed', required=True)
    parser.add_argument('-a','--alpha', type=float, help='alpha', required=True)
    parser.add_argument('-f','--frames', type=int, help='amount of frames', required=True)
    parser.add_argument('-t','--type', default='ss3', type=str, help='type', required=False)
    parser.add_argument('-sc','--skipcoefficients', default=1, type=int, help='number of coefficients to skip for embedding (1)', required=False)
    
    args = vars(parser.parse_args())
    
    width = args['width']
    height = args['height']
    amount_of_frames = args['frames']
    
    watermark_size = args['watermarksize']
    alpha = args['alpha']
    seed = args['seed']
    skip_coefficients = args['skipcoefficients']
    
    input = args['input']
    output = args['output']
    
    type = args['type']
    start_time = time.time()
    if type == 'ss' or type == 'ss2':
        embed_ss2_watermark(input, output, width, height, amount_of_frames, watermark_size, alpha, seed, skip_coefficients)
    elif type == 'ss3':
        embed_ss3_watermark(input, output, width, height, amount_of_frames, watermark_size, alpha, seed, skip_coefficients)
    else: 
        print("Type %s not supported" % type)
    
    elapsed_time = time.time() - start_time
    print("Elapsed time: %f" % elapsed_time)
    