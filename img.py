import math
import numpy as np
import sys
import cv2
import os

def setup_folders(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

def mild_white_balance(originalImage):
    lab = cv2.cvtColor(originalImage, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # Color correction
    correction_strength = 0.1  
    a_corrected = a + (128 - a_mean) * correction_strength
    b_corrected = b + (128 - b_mean) * correction_strength
    
    # Clamp values to valid range
    a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    
    # Merge channels and convert back to BGR
    balanced_lab = cv2.merge([l, a_corrected, b_corrected])
    balanced_bgr = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)
    
    return balanced_bgr

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def mild_atm_light(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
 
    numpx = int(max(math.floor(imsz / 2000), 1))
    
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    
    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]
    
    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    
    A = atmsum / numpx
    
    A[0, 0] = A[0, 0] * 0.98  
    A[0, 2] = A[0, 2] * 1.02  
    return A

def mild_transmission_estimate(im, A, sz):
    omega = 0.95 - 0.03 * (np.mean(A) / 255) 
    im3 = np.empty(im.shape, im.dtype)
    
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    
    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission

# Guided filter
def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * im + mean_b
    return q

def transmission_refine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 20  
    eps = 0.005 
    t = guided_filter(gray, et, r, eps)
    return t

def mild_recover(im, t, A, tx=0.4):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx) 

    red_coeff = 1.02   
    green_coeff = 1.0  
    blue_coeff = 0.98  
    
    res[:, :, 0] = ((im[:, :, 0] - A[0, 0]) / t + A[0, 0]) * red_coeff
    res[:, :, 1] = ((im[:, :, 1] - A[0, 1]) / t + A[0, 1]) * green_coeff
    res[:, :, 2] = ((im[:, :, 2] - A[0, 2]) / t + A[0, 2]) * blue_coeff
    
    # Clamp pixel values to avoid overexposure
    res = np.clip(res, 0, 1)
    return res

def mild_contrast_enhance(im):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(20, 20))  # Lower clipLimit and larger grid size
    y_enhanced = clahe.apply(y)
    
    # Merge channels
    ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
    bgr_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)
    
    kernel = np.array([[0, -0.1, 0], [-0.1, 1.4, -0.1], [0, -0.1, 0]], np.float32)
    sharpened = cv2.filter2D(bgr_enhanced, -1, kernel)
    
    return sharpened

# Main processing pipeline
def process_underwater_image(img_path, output_path):
    # Read image
    src = cv2.imread(img_path)
    if src is None:
        print(f"Failed to read image: {img_path}")
        return
    
    # Step 1: White balance correction
    balanced = mild_white_balance(src)
    
    # Step 2: Dehazing processing
    I = balanced.astype('float64') / 255
    dark = dark_channel(I, 8)  
    A = mild_atm_light(I, dark)
    te = mild_transmission_estimate(I, A, 8)
    t = transmission_refine(balanced, te)
    dehazed = mild_recover(I, t, A, 0.4)  

    dehazed_8u = (dehazed * 255).astype(np.uint8)
    
    enhanced = mild_contrast_enhance(dehazed_8u)
    
    enhanced = np.clip(enhanced * 0.99, 0, 255).astype(np.uint8) 
    
    # Save result
    cv2.imwrite(output_path, enhanced)
    print(f"Processing completed: {output_path}")

if __name__ == '__main__':
    try:
        input_dir = sys.argv[1]
    except:
        input_dir = r" "  # input path
    
    output_dir = r" "  # output path
    
    # Set up folders
    setup_folders(input_dir, output_dir)
    
    # Process all images
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"mild_enhanced_{img_file}")
            process_underwater_image(img_path, output_path)
    
    print("All image processing completed!")
    