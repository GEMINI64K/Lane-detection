# Lane-detection
a simple procedural lane detection program written in python

Original Frame
Image:![Screenshot 2025-05-09 134157](https://github.com/user-attachments/assets/da0f8227-5463-4f5f-8132-1d22003dcd93)
This is the raw input video frame, resized for display. It serves as the starting point for all further image processing.

Canny Edge Detection
Image: ![Screenshot 2025-05-09 134240](https://github.com/user-attachments/assets/98803637-96fe-4ba3-9e22-2137a1c8470a)
Applies the Canny edge detector to identify strong edges in the image, which are likely to correspond to lane markings or road boundaries.

Masked Output
Image: ![Screenshot 2025-05-09 134328](https://github.com/user-attachments/assets/f55a9710-632d-4998-bc20-8746bcd03a00)
Applies a region of interest (ROI) mask to keep only the part of the image where lanes are likely to appear (usually the lower half or trapezoid area). 
This removes irrelevant edges outside the driving area.

Warped Binary Image (Bird’s Eye View)
Image: ![Screenshot 2025-05-09 134410](https://github.com/user-attachments/assets/232b1a60-d9f9-4d07-880a-48f3184502a1)
Transforms the masked image into a top-down (bird’s eye) view using a perspective transform. This helps to make lane lines appear more parallel and easier to detect.

Final Lane Detection Result
Image: ![Screenshot 2025-05-09 134441](https://github.com/user-attachments/assets/df2692fb-20e5-44c9-8a0d-9fd7a90c1218)
Displays the original image with detected lane lines overlaid. This is the final visual output for the driver or system.
