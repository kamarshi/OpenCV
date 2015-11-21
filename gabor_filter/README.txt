Looking at Gabor filter properties.  Invocation:
cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma,  psi, ktype)

Understanding so far:
  ksize - kernel size in pixels
  sigma - appears to be peak-to-peak size
  theta - is the orientation of filter
  lambd - relates to spatial frequency of filter
  gamma - extent of filter in the window - smaller means longer
  psi   - appears to attenuate one side of the filter
